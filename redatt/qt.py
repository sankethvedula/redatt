"""QT adapter: builds the scanner-specific inputs of the estimator from one QT exam directory.

Uses `qt_detgc` (numpy only), which reproduces the scanner's time-gain ramps, filter banks and demodulation table from
conf/unit<k>.xml.  A copy is bundled in redatt/_vendor and is used unless an installed qt_detgc is found.  Everything
else in redatt is scanner-agnostic.

    probe = QTProbe("/path/to/QT007-S1-BL", unit=1)               # stepped demodulation (the scanner's table)
    frame = probe.frame(view=5, level=16)                          # redatt.Frame
    est = probe.estimator()                                        # Estimator with QT noise and filter limits

Per stored sample m of a line the adapter computes
  tgc_db[m]        analog + digital time-gain (amplitude dB), exactly as played back
  phase[m]         accumulated demodulation phase at the output's filter centre (stepped table, phase-continuous;
                   or 2 pi demod_freq t for demod='fixed')
  t[m]             time of that filter centre (s), for the common carrier
  filter_db[b, m]  band-averaged power response of the active filter bank at band b (capped `cap_db` below the
                   bank's peak), and limited[b, m] where that response is more than `cap_db` down
  depth_mm[m], beam_spacing_mm[m] from QT's reflection parameters (nt_iq_shift, sample_rate, water temperature,
                   rad_trans, element_width); temperature='first' uses the first value of temperature.dat as QT's
                   own pipelines do, 'mean' its mean
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .config import Config
from .frontend import band_envelopes, rereference, stepped_phase, undo_gain
from .pipeline import Estimator, Frame

DATA = Path(__file__).resolve().parent / "data"


def _qt_detgc():
    """The qt_detgc package: an installed one if present, else the copy bundled in redatt/_vendor."""
    try:
        import qt_detgc as q
        from qt_detgc import fir, hardware
    except ImportError:
        from ._vendor import qt_detgc as q
        from ._vendor.qt_detgc import fir, hardware
    q.fir_module, q.hardware_module = fir, hardware
    return q


def _param(text: str, key: str) -> float:
    return float(re.search(rf"\b{key}\s*=\s*([0-9.eE+-]+)", text).group(1))


class QTProbe:
    def __init__(self, exam_dir, unit: int = 1, cfg: Config | None = None, demod: str = "stepped",
                 f_ref_hz: float = 3.5e6, cap_db: float = 15.0, temperature: str = "first"):
        qt_detgc = _qt_detgc()
        _fir, hw, gain_curve = qt_detgc.fir_module, qt_detgc.hardware_module, qt_detgc.gain_curve
        self.exam = Path(exam_dir); self.unit = unit
        self.cfg = cfg if cfg is not None else Config(block_samples=49)
        self.scan = qt_detgc.Scan(str(self.exam))
        acq = self.scan.config(unit)
        n = acq.samples_per_beam(); self.n = n
        curve = gain_curve(acq, n, fir_mode="pulse")
        self.tgc_db = np.asarray(curve.tgc_db, float)                  # analog + digital, amplitude dB
        self.bank = np.asarray(curve.fir_bank)                          # active filter bank per stored sample
        taps = np.stack([_fir.bank_taps(r) for r in acq.fir_rows])      # [32, 128] at 40 MHz
        gd = (taps.shape[1] - 1) / 2.0
        crop = int(acq.start_depth / hw.delta_rx(acq.c) + acq.rx_buffer + hw.SYSTEM_DELAY_OFFSET * acq.decimation)
        raw = crop + acq.decimation * np.arange(n) - gd                 # filter centre of each output, raw samples
        self.t = raw / hw.FS_RX
        recon = (self.exam / "conf" / "reflection_params.txt").read_text()
        self.demod_freq = _param(recon, "demod_freq") * 1e6            # QT's fixed value (3.6 MHz)
        if demod == "stepped":
            d = [0.0] + list(acq.tfc_depths)
            ints = [int(2 * (d[i + 1] - d[i]) * hw.FS_RX / acq.c) for i in range(len(d) - 1)]
            self.phase = stepped_phase(acq.demod_freqs, ints, hw.FS_RX, raw)
            f_bank = np.asarray(acq.demod_freqs, float)
        elif demod == "fixed":
            self.phase = 2 * np.pi * self.demod_freq * self.t
            f_bank = np.full(len(acq.demod_freqs), self.demod_freq)
        else:
            raise ValueError("demod must be 'stepped' or 'fixed'")
        self.f_ref = f_ref_hz
        # band-averaged filter-bank response on the RF axis, per band and stored sample
        c = self.cfg.centres() * 1e6
        sig = self.cfg.df * 1e6 / (2 * np.sqrt(np.log(2)))
        ff = np.arange(1.0e6, 6.0e6 + 1, 0.005e6)
        G2 = np.exp(-((ff[None, :] - c[:, None]) / sig) ** 2)                               # [nband, nf]
        nu = f_bank[:, None] - ff[None, :]                                                  # [32, nf]
        H = np.abs(np.einsum("kn,kfn->kf", taps, np.exp(-2j * np.pi * nu[..., None] * np.arange(taps.shape[1])
                                                        / hw.FS_RX))) ** 2
        eff = (G2 @ H.T) / G2.sum(1, keepdims=True)                                        # [nband, 32]
        Hs = 10 * np.log10(eff[:, self.bank])
        self.limited = (10 * np.log10(eff / H.max(1)[None, :]))[:, self.bank] < -cap_db
        peak = (10 * np.log10(H.max(1)))[self.bank]
        self.filter_db = np.maximum(Hs, peak[None, :] - cap_db) if cap_db is not None else Hs
        # geometry
        qp = self.scan.qt_params()
        if temperature == "mean":                                   # mean of temperature.dat (QT pipelines: first)
            water_speed_mm_per_us = qt_detgc.water_speed_mm_per_us
            tv = np.array(open(self.exam / "da" / "refl" / "temperature.dat").read().split(), float)
            c0 = water_speed_mm_per_us(float(tv.mean()))
        else:
            c0 = qp.c0_mm_per_us
        mm = c0 * qp.dt_us / 2.0
        self.depth_mm = (qp.nt_shift[unit] + np.arange(n)) * mm
        R = _param(recon, "rad_trans"); dphi = _param(recon, "element_width") / R
        self.beam_spacing_mm = 2 * (self.depth_mm + R) * np.sin(dphi / 2)

    # ------------------------------------------------------------------------------------------------
    def read(self, view: int) -> np.ndarray:
        """Raw stored IQ of one view, complex [levels, beams, samples]."""
        return self.scan.read(self.unit, view)

    def frame_from_iq(self, iq_frame: np.ndarray) -> Frame:
        """iq_frame [beams, samples] (one level of one view) -> Frame."""
        z = rereference(undo_gain(iq_frame, self.tgc_db), self.phase, self.t, self.f_ref)
        env = band_envelopes(z, 5e6, self.f_ref, self.cfg.centres(), self.cfg.df, self.cfg.nfft, self.filter_db)
        return Frame(env, self.cfg.centres(), self.depth_mm, self.beam_spacing_mm)

    def frame(self, view: int, level: int) -> Frame:
        return self.frame_from_iq(self.read(view)[level])

    def noise(self) -> np.ndarray | None:
        """The QT system receive-noise curve shipped with the package (far probe, stepped demodulation, default
        bands), or None if it does not match this probe's samples or bands."""
        p = DATA / "qt_unit1_system_noise.npz"
        if not p.exists():
            return None
        d = np.load(p)
        if d["N"].shape != (len(self.cfg.centres()), self.n) or not np.allclose(d["f"], self.cfg.centres()):
            return None
        return d["N"]

    def estimator(self) -> Estimator:
        return Estimator(self.cfg, noise=self.noise(), limited=self.limited)
