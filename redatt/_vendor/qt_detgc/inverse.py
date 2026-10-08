"""The two gain inversions, behind one interface.

    ExactInverse   engineered from the driver/FPGA source: exact analog + digital TGC and the FIR-bank
                   gain, undone per file sample in reverse acquisition order
    QTInverse      QT's reconstruction model (remove_analog_gain): 0.3 dB/mm by QT's straight-ray distance.
                   QT's ray-traced `apply_digital_gain` (a ramp multiplied back in inside the breast
                   support) is deliberately NOT applied, nor is refraction correction.

Both return data on the same absolute scale (divided by the absolute gain each model believes was
applied, in the XML's nominal dB), so results are directly comparable.

    inv = ExactInverse(cfg)                      # or QTInverse(qt_params, unit=1, cfg=cfg)
    res = inv.apply(iq)                          # iq: [..., samples] complex, raw file values
    res.iq, res.saturated, res.gain_db, res.depth_mm
"""
import dataclasses
from typing import Optional

import numpy as np

from . import hardware as hw
from .curve import GainCurve, gain_curve
from .iq import saturation_mask


@dataclasses.dataclass
class Corrected:
    iq: np.ndarray                 # corrected samples (same shape as input)
    saturated: np.ndarray          # bool, same shape: clipped in the raw data (lower bounds only -- mask them)
    gain_db: np.ndarray            # per-sample gain that was divided out (dB)
    depth_mm: np.ndarray           # per-sample depth in the driver's time base
    method: str


class GainInverse:
    name = 'base'

    def gain_db(self, n_samples) -> np.ndarray:
        raise NotImplementedError

    def depth_mm(self, n_samples) -> np.ndarray:
        raise NotImplementedError

    def apply(self, iq, reference: Optional[str] = None) -> Corrected:
        """Divide the gain out of raw file samples.  reference='first' rescales so that the first sample of
        each line is unchanged (useful for display); default keeps the absolute nominal scale."""
        raise NotImplementedError


class ExactInverse(GainInverse):
    """Exact inversion of the acquisition gain.

    Acquisition:  x_ch -[x G_A(t)]-> ADC/HPF/delay-sum/mixer -[FIR bank H_k]-> sat -[x c_D/32, floor, sat]-> file
    Inverse:      (0) saturation flagged on the raw integers   (the rail depends on c_D)
                  (1) / (c_D/32)       digital TGC   -- last applied, first undone
                  (2) / H_k / H_0      FIR-bank gain (scalar model: fir_mode 'pulse' | 'dc' | 'noise')
                  (3) / 10^(G_A/20)    analog TGC
    (1)-(3) are positive depth-only scalars and commute exactly; the order matters only for (0)."""
    name = 'exact'

    def __init__(self, cfg, include_fir=True, fir_mode='pulse', frac_bw=0.7,
                 atgc_offset=0, dtgc_offset=0, fir_offset=0):
        self.cfg = cfg
        self.include_fir = include_fir
        self.kw = dict(fir_mode=fir_mode, frac_bw=frac_bw, atgc_offset=atgc_offset,
                       dtgc_offset=dtgc_offset, fir_offset=fir_offset)
        self._cache = {}

    def curve(self, n_samples) -> GainCurve:
        if n_samples not in self._cache:
            self._cache[n_samples] = gain_curve(self.cfg, n_samples, **self.kw)
        return self._cache[n_samples]

    def gain_db(self, n_samples):
        return self.curve(n_samples).total_db(self.include_fir)

    def depth_mm(self, n_samples):
        return self.curve(n_samples).depth_mm

    def apply(self, iq, reference=None):
        c = self.curve(iq.shape[-1])
        sat = saturation_mask(iq, c)                               # (0) before any division
        out = iq / c.dtgc                                          # (1)
        if self.include_fir:
            out = out / c.fir_rel                                  # (2)
        out = out / c.atgc                                         # (3)
        g_db = c.total_db(self.include_fir)
        if reference == 'first':
            out = out * 10 ** (g_db[0] / 20)
        return Corrected(out, sat, g_db, c.depth_mm, self.name + ('' if self.include_fir else '-tgc'))


class QTInverse(GainInverse):
    """QT's gain removal (refl_flip_image.f:208-224 / interfaceLayerRefl.cpp:2103-2124), straight ray:

        envelope index   L    = m + 1 + nt_shift[unit]
        ray step         lray = (L - time_shift) * c0 * dt / (2 * hray)     (inverse of lt = round(tray/dt + time_shift))
        distance         dist = (lray - 1) * hray
        removed gain     g    = clip(slope * (dist - start), 0, max)        [dB]

    Not included, by design: QT's `apply_digital_gain` (re-added along each ray inside the support during
    ray tracing) and the refraction-corrected ray length (see refraction_gain_error_db)."""
    name = 'qt'

    def __init__(self, params, unit, cfg=None):
        if params.c0_mm_per_us is None:
            raise ValueError('QTReconParams needs the water temperature (c0); use .with_temperature()')
        self.p = params
        self.unit = unit
        self.cfg = cfg                         # optional: gives the hardware depth axis and saturation rail

    def distance_mm(self, n_samples):
        """QT's ray distance for each file sample."""
        p = self.p
        L = np.arange(n_samples) + 1 + p.nt_shift[self.unit]
        lray = (L - p.time_shift) * p.c0_mm_per_us * p.dt_us / (2 * p.hray_mm)
        return (lray - 1) * p.hray_mm

    def gain_db(self, n_samples):
        p = self.p
        if p.remove_analog_gain.lower() != 'y':
            return np.zeros(n_samples)
        return np.clip(p.analog_gain_slope_db_per_mm * (self.distance_mm(n_samples) - p.analog_gain_start_mm),
                       0, p.analog_gain_max_db)

    def depth_mm(self, n_samples):
        if self.cfg is not None:
            return (self.cfg.start_depth + np.arange(n_samples) * self.cfg.decimation * hw.delta_rx(self.cfg.c)) * 1e3
        return self.distance_mm(n_samples)

    def window(self, n_samples):
        """Samples QT actually uses for this probe (range_min < dist <= range_max)."""
        d = self.distance_mm(n_samples)
        return (d > self.p.range_min_mm[self.unit]) & (d <= self.p.range_max_mm[self.unit])

    def apply(self, iq, reference=None):
        n = iq.shape[-1]
        g_db = self.gain_db(n)
        if self.cfg is not None:
            sat = saturation_mask(iq, gain_curve(self.cfg, n))    # QT itself does not mask; reported only
        else:
            sat = np.zeros(iq.shape, bool)
        out = iq / 10 ** (g_db / 20)
        if reference == 'first':
            out = out * 10 ** (g_db[0] / 20)
        return Corrected(out, sat, g_db, self.depth_mm(n), self.name)


def refraction_gain_error_db(tissue_path_mm, c_tissue_m_s, c0_mm_per_us, slope_db_per_mm=0.3):
    """Additional error of QT's model when its ray tracer uses correct_refraction='y': the gain is looked
    up by geometric distance while the data (and the hardware TGC) follow travel time.  Negative = QT
    removes too little (tissue slower than water)."""
    return -slope_db_per_mm * np.asarray(tissue_path_mm) * (c0_mm_per_us * 1e3 / np.asarray(c_tissue_m_s) - 1)
