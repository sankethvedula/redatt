"""Per-sample acquisition gain for one probe (unit): analog TGC, digital TGC and FIR-bank gain."""
import dataclasses

import numpy as np

from . import fir as _fir
from . import hardware as hw
from .tgc import analog_code_to_db, play_ramps, settgc_tables


@dataclasses.dataclass(frozen=True)
class GainCurve:
    """All arrays have one entry per file sample m = 0..n-1 (fast time)."""
    depth_mm: np.ndarray          # startDepth + m * decim * c/(2 fs), in the driver's 1540 m/s time base
    atgc_code: np.ndarray         # analog TGC DAC code
    atgc_db: np.ndarray           # analog gain, driver's nominal law (10 dB at code 0)
    dtgc_code: np.ndarray         # digital TGC code (32 = unity)
    fir_bank: np.ndarray          # active demod/decimation filter bank
    fir_gain: np.ndarray          # scalar gain of that bank (absolute)
    fir_rel: np.ndarray           # ... relative to bank 0
    fir_mode: str = 'pulse'

    @property
    def atgc(self):               # linear amplitude factor of the analog stage
        return 10 ** (self.atgc_db / 20)

    @property
    def dtgc(self):               # linear amplitude factor of the digital stage
        return self.dtgc_code / hw.DTGC_UNITY

    @property
    def dtgc_db(self):
        return 20 * np.log10(self.dtgc)

    @property
    def tgc_db(self):
        """Total hardware TGC (analog + digital), on the same nominal scale as the XML request."""
        return self.atgc_db + self.dtgc_db

    @property
    def fir_db(self):
        return 20 * np.log10(self.fir_rel)

    def total_db(self, include_fir=True):
        return self.tgc_db + (self.fir_db if include_fir else 0.0)

    def saturation_rail(self):
        """|I| or |Q| value at which a sample is clipped (FIR / FPGA-sum saturation x code/32, or the final
        24-bit MultFactor saturation, whichever is lower)."""
        return np.minimum(hw.IQ_FULL_SCALE, np.floor(hw.IQ_FULL_SCALE * self.dtgc))


def gain_curve(cfg, n_samples, fir_mode='pulse', frac_bw=0.7, atgc_offset=0, dtgc_offset=0, fir_offset=0):
    """Gain applied to file samples m = 0..n_samples-1 of an acquisition with config `cfg`.

    TGC clock of sample m : round(z_m / delta) + offset  (SetTGC lays the ramps out in depth from 0; the
                            offsets, in 40 MHz clocks, allow a measured misalignment to be applied)
    FIR bank of sample m  : raw index startDepth/delta + rxBuffer + 8*SYSTEM_DELAY_OFFSET + decim*m (+ offset)
    """
    d = hw.delta_rx(cfg.c)
    m = np.arange(n_samples)
    z = cfg.start_depth + m * cfg.decimation * d
    n_clk = np.round(z / d).astype(int)
    ra, rd = settgc_tables(cfg)
    n_tot = int(n_clk.max() + abs(atgc_offset) + abs(dtgc_offset) + 2)
    a_code = play_ramps(ra, n_tot)[np.clip(n_clk + atgc_offset, 0, n_tot - 1)]
    d_code = play_ramps(rd, n_tot)[np.clip(n_clk + dtgc_offset, 0, n_tot - 1)]
    crop = int(cfg.start_depth / d + cfg.rx_buffer + hw.SYSTEM_DELAY_OFFSET * cfg.decimation)
    bank = _fir.bank_index(cfg, crop + cfg.decimation * m + fir_offset)
    gb = _fir.bank_gains(cfg, fir_mode, frac_bw)
    return GainCurve(depth_mm=z * 1e3, atgc_code=a_code, atgc_db=analog_code_to_db(a_code), dtgc_code=d_code,
                     fir_bank=bank, fir_gain=gb[bank], fir_rel=gb[bank] / gb[0], fir_mode=fir_mode)
