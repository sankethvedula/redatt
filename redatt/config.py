"""All parameters of the estimator, in one place.  Defaults are the configuration that worked best on QT breast data
("self_nn": self reference, no noise handling, 15-wavelength blocks, RED with mu = 100 and a 7 x 7 median)."""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Config:
    # ---- frequency bands (MHz): Gaussian band-passes whose POWER response has FWHM df, centred every `step`
    f_lo: float = 2.04
    f_hi: float = 4.24
    df: float = 0.3
    step: float = 0.15
    nfft: int = 2048                    # FFT length of the band decomposition (zero-padded lines)

    # ---- blocks
    n_lambda: float = 15.0              # block side in wavelengths
    c0_mm_us: float = 1.51              # speed used for the wavelength (apparent range)
    f_lambda: float | None = None       # MHz for the wavelength; None: midway between the first and last band centre
    block_samples: int | None = None    # fix the block length in samples (overrides n_lambda axially)
    overlap: float = 0.8                # fraction of a block shared with the next one (rows and columns)
    win_frac: float = 0.5               # proximal / distal window length as a fraction of the block

    # ---- which blocks and bands enter
    subtract_noise: bool = False        # True: subtract the noise per band and apply band_snr_db / window_snr_db
    band_snr_db: float = 10.0           # (subtract_noise only) block SNR of a band
    window_snr_db: float = 3.0          # (subtract_noise only) both windows this far above the noise
    drop_db: float = 20.0               # drop bands more than this below the block's strongest band
    min_bands: int = 3                  # a block needs at least this many bands
    pixel_snr_db: float = 10.0          # a pixel holds echo when its windowed full-band power is this far above noise
    valid_frac: float = 0.9             # a block is used when this fraction of its pixels holds echo
    win_mm: float = 3.0                 # window (mm x mm) of the pixel-validity power

    # ---- edge-aware estimation (redatt.gpu only; needs a guide image per frame, e.g. a compounded B-mode in dB)
    edge_sigma_db: float | None = None  # spectra: weight each pixel of a block by exp(-(g - g_centre)^2 / 2 s^2)
    edge_min_frac: float = 0.3          # ... and drop blocks whose proximal or distal window keeps less weight than this
    med_sigma_db: float | None = None   # RED: weighted median, neighbour weight exp(-(g_j - g_i)^2 / 2 s^2)
    guide_detrend: bool = True          # subtract the guide's per-sample median over beams (its depth trend) first

    # ---- RED solver
    mu: float = 100.0                   # regularisation weight
    k_med: int = 7                      # median denoiser size (blocks)
    L: float = 1.01                     # RED-PG step parameter
    lam: float | None = None            # soft threshold on the backscatter channel; None: 1 / mu
    tol: float = 1e-5                   # relative-change stopping tolerance
    max_iter: int = 3000
    accel: str = "none"                 # 'none': plain RED-PG; 'anderson': Anderson-accelerated (same fixed point,
                                        # reached far more closely: use with tol ~1e-8)
    mem: int = 10                       # Anderson memory

    # ---- reference
    ref_min_blocks: int = 200           # a row / band of the self reference needs at least this many blocks

    def centres(self) -> np.ndarray:
        """Band centres (MHz)."""
        return np.arange(self.f_lo + self.df / 2, self.f_hi - self.df / 2 + 1e-9, self.step)

    def block_mm(self) -> float:
        """Block side (mm): n_lambda wavelengths at f_lambda (default: midway between the first and last band
        centre)."""
        c = self.centres()
        f = self.f_lambda if self.f_lambda else 0.5 * (c[0] + c[-1])
        return self.n_lambda * self.c0_mm_us / f


# Solver scaling: inside the solver the attenuation column of the model is g_b = 2 dz[cm] f_b, i.e. the slope is in
# dB/cm/MHz (the scaling of Merino et al. 2026, which keeps mu comparable to the paper).  Results are reported in
# dB/mm/MHz.
TO_MM = 0.1
