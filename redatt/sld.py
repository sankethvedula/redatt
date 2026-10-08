"""Spectral log difference (SLD) measurements of one frame on a regular grid of blocks.

The frame [beam, sample] is tiled with square blocks (side n_lambda wavelengths, 80 % overlap), whose centres form an
[R, C] lattice.  For each block and band b the band power is averaged over the block's beams, and

    Y_b = 10 log10( S_b(proximal window) / S_b(distal window) )                             [dB]

with the two windows (win_frac of the block) at its top and bottom, dz apart.  The model (Merino et al. 2026, eq. 3):

    Y_b - D(z, f_b) = 2 dz f_b beta + c

beta the attenuation slope (one-way), c a backscatter term, D the reference (see reference.py).

Which bands and blocks enter:
  * a pixel holds echo when its windowed full-band power is >= pixel_snr_db above the noise; a block is used when
    >= valid_frac of its pixels do (keeps water and empty regions out);
  * a band enters a block when it is within drop_db of the block's strongest band and both windows are positive;
  * with subtract_noise, the noise is subtracted per band and the band / window SNR cuts apply as well;
  * a block keeps >= min_bands bands.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.ndimage import uniform_filter

from .config import Config


@dataclasses.dataclass
class Grid:
    s0: np.ndarray          # [R] first sample of each row of blocks
    ns: int                 # samples per block
    nb: np.ndarray          # [R] beams per block in each row (blocks are square in mm; beams spread with range)
    cb: np.ndarray          # [C] centre beam of each column
    ss: int                 # row step (samples)
    sb: int                 # column step (beams)
    z_mm: np.ndarray        # [R] centre range of each row
    wp: int                 # window length (samples)
    dz_cm: float            # distance between the proximal and distal window centres

    @property
    def shape(self):
        return len(self.s0), len(self.cb)

    def b0(self, r: int) -> np.ndarray:
        return self.cb - self.nb[r] // 2

    @property
    def cs(self) -> np.ndarray:
        return self.s0 + (self.ns - 1) / 2.0


def make_grid(depth_mm: np.ndarray, beam_spacing_mm: np.ndarray, nbeams: int, cfg: Config) -> Grid:
    """Block grid of a frame with ranges depth_mm [S] and lateral beam spacing beam_spacing_mm [S]."""
    mm = float(np.median(np.diff(depth_mm)))
    size = cfg.block_mm()
    ns = cfg.block_samples if cfg.block_samples else max(4, int(round(size / mm)))
    ss = max(1, int(round(ns * (1 - cfg.overlap))))
    s0 = np.arange(0, len(depth_mm) - ns + 1, ss)
    nb = np.array([int(min(nbeams, max(2, round(size / float(beam_spacing_mm[s + ns // 2]))))) for s in s0])
    sb = max(1, int(round(size * (1 - cfg.overlap) / float(np.median(beam_spacing_mm)))))
    nmax = int(nb.max())
    cb = np.arange(nmax // 2, nbeams - (nmax - nmax // 2) + 1, sb)
    wp = max(2, int(round(cfg.win_frac * ns)))
    return Grid(s0=s0, ns=ns, nb=nb, cb=cb, ss=ss, sb=sb, z_mm=np.array([depth_mm[s:s + ns].mean() for s in s0]),
                wp=wp, dz_cm=(ns - wp) * mm / 10.0)


def windowed_power(env_db: np.ndarray, mm: float, beam_mm: float, win_mm: float):
    """Band power averaged over a win_mm x win_mm window, and the frame's own noise estimate (10th percentile over
    beams, smoothed along range) used when no noise is given.  Returns Pw [nband, B, S], N [nband, 1, S]."""
    P = np.power(10.0, np.asarray(env_db, np.float64) / 10)
    ns = max(1, int(round(win_mm / mm)))
    nbw = max(1, int(round(win_mm / beam_mm)))
    Pw = uniform_filter(P, size=(1, nbw, ns), mode="nearest")
    N = uniform_filter(np.percentile(Pw, 10, axis=1), size=(1, ns), mode="nearest")[:, None, :]
    return Pw, N


def _integral(img):
    I = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
    I[1:, 1:] = np.cumsum(np.cumsum(np.asarray(img, np.float64), 0), 1)
    return I


def block_fraction(grid: Grid, mask: np.ndarray) -> np.ndarray:
    """[R, C] fraction of each block's pixels where mask [B, S] is True."""
    I = _integral(mask)
    out = np.zeros(grid.shape)
    for r in range(len(grid.s0)):
        b0 = grid.b0(r); b1 = b0 + grid.nb[r]; s0 = grid.s0[r]; s1 = s0 + grid.ns
        out[r] = (I[b1, s1] - I[b0, s1] - I[b1, s0] + I[b0, s0]) / (grid.nb[r] * grid.ns)
    return out


def measure(env_db: np.ndarray, centres_mhz, depth_mm: np.ndarray, beam_spacing_mm: np.ndarray, cfg: Config,
            noise: np.ndarray | None = None, limited: np.ndarray | None = None, grid: Grid | None = None) -> dict:
    """SLD measurements of one frame.

    env_db [nband, B, S] band envelopes (dB); centres_mhz [nband]; depth_mm [S]; beam_spacing_mm [S];
    noise [nband, S] linear band power of the receive noise (None: the frame's own estimate);
    limited [nband, S] bool, bands to exclude per sample (e.g. outside the receive filter).
    Returns dict: Y [nband, R, C] (dB, NaN where unused), W [nband, R, C] bool, valid [R, C], f, grid,
    snr_p / snr_d [nband, R, C] (window SNRs, for the block weights)."""
    f = np.asarray(centres_mhz, float)
    nband, B, S = env_db.shape
    mm = float(np.median(np.diff(depth_mm)))
    if grid is None:
        grid = make_grid(depth_mm, beam_spacing_mm, B, cfg)
    P = np.power(10.0, np.asarray(env_db, np.float64) / 10)
    Pw, N = windowed_power(env_db, mm, float(np.median(beam_spacing_mm)), cfg.win_mm)
    if noise is not None:
        N = np.asarray(noise, np.float64)[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        valid_px = 10 * np.log10(Pw.sum(0) / N.sum(0)) >= cfg.pixel_snr_db
    N = N[:, 0, :]
    lim = np.zeros((nband, S), bool) if limited is None else np.asarray(limited, bool)
    cP = np.concatenate([np.zeros((nband, 1, S)), np.cumsum(P, 1)], 1)
    vf = block_fraction(grid, valid_px)
    R, C = grid.shape
    Y = np.full((nband, R, C), np.nan); W = np.zeros((nband, R, C), bool)
    SNRp = np.full((nband, R, C), np.nan); SNRd = np.full((nband, R, C), np.nan)
    ns, wp = grid.ns, grid.wp
    for r in range(R):
        b0 = grid.b0(r); nb = grid.nb[r]
        sl = slice(grid.s0[r], grid.s0[r] + ns)
        Pm = (cP[:, b0 + nb, sl] - cP[:, b0, sl]) / nb                     # [nband, C, ns] beam-averaged power
        Nn = N[:, None, sl]
        X = Pm - Nn if cfg.subtract_noise else Pm
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = 10 * np.log10(Pm.mean(-1) / Nn.mean(-1))
            Xm = X.mean(-1)
            peak = np.nanmax(np.where(Xm > 0, Xm, np.nan), 0)
            near_peak = Xm >= peak[None, :] * 10 ** (-cfg.drop_db / 10)
            Sp, Sd = X[..., :wp].mean(-1), X[..., ns - wp:].mean(-1)
            Np_, Nd_ = Nn[..., :wp].mean(-1), Nn[..., ns - wp:].mean(-1)
            if cfg.subtract_noise:
                thr = 10 ** (cfg.window_snr_db / 10)
                pos = (Sp > thr * Np_) & (Sd > thr * Nd_)
            else:
                pos = (Sp > 0) & (Sd > 0)
            y = 10 * np.log10(np.where(pos, Sp, 1.0) / np.where(pos, Sd, 1.0))
            sp_, sd_ = Sp / Np_, Sd / Nd_
        ok = near_peak & pos & ~lim[:, sl].any(-1)[:, None]
        if cfg.subtract_noise:
            ok &= snr >= cfg.band_snr_db
        ok &= (vf[r] >= cfg.valid_frac)[None, :]
        ok &= (ok.sum(0) >= cfg.min_bands)[None, :]
        Y[:, r] = np.where(ok, y, np.nan); W[:, r] = ok
        SNRp[:, r] = np.where(ok, sp_, np.nan); SNRd[:, r] = np.where(ok, sd_, np.nan)
    return dict(Y=Y, W=W, valid=W.any(0), f=f, grid=grid, snr_p=SNRp, snr_d=SNRd)


def g_vector(f_mhz, dz_cm: float) -> np.ndarray:
    """Attenuation column of the model, 2 dz[cm] f (so beta is solved in dB/cm/MHz)."""
    return 2.0 * dz_cm * np.asarray(f_mhz, float)


def block_weight(meas: dict) -> np.ndarray:
    """[R, C] reliability of each block's attenuation from the SNR it has: the inverse predicted variance of the
    slope over frequency, with per-band variance v_b = (1 + 1/SNR_prox)^2 + (1 + 1/SNR_dist)^2.  0 without data.
    Used to weight views when compounding."""
    g = 2.0 * meas["grid"].dz_cm * 10.0 * np.asarray(meas["f"], float)[:, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (1 + 1 / meas["snr_p"]) ** 2 + (1 + 1 / meas["snr_d"]) ** 2
        w = np.where(meas["W"] & np.isfinite(v), 1.0 / v, 0.0)
        gbar = (w * g).sum(0) / w.sum(0)
        info = (w * (g - gbar[None]) ** 2).sum(0)
    return np.where(meas["valid"] & np.isfinite(info), info, 0.0)


def to_pixels(grid: Grid, img: np.ndarray, shape) -> np.ndarray:
    """[R, C] block image -> [B, S] pixels (each pixel takes the nearest block centre; NaN outside the lattice)."""
    B, S = shape
    cs = grid.cs
    s = np.arange(S)
    ir = np.clip(np.rint((s - cs[0]) / grid.ss).astype(int), 0, len(cs) - 1)
    s_ok = np.abs(s - cs[ir]) <= grid.ss / 2 + 1e-9
    b = np.arange(B)
    jc = np.clip(np.rint((b - grid.cb[0]) / grid.sb).astype(int), 0, len(grid.cb) - 1)
    b_ok = np.abs(b - grid.cb[jc]) <= grid.sb / 2 + 1e-9
    v = np.asarray(img, float)[ir[None, :], jc[:, None]]
    return np.where(b_ok[:, None] & s_ok[None, :], v, np.nan)
