"""The reference term D(z, f) of the model, which removes the system's own depth-dependent spectral shaping (beam
diffraction and focusing, residual receive-chain effects).  Two reference-free options:

  self reference   D[band, row] = median of Y over every tissue block of the exam at that row (many views and
                   levels; blocks touching a lesion excluded).  The result is attenuation relative to this exam's
                   typical tissue at the same range (0 = typical).  Needs no phantom.
  depth trend      solve without a reference, then subtract a smooth (quadratic) function of range fitted to the
                   exam's tissue.  Same meaning; three numbers instead of a table.

A phantom reference is the same table built from a uniform phantom scanned with identical settings (pass its
measurements to `self_reference` and, if its attenuation a_ref is known, use `a_ref` to make the result absolute).
"""
from __future__ import annotations

import warnings

import numpy as np

from .config import TO_MM
from .sld import block_fraction, g_vector


def exclude_blocks(meas: dict, pixel_mask: np.ndarray) -> dict:
    """Copy of a measurement with every block that touches pixel_mask [B, S] removed (e.g. a lesion, dilated)."""
    out = dict(meas)
    bad = block_fraction(meas["grid"], pixel_mask) > 0
    out["Y"] = np.where(bad[None], np.nan, meas["Y"])
    out["W"] = meas["W"] & ~bad[None]
    out["valid"] = meas["valid"] & ~bad
    return out


def self_reference(measurements, min_blocks: int = 200, a_ref: float = 0.0) -> dict:
    """D [nband, R] = median over the blocks of all measurements (same grid) of (Y - 2 dz f a_ref).  a_ref in
    dB/mm/MHz (0: relative).  NaN where fewer than min_blocks blocks."""
    g = measurements[0]["grid"]; f = measurements[0]["f"]
    Ys = np.concatenate([np.moveaxis(np.asarray(m["Y"], np.float32), 2, 0) for m in measurements], 0)  # [n*C, nb, R]
    n = np.isfinite(Ys).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(Ys, 0)
    D = np.where(n >= min_blocks, med - (g_vector(f, g.dz_cm) * a_ref / TO_MM)[:, None], np.nan)
    return dict(D=D, n=n, f=np.asarray(f), s0=g.s0, ns=g.ns)


def apply_reference(meas: dict, ref: dict | None) -> dict:
    """Subtract D from a measurement on the same grid; bands / rows without a reference drop out."""
    if ref is None:
        return meas
    g = meas["grid"]
    if not (np.array_equal(g.s0, ref["s0"]) and g.ns == ref["ns"] and np.allclose(meas["f"], ref["f"])):
        raise ValueError("reference and measurement use different grid rows or bands")
    out = dict(meas)
    Y = meas["Y"] - ref["D"][:, :, None]
    W = meas["W"] & np.isfinite(Y)
    out["Y"] = np.where(W, Y, np.nan); out["W"] = W
    out["valid"] = meas["valid"] & (W.sum(0) >= 2)
    return out


def fit_depth_trend(betas, z_mm: np.ndarray, deg: int = 2, min_blocks: int = 50) -> np.ndarray:
    """Polynomial (coefficients, np.polyval order) of the per-row median of unreferenced attenuation maps betas
    (list of [R, C], same grid with row centres z_mm), weighted by sqrt(blocks per row)."""
    vals = [[] for _ in z_mm]
    for b in betas:
        for r in range(len(z_mm)):
            x = b[r][np.isfinite(b[r])]
            vals[r].extend(x.tolist())
    n = np.array([len(v) for v in vals]); med = np.array([np.median(v) if v else np.nan for v in vals])
    ok = n >= min_blocks
    return np.polyfit(z_mm[ok], med[ok], deg, w=np.sqrt(n[ok]))
