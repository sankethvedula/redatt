"""RED: regularisation by denoising for SLD attenuation imaging (Merino et al., IEEE TUFFC 73(8), 2026).

Per block, per band b (weight w_b in {0, 1}):   Y_b = g_b beta + c,   g_b = 2 dz f_b.
Unknowns x = (beta, c) on the blocks of a domain [R, C].

  least squares   x_LS = argmin 1/2 ||A x - y||^2           (no spatial coupling: one 2x2 solve per block)
  RED             argmin 1/2 ||A x - y||^2 + mu/2 x^T (x - D(x)),   D(x) = (median_k(beta), soft_{1/mu}(c))
  RED-PG          x_0 = x_LS,  v_0 = D(x_0)/L - (1 - L)/L x_0
                  x_k = argmin 1/2 ||A x - y||^2 + mu L/2 ||x - v_{k-1}||^2     (exact, 2x2 per block)
                  v_k = D(x_k)/L - (1 - L)/L x_k,    until ||x_k - x_{k-1}|| / ||x_k|| < tol
The median runs over the domain's blocks only (symmetric extension at the array edges).
"""
from __future__ import annotations

import dataclasses

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter

from .config import TO_MM, Config
from .sld import g_vector


@dataclasses.dataclass
class _Normal:
    a11: np.ndarray
    a12: np.ndarray
    a22: np.ndarray
    b1: np.ndarray
    b2: np.ndarray

    def solve(self, rho=0.0, v1=0.0, v2=0.0):
        a11, a22 = self.a11 + rho, self.a22 + rho
        r1, r2 = self.b1 + rho * v1, self.b2 + rho * v2
        det = a11 * a22 - self.a12 ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return (a22 * r1 - self.a12 * r2) / det, (a11 * r2 - self.a12 * r1) / det


def _normal(Y, W, g) -> _Normal:
    w = np.asarray(W, float)
    y = np.where(W, Y, 0.0)
    g = np.asarray(g, float)[:, None, None]
    return _Normal((w * g * g).sum(0), (w * g).sum(0), w.sum(0), (w * g * y).sum(0), (w * y).sum(0))


def _crop(dom):
    r = np.nonzero(dom.any(1))[0]; c = np.nonzero(dom.any(0))[0]
    return slice(r[0], r[-1] + 1), slice(c[0], c[-1] + 1)


def least_squares(Y, W, g, dom=None):
    """Unregularised SLD (beta, c) [R, C]; NaN outside dom or where the 2x2 system is singular."""
    dom = np.asarray(W).any(0) if dom is None else dom
    beta, c = _normal(Y, W, g).solve()
    ok = dom & np.isfinite(beta) & (np.asarray(W).sum(0) >= 2)
    return np.where(ok, beta, np.nan), np.where(ok, c, np.nan)


def masked_median(x, dom, k: int = 7):
    """k x k median over the domain's blocks only; NaN outside the domain."""
    h = k // 2
    if dom.all():
        return median_filter(x, size=k, mode="reflect")
    xp = np.pad(np.where(dom, x, np.nan), h, mode="symmetric")
    win = np.sort(sliding_window_view(xp, (k, k)).reshape(x.shape[0], x.shape[1], -1), -1)
    n = np.isfinite(win).sum(-1)
    lo = np.take_along_axis(win, np.maximum((n - 1) // 2, 0)[..., None], -1)[..., 0]
    hi = np.take_along_axis(win, np.maximum(n // 2, 0)[..., None], -1)[..., 0]
    return np.where(dom & (n > 0), 0.5 * (lo + hi), np.nan)


def _soft(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


@dataclasses.dataclass
class Solution:
    beta: np.ndarray        # [R, C] attenuation slope, dB/mm/MHz (NaN outside the domain)
    c: np.ndarray           # [R, C] backscatter term, dB
    iters: int
    converged: bool


def red_pg(Y, W, g, mu, dom=None, k_med=7, L=1.01, lam=None, tol=1e-5, max_iter=3000):
    """RED-PG with D = (k_med x k_med median on beta, soft-threshold lam = 1/mu on c).  beta in the units of g."""
    W = np.asarray(W, bool)
    dom = W.any(0) if dom is None else np.asarray(dom, bool)
    rs, cs = _crop(dom)
    Yc, Wc, D_ = Y[:, rs, cs], W[:, rs, cs], dom[rs, cs]
    lam = 1.0 / mu if lam is None else lam
    Nq = _normal(Yc, Wc, g)
    b, c = least_squares(Yc, Wc, g, D_)
    fill_b = np.nanmedian(b[D_]) if np.isfinite(b[D_]).any() else 0.0
    b = np.where(D_, np.where(np.isfinite(b), b, fill_b), 0.0)
    c = np.where(D_, np.nan_to_num(c), 0.0)

    def denoise(b, c):
        db = masked_median(b, D_, k_med)
        return np.where(D_ & np.isfinite(db), db, b), _soft(c, lam)

    db, dc = denoise(b, c)
    vb, vc = db / L - (1 - L) / L * b, dc / L - (1 - L) / L * c
    rho = mu * L
    conv = False
    for k in range(1, max_iter + 1):
        nb_, nc_ = Nq.solve(rho, vb, vc)
        nb_ = np.where(D_, nb_, 0.0); nc_ = np.where(D_, nc_, 0.0)
        num = np.sqrt(((nb_ - b) ** 2).sum() + ((nc_ - c) ** 2).sum())
        den = np.sqrt((nb_ ** 2).sum() + (nc_ ** 2).sum()) + 1e-30
        b, c = nb_, nc_
        db, dc = denoise(b, c)
        vb, vc = db / L - (1 - L) / L * b, dc / L - (1 - L) / L * c
        if num / den < tol:
            conv = True
            break
    Bo = np.full(dom.shape, np.nan); Co = np.full(dom.shape, np.nan)
    Bo[rs, cs] = np.where(D_, b, np.nan); Co[rs, cs] = np.where(D_, c, np.nan)
    return Bo, Co, k, conv


def solve(meas: dict, cfg: Config, method: str = "red", dom=None) -> Solution:
    """Attenuation of one measured (and referenced) frame.  method 'red' or 'ls'.  beta returned in dB/mm/MHz."""
    Y, W = meas["Y"], meas["W"]
    g = g_vector(meas["f"], meas["grid"].dz_cm)
    dom = meas["valid"] if dom is None else (np.asarray(dom, bool) & meas["valid"])
    if not dom.any():
        nan = np.full(dom.shape, np.nan)
        return Solution(nan, nan.copy(), 0, True)
    if method == "ls":
        b, c = least_squares(Y, W, g, dom)
        return Solution(b * TO_MM, c, 0, True)
    b, c, it, conv = red_pg(Y, W, g, cfg.mu, dom, cfg.k_med, cfg.L, cfg.lam, cfg.tol, cfg.max_iter)
    return Solution(b * TO_MM, c, it, conv)
