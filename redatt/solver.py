"""RED: regularisation by denoising for SLD attenuation imaging (Merino et al., IEEE TUFFC 73(8), 2026).

Per block, per band b (weight w_b in {0, 1}):   Y_b = g_b beta + c,   g_b = 2 dz f_b.
Unknowns x = (beta, c) on the blocks of a domain [R, C].

  least squares   x_LS = argmin 1/2 ||A x - y||^2           (no spatial coupling: one 2x2 solve per block)
  RED             argmin 1/2 ||A x - y||^2 + mu/2 x^T (x - D(x)),   D(x) = (median_k(beta), soft_{1/mu}(c))
  RED-PG          x_0 = x_LS,  v_0 = D(x_0)/L - (1 - L)/L x_0
                  x_k = argmin 1/2 ||A x - y||^2 + mu L/2 ||x - v_{k-1}||^2     (exact, 2x2 per block)
                  v_k = D(x_k)/L - (1 - L)/L x_k,    until ||x_k - x_{k-1}|| / ||x_k|| < tol
The median runs over the domain's blocks only (symmetric extension at the array edges).

RED-PG is a fixed-point iteration x_k = T(x_{k-1}) that contracts slowly (the median spreads information by about one
block per iteration), so stopping on a small relative change still leaves x some way from the fixed point (on QT data,
tol 1e-5 after ~900 iterations: typically ~4e-3, up to ~1.4e-2 dB/mm/MHz;
converging fully takes up to ~10000).  accel='anderson' applies Anderson acceleration (type II,
memory `mem`, safeguarded by a restart) to the same map T: same fixed point, and the residual ||T(x) - x|| / ||T(x)||
falls below 1e-8 in ~500 iterations (at most ~1300 on QT data).
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


def _anderson(T, b, c, dom, tol, max_iter, mem, reg=1e-10, restart=1.0):
    """Anderson acceleration (type II) of x = T(x) on the domain's (beta, c).  Stops when ||T(x) - x|| / ||T(x)|| <
    tol and returns T(x).  A step whose residual grows (more than `restart` times) clears the history, and a mixed
    one is replaced by the plain step T(x); mixing resumes once a plain step has reduced the residual.  With a looser
    safeguard (restart 10, or never restarting on plain steps) a few frames stall."""
    n = int(dom.sum())

    def Tv(x):
        bb = np.zeros(dom.shape); cc = np.zeros(dom.shape)
        bb[dom], cc[dom] = x[:n], x[n:]
        nb_, nc_ = T(bb, cc)
        return np.concatenate([nb_[dom], nc_[dom]])

    x = np.concatenate([b[dom], c[dom]])
    gx = Tv(x); f = gx - x
    G, F = [], []
    conv = False
    for k in range(1, max_iter + 1):
        if np.linalg.norm(f) < tol * (np.linalg.norm(gx) + 1e-30):
            conv = True
            break
        G.append(gx); F.append(f)
        if len(F) > mem + 1:
            G.pop(0); F.pop(0)
        xn = gx
        if len(F) > 1:
            dF = np.diff(np.array(F), axis=0); dG = np.diff(np.array(G), axis=0)
            A = dF @ dF.T
            tr = np.trace(A) if np.trace(A) > 0 else 1.0 / reg            # no change in the history: gamma = 0
            gam = np.linalg.solve(A + reg * tr * np.eye(len(A)), dF @ f)
            xn = gx - gam @ dG
        gn = Tv(xn); fn = gn - xn
        if np.linalg.norm(fn) > restart * np.linalg.norm(f):
            G, F = [], []
            xn = gx; gn = Tv(xn); fn = gn - xn
        x, gx, f = xn, gn, fn
    bo = np.zeros(dom.shape); co = np.zeros(dom.shape)
    bo[dom], co[dom] = gx[:n], gx[n:]
    return bo, co, k, conv


@dataclasses.dataclass
class Solution:
    beta: np.ndarray        # [R, C] attenuation slope, dB/mm/MHz (NaN outside the domain)
    c: np.ndarray           # [R, C] backscatter term, dB
    iters: int
    converged: bool


def red_pg(Y, W, g, mu, dom=None, k_med=7, L=1.01, lam=None, tol=1e-5, max_iter=3000, accel="none", mem=10):
    """RED-PG with D = (k_med x k_med median on beta, soft-threshold lam = 1/mu on c).  beta in the units of g.
    accel 'none' (plain RED-PG) or 'anderson' (Anderson-accelerated, same fixed point; use a smaller tol, e.g. 1e-8)."""
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

    rho = mu * L

    def T(b, c):                                    # one RED-PG step: x_k = T(x_{k-1})
        db, dc = denoise(b, c)
        vb, vc = db / L - (1 - L) / L * b, dc / L - (1 - L) / L * c
        nb_, nc_ = Nq.solve(rho, vb, vc)
        return np.where(D_, nb_, 0.0), np.where(D_, nc_, 0.0)

    if accel == "anderson":
        b, c, k, conv = _anderson(T, b, c, D_, tol, max_iter, mem)
    elif accel == "none":
        conv = False
        for k in range(1, max_iter + 1):
            nb_, nc_ = T(b, c)
            num = np.sqrt(((nb_ - b) ** 2).sum() + ((nc_ - c) ** 2).sum())
            den = np.sqrt((nb_ ** 2).sum() + (nc_ ** 2).sum()) + 1e-30
            b, c = nb_, nc_
            if num / den < tol:
                conv = True
                break
    else:
        raise ValueError("accel must be 'none' or 'anderson'")
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
    b, c, it, conv = red_pg(Y, W, g, cfg.mu, dom, cfg.k_med, cfg.L, cfg.lam, cfg.tol, cfg.max_iter, cfg.accel,
                           cfg.mem)
    return Solution(b * TO_MM, c, it, conv)
