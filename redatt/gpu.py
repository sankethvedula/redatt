"""GPU (PyTorch) implementation of the estimator, batched over frames.

Same computations as frontend.py, sld.py, reference.py and solver.py, but on a batch of N frames that share one
geometry (same samples, beams, depth and beam spacing: every frame of one probe and acquisition setting).  Results
match the CPU path (the RED map to ~1e-6 dB/mm/MHz in float64).  Needs torch; the rest of redatt does not import this
module.

    import redatt as ra, redatt.gpu as rg
    est = rg.GPUEstimator(ra.Config(), noise=noise, limited=limited, device="cuda")
    env = rg.band_envelopes(z, fs, f_ref, cfg.centres(), cfg.df, cfg.nfft)       # z [N, B, S] -> [N, nband, B, S]
    fb = rg.FrameBatch(env, cfg.centres(), depth_mm, beam_spacing_mm)
    ref = est.self_reference([fb1, fb2, ...])                                   # FrameBatch or Frame iterables
    outs = est.estimate(fb, ref)                                                # list of ra.Estimate, one per frame

With QT data, `QTProbe.frame_batch(view, device=...)` builds the FrameBatch of every level of a view on the GPU.

The RED solver runs all frames of a batch together; each frame stops at its own iteration, exactly as when solved
alone.  accel='anderson' (Config.accel) is batched as well.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from .config import TO_MM, Config
from .pipeline import Estimate, Frame
from .sld import Grid, make_grid

# ------------------------------------------------------------------------------------------------------------------
# front end


def undo_gain(iq: torch.Tensor, gain_db) -> torch.Tensor:
    """iq [..., S] / 10^(gain_db [S] / 20)."""
    g = torch.as_tensor(np.asarray(gain_db, float), device=iq.device)
    return iq / torch.pow(10.0, g / 20.0)


def rereference(iq: torch.Tensor, phase, t_s, f_ref_hz: float) -> torch.Tensor:
    """Remove the demodulation phase phase [S] and apply the carrier f_ref at times t_s [S] (see frontend.py)."""
    arg = -np.asarray(phase, float) + 2 * np.pi * f_ref_hz * np.asarray(t_s, float)
    rot = torch.as_tensor(np.exp(1j * arg), device=iq.device)
    return iq * rot.to(iq.dtype)


def band_envelopes(z: torch.Tensor, fs_hz: float, f_ref_hz: float, centres_mhz, df_mhz: float, nfft: int = 2048,
                   filter_db=None) -> torch.Tensor:
    """Band envelopes [N, nband, B, S] (float32 dB) of re-referenced IQ z [N, B, S] (or [B, S]).  As
    frontend.band_envelopes; computed in z's precision (pass complex128 to match the CPU path)."""
    if z.ndim == 2:
        z = z[None]
    N, B, S = z.shape
    real = torch.float64 if z.dtype == torch.complex128 else torch.float32
    nu = torch.fft.fftfreq(nfft, 1.0 / fs_hz, dtype=torch.float64, device=z.device)
    centres = torch.as_tensor(np.asarray(centres_mhz, float) * 1e6, device=z.device)
    sig = df_mhz * 1e6 / (2 * math.sqrt(math.log(2)))
    gauss = torch.exp(-0.5 * ((nu[None, :] - (f_ref_hz - centres[:, None])) / sig) ** 2).to(real)
    Z = torch.fft.fft(z, nfft, dim=-1)
    fdb = None if filter_db is None else torch.as_tensor(np.asarray(filter_db, float), device=z.device).to(real)
    env = torch.empty((N, len(centres), B, S), dtype=torch.float32, device=z.device)
    for b in range(len(centres)):
        y = torch.fft.ifft(Z * gauss[b], dim=-1)[..., :S]
        e = 20 * torch.log10(torch.abs(y) + 1e-9)
        if fdb is not None:
            e = e - fdb[b][None, None, :]
        env[:, b] = e.to(torch.float32)
    return env


# ------------------------------------------------------------------------------------------------------------------
# frames and measurements


@dataclasses.dataclass
class FrameBatch:
    """N frames of one geometry.  env_db [N, nband, B, S] (tensor or array); centres_mhz [nband]; depth_mm [S];
    beam_spacing_mm [S]."""
    env_db: torch.Tensor
    centres_mhz: np.ndarray
    depth_mm: np.ndarray
    beam_spacing_mm: np.ndarray

    @classmethod
    def from_frames(cls, frames, device=None) -> "FrameBatch":
        frames = list(frames)
        env = torch.as_tensor(np.stack([np.asarray(f.env_db) for f in frames]), device=device)
        f0 = frames[0]
        return cls(env, np.asarray(f0.centres_mhz), np.asarray(f0.depth_mm), np.asarray(f0.beam_spacing_mm))

    def __len__(self):
        return self.env_db.shape[0]

    @property
    def shape(self):
        return tuple(self.env_db.shape[2:])

    def __getitem__(self, i) -> "FrameBatch":
        i = slice(i, i + 1) if isinstance(i, int) else i
        return FrameBatch(self.env_db[i], self.centres_mhz, self.depth_mm, self.beam_spacing_mm)


@dataclasses.dataclass
class Measurement:
    """SLD measurements of N frames on one grid (tensors).  Y, W, snr_p, snr_d [N, nband, R, C]; valid [N, R, C]."""
    Y: torch.Tensor
    W: torch.Tensor
    valid: torch.Tensor
    snr_p: torch.Tensor
    snr_d: torch.Tensor
    f: np.ndarray
    grid: Grid
    omega: torch.Tensor | None = None       # [N, R, C] edge-aware data weight of each block (None: 1)
    gblock: torch.Tensor | None = None      # [N, R, C] guide value at each block centre (edge-aware RED)
    kappa: torch.Tensor | None = None       # [N, R, C] effective window separation / dz (edge-aware spectra)

    def __len__(self):
        return self.Y.shape[0]

    def replace(self, **kw) -> "Measurement":
        return dataclasses.replace(self, **kw)

    def to_numpy(self) -> list:
        """One CPU measurement dict (as sld.measure returns) per frame."""
        a = {k: getattr(self, k).cpu().numpy() for k in ("Y", "W", "valid", "snr_p", "snr_d")}
        return [dict(Y=a["Y"][i], W=a["W"][i], valid=a["valid"][i], snr_p=a["snr_p"][i], snr_d=a["snr_d"][i],
                     f=self.f, grid=self.grid) for i in range(len(self))]


def _box_mean(x: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    """scipy.ndimage.uniform_filter1d(x, size, axis=dim, mode='nearest')."""
    if size <= 1:
        return x
    x = x.movedim(dim, -1)
    lo, hi = size // 2, size - 1 - size // 2
    xp = torch.cat([x[..., :1].expand(*x.shape[:-1], lo), x, x[..., -1:].expand(*x.shape[:-1], hi)], -1)
    cs = torch.cat([torch.zeros_like(xp[..., :1]), torch.cumsum(xp, -1)], -1)
    out = (cs[..., size:] - cs[..., :-size]) / size
    return out.movedim(-1, dim)


def _percentile(x: torch.Tensor, q: float, dim: int) -> torch.Tensor:
    """np.percentile(x, q, axis=dim) (linear interpolation)."""
    xs = torch.sort(x, dim=dim).values
    pos = q / 100.0 * (x.shape[dim] - 1)
    i0 = int(math.floor(pos)); i1 = min(i0 + 1, x.shape[dim] - 1); fr = pos - i0
    return xs.select(dim, i0) * (1 - fr) + xs.select(dim, i1) * fr


def _block_fraction(grid: Grid, mask: torch.Tensor) -> torch.Tensor:
    """[N, R, C] fraction of each block's pixels where mask [N, B, S] is True."""
    N, B, S = mask.shape
    I = torch.zeros((N, B + 1, S + 1), dtype=torch.float64, device=mask.device)
    I[:, 1:, 1:] = torch.cumsum(torch.cumsum(mask.to(torch.float64), 1), 2)
    b0, b1, s0, s1 = _block_index(grid, mask.device)
    num = I[:, b1, s1] - I[:, b0, s1] - I[:, b1, s0] + I[:, b0, s0]
    nb = torch.as_tensor(grid.nb, dtype=torch.float64, device=mask.device)[:, None]
    return num / (nb * grid.ns)


def _block_index(grid: Grid, device):
    """[R, C] beam and sample bounds of every block."""
    b0 = grid.cb[None, :] - grid.nb[:, None] // 2
    b1 = b0 + grid.nb[:, None]
    s0 = np.broadcast_to(grid.s0[:, None], b0.shape)
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long, device=device)
    return t(b0), t(b1), t(s0), t(s0 + grid.ns)


def measure(batch: FrameBatch, cfg: Config, noise=None, limited=None, grid: Grid | None = None,
            chunk: int = 16, guide: torch.Tensor | None = None) -> Measurement:
    """SLD measurements of every frame of a batch (sld.measure, batched).  noise [nband, S], limited [nband, S] as
    there.  Frames are processed `chunk` at a time to bound memory.

    guide [N, B, S] (e.g. a compounded B-mode in dB, mapped into each frame) makes the estimate edge-aware: with
    cfg.edge_sigma_db, each block's window spectra weight its pixels by their guide similarity to the block centre,
    and the block's data weight omega is the smaller weight fraction kept by its two windows; the windows' weighted
    centroids give the block's effective separation kappa dz, used in the model (weights that favour the block centre
    shorten it).  The guide at the block centres (gblock) feeds the weighted median of cfg.med_sigma_db.  With
    cfg.guide_detrend the guide's per-sample median over beams (its depth trend) is removed first."""
    env = batch.env_db
    N, nband, B, S = env.shape
    dev = env.device
    f = np.asarray(batch.centres_mhz, float)
    depth = np.asarray(batch.depth_mm, float)
    mm = float(np.median(np.diff(depth)))
    if grid is None:
        grid = make_grid(depth, np.asarray(batch.beam_spacing_mm), B, cfg)
    if guide is not None:
        guide = torch.as_tensor(guide, dtype=torch.float64, device=dev)
        if cfg.guide_detrend:                       # remove the depth trend (shading), keep the edges
            guide = guide - torch.nanmedian(guide, 1, keepdim=True).values
        guide = torch.nan_to_num(guide, nan=0.0)
        if cfg.edge_sigma_db:
            chunk = min(chunk, 4)                                      # the edge-aware spectra gather every pixel
    parts = [_measure_chunk(env[i:i + chunk], grid, mm, float(np.median(batch.beam_spacing_mm)), cfg, noise, limited,
                            None if guide is None else guide[i:i + chunk])
             for i in range(0, N, chunk)]
    cat = lambda k: None if parts[0].get(k) is None else torch.cat([p[k] for p in parts], 0)
    return Measurement(cat("Y"), cat("W"), cat("valid"), cat("snr_p"), cat("snr_d"), f, grid, cat("omega"),
                       cat("gblock"), cat("kappa"))


def _measure_chunk(env, grid: Grid, mm: float, beam_mm: float, cfg: Config, noise, limited, guide=None) -> dict:
    N, nband, B, S = env.shape
    dev = env.device
    P = torch.pow(10.0, env.to(torch.float64) / 10)                          # [N, nband, B, S]
    # pixel validity: windowed full-band power vs noise (sld.windowed_power)
    ns_w = max(1, int(round(cfg.win_mm / mm)))
    nbw = max(1, int(round(cfg.win_mm / beam_mm)))
    Pw = _box_mean(_box_mean(P, nbw, 2), ns_w, 3)
    if noise is not None:
        Nz = torch.as_tensor(np.asarray(noise, np.float64), device=dev)[None].expand(N, -1, -1)   # [N, nband, S]
    else:
        Nz = _box_mean(_percentile(Pw, 10, 2), ns_w, 2)
    valid_px = 10 * torch.log10(Pw.sum(1) / Nz.sum(1)[:, None, :]) >= cfg.pixel_snr_db         # [N, B, S]
    del Pw
    vf = _block_fraction(grid, valid_px)                                                        # [N, R, C]
    lim = (torch.zeros((nband, S), dtype=torch.bool, device=dev) if limited is None
           else torch.as_tensor(np.asarray(limited, bool), device=dev))
    b0, b1, s0, _ = _block_index(grid, dev)
    ns, wp = grid.ns, grid.wp
    rows = torch.as_tensor(grid.s0, dtype=torch.long, device=dev)[:, None] + torch.arange(ns, device=dev)  # [R, ns]
    Nn = Nz[:, :, rows][:, :, :, None, :]                                                       # [N, nband, R, 1, ns]
    omega = gblock = kappa = None
    if guide is not None:
        cb = torch.as_tensor(grid.cb, dtype=torch.long, device=dev)
        gblock = guide[:, cb[None, :], (rows[:, ns // 2])[:, None]]                              # [N, R, C]
    if guide is not None and cfg.edge_sigma_db:
        # edge-aware spectra: every pixel of a block weighted by its guide similarity to the block centre
        nbmax = int(grid.nb.max())
        j = torch.arange(nbmax, device=dev)
        bidx = (b0[..., None] + j).clamp(max=B - 1)                                             # [R, C, nbmax]
        inb = (j[None, :] < torch.as_tensor(grid.nb, device=dev)[:, None]).to(torch.float64)[:, None, :, None]
        gp = guide[:, bidx[..., None], rows[:, None, None, :]]                                  # [N, R, C, nbmax, ns]
        u = torch.exp(-0.5 * ((gp - gblock[..., None, None]) / cfg.edge_sigma_db) ** 2) * inb
        del gp
        us = u.sum(3)                                                                            # [N, R, C, ns]
        num = (P[:, :, bidx[..., None], rows[:, None, None, :]] * u[:, None]).sum(4)            # [N, nband, R, C, ns]
        del P, u
        wsum = lambda a, sl: (a[..., sl]).sum(-1)
        up, ud, ua = wsum(us, slice(0, wp)), wsum(us, slice(ns - wp, ns)), us.sum(-1)
        Pp, Pd, Pa = (wsum(num, slice(0, wp)) / up[:, None], wsum(num, slice(ns - wp, ns)) / ud[:, None],
                      num.sum(-1) / ua[:, None])
        Nw = Nn * us[:, None]
        Np_, Nd_, Na = wsum(Nw, slice(0, wp)) / up[:, None], wsum(Nw, slice(ns - wp, ns)) / ud[:, None], \
            Nw.sum(-1) / ua[:, None]
        nbw_ = torch.as_tensor(grid.nb, dtype=torch.float64, device=dev)[:, None] * wp
        omega = torch.minimum(up, ud) / nbw_
        spos = torch.arange(ns, dtype=torch.float64, device=dev)
        cp = (us[..., :wp] * spos[:wp]).sum(-1) / up
        cd = (us[..., ns - wp:] * spos[ns - wp:]).sum(-1) / ud
        kappa = (cd - cp) / (ns - wp)
        snr = 10 * torch.log10(Pa / Na)
        if cfg.subtract_noise:
            Sp, Sd, Xm = Pp - Np_, Pd - Nd_, Pa - Na
        else:
            Sp, Sd, Xm = Pp, Pd, Pa
    else:
        # beam-averaged power of every block and sample: Pm [N, nband, R, C, ns]
        cP = torch.cat([torch.zeros_like(P[:, :, :1]), torch.cumsum(P, 2)], 2)
        del P
        sidx = s0[..., None] + torch.arange(ns, device=dev)                                     # [R, C, ns]
        nb = torch.as_tensor(grid.nb, dtype=torch.float64, device=dev)[:, None, None]
        Pm = (cP[:, :, b1[..., None], sidx] - cP[:, :, b0[..., None], sidx]) / nb
        del cP
        X = Pm - Nn if cfg.subtract_noise else Pm
        snr = 10 * torch.log10(Pm.mean(-1) / Nn.mean(-1))
        Xm = X.mean(-1)
        Sp, Sd = X[..., :wp].mean(-1), X[..., ns - wp:].mean(-1)
        Np_, Nd_ = Nn[..., :wp].mean(-1), Nn[..., ns - wp:].mean(-1)
    peak = torch.where(Xm > 0, Xm, torch.full_like(Xm, -math.inf)).amax(1)                      # [N, R, C]
    near_peak = (Xm >= peak[:, None] * 10 ** (-cfg.drop_db / 10)) & torch.isfinite(peak)[:, None]
    if cfg.subtract_noise:
        thr = 10 ** (cfg.window_snr_db / 10)
        pos = (Sp > thr * Np_) & (Sd > thr * Nd_)
    else:
        pos = (Sp > 0) & (Sd > 0)
    one = torch.ones_like(Sp)
    y = 10 * torch.log10(torch.where(pos, Sp, one) / torch.where(pos, Sd, one))
    sp_, sd_ = Sp / Np_, Sd / Nd_
    ok = near_peak & pos & ~lim[:, rows].any(-1)[None, :, :, None]
    if cfg.subtract_noise:
        ok &= snr >= cfg.band_snr_db
    ok &= (vf >= cfg.valid_frac)[:, None]
    if omega is not None:
        ok &= (omega >= cfg.edge_min_frac)[:, None]
    ok &= (ok.sum(1) >= cfg.min_bands)[:, None]
    nan = torch.full_like(y, math.nan)
    return dict(Y=torch.where(ok, y, nan), W=ok, valid=ok.any(1), snr_p=torch.where(ok, sp_, nan),
                snr_d=torch.where(ok, sd_, nan), omega=omega, gblock=gblock, kappa=kappa)


def block_weight(meas: Measurement) -> torch.Tensor:
    """[N, R, C] SNR reliability of each block (sld.block_weight)."""
    g = torch.as_tensor(2.0 * meas.grid.dz_cm * 10.0 * np.asarray(meas.f, float), device=meas.Y.device)
    g = g[None, :, None, None]
    v = (1 + 1 / meas.snr_p) ** 2 + (1 + 1 / meas.snr_d) ** 2
    w = torch.where(meas.W & torch.isfinite(v), 1.0 / v, torch.zeros_like(v))
    gbar = (w * g).sum(1) / w.sum(1)
    info = (w * (g - gbar[:, None]) ** 2).sum(1)
    if meas.kappa is not None:
        info = info * meas.kappa ** 2
    return torch.where(meas.valid & torch.isfinite(info), info, torch.zeros_like(info))


# ------------------------------------------------------------------------------------------------------------------
# reference


def exclude_blocks(meas: Measurement, pixel_masks) -> Measurement:
    """Remove every block that touches pixel_masks [N, B, S] (bool) of the matching frame."""
    m = torch.as_tensor(np.asarray(pixel_masks, bool), device=meas.Y.device)
    bad = _block_fraction(meas.grid, m) > 0
    W = meas.W & ~bad[:, None]
    return meas.replace(Y=torch.where(W, meas.Y, torch.full_like(meas.Y, math.nan)), W=W, valid=meas.valid & ~bad)


def _nanmedian(x: torch.Tensor, dim: int) -> tuple:
    """np.nanmedian along dim (mean of the two middle values) and the count of finite values."""
    fin = torch.isfinite(x)
    n = fin.sum(dim, keepdim=True)
    xs = torch.sort(torch.where(fin, x, torch.full_like(x, math.inf)), dim=dim).values
    lo = torch.gather(xs, dim, ((n - 1) // 2).clamp(min=0))
    hi = torch.gather(xs, dim, (n // 2).clamp(max=x.shape[dim] - 1))
    med = torch.where(n > 0, 0.5 * (lo + hi), torch.full_like(lo, math.nan))
    return med.squeeze(dim), n.squeeze(dim)


def self_reference(measurements, min_blocks: int = 200, a_ref: float = 0.0) -> dict:
    """reference.self_reference over Measurement batches (same grid).  Returns the same numpy dict."""
    ms = [measurements] if isinstance(measurements, Measurement) else list(measurements)
    g, f = ms[0].grid, ms[0].f
    Ys = torch.cat([m.Y.to(torch.float32).permute(0, 3, 1, 2).reshape(-1, *m.Y.shape[1:3]) for m in ms], 0)
    med, n = _nanmedian(Ys, 0)                                                     # [nband, R]
    med = med.cpu().numpy(); n = n.cpu().numpy()
    D = np.where(n >= min_blocks, med - (2.0 * g.dz_cm * np.asarray(f) * a_ref / TO_MM)[:, None], np.nan)
    return dict(D=D, n=n, f=np.asarray(f), s0=g.s0, ns=g.ns)


def apply_reference(meas: Measurement, ref: dict | None) -> Measurement:
    if ref is None:
        return meas
    g = meas.grid
    if not (np.array_equal(g.s0, ref["s0"]) and g.ns == ref["ns"] and np.allclose(meas.f, ref["f"])):
        raise ValueError("reference and measurement use different grid rows or bands")
    D = torch.as_tensor(ref["D"], dtype=meas.Y.dtype, device=meas.Y.device)[None, :, :, None]
    Y = meas.Y - (D if meas.kappa is None else D * meas.kappa[:, None])   # the system term over kappa dz
    W = meas.W & torch.isfinite(Y)
    return meas.replace(Y=torch.where(W, Y, torch.full_like(Y, math.nan)), W=W, valid=meas.valid & (W.sum(1) >= 2))


# ------------------------------------------------------------------------------------------------------------------
# RED solver


def _median_index(dom: np.ndarray, k: int) -> np.ndarray:
    """[R*C, k*k] flat source index of every block's k x k window, with the symmetric extension at the edges of the
    domain's bounding box (as solver.masked_median after the crop in red_pg)."""
    R, C = dom.shape
    r = np.nonzero(dom.any(1))[0]; c = np.nonzero(dom.any(0))[0]
    h = k // 2

    def refl(i, a, n):
        m = np.mod(i - a, 2 * n)
        return a + np.where(m >= n, 2 * n - 1 - m, m)

    rr = refl(np.arange(R)[:, None] + np.arange(-h, h + 1)[None], r[0], r[-1] - r[0] + 1)        # [R, k]
    cc = refl(np.arange(C)[:, None] + np.arange(-h, h + 1)[None], c[0], c[-1] - c[0] + 1)        # [C, k]
    idx = rr[:, None, :, None] * C + cc[None, :, None, :]                                         # [R, C, k, k]
    return idx.reshape(R * C, k * k)


class _Problem:
    """The batched RED problem of N frames: normal equations, domains, and the median gather.

    The median runs on the domain blocks only (rows of a packed list over all frames).  A window slot outside the
    domain is filled with -inf or +inf so that the plain median of the k*k slots is the median of the domain values
    (lower and upper middle: the first floor(m/2) resp. ceil(m/2) of the m missing slots get -inf, the rest +inf);
    rows with an even number of domain values average the two.  `select` keeps a subset of the frames."""

    def __init__(self, Y, W, g, dom, mu, k_med, L, lam, omega=None, gblock=None, med_sigma=None, kappa=None):
        self.N, _, self.R, self.C = Y.shape
        dev, dt = Y.device, Y.dtype
        w = W.to(dt) if omega is None else W.to(dt) * omega.to(dt)[:, None]
        y = torch.where(W, Y, torch.zeros_like(Y))
        gg = torch.as_tensor(np.asarray(g, float), dtype=dt, device=dev)[None, :, None, None]
        if kappa is not None:
            gg = gg * kappa.to(dt)[:, None]
        self.a11, self.a12, self.a22 = (w * gg * gg).sum(1), (w * gg).sum(1), w.sum(1)
        self.b1, self.b2 = (w * gg * y).sum(1), (w * y).sum(1)
        self.nW = W.sum(1)
        self.dom = dom
        self.mu, self.L, self.lam, self.rho = mu, L, lam, mu * L
        # packed domain blocks: frame slot, own flat position, window source positions and which are missing
        d = dom.cpu().numpy().reshape(self.N, -1)
        frame, pos, src = [], [], []
        for i in range(self.N):
            p = np.nonzero(d[i])[0]
            if len(p):
                frame.append(np.full(len(p), i)); pos.append(p)
                src.append(_median_index(d[i].reshape(self.R, self.C), k_med)[p])
        t = lambda a, dt=torch.long: torch.as_tensor(np.concatenate(a) if a else np.zeros(0, int), dtype=dt,
                                                         device=dev)
        self.frame, self.pos = t(frame), t(pos)
        self.src = t(src).reshape(-1, k_med * k_med)
        miss = ~dom.reshape(self.N, -1)[self.frame[:, None], self.src]
        m = miss.sum(1, keepdim=True)
        cm = torch.cumsum(miss.to(torch.int32), 1)
        inf = torch.full(miss.shape, math.inf, dtype=dt, device=dev)
        self.fill_lo = torch.where(miss, torch.where(cm <= m // 2, -inf, inf), torch.zeros_like(inf))
        self.odd = torch.nonzero(m[:, 0] % 2 == 1)[:, 0]                   # even number of domain values
        self.fill_hi = torch.where(miss[self.odd], torch.where(cm[self.odd] <= (m[self.odd] + 1) // 2,
                                                                -inf[self.odd], inf[self.odd]),
                                   torch.zeros_like(inf[self.odd]))
        self.miss = miss
        self.mw = None                                                      # weighted median: neighbour weights
        if gblock is not None and med_sigma:
            gb = gblock.to(dt).reshape(self.N, -1)
            gsrc = gb[self.frame[:, None], self.src]
            gc = gb[self.frame, self.pos][:, None]
            self.mw = torch.exp(-0.5 * ((gsrc - gc) / med_sigma) ** 2) * (~miss).to(dt)

    def select(self, keep: torch.Tensor) -> "_Problem":
        """The problem restricted to the frames where keep [N] is True (in order)."""
        new = object.__new__(_Problem)
        new.__dict__.update(self.__dict__)
        for k in ("a11", "a12", "a22", "b1", "b2", "nW", "dom"):
            setattr(new, k, getattr(self, k)[keep])
        new.N = int(keep.sum())
        rows = keep[self.frame]
        slot = torch.cumsum(keep.to(torch.long), 0) - 1
        new.frame, new.pos, new.src = slot[self.frame[rows]], self.pos[rows], self.src[rows]
        new.miss, new.fill_lo = self.miss[rows], self.fill_lo[rows]
        new.mw = None if self.mw is None else self.mw[rows]
        odd_rows = rows[self.odd]
        new.fill_hi = self.fill_hi[odd_rows]
        rank = torch.cumsum(rows.to(torch.long), 0) - 1
        new.odd = rank[self.odd[odd_rows]]
        return new

    def solve(self, rho=0.0, v1=0.0, v2=0.0):
        a11, a22 = self.a11 + rho, self.a22 + rho
        r1, r2 = self.b1 + rho * v1, self.b2 + rho * v2
        det = a11 * a22 - self.a12 ** 2
        return (a22 * r1 - self.a12 * r2) / det, (a11 * r2 - self.a12 * r1) / det

    def median(self, b):
        """Masked k x k median of b [N, R, C] over each frame's domain (solver.masked_median); b outside."""
        bf = b.reshape(self.N, -1)
        v = bf[self.frame[:, None], self.src]
        if self.mw is not None:                                           # weighted (lower) median
            vs, order = torch.sort(torch.where(self.miss, torch.full_like(v, math.inf), v), 1)
            cw = torch.cumsum(torch.gather(self.mw, 1, order), 1)
            i = (cw < 0.5 * cw[:, -1:]).sum(1, keepdim=True).clamp(max=v.shape[1] - 1)
            out = bf.clone()
            out[self.frame, self.pos] = torch.gather(vs, 1, i)[:, 0]
            return out.reshape(b.shape)
        med = torch.median(v + self.fill_lo, 1).values                    # domain slots: + 0
        if len(self.odd):
            hi = torch.median(v[self.odd] + self.fill_hi, 1).values
            med = med.index_put((self.odd,), 0.5 * (med[self.odd] + hi))
        out = bf.clone()
        out[self.frame, self.pos] = med
        return out.reshape(b.shape)

    def T(self, b, c):
        """One RED-PG step (solver.red_pg)."""
        db = self.median(b)
        dc = torch.sign(c) * torch.clamp(torch.abs(c) - self.lam, min=0.0)
        L = self.L
        vb, vc = db / L - (1 - L) / L * b, dc / L - (1 - L) / L * c
        nb_, nc_ = self.solve(self.rho, vb, vc)
        z = torch.zeros_like(nb_)
        return torch.where(self.dom, nb_, z), torch.where(self.dom, nc_, z)

    def init(self):
        """Least-squares start (solver.least_squares + the fill of red_pg)."""
        beta, c = self.solve()
        ok = self.dom & torch.isfinite(beta) & (self.nW >= 2)
        nan = torch.full_like(beta, math.nan)
        beta = torch.where(ok, beta, nan); c = torch.where(ok, c, nan)
        fill, _ = _nanmedian(torch.where(self.dom, beta, nan).reshape(self.N, -1), 1)
        fill = torch.nan_to_num(fill, nan=0.0)[:, None, None]
        z = torch.zeros_like(beta)
        b = torch.where(self.dom, torch.where(torch.isfinite(beta), beta, fill.expand_as(beta)), z)
        c = torch.where(self.dom, torch.nan_to_num(c), z)
        return b, c


def red_pg(Y: torch.Tensor, W: torch.Tensor, g, mu: float, dom: torch.Tensor, k_med: int = 7, L: float = 1.01,
           lam: float | None = None, tol: float = 1e-5, max_iter: int = 3000, accel: str = "none", mem: int = 10,
           check_every: int = 8, compact: float = 0.75, omega=None, gblock=None, med_sigma=None, kappa=None):
    """Batched solver.red_pg.  Y, W [N, nband, R, C]; dom [N, R, C].  Returns beta, c [N, R, C] (NaN outside the
    domains, beta in the units of g), iters [N], converged [N].  Every frame runs until its own stopping test, as when
    solved alone.  Every `check_every` iterations the batch is checked, and shrunk to the running frames once they are
    fewer than `compact` of it.  omega [N, R, C]: block data weights; gblock [N, R, C] with med_sigma: the median is
    weighted by the guide similarity of each neighbour to the block (edge-aware RED); kappa [N, R, C] scales g per
    block."""
    lam = 1.0 / mu if lam is None else lam
    pb = _Problem(Y, W, g, dom, mu, k_med, L, lam, omega, gblock, med_sigma, kappa)
    b, c = pb.init()
    if accel == "anderson":
        b, c, it, conv = _anderson(pb, b, c, tol, max_iter, mem, check_every, compact)
    elif accel == "none":
        b, c, it, conv = _plain(pb, b, c, tol, max_iter, check_every, compact)
    else:
        raise ValueError("accel must be 'none' or 'anderson'")
    nan = torch.full_like(b, math.nan)
    return torch.where(dom, b, nan), torch.where(dom, c, nan), it, conv


class _Batch:
    """Per-frame state of a running batch, with the results of finished frames kept at their original index."""

    def __init__(self, pb: _Problem, state: dict, results: dict):
        self.pb, self.state = pb, state
        self.ids = torch.arange(pb.N, device=pb.dom.device)
        self.out = {k: v.clone() for k, v in results.items()}
        self.active = torch.ones(pb.N, dtype=torch.bool, device=pb.dom.device)

    def maybe_compact(self, results: dict, compact: float) -> bool:
        """False when every frame has stopped.  Writes the results of all frames, then drops the stopped ones if
        they are more than 1 - compact of the batch."""
        n = int(self.active.sum())
        if n == 0 or n <= compact * len(self.active):
            for k, v in results.items():
                self.out[k][self.ids] = v
            if n == 0:
                return False
            keep = self.active
            self.pb = self.pb.select(keep)
            self.ids = self.ids[keep]
            self.state = {k: v[keep] for k, v in self.state.items()}
            self.active = self.active[keep]
        return True


def _plain(pb: _Problem, b, c, tol, max_iter, check_every, compact):
    N, dev = pb.N, b.device
    bt = _Batch(pb, dict(b=b, c=c), dict(b=b, c=c, it=torch.full((N,), max_iter, dtype=torch.long, device=dev),
                                          conv=torch.zeros(N, dtype=torch.bool, device=dev)))
    st = bt.state
    st["it"] = bt.out["it"].clone(); st["conv"] = bt.out["conv"].clone()
    for k in range(1, max_iter + 1):
        pb, st = bt.pb, bt.state
        b, c = st["b"], st["c"]
        nb_, nc_ = pb.T(b, c)
        num = torch.sqrt(((nb_ - b) ** 2).sum((1, 2)) + ((nc_ - c) ** 2).sum((1, 2)))
        den = torch.sqrt((nb_ ** 2).sum((1, 2)) + (nc_ ** 2).sum((1, 2))) + 1e-30
        a = bt.active[:, None, None]
        st["b"] = torch.where(a, nb_, b); st["c"] = torch.where(a, nc_, c)
        done = bt.active & (num / den < tol)
        st["it"] = torch.where(done, torch.full_like(st["it"], k), st["it"])
        st["conv"] = st["conv"] | done
        bt.active = bt.active & ~done
        if (k % check_every == 0 or k == max_iter) and not bt.maybe_compact(bt.state, compact):
            break
    bt.maybe_compact(bt.state, 1.0)
    o = bt.out
    return o["b"], o["c"], o["it"], o["conv"]


def _anderson(pb: _Problem, b, c, tol, max_iter, mem, check_every, compact, reg=1e-10, restart=1.0):
    """Batched solver._anderson: the same iterates; a rejected step counts as an extra iteration.  Each frame keeps
    its own history."""
    N, dev = pb.N, b.device
    nrm = lambda a: torch.sqrt((a ** 2).sum((1, 2, 3)))

    def Tv(pb, x):
        nb_, nc_ = pb.T(x[:, 0], x[:, 1])
        return torch.stack([nb_, nc_], 1)

    x = torch.stack([b, c], 1)                                              # [N, 2, R, C]
    gx = Tv(pb, x)
    d = x[0].numel()
    st = dict(x=x, gx=gx, f=gx - x, out=gx.clone(),
              G=torch.zeros((N, mem + 1, d), dtype=x.dtype, device=dev),
              F=torch.zeros((N, mem + 1, d), dtype=x.dtype, device=dev),
              h=torch.zeros(N, dtype=torch.long, device=dev),               # history length per frame
              it=torch.full((N,), max_iter, dtype=torch.long, device=dev),
              conv=torch.zeros(N, dtype=torch.bool, device=dev))
    res_keys = ("out", "it", "conv")
    bt = _Batch(pb, st, {k: st[k] for k in res_keys})
    eye = torch.eye(mem, dtype=x.dtype, device=dev)
    for k in range(1, max_iter + 1):
        pb, st, active = bt.pb, bt.state, bt.active
        n = pb.N
        x, gx, f = st["x"], st["gx"], st["f"]
        done = active & (nrm(f) < tol * (nrm(gx) + 1e-30))
        st["out"] = torch.where(done[:, None, None, None], gx, st["out"])
        st["it"] = torch.where(done, torch.full_like(st["it"], k), st["it"])
        st["conv"] = st["conv"] | done
        active = bt.active = active & ~done
        if k % check_every == 0 and not bt.maybe_compact({kk: bt.state[kk] for kk in res_keys}, compact):
            break
        pb, st, active = bt.pb, bt.state, bt.active
        n = pb.N
        x, gx, f, G, F, h = st["x"], st["gx"], st["f"], st["G"], st["F"], st["h"]
        G = torch.roll(G, -1, 1); F = torch.roll(F, -1, 1)
        G[:, -1] = gx.reshape(n, d); F[:, -1] = f.reshape(n, d)
        h = torch.clamp(h + 1, max=mem + 1)                                # valid entries
        dF = F[:, 1:] - F[:, :-1]; dG = G[:, 1:] - G[:, :-1]                # [n, mem, d], newest last
        okf = (torch.arange(mem, device=dev)[None] >= (mem + 1 - h)[:, None]).to(x.dtype)   # valid differences
        A = dF @ dF.transpose(1, 2) * okf[:, :, None] * okf[:, None, :]
        tr = torch.diagonal(A, dim1=1, dim2=2).sum(-1)[:, None, None]
        tr = torch.where(tr > 0, tr, torch.ones_like(tr) / reg)              # no change in the history: gamma = 0
        A = A + reg * tr * eye + torch.diag_embed(1.0 - okf)
        rhs = (dF @ f.reshape(n, d, 1))[..., 0] * okf
        gam = torch.linalg.solve(A, rhs)                                    # zero where the history is short
        xn = gx - (gam[:, None, :] @ dG)[:, 0].reshape(gx.shape)
        gn = Tv(pb, xn); fn = gn - xn
        # a step whose residual grows clears the history; a mixed one is also rejected (x kept, h = -1: the next two
        # steps are plain, as solver._anderson, which takes the first of them in the same iteration)
        bad = active & (nrm(fn) > restart * nrm(f))
        mixed = h > 1
        a = (active & ~(bad & mixed))[:, None, None, None]
        st.update(x=torch.where(a, xn, x), gx=torch.where(a, gn, gx), f=torch.where(a, fn, f), G=G, F=F,
                  h=torch.where(bad, torch.where(mixed, torch.full_like(h, -1), torch.zeros_like(h)), h))
    st = bt.state
    st["out"] = torch.where(bt.active[:, None, None, None], st["gx"], st["out"])
    bt.maybe_compact({kk: st[kk] for kk in res_keys}, 1.0)
    o = bt.out
    return o["out"][:, 0], o["out"][:, 1], o["it"], o["conv"]


# ------------------------------------------------------------------------------------------------------------------
# estimator


class GPUEstimator:
    def __init__(self, cfg: Config = Config(), noise=None, limited=None, device="cuda", dtype=torch.float64,
                 chunk: int = 16):
        """As pipeline.Estimator, on `device`.  dtype of the measurement and the solver (float64 matches the CPU
        path; float32 is faster on consumer GPUs).  chunk: frames measured at a time."""
        self.cfg, self.noise, self.limited = cfg, noise, limited
        self.device, self.dtype, self.chunk = torch.device(device), dtype, chunk

    def _batch(self, frames) -> FrameBatch:
        if isinstance(frames, FrameBatch):
            return FrameBatch(frames.env_db.to(self.device), frames.centres_mhz, frames.depth_mm,
                              frames.beam_spacing_mm)
        if isinstance(frames, Frame):
            frames = [frames]
        return FrameBatch.from_frames(frames, self.device)

    def measure(self, frames, guide=None) -> Measurement:
        """frames: FrameBatch, Frame or iterable of Frame (one geometry); guide [N, B, S] (see measure)."""
        return measure(self._batch(frames), self.cfg, self.noise, self.limited, chunk=self.chunk, guide=guide)

    def self_reference(self, batches, exclude=None) -> dict:
        """batches: iterable of FrameBatch / Frame lists (or one FrameBatch); exclude: matching iterable of
        [N, B, S] pixel masks or None.  Measured without a guide (full windows); apply_reference scales it by kappa
        for edge-aware measurements."""
        if isinstance(batches, (FrameBatch, Frame)):
            batches = [batches]
        elif isinstance(batches, (list, tuple)) and batches and isinstance(batches[0], Frame):
            batches = [batches]
        ms = []
        for i, fb in enumerate(batches):
            m = self.measure(fb)
            mask = None if exclude is None else exclude[i]
            if mask is not None and np.any(mask):
                m = exclude_blocks(m, mask)
            ms.append(m.replace(snr_p=None, snr_d=None, W=None, valid=None, omega=None, gblock=None))  # only Y
        return self_reference(ms, self.cfg.ref_min_blocks)

    def solve(self, meas: Measurement, method: str = "red", dom=None):
        """beta [N, R, C] (dB/mm/MHz), c, iters, converged (tensors)."""
        cfg = self.cfg
        Y = meas.Y.to(self.dtype)
        g = 2.0 * meas.grid.dz_cm * np.asarray(meas.f, float)
        d = meas.valid if dom is None else (torch.as_tensor(np.asarray(dom, bool), device=Y.device) & meas.valid)
        if method == "ls":
            pb = _Problem(Y, meas.W, g, d, cfg.mu, cfg.k_med, cfg.L, 0.0, meas.omega, kappa=meas.kappa)
            beta, c = pb.solve()
            ok = d & torch.isfinite(beta) & (pb.nW >= 2)
            nan = torch.full_like(beta, math.nan)
            n = len(meas)
            return (torch.where(ok, beta, nan) * TO_MM, torch.where(ok, c, nan),
                    torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.bool))
        b, c, it, conv = red_pg(Y, meas.W, g, cfg.mu, d, cfg.k_med, cfg.L, cfg.lam, cfg.tol, cfg.max_iter, cfg.accel,
                                cfg.mem, omega=meas.omega, gblock=meas.gblock, med_sigma=cfg.med_sigma_db,
                                kappa=meas.kappa)
        return b * TO_MM, c, it, conv

    def estimate(self, frames, reference: dict | None = None, method: str = "red", guide=None) -> list:
        """One Estimate per frame of `frames` (FrameBatch, Frame or iterable of Frame); guide [N, B, S] or None.
        The compounding weight of a block is its SNR weight times its edge-aware data weight omega."""
        fb = self._batch(frames)
        m = apply_reference(self.measure(fb, guide), reference)
        beta, c, _, conv = self.solve(m, method)
        w = block_weight(m) if m.omega is None else block_weight(m) * m.omega
        beta, c, w, conv = beta.cpu().numpy(), c.cpu().numpy(), w.cpu().numpy(), conv.cpu().numpy()
        return [Estimate(beta[i], c[i], w[i], m.grid, fb.shape, bool(conv[i])) for i in range(len(beta))]
