"""Synthetic checks: a frame with known attenuation (background + a more attenuating inclusion) and speckle."""
import numpy as np

import redatt as ra


def synthetic_frame(beta_bg=0.06, beta_inc=0.12, B=160, S=550, seed=0):
    cfg = ra.Config(block_samples=49)
    rng = np.random.default_rng(seed)
    f = cfg.centres()                                            # MHz
    depth = 50 + 0.151 * np.arange(S)                            # mm
    bspace = np.full(S, 0.45)                                    # mm between beams
    beta = np.full((B, S), beta_bg)
    xb, zs = np.meshgrid(np.arange(B), depth, indexing="ij")
    inc = (xb - B / 2) ** 2 / 25 ** 2 + (zs - 90) ** 2 / 12 ** 2 < 1
    beta[inc] = beta_inc
    att_db = 2 * np.cumsum(beta, 1) * 0.151                      # one-way dB/MHz, two-way path
    P = np.exp(-0.5 * ((f[:, None, None] - 3.1) / 0.6) ** 2) * 10 ** (-att_db[None] * f[:, None, None] / 10)
    P = P * rng.exponential(1.0, size=P.shape)                   # speckle
    env = 10 * np.log10(P)
    noise = np.full((len(f), S), 1e-12)
    return ra.Frame(env.astype(np.float32), f, depth, bspace), cfg, noise, inc


def test_uniform_background_recovered():
    fr, cfg, noise, inc = synthetic_frame(beta_bg=0.06, beta_inc=0.06)
    est = ra.Estimator(cfg, noise=noise)
    out = est.estimate(fr)                                       # no reference: synthetic has no system term
    assert out.converged
    assert abs(np.nanmedian(out.beta) - 0.06) < 0.01


def test_inclusion_contrast_recovered():
    fr, cfg, noise, inc = synthetic_frame()
    out = ra.Estimator(cfg, noise=noise).estimate(fr)
    pix = out.pixels()
    contrast = np.nanmedian(pix[inc]) - np.nanmedian(pix[~inc])
    assert 0.03 < contrast < 0.09                                # true 0.06, blurred by 7.3 mm blocks


def test_self_reference_removes_common_term():
    # a range-dependent system term common to every frame must cancel with the self reference
    frs = [synthetic_frame(beta_inc=0.06, seed=s)[0] for s in range(6)]
    _, cfg, noise, _ = synthetic_frame()
    sys_db = 0.8 * (frs[0].depth_mm - 50)[None, None, :] * (frs[0].centres_mhz[:, None, None] - 3.1) / 10
    frs = [ra.Frame(f.env_db + sys_db.astype(np.float32), f.centres_mhz, f.depth_mm, f.beam_spacing_mm) for f in frs]
    est = ra.Estimator(cfg, noise=noise)
    est.cfg.ref_min_blocks = 20
    ref = est.self_reference(frs[1:])
    out = est.estimate(frs[0], ref)
    assert abs(np.nanmedian(out.beta)) < 0.01                    # relative to the exam's own tissue: ~0


def test_stepped_phase_constant_frequency():
    t = np.arange(0, 400) * 8.0
    ph = ra.stepped_phase([3.6e6] * 4, [100, 100, 100, 100], 40e6, t)
    assert np.allclose(ph, 2 * np.pi * 3.6e6 * t / 40e6)
