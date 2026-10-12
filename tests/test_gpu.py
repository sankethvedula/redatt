"""The GPU (torch) path against the CPU path on synthetic frames, and Anderson-accelerated RED against plain RED run
to its fixed point.  GPU tests are skipped without torch / CUDA."""
import dataclasses

import numpy as np
import pytest

import redatt as ra
from test_synthetic import synthetic_frame


def test_anderson_reaches_the_red_fixed_point():
    fr, cfg, noise, _ = synthetic_frame()
    m = ra.Estimator(cfg, noise=noise).measure(fr)
    g = ra.sld.g_vector(m["f"], m["grid"].dz_cm)
    b_ref, *_ = ra.red_pg(m["Y"], m["W"], g, cfg.mu, m["valid"], tol=1e-11, max_iter=40000)
    b_aa, _, it, conv = ra.red_pg(m["Y"], m["W"], g, cfg.mu, m["valid"], tol=1e-9, accel="anderson")
    assert conv
    assert np.nanmax(np.abs(b_aa - b_ref)) * ra.TO_MM < 1e-6


def _gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from redatt import gpu
    return gpu


@pytest.mark.parametrize("accel,tol,atol", [("none", 1e-5, 1e-10), ("anderson", 1e-9, 1e-6)])
def test_gpu_matches_cpu(accel, tol, atol):
    gpu = _gpu()
    frs = [synthetic_frame(seed=s)[0] for s in range(3)]
    _, cfg, noise, _ = synthetic_frame()
    cfg = dataclasses.replace(cfg, accel=accel, tol=tol, ref_min_blocks=20)
    est = ra.Estimator(cfg, noise=noise)
    gest = gpu.GPUEstimator(cfg, noise=noise)
    ref_c = est.self_reference(frs)
    ref_g = gest.self_reference(frs)
    assert np.allclose(ref_c["D"], ref_g["D"], atol=1e-6, equal_nan=True)
    outs = gest.estimate(frs, ref_c)
    for fr, og in zip(frs, outs):
        oc = est.estimate(fr, ref_c)
        assert og.converged == oc.converged
        assert np.allclose(og.beta, oc.beta, atol=atol, equal_nan=True)
        assert np.allclose(og.weight, oc.weight, rtol=1e-9, equal_nan=True)


def test_gpu_measure_matches_cpu():
    gpu = _gpu()
    fr, cfg, noise, _ = synthetic_frame()
    for nz in (noise, None):                                    # given noise, and the frame's own estimate
        mc = ra.Estimator(cfg, noise=nz).measure(fr)
        mg = gpu.GPUEstimator(cfg, noise=nz).measure(fr).to_numpy()[0]
        assert np.array_equal(mc["W"], mg["W"])
        assert np.allclose(mc["Y"], mg["Y"], atol=1e-9, equal_nan=True)


def test_gpu_band_envelopes_match_cpu():
    gpu = _gpu()
    import torch
    rng = np.random.default_rng(0)
    z = rng.standard_normal((2, 16, 300)) + 1j * rng.standard_normal((2, 16, 300))
    cfg = ra.Config()
    fdb = rng.standard_normal((len(cfg.centres()), 300))
    e_g = gpu.band_envelopes(torch.as_tensor(z, device="cuda"), 5e6, 3.5e6, cfg.centres(), cfg.df, 2048, fdb)
    for i in range(2):
        e_c = ra.band_envelopes(z[i], 5e6, 3.5e6, cfg.centres(), cfg.df, 2048, fdb)
        assert np.allclose(e_g[i].cpu().numpy(), e_c, atol=1e-4)


def test_edge_aware_with_flat_guide_is_plain():
    # a constant guide keeps every pixel (weights 1, kappa 1, omega 1): edge-aware spectra reduce to the plain ones
    gpu = _gpu()
    import torch
    fr, cfg, noise, _ = synthetic_frame()
    cfg = dataclasses.replace(cfg, accel="anderson", tol=1e-9)
    plain = gpu.GPUEstimator(cfg, noise=noise).estimate([fr])[0]
    flat = torch.zeros((1,) + fr.shape, dtype=torch.float64, device="cuda")
    edge = gpu.GPUEstimator(dataclasses.replace(cfg, edge_sigma_db=2.0), noise=noise).estimate([fr], guide=flat)[0]
    assert np.allclose(edge.beta, plain.beta, atol=1e-6, equal_nan=True)


def test_guided_median_keeps_a_guided_edge():
    # an inclusion that the guide outlines keeps more contrast with the guided median than with the plain one
    gpu = _gpu()
    import torch
    fr, cfg, noise, inc = synthetic_frame(B=160, S=550)
    guide = torch.as_tensor(np.where(inc, -6.0, 0.0)[None], device="cuda")
    out = {}
    for name, c in (("plain", cfg), ("guided", dataclasses.replace(cfg, med_sigma_db=2.0))):
        pix = gpu.GPUEstimator(c, noise=noise).estimate([fr], guide=guide)[0].pixels()
        out[name] = np.nanmedian(pix[inc]) - np.nanmedian(pix[~inc])
    assert out["guided"] > out["plain"]
