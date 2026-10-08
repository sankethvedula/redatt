"""Equivalence with the research pipeline this package was extracted from (phantom_detgc/pdg in the qti-specular
repository): same band envelopes, same SLD measurements, same RED map on a real QT frame.  Skipped where that
repository or the exam data are not available."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
EXAM, VIEW, LEVEL = "QT007-S1-BL", 5, 16


def _pdg():
    sys.path.insert(0, str(REPO / "phantom_detgc"))
    os.environ.setdefault("PDG_DEMOD", "xml")
    os.environ.setdefault("PDG_NOISE", "system")
    try:
        from pdg import red, redqt, sldgrid as sg
        from pdg.paths import exam_dir
    except Exception as e:                                          # noqa: BLE001
        pytest.skip(f"research pipeline not importable: {e}")
    if not (exam_dir(EXAM) / "da" / "refl").exists():
        pytest.skip("exam data not available")
    return red, redqt, sg, exam_dir


def test_matches_research_pipeline():
    red, redqt, sg, exam_dir = _pdg()
    import dataclasses
    import redatt as ra
    from redatt.qt import QTProbe
    # research pipeline (self reference, no noise handling)
    ctx = redqt.context(exam_dir(EXAM))
    N = redqt.water_noise(ctx, cache=redqt.noise_path(EXAM), log=lambda *a: None)
    cfg_old = dataclasses.replace(redqt.grid_config(), subtract_noise=False)
    F = ctx.frame(VIEW, LEVEL, lesion=False)
    m_old = sg.measure(F["env"], ctx.centres_mhz, ctx.depth, redqt.beam_spacing(ctx), cfg_old, limited=ctx.limited,
                       noise=N)
    ref_old = redqt.load_reference(name=f"self_nn_{EXAM}", min_n=200)
    b_old = red.solve("red", sg.apply_reference(m_old, ref_old), 100.0, tol=1e-5, max_iter=3000).beta
    # this package
    probe = QTProbe(exam_dir(EXAM), unit=1, temperature="mean")   # as the research pipeline
    fr = probe.frame(VIEW, LEVEL)
    est = probe.estimator()
    assert np.allclose(fr.depth_mm, ctx.depth, atol=1e-9)
    assert np.allclose(fr.beam_spacing_mm, redqt.beam_spacing(ctx), atol=1e-9)
    assert np.allclose(fr.env_db, F["env"], atol=1e-4)
    m_new = est.measure(fr)
    assert np.array_equal(m_new["W"], m_old["W"])
    assert np.allclose(m_new["Y"], m_old["Y"], atol=1e-4, equal_nan=True)
    ref_new = dict(D=ref_old["D"], s0=ref_old["s0"], ns=ref_old["ns"], f=ref_old["f"])
    out = est.estimate(fr, ref_new)
    assert np.allclose(out.beta, b_old, atol=1e-5, equal_nan=True)
    # the self reference itself, on a handful of frames
    frames_old = [sg.measure(ctx.frame(v, L, lesion=False)["env"], ctx.centres_mhz, ctx.depth, redqt.beam_spacing(ctx),
                             cfg_old, limited=ctx.limited, noise=N) for v in ctx.views[::20] for L in (14, 18)]
    t_old = sg.reference_table([dict(Y=m["Y"].astype(np.float32), grid=m["grid"], f=m["f"]) for m in frames_old],
                               min_n=20)
    t_new = ra.self_reference([est.measure(probe.frame(v, L)) for v in ctx.views[::20] for L in (14, 18)], 20)
    assert np.allclose(t_new["D"], t_old["D"], atol=1e-4, equal_nan=True)
