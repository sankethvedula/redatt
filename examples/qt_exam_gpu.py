#!/usr/bin/env python3
"""A whole QT exam on the GPU: self reference from the exam, then RED attenuation of every level of every view
(one batch per view).  Saves beta, weight and convergence for each frame.

    python examples/qt_exam_gpu.py /path/to/QT007-S1-BL [--every-view 3 --every-level 2] [--accel anderson]
"""
import argparse
import time

import numpy as np
import torch

from redatt.qt import QTProbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exam"); ap.add_argument("--every-view", type=int, default=3)
    ap.add_argument("--every-level", type=int, default=2); ap.add_argument("--views", type=int, default=None)
    ap.add_argument("--accel", default="none", choices=["none", "anderson"])
    ap.add_argument("--tol", type=float, default=None, help="default: 1e-5 (none), 1e-8 (anderson)")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="qt_exam_gpu.npz")
    a = ap.parse_args()

    probe = QTProbe(a.exam, unit=1)
    probe.cfg.accel = a.accel
    probe.cfg.tol = a.tol if a.tol else (1e-8 if a.accel == "anderson" else 1e-5)
    est = probe.gpu_estimator(a.device)
    views = probe.scan.views(1)[:a.views]
    sync = torch.cuda.synchronize if a.device.startswith("cuda") else (lambda: None)

    t0 = time.time()
    ref = est.self_reference([probe.frame_batch(v, slice(None, None, a.every_level), a.device)
                              for v in views[::a.every_view]])
    sync(); t_ref = time.time() - t0
    print(f"self reference: {t_ref:.1f} s")

    beta, weight, conv, t_io, t_est = [], [], [], 0.0, 0.0
    for v in views:
        t = time.time(); fb = probe.frame_batch(v, device=a.device); sync(); t_io += time.time() - t
        t = time.time(); outs = est.estimate(fb, ref); sync(); t_est += time.time() - t
        beta.append(np.stack([o.beta for o in outs])); weight.append(np.stack([o.weight for o in outs]))
        conv.append([o.converged for o in outs])
    n = sum(len(b) for b in beta)
    print(f"{n} frames: read + front end {t_io:.1f} s, measure + RED {t_est:.1f} s "
          f"({1e3 * t_est / n:.0f} ms/frame); all converged: {bool(np.all(conv))}")
    np.savez_compressed(a.out, beta=np.stack(beta), weight=np.stack(weight), converged=np.array(conv),
                        views=np.array(views), D=ref["D"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
