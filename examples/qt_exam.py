#!/usr/bin/env python3
"""End-to-end on one QT exam: self reference from the exam, RED attenuation of one view, figure.

    python examples/qt_exam.py /path/to/QT007-S1-BL --view 5 --level 16 [--every-view 3 --every-level 2]
"""
import argparse
import time

import numpy as np

import redatt as ra
from redatt.qt import QTProbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exam"); ap.add_argument("--view", type=int, default=5); ap.add_argument("--level", type=int, default=16)
    ap.add_argument("--every-view", type=int, default=3); ap.add_argument("--every-level", type=int, default=2)
    ap.add_argument("--out", default="qt_exam_red.png")
    a = ap.parse_args()

    t0 = time.time()
    probe = QTProbe(a.exam, unit=1)                      # stepped demodulation, TGC undone, filter bank equalised
    est = probe.estimator()                              # QT system noise + filter limits, default Config
    views = probe.scan.views(1)

    # 1) self reference: median tissue spectrum per range row over a sample of the exam's frames.
    #    (Pass exclude=[pixel masks] to leave a lesion out; the median is robust to a small one.)
    frames = []
    for v in views[::a.every_view]:
        iq = probe.read(v)
        for L in range(0, iq.shape[0], a.every_level):
            frames.append(probe.frame_from_iq(iq[L]))
    ref = est.self_reference(frames)
    print(f"self reference from {len(frames)} frames: {time.time() - t0:.0f} s")

    # 2) one view: RED attenuation relative to this breast's typical tissue at the same range
    fr = probe.frame(a.view, a.level)
    out = est.estimate(fr, ref)
    print(f"view {a.view} level {a.level}: converged {out.converged}; median beta {np.nanmedian(out.beta):+.4f} "
          f"dB/mm/MHz; {time.time() - t0:.0f} s total")

    # 3) figure: B-mode and the attenuation map on the frame's (beam, sample) grid
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bm = 10 * np.log10(np.power(10.0, fr.env_db.astype(float) / 10).sum(0))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ext = [0, bm.shape[0], fr.depth_mm[-1], fr.depth_mm[0]]
    ax[0].imshow(bm.T, cmap="gray", vmin=np.nanpercentile(bm, 99.5) - 50, vmax=np.nanpercentile(bm, 99.5),
                 aspect="auto", extent=ext)
    im = ax[1].imshow(out.pixels().T, cmap="RdBu_r", vmin=-0.06, vmax=0.06, aspect="auto", extent=ext)
    ax[0].set_ylabel("depth (mm)"); [x.set_xlabel("beam") for x in ax]
    fig.colorbar(im, ax=ax[1], label="dB/mm/MHz (rel. own tissue)")
    fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
