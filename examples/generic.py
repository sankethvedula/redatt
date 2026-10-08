#!/usr/bin/env python3
"""Using redatt with any scanner: supply your own IQ, gains and demodulation, then the same estimator.

Replace the synthetic arrays below with your data.  What you need per frame (all beams of one image line set):
  iq [B, S]            complex stored IQ (beams x range samples)
  depth_mm [S]         range of each sample
  beam_spacing_mm [S]  lateral distance between neighbouring beams at each range (constant for linear arrays)
and per scanner setting:
  gain_db [S]          the time-gain your scanner applied (amplitude dB), to divide out (optional)
  phase [S], t_s [S]   the demodulation phase your scanner applied and the sample times (s); for a constant
                       demodulation frequency f0, phase = 2 pi f0 t_s.  For a stepped table use ra.stepped_phase.
  filter_db [nb, S]    the receive filter's power response per band (dB) to equalise (optional)
  noise [nb, S]        the receive-noise band power (optional; decides which pixels hold echo)
"""
import numpy as np

import redatt as ra

cfg = ra.Config()                                 # defaults: 13 bands 2.04-4.24 MHz, 15-lambda blocks, RED mu 100
fs, f0 = 5e6, 3.5e6                               # IQ sample rate, demodulation frequency (constant here)
B, S = 128, 600
depth = 20 + (1.54 / 2) / fs * 1e6 * np.arange(S)          # mm
bspace = np.full(S, 0.3)                                    # mm, linear array
t = 2 * depth / 1.54 * 1e-6                                 # s
phase = 2 * np.pi * f0 * t
gain_db = 0.3 * (depth - depth[0])                          # e.g. a 0.3 dB/mm TGC ramp


def my_frame(iq):
    z = ra.rereference(ra.undo_gain(iq, gain_db), phase, t, f_ref_hz=f0)
    env = ra.band_envelopes(z, fs, f0, cfg.centres(), cfg.df, cfg.nfft)
    return ra.Frame(env, cfg.centres(), depth, bspace)


def synthetic_iq(alpha=0.05, seed=0):
    """Stand-in data: speckle from a 3.4 MHz pulse (Gaussian spectrum) attenuated by alpha dB/mm/MHz, demodulated at
    f0 and amplified by gain_db.  Built in 2 mm segments, each with its own attenuated spectrum."""
    r = np.random.default_rng(seed)
    nu = np.fft.fftfreq(S, 1 / fs); f = (f0 - nu) / 1e6                  # RF frequency of each baseband bin (MHz)
    w = r.standard_normal((B, S)) + 1j * r.standard_normal((B, S)); Wf = np.fft.fft(w, axis=1)
    iq = np.zeros((B, S), complex); seg = int(round(2.0 / np.median(np.diff(depth))))
    for s0 in range(0, S, seg):
        zc = depth[min(s0 + seg // 2, S - 1)] - depth[0]
        H = np.exp(-0.5 * ((f - 3.4) / 0.7) ** 2) * 10 ** (-2 * alpha * f * zc / 20)   # amplitude, two-way
        iq[:, s0:s0 + seg] = np.fft.ifft(Wf * H, axis=1)[:, s0:s0 + seg]
    return iq * 10 ** (gain_db / 20)


frames = [my_frame(synthetic_iq(seed=k)) for k in range(4)]
noise = np.full((len(cfg.centres()), S), 1e-12)   # known noise floor (here: none); None uses each frame's estimate
est = ra.Estimator(cfg, noise=noise)

# no system term in this synthetic, so no reference is needed: beta comes out absolute
out = est.estimate(frames[0])
print(f"true 0.050, estimated {np.nanmedian(out.beta):.3f} dB/mm/MHz on {int(np.isfinite(out.beta).sum())} blocks")

# real data: build the self reference from the exam's own frames, then estimate relative to it
cfg.ref_min_blocks = 20
ref = est.self_reference(frames[1:])
print(f"relative to own tissue: {np.nanmedian(est.estimate(frames[0], ref).beta):+.3f} dB/mm/MHz (uniform: ~0)")

# compounding several views into a volume: your geometry maps each view's (beam, sample) to voxel indices
vol = ra.WeightedCompound(n_voxels=B * S)
for fr in frames:
    o = est.estimate(fr)
    vol.add(np.arange(B * S), o.pixels().ravel(), o.pixels("weight").ravel())
print(f"compound: {int(np.isfinite(vol.mean(3)).sum())} voxels with >= 3 views, median {np.nanmedian(vol.mean(3)):.3f}")
