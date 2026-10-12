# redatt

RED-regularised spectral-log-difference (SLD) attenuation imaging from pulse-echo IQ data, in dB/mm/MHz.
The method follows Merino et al., "Attenuation coefficient imaging using regularization by denoising",
IEEE TUFFC 73(8), 2026, with these choices:
- each exam is its own reference;
- no noise handling;
- SNR-weighted compounding across views.

These are the settings that worked best on QT breast data. Dependencies are numpy and scipy. The optional QT adapter
uses a bundled copy of `qt_detgc` (numpy only).

```
pip install -e .            # or put this folder on PYTHONPATH
```

## What it computes

For each frame (all beams of one view and level), from the stored complex IQ:

1. **Front end** (`frontend.py`)
   - **Gain:** divide out the scanner's time-gain, `undo_gain`.
   - **Demodulation:** remove the scanner's demodulation phase and apply one common carrier, `rereference`:
     `z = x · exp(-jφ(t)) · exp(+j2π f_ref t)`.
     - For a stepped demodulation table, `stepped_phase` builds φ.
     - For a constant one, use `φ = 2π f₀ t`.
   - **Bands:** Gaussian bands along range (`band_envelopes`), by default 13 bands of 0.3 MHz FWHM every 0.15 MHz over
     2.04–4.24 MHz. Each band can be divided by the receive filter's response.
2. **Measurement** (`sld.py`)
   - **Blocks:** 15 wavelengths square, 80 % overlap.
   - **Per block and band:** `Y_b = 10 log10(S_proximal / S_distal)`, from two half-block windows `dz` apart.
   - **Which blocks:** a block is used when ≥ 90 % of its pixels hold echo, i.e. windowed full-band power ≥ 10 dB above
     the receive noise.
   - **Which bands:** a band is used when it is within 20 dB of the block's strongest band. A block needs ≥ 3 bands.
3. **Model:** `Y_b − D(z, f_b) = 2 dz f_b β + c`.
   - `β` is the attenuation slope (one-way, dB/mm/MHz). `c` is a backscatter term.
   - `D` is the reference (`reference.py`).
4. **Reference-free reference**
   - **Self reference (default):** `D[band, row]` is the median of `Y` over the exam's own tissue blocks at that range
     row. Leave a lesion out with `exclude`.
   - **What β then means:** attenuation relative to this exam's typical tissue at the same range (0 = typical).
   - **Why it works:** it cancels the system's depth-dependent spectral shaping (diffraction, focusing, receive chain)
     without a phantom.
   - **Alternative:** `fit_depth_trend` removes a quadratic range trend from unreferenced maps instead.
   - **Phantom reference:** build `D` with `self_reference` from phantom frames and pass `a_ref` for absolute values.
5. **RED solver** (`solver.py`)
   - **Problem:** `argmin ½‖Ax − y‖² + (μ/2) xᵀ(x − D(x))` with `D = (7×7 median on β, soft-threshold 1/μ on c)`,
     `μ = 100`.
   - **Algorithm:** RED proximal gradient. Each iteration is an exact 2×2 solve per block.
   - **Acceleration (optional, `Config(accel="anderson", tol=1e-8)`):** Anderson acceleration of the same fixed-point
     map, with a monotone safeguard.
     - It reaches the same fixed point. Plain RED-PG contracts slowly: at `tol=1e-5` (~900 iterations) it stops
       typically 4e-3 and up to 1.4e-2 dB/mm/MHz from the fixed point, and fully converging takes up to ~10 000
       iterations.
     - Anderson gets within ~4e-5 in a median of ~500 iterations (at most ~1400) on QT data.
     - The default stays `accel="none"` so results reproduce the research pipeline.
6. **Compounding** (`compound.py`)
   - **Weight:** each block's reliability, `block_weight`, is the inverse predicted variance of the slope from its
     window SNRs.
   - **Combination:** views are combined as `Σwβ / Σw` with `WeightedCompound`. The mapping from (beam, sample) to
     voxels is your scanner geometry.

All parameters live in `Config` (`config.py`). The defaults are the configuration above.

## Use

```python
import redatt as ra

est = ra.Estimator(ra.Config(), noise=noise, limited=limited)    # scanner-level inputs, both optional
frames = [ra.Frame(env_db, centres_mhz, depth_mm, beam_spacing_mm) for ...]   # from ra.band_envelopes(...)
ref = est.self_reference(frames[::4], exclude=lesion_masks)      # once per exam
out = est.estimate(frames[k], ref)       # out.beta [R, C] (dB/mm/MHz), out.weight, out.pixels() -> [beams, samples]
```

**QT data:** `redatt.qt.QTProbe` builds every scanner-specific input from an exam folder (`conf/`, `da/refl/`):
- the exact time-gain curve;
- the stepped, phase-continuous demodulation phase from the acquisition's 32-entry table;
- the filter-bank response per band (capped 15 dB below each bank's peak);
- depth and beam spacing from QT's reflection parameters;
- a shipped system receive-noise curve for the far probe.

```python
from redatt.qt import QTProbe
probe = QTProbe("/path/to/QT007-S1-BL", unit=1)       # demod="stepped" (default) or "fixed" (QT's demod_freq)
est = probe.estimator()
ref = est.self_reference([probe.frame(v, L) for v in probe.scan.views(1)[::3] for L in range(0, 60, 2)])
out = est.estimate(probe.frame(5, 16), ref)
```

`examples/qt_exam.py` does this end to end and saves a figure: about 30 s for a 440-frame reference plus one view, on
one core. `examples/generic.py` shows the same with your own arrays, on synthetic IQ with known attenuation.

## GPU

`redatt.gpu` (needs PyTorch, `pip install -e .[gpu]`) runs the same computations batched over frames on a GPU:
- the front end;
- the measurement;
- the self reference;
- the RED solver, plain or Anderson. Each frame stops at its own iteration, and finished frames are dropped from the
  batch.

Results match the CPU path (float64): measurements to ~1e-12, the β map to ~1e-14 (plain) or within the tolerance
(Anderson).

```python
import redatt.gpu as rg
est = rg.GPUEstimator(cfg, noise=noise, limited=limited, device="cuda")
fb = rg.FrameBatch(env, centres_mhz, depth_mm, beam_spacing_mm)   # env [N, nband, B, S]; or from rg.band_envelopes
ref = est.self_reference([fb1, fb2, ...])
outs = est.estimate(fb, ref)                                       # list of Estimate, one per frame
```

With QT data, `probe.frame_batch(view)` builds all levels of a view on the GPU and `probe.gpu_estimator()` gives the
estimator. `examples/qt_exam_gpu.py` runs a whole exam: the self reference, then all 60 views × 43 levels.

On one H100, QT007 (2580 frames) takes about 1 minute:

| step | time |
|---|---|
| self reference | 2.5 s |
| front end | 3 s |
| measure + RED, plain | 55 s (21 ms/frame) |
| measure + RED, Anderson | 67 s (26 ms/frame) |

On the CPU this is ~1 s per frame (~45 min on one core).

**Edge-aware estimation (experimental, GPU only).** Pass a guide image per frame (e.g. a compounded B-mode in dB,
mapped back into the frame) as `est.estimate(fb, ref, guide=guide)`. Two options use it:
- `Config.med_sigma_db`: RED's median becomes a median weighted by how similar each neighbouring block's guide value is
  to the block's own, so smoothing does not cross guide edges.
- `Config.edge_sigma_db`: each block's window spectra weight pixels by their guide similarity to the block centre.
  - A block's data weight is the fraction of weight its windows keep.
  - Its effective window separation comes from the weight centroids.
  - The guide's depth trend is removed first (`guide_detrend`).

On a synthetic phantom where echo edges coincide with attenuation edges, the guided median recovers a 16 mm
inclusion's contrast (0.043 of 0.050 vs 0.016) with a 4 mm instead of a 22 mm edge, at the same noise. Edge-aware
spectra add contrast but also create false contrast at echo-only edges.

The masked 7×7 median dominates each iteration. On the GPU it runs on the domain blocks only, as a selection rather
than a sort: missing window slots are padded with ±inf so that the plain median is the masked one.

## Checks

```
PYTHONPATH=. pytest -q tests
```

- **`test_synthetic.py`:**
  - a uniform medium is recovered;
  - an inclusion's contrast is recovered;
  - a range-dependent system term cancels with the self reference;
  - the stepped phase reduces to a constant carrier.
- **`test_gpu.py`:**
  - Anderson-accelerated RED reaches the fixed point of plain RED;
  - the GPU front end, measurement, reference and RED map match the CPU. Skipped without CUDA.
- **`test_against_pdg.py`:** reproduces the research pipeline this was extracted from (`phantom_detgc/pdg` in the
  qti-specular repository) on a real QT frame. The band envelopes, band masks, measurements, self reference and RED map
  all match, the map to 1e-5 dB/mm/MHz. It is skipped where that repository or the data are absent.

## Notes and limits

- **Units:** dB/mm/MHz throughout. Inside the solver β is scaled per cm, the paper's convention, so μ compares with
  the paper.
- **Contrast only:** with the self reference, values are relative to the exam's own tissue at the same range. Absolute
  values need a phantom reference of known attenuation.
- **Demodulation:** QT stores IQ demodulated with a depth-stepped frequency (3.76 → 3.24 MHz over 50–98 mm), so it
  must be re-referenced with that table before any spectral analysis. `demod="fixed"` is there only for comparison.
- **Temperature:** `QTProbe(temperature="first")` uses the first value of `temperature.dat`, as QT's pipelines do.
  `"mean"` reproduces the research pipeline; the depth axis differs by 0.006 %.
