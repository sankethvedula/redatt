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

## Checks

```
PYTHONPATH=. pytest -q tests
```

- **`test_synthetic.py`:**
  - a uniform medium is recovered;
  - an inclusion's contrast is recovered;
  - a range-dependent system term cancels with the self reference;
  - the stepped phase reduces to a constant carrier.
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
