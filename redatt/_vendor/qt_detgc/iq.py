"""Reading STAGE0 IQ files and locating clipped samples."""
import numpy as np

from . import hardware as hw


def read_iq(path, cfg=None, num_beams=192, samples_per_beam=None):
    """Read a STAGE0 IQ file, or a QT/refl 'stack' file (unit<k>_<view>: one frame per level, back to back).

    Layout: little-endian int32 pairs (I, Q), each a sign-extended 24-bit S24Q7 value, beam-major, no header.
    Returns complex128 [frames, beams, samples] in integer file units (divide by 128 for S24Q7).
    With `cfg`, beams and samples per beam are taken from the acquisition XML (the last sample of every
    beam is alignment padding; it is kept so that indices match the files and QT's nt_iq_probe)."""
    if cfg is not None:
        num_beams = cfg.num_beams
        samples_per_beam = samples_per_beam or cfg.samples_per_beam()
    if samples_per_beam is None:
        raise ValueError('samples_per_beam (or cfg) is required')
    a = np.fromfile(path, dtype='<i4')
    per = num_beams * samples_per_beam * 2
    if a.size % per:
        raise ValueError(f'{path}: {a.size} int32 is not a multiple of {per} '
                         f'(beams={num_beams}, samples/beam={samples_per_beam})')
    a = ((a << 8) >> 8).reshape(-1, num_beams, samples_per_beam, 2)
    return a[..., 0] + 1j * a[..., 1]


def saturation_mask(iq, curve):
    """True where I or Q sits at the digital clip rail for that depth.  Must be evaluated on the raw file
    values, before any gain is divided out.  Per-channel ADC clipping is not visible this way (it is mixed
    into the beam sum) -- see clip_levels()."""
    lim = curve.saturation_rail() - 1
    return (np.abs(iq.real) >= lim) | (np.abs(iq.imag) >= lim)


def clip_levels(curve, n_active=64):
    """Amplitude limits per sample in file units.

      rail            value an I/Q component takes when a digital saturation occurred (detectable)
      adc_fullscale   |IQ| of a narrow-band echo with *every* one of `n_active` channels at ADC full scale;
                      compare |IQ| with a fraction of this to flag *possible* per-channel ADC clipping
      *_pre           the same after dividing out the TGC (sample-0 referenced)"""
    rail = curve.saturation_rail()
    fs_adc = 128 * curve.dtgc * curve.fir_gain * 0.5 * np.asarray(n_active) * 2047
    g = curve.atgc * curve.dtgc
    g = g / g[0]
    return dict(rail=rail, adc_fullscale=fs_adc, rail_pre=rail / g, adc_fullscale_pre=fs_adc / g)
