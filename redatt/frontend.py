"""From stored complex IQ to band envelopes on the true RF frequency axis.

Three generic operations, each optional:
  undo_gain      divide out a per-sample amplitude gain (e.g. the scanner's time-gain compensation)
  rereference    remove the scanner's (possibly stepped) demodulation phase and apply one common carrier f_ref,
                     z[m] = x[m] exp(-j phi[m]) exp(+j 2 pi f_ref t[m]);
                 afterwards an RF component at frequency f sits at baseband nu = f_ref - f at every sample
  band_envelopes Gaussian band-pass filters along range (FFT of each zero-padded line), envelope in dB, minus an
                 optional per-band, per-sample filter response in dB (the scanner's receive filter, to equalise it)

Sign convention: baseband nu = f_demod - f (the component at RF f appears at nu = f_demod - f after mixing with
exp(+j 2 pi f_demod t)).  If your data use the opposite convention, pass f_ref with the opposite sign handling by
flipping the spectrum (np.conj on the IQ).
"""
from __future__ import annotations

import numpy as np


def undo_gain(iq: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    """iq [..., S] / 10^(gain_db [S] / 20)."""
    return iq / np.power(10.0, np.asarray(gain_db, float) / 20.0)


def rereference(iq: np.ndarray, phase: np.ndarray, t_s: np.ndarray, f_ref_hz: float) -> np.ndarray:
    """Remove the demodulation phase phase [S] (rad, accumulated as the scanner did) and apply the carrier f_ref at
    times t_s [S] (seconds).  For a constant scanner demodulation f0, phase = 2 pi f0 t."""
    return iq * np.exp(-1j * np.asarray(phase) + 2j * np.pi * f_ref_hz * np.asarray(t_s))


def stepped_phase(freqs_hz, interval_lengths, fs_raw_hz: float, sample_index_raw: np.ndarray) -> np.ndarray:
    """Accumulated phase of a phase-continuous oscillator whose frequency steps through `freqs_hz` (one per interval,
    `interval_lengths` raw samples each; the last frequency is held), evaluated at fractional raw-sample indices
    `sample_index_raw` (e.g. each stored sample's filter centre).  phi[n+1] = phi[n] + 2 pi f_k[n] / fs_raw."""
    n_raw = int(np.ceil(np.max(sample_index_raw))) + 2
    k = np.minimum(np.searchsorted(np.cumsum(interval_lengths), np.arange(n_raw), side="right"), len(freqs_hz) - 1)
    f = np.asarray(freqs_hz, float)[k]
    phi = 2 * np.pi * np.concatenate([[0.0], np.cumsum(f)]) / fs_raw_hz
    return np.interp(sample_index_raw, np.arange(n_raw + 1), phi)


def band_envelopes(z: np.ndarray, fs_hz: float, f_ref_hz: float, centres_mhz, df_mhz: float, nfft: int = 2048,
                   filter_db: np.ndarray | None = None) -> np.ndarray:
    """Band envelopes [nband, B, S] in dB of re-referenced IQ z [B, S].  Band b is a Gaussian whose power response
    exp(-((nu - nu_b)/sig)^2) has FWHM df, centred at nu_b = f_ref - f_b.  filter_db [nband, S] (optional) is
    subtracted (an equaliser for the scanner's receive filter, in power dB)."""
    B, S = z.shape
    nu = np.fft.fftfreq(nfft, 1.0 / fs_hz)
    centres = np.asarray(centres_mhz, float) * 1e6
    sig = df_mhz * 1e6 / (2 * np.sqrt(np.log(2)))
    gauss = np.exp(-0.5 * ((nu[None, :] - (f_ref_hz - centres[:, None])) / sig) ** 2)
    Z = np.fft.fft(z, nfft, axis=-1)
    env = np.empty((len(centres), B, S), np.float32)
    for b in range(len(centres)):
        y = np.fft.ifft(Z * gauss[b], axis=-1)[:, :S]
        e = 20 * np.log10(np.abs(y) + 1e-9)
        if filter_db is not None:
            e = e - np.asarray(filter_db[b], float)[None, :]
        env[b] = e
    return env
