"""The 32 depth-switched demodulation/decimation filters and their (depth-dependent) gain.

Taps: 64 coefficients per bank in the XML, 13-bit Q11, folded into a 128-tap symmetric FIR
h = [c0..c63, c63..c0] / 2048 (cquest-cicada-hw/ip/ip_iqddc/ip_iqddc_hw/datapath_rtl_4beam/firDecimator.v).
The banks are peak-normalised, not gain-normalised, so their gain rises with depth.
"""
import math

import numpy as np

from . import hardware as hw

FIR_MODES = ('dc', 'pulse', 'noise')


def bank_taps(row):
    return np.array(list(row) + list(row)[::-1], float) / 2048.0


def bank_gains(cfg, mode='pulse', frac_bw=0.7):
    """Scalar gain of each bank.
      'dc'    : |H(0)| (narrow-band echo centred on the demod frequency)
      'pulse' : peak response to a Gaussian echo with -6 dB fractional bandwidth `frac_bw`, centred on the
                bank's demod frequency
      'noise' : sqrt(sum h^2) (white noise)"""
    g = []
    for k, row in enumerate(cfg.fir_rows):
        h = bank_taps(row)
        if mode == 'dc':
            g.append(h.sum())
        elif mode == 'noise':
            g.append(np.sqrt((h ** 2).sum()))
        elif mode == 'pulse':
            f0 = cfg.demod_freqs[k]
            sig = math.sqrt(2 * math.log(2)) / (math.pi * frac_bw * f0)
            env = np.exp(-(np.arange(-400, 401) / hw.FS_RX) ** 2 / (2 * sig ** 2))
            g.append(np.abs(np.convolve(env, h)).max() / env.max())
        else:
            raise ValueError(f'fir mode must be one of {FIR_MODES}')
    return np.array(g)


def bank_index(cfg, raw_index):
    """Active bank at beamformer-output sample `raw_index` (40 MHz clocks, counted from the first
    beamformed sample, i.e. including the rx buffer + 80 samples that are cropped later).
    Interval lengths = (uint16) d2sRX(delta depth), no offset for convex probes (RXtoBFHW.cc:1015-1030)."""
    d = [0.0] + list(cfg.tfc_depths)
    ints = [int(2 * (d[i + 1] - d[i]) * hw.FS_RX / cfg.c) for i in range(len(d) - 1)]
    return np.minimum(np.searchsorted(np.cumsum(ints), raw_index, side='right'), len(ints) - 1)
