"""redatt: regularised (RED) spectral-log-difference attenuation imaging for pulse-echo IQ data.

Reference-free by default: each exam is its own reference (median tissue spectrum per range row), so results are
attenuation relative to that exam's typical tissue at the same range, in dB/mm/MHz.  See README.md.
"""
from .compound import WeightedCompound
from .config import TO_MM, Config
from .frontend import band_envelopes, rereference, stepped_phase, undo_gain
from .pipeline import Estimate, Estimator, Frame
from .reference import apply_reference, exclude_blocks, fit_depth_trend, self_reference
from .sld import Grid, block_fraction, block_weight, make_grid, measure, to_pixels
from .solver import least_squares, masked_median, red_pg, solve

__all__ = ["Config", "TO_MM", "Frame", "Estimator", "Estimate", "WeightedCompound", "band_envelopes", "rereference",
           "stepped_phase", "undo_gain", "measure", "make_grid", "Grid", "block_weight", "block_fraction", "to_pixels",
           "solve", "red_pg", "least_squares", "masked_median", "self_reference", "apply_reference", "exclude_blocks",
           "fit_depth_trend"]
__version__ = "0.1.0"
