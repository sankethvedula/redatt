"""High-level use: one estimator per probe / acquisition setting.

    est = Estimator(cfg, noise=noise, limited=limited)          # scanner-level inputs (optional)
    frames = [...]                                               # Frame objects of one exam
    ref = est.self_reference(frames[::k], exclude=masks)         # once per exam
    out = est.estimate(frame, ref)                               # per frame: out.beta [R, C], out.pixels() [B, S]
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .config import Config
from .reference import apply_reference, exclude_blocks, self_reference
from .sld import Grid, block_weight, measure, to_pixels
from .solver import solve


@dataclasses.dataclass
class Frame:
    """One frame (all beams of one view and level) after the front end.
    env_db [nband, B, S] band envelopes (dB); centres_mhz [nband]; depth_mm [S]; beam_spacing_mm [S]."""
    env_db: np.ndarray
    centres_mhz: np.ndarray
    depth_mm: np.ndarray
    beam_spacing_mm: np.ndarray

    @property
    def shape(self):
        return self.env_db.shape[1:]


@dataclasses.dataclass
class Estimate:
    beta: np.ndarray        # [R, C] attenuation, dB/mm/MHz (relative to the reference)
    c: np.ndarray           # [R, C] backscatter term, dB
    weight: np.ndarray      # [R, C] SNR reliability weight (for compounding)
    grid: Grid
    shape: tuple            # (B, S) of the frame
    converged: bool

    def pixels(self, what: str = "beta") -> np.ndarray:
        """[B, S] map of beta (or 'weight') on the frame's pixels."""
        return to_pixels(self.grid, getattr(self, what), self.shape)


class Estimator:
    def __init__(self, cfg: Config = Config(), noise: np.ndarray | None = None, limited: np.ndarray | None = None):
        """noise [nband, S]: linear band power of the receive noise (decides which pixels hold echo; None: each
        frame's own estimate).  limited [nband, S]: bands to exclude per sample (outside the receive filter)."""
        self.cfg, self.noise, self.limited = cfg, noise, limited

    def measure(self, frame: Frame) -> dict:
        return measure(frame.env_db, frame.centres_mhz, frame.depth_mm, frame.beam_spacing_mm, self.cfg,
                       noise=self.noise, limited=self.limited)

    def self_reference(self, frames, exclude=None) -> dict:
        """Reference from the exam's own tissue.  frames: iterable of Frame; exclude: matching iterable of [B, S]
        pixel masks (e.g. a dilated lesion) or None."""
        ms = []
        for i, fr in enumerate(frames):
            m = self.measure(fr)
            mask = None if exclude is None else exclude[i]
            if mask is not None and np.any(mask):
                m = exclude_blocks(m, mask)
            ms.append(m)
        return self_reference(ms, self.cfg.ref_min_blocks)

    def estimate(self, frame: Frame, reference: dict | None = None, method: str = "red") -> Estimate:
        m = apply_reference(self.measure(frame), reference)
        s = solve(m, self.cfg, method)
        return Estimate(s.beta, s.c, block_weight(m), m["grid"], frame.shape, s.converged)
