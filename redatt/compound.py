"""SNR-weighted compounding of per-view attenuation maps into an image or volume.

Each view contributes, at the voxels it covers, its attenuation and a weight (block_weight of the block covering the
voxel).  The compound is sum(w beta) / sum(w); n_eff = (sum w)^2 / sum w^2 is the effective number of views.
How a view's (beam, sample) pixels map to voxels is scanner geometry and is supplied by the caller (`idx`).
"""
from __future__ import annotations

import numpy as np


class WeightedCompound:
    def __init__(self, n_voxels: int):
        self.sw = np.zeros(n_voxels); self.swx = np.zeros(n_voxels); self.sw2 = np.zeros(n_voxels)
        self.swxx = np.zeros(n_voxels); self.count = np.zeros(n_voxels, int)

    def add(self, idx: np.ndarray, values: np.ndarray, weights: np.ndarray):
        """idx [n] voxel indices, values [n] attenuation, weights [n] (>= 0); NaN values or zero weights skipped."""
        idx = np.asarray(idx, np.int64); v = np.asarray(values, float); w = np.asarray(weights, float)
        ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
        idx, v, w = idx[ok], v[ok], w[ok]
        np.add.at(self.sw, idx, w); np.add.at(self.swx, idx, w * v); np.add.at(self.sw2, idx, w * w)
        np.add.at(self.swxx, idx, w * v * v); np.add.at(self.count, idx, 1)

    def mean(self, min_views: int = 3) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            m = self.swx / self.sw
        return np.where(self.count >= min_views, m, np.nan)

    def sd(self, min_views: int = 3) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            m = self.swx / self.sw
            v = self.swxx / self.sw - m * m
        return np.where(self.count >= min_views, np.sqrt(np.clip(v, 0, None)), np.nan)

    def n_eff(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.sw2 > 0, self.sw ** 2 / self.sw2, 0.0)
