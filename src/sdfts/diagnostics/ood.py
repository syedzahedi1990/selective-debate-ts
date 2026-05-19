"""Out-of-distribution score for an input window.

We summarize each window into a small feature vector (mean / std / trend / acf1)
and compute its scaled L2 distance to the nearest training window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _featurize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4:
        return np.array([float(x.mean()), float(x.std()), 0.0, 0.0], dtype=np.float64)
    x0 = x - x.mean()
    denom = float(np.sum(x0 ** 2)) + 1e-12
    ac1 = float(np.sum(x0[:-1] * x0[1:]) / denom)
    t = np.arange(len(x), dtype=np.float64)
    slope = float(np.polyfit(t, x, 1)[0]) if len(x) >= 2 else 0.0
    return np.array([float(x.mean()), float(x.std()), slope, ac1], dtype=np.float64)


@dataclass
class OODReference:
    feats: np.ndarray          # (N, D)
    mu: np.ndarray
    sigma: np.ndarray

    @classmethod
    def from_windows(cls, windows: list) -> "OODReference":
        feats = np.stack([_featurize(w.x) for w in windows], axis=0) if windows else np.zeros((1, 4))
        mu = feats.mean(axis=0)
        sigma = feats.std(axis=0) + 1e-6
        return cls(feats=feats, mu=mu, sigma=sigma)


def ood_score(x: np.ndarray, ref: OODReference, neighbors: int = 20) -> float:
    """Mean scaled L2 distance to the ``k`` nearest training windows."""
    q = (_featurize(x) - ref.mu) / ref.sigma
    R = (ref.feats - ref.mu) / ref.sigma
    d = np.linalg.norm(R - q[None, :], axis=1)
    k = max(1, min(neighbors, d.shape[0]))
    return float(np.sort(d)[:k].mean())
