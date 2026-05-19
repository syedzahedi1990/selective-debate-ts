"""Changepoint score.

Default is a robust *cumulative-sum* style detector: scan candidate split
points and report the maximum normalized mean-shift across the input window.
This is cheap, dependency-free, and good enough for the smoke test.
``ruptures``-based variants can be added later.
"""
from __future__ import annotations

import numpy as np


def changepoint_score(x: np.ndarray, min_segment: int = 4) -> float:
    """Return the largest absolute mean-shift across any single split point,
    normalized by overall standard deviation. In [0, ~4] typically.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2 * min_segment:
        return 0.0
    s = float(np.std(x)) + 1e-12
    csum = np.cumsum(x)
    best = 0.0
    for t in range(min_segment, n - min_segment):
        m1 = csum[t - 1] / t
        m2 = (csum[-1] - csum[t - 1]) / (n - t)
        score = abs(m1 - m2) * np.sqrt(t * (n - t) / n) / s
        if score > best:
            best = float(score)
    return best
