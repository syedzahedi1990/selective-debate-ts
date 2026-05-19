"""Bootstrap confidence intervals + paired bootstrap differences."""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    stats = np.zeros(n_samples)
    n = v.size
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        stats[i] = statistic(v[idx])
    return {
        "mean": float(statistic(v)),
        "lo": float(np.quantile(stats, alpha / 2)),
        "hi": float(np.quantile(stats, 1 - alpha / 2)),
    }


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI for mean(a) - mean(b) using paired resampling of indices."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap requires same shape")
    if a.size == 0:
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_value_two_sided": float("nan")}
    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_samples)
    n = a.size
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(a[idx]) - np.mean(b[idx]))
    point = float(np.mean(a) - np.mean(b))
    p = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))
    return {
        "mean_diff": point,
        "lo": float(np.quantile(diffs, alpha / 2)),
        "hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "p_value_two_sided": float(p),
    }
