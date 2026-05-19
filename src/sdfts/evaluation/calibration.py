"""Calibration: reliability curve, ECE, Brier."""
from __future__ import annotations

import numpy as np


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.size == 0:
        return {"bin_edges": np.array([0, 1]), "mean_predicted": np.array([]), "mean_observed": np.array([]), "count": np.array([])}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    mean_p = np.zeros(n_bins)
    mean_y = np.zeros(n_bins)
    count = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            mean_p[b] = p[mask].mean()
            mean_y[b] = y[mask].mean()
            count[b] = mask.sum()
    return {"bin_edges": edges, "mean_predicted": mean_p, "mean_observed": mean_y, "count": count}


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    n = p.size
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += float(mask.sum() / n) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)
