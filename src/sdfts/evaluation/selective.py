"""Selective forecasting / abstention metrics."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def risk_coverage_curve(errors: np.ndarray, confidence: np.ndarray) -> dict[str, np.ndarray]:
    """Sort by *descending* confidence; report cumulative risk vs coverage."""
    e = np.asarray(errors, dtype=np.float64)
    c = np.asarray(confidence, dtype=np.float64)
    if e.size == 0:
        return {"coverage": np.array([]), "risk": np.array([])}
    order = np.argsort(-c)
    e_sorted = e[order]
    n = e.size
    coverage = np.arange(1, n + 1, dtype=np.float64) / n
    risk = np.cumsum(e_sorted) / np.arange(1, n + 1, dtype=np.float64)
    return {"coverage": coverage, "risk": risk}


def selective_risk_at_coverage(
    errors: np.ndarray,
    confidence: np.ndarray,
    coverages: Sequence[float],
) -> dict[float, float]:
    curve = risk_coverage_curve(errors, confidence)
    out: dict[float, float] = {}
    cov = curve["coverage"]
    rsk = curve["risk"]
    if cov.size == 0:
        return {float(c): float("nan") for c in coverages}
    for c in coverages:
        idx = int(np.searchsorted(cov, c))
        idx = min(idx, len(rsk) - 1)
        out[float(c)] = float(rsk[idx])
    return out


def abstention_precision(abstained: np.ndarray, would_fail: np.ndarray) -> float:
    """Fraction of abstained cases that *would* have failed."""
    a = np.asarray(abstained, dtype=bool)
    f = np.asarray(would_fail, dtype=bool)
    if a.sum() == 0:
        return 0.0
    return float(f[a].mean())


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Implementation independent of sklearn, for environments without it."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    P = y.sum()
    N = y.size - P
    tp = 0
    fp = 0
    prev_tp = 0
    prev_fp = 0
    area = 0.0
    prev_score = None
    for yi, si in zip(y_sorted, s[order]):
        if prev_score is not None and si != prev_score:
            area += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_fp, prev_tp = fp, tp
        if yi == 1:
            tp += 1
        else:
            fp += 1
        prev_score = si
    area += (fp - prev_fp) * (tp + prev_tp) / 2.0
    if P == 0 or N == 0:
        return float("nan")
    return float(area / (P * N))


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average-precision style AUPRC (trapezoidal)."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = tp / max(int(y.sum()), 1)
    # Trapezoidal AUC over (recall, precision)
    return float(np.trapz(precision, recall))
