"""Input-window diagnostics.

Lightweight, NumPy-only implementations. Robust to short windows and NaNs.
"""
from __future__ import annotations

import numpy as np


def _linreg_slope(x: np.ndarray) -> tuple[float, float]:
    """Return (slope, var_resid) of a 1-D OLS y ~ a + b*t."""
    n = len(x)
    t = np.arange(n, dtype=np.float64)
    if n < 2:
        return 0.0, 0.0
    tm = t.mean()
    xm = x.mean()
    cov = np.sum((t - tm) * (x - xm))
    var_t = np.sum((t - tm) ** 2) + 1e-12
    b = cov / var_t
    a = xm - b * tm
    resid = x - (a + b * t)
    return float(b), float(np.var(resid))


def trend_strength(x: np.ndarray) -> float:
    """1 - Var(resid) / Var(x).  Bounded to [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    var_x = float(np.var(x)) + 1e-12
    _, vr = _linreg_slope(x)
    return float(max(0.0, min(1.0, 1.0 - vr / var_x)))


def seasonality_strength(x: np.ndarray, max_lag: int | None = None) -> float:
    """Peak of ACF for lag >= 2, clipped to [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        return 0.0
    x0 = x - x.mean()
    denom = float(np.sum(x0 ** 2)) + 1e-12
    cap = max_lag or min(n - 1, n // 2)
    peak = 0.0
    for k in range(2, cap + 1):
        ac = float(np.sum(x0[:-k] * x0[k:]) / denom)
        if ac > peak:
            peak = ac
    return float(max(0.0, min(1.0, peak)))


def volatility(x: np.ndarray) -> float:
    """Standard deviation of first differences."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    return float(np.std(np.diff(x)))


def missingness_rate(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.isnan(x)))


def recent_level_shift(x: np.ndarray, window_frac: float = 0.25) -> float:
    """|mean(last w) - mean(first n - w)| / std(x)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    w = max(2, int(window_frac * n))
    if n <= w + 2:
        return 0.0
    s = float(np.std(x)) + 1e-12
    return float(abs(x[-w:].mean() - x[:-w].mean()) / s)


def residual_autocorrelation(resid: np.ndarray, lag: int = 1) -> float:
    """Pearson autocorrelation of residuals at given lag."""
    r = np.asarray(resid, dtype=np.float64)
    n = len(r)
    if n <= lag + 1:
        return 0.0
    r0 = r - r.mean()
    denom = float(np.sum(r0 ** 2)) + 1e-12
    return float(np.sum(r0[:-lag] * r0[lag:]) / denom)
