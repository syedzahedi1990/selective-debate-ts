"""Forecast disagreement features.

Decomposes a forecast tensor of shape (K, H) (or higher-dimensional
``arch x regime x H``) into architecture-level, regime-level, and horizon-level
disagreement signals.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def disagreement_features(
    forecasts: np.ndarray,
    archs: list[str],
    regimes: list[str],
) -> dict[str, Any]:
    """``forecasts`` is (K, H) over candidates, ordered to match ``archs``/``regimes``.

    Returns a dict of scalars + ``horizonwise_disagreement`` array (H,).
    """
    if forecasts.ndim != 2 or forecasts.size == 0:
        raise ValueError("forecasts must be 2-D (K, H) and non-empty")

    K, H = forecasts.shape
    # Overall dispersion: mean per-horizon std across candidates.
    horizonwise = forecasts.std(axis=0)             # (H,)
    overall = float(horizonwise.mean())

    # Architecture-level: average of group means per arch, then variance.
    arch_disagreement = _group_disagreement(forecasts, archs)
    regime_disagreement = _group_disagreement(forecasts, regimes)

    # Disagreement slope (does dispersion grow with horizon?)
    if H >= 2:
        t = np.arange(H, dtype=np.float64)
        slope = float(np.polyfit(t, horizonwise.astype(np.float64), 1)[0])
    else:
        slope = 0.0

    # Pairwise mean-abs-diff matrix (compact summary).
    pair_mat = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            pair_mat[i, j] = float(np.mean(np.abs(forecasts[i] - forecasts[j])))
    pair_offdiag_mean = float((pair_mat.sum() - np.trace(pair_mat)) / max(1, (K * K - K)))

    return {
        "horizonwise_disagreement": horizonwise.astype(np.float32).tolist(),
        "overall_forecast_dispersion": overall,
        "architecture_disagreement": arch_disagreement,
        "training_regime_disagreement": regime_disagreement,
        "disagreement_slope_over_horizon": slope,
        "pairwise_mad_mean": pair_offdiag_mean,
    }


def _group_disagreement(forecasts: np.ndarray, labels: list[str]) -> float:
    """Average horizon-wise variance across group means."""
    uniq = sorted(set(labels))
    if len(uniq) < 2:
        return 0.0
    means = []
    for u in uniq:
        idx = [i for i, l in enumerate(labels) if l == u]
        means.append(forecasts[idx].mean(axis=0))
    M = np.stack(means, axis=0)        # (G, H)
    return float(M.var(axis=0).mean())


def foundation_vs_supervised_disagreement(
    foundation_forecasts: np.ndarray | None,
    supervised_forecasts: np.ndarray,
) -> float | None:
    if foundation_forecasts is None or foundation_forecasts.size == 0:
        return None
    f_mean = foundation_forecasts.mean(axis=0)
    s_mean = supervised_forecasts.mean(axis=0)
    return float(np.mean(np.abs(f_mean - s_mean)))
