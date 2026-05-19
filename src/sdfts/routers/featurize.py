"""Convert a forecast card into a fixed-length feature vector for the router."""
from __future__ import annotations

from typing import Any

import numpy as np


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:  # noqa: BLE001
        return float(default)


def card_to_router_features(card: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Return feature vector + parallel list of feature names.

    Uses only diagnostics + validation metrics — never any test-time error.
    """
    d = card.get("diagnostics", {})
    feats: list[float] = []
    names: list[str] = []

    diag_keys = [
        "architecture_disagreement",
        "training_regime_disagreement",
        "overall_forecast_dispersion",
        "disagreement_slope_over_horizon",
        "trend_strength",
        "seasonality_strength",
        "changepoint_score",
        "missingness_rate",
        "input_volatility",
        "recent_level_shift_score",
        "ood_score",
    ]
    for k in diag_keys:
        feats.append(_safe(d.get(k, 0.0)))
        names.append(f"diag.{k}")

    # Horizon-wise disagreement: 3 summary stats (mean, max, last).
    hw = np.asarray(d.get("horizonwise_disagreement") or [0.0], dtype=np.float64)
    feats.extend([float(hw.mean()), float(hw.max()), float(hw[-1])])
    names.extend(["diag.hw_disag.mean", "diag.hw_disag.max", "diag.hw_disag.last"])

    # Candidate-level summaries: mean/min/max of validation MAE across candidates.
    vmaes = np.array([m["validation_metrics"]["mae"] for m in card["candidate_models"]], dtype=np.float64)
    feats.extend([float(vmaes.mean()), float(vmaes.min()), float(vmaes.max()), float(vmaes.std())])
    names.extend(["cand.val_mae.mean", "cand.val_mae.min", "cand.val_mae.max", "cand.val_mae.std"])

    # Recent residual summaries: pooled magnitude.
    rstds = np.array([
        _safe(m.get("recent_residual_summary", {}).get("std", 0.0))
        for m in card["candidate_models"]
    ])
    racs = np.array([
        _safe(m.get("recent_residual_summary", {}).get("autocorr_lag1", 0.0))
        for m in card["candidate_models"]
    ])
    feats.extend([float(rstds.mean()), float(rstds.max()), float(racs.mean())])
    names.extend(["cand.resid_std.mean", "cand.resid_std.max", "cand.resid_ac1.mean"])

    return np.asarray(feats, dtype=np.float32), names
