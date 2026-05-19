"""Forecasting, arbitration, and cost metrics."""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / sMAPE / MASE (naive seasonal=1)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch {yt.shape} vs {yp.shape}")
    if yt.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "smape": 0.0, "mase": 0.0}
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    smape = float(np.mean(np.where(denom < 1e-9, 0.0, np.abs(yt - yp) / np.maximum(denom, 1e-9))))
    # MASE: scale by mean abs first-difference of y_true along horizon axis when 2D.
    if yt.ndim == 2 and yt.shape[1] >= 2:
        scale = np.mean(np.abs(np.diff(yt, axis=1))) + 1e-9
    elif yt.ndim == 1 and yt.size >= 2:
        scale = float(np.mean(np.abs(np.diff(yt)))) + 1e-9
    else:
        scale = 1e-9
    mase = float(mae / scale)
    return {"mae": mae, "rmse": rmse, "smape": smape, "mase": mase}


def per_instance_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=1)


def arbitration_metrics(
    selected_errors: np.ndarray,
    oracle_errors: np.ndarray,
    default_errors: np.ndarray,
    selected_ids: list[str] | None = None,
    oracle_ids: list[str] | None = None,
) -> dict[str, float]:
    sel = np.asarray(selected_errors, dtype=np.float64)
    orc = np.asarray(oracle_errors, dtype=np.float64)
    dflt = np.asarray(default_errors, dtype=np.float64)
    regret = sel - orc
    rel_regret = regret / np.maximum(orc, 1e-9)
    out = {
        "selected_mae_mean": float(sel.mean()),
        "oracle_mae_mean": float(orc.mean()),
        "default_mae_mean": float(dflt.mean()),
        "regret_mean": float(regret.mean()),
        "relative_regret_mean": float(rel_regret.mean()),
        "improvement_over_default_pct": float((dflt.mean() - sel.mean()) / max(dflt.mean(), 1e-9) * 100),
    }
    if selected_ids is not None and oracle_ids is not None:
        match = [int(s == o) for s, o in zip(selected_ids, oracle_ids)]
        out["selection_accuracy"] = float(np.mean(match)) if match else 0.0
    return out


def cost_metrics(
    n_calls: int,
    prompt_tokens: int,
    completion_tokens: int,
    pricing_prompt_per_1k: float = 0.0,
    pricing_completion_per_1k: float = 0.0,
    elapsed_seconds: float = 0.0,
) -> dict[str, float]:
    cost = (prompt_tokens / 1000.0) * pricing_prompt_per_1k + (completion_tokens / 1000.0) * pricing_completion_per_1k
    return {
        "n_calls": int(n_calls),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cost_usd": float(cost),
        "latency_seconds": float(elapsed_seconds),
    }
