"""Forecast-only decision systems that operate directly on cards (no labels).

Each function returns ``(forecast, selected_model_id, confidence)`` per
instance. ``selected_model_id`` is informational; for true ensembles it is the
string "ensemble".
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _candidate_matrix(card: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    cand_ids = [m["model_id"] for m in card["candidate_models"]]
    F = np.stack([np.asarray(m["forecast"], dtype=np.float64) for m in card["candidate_models"]], axis=0)
    return cand_ids, F


def _val_mae_vector(card: dict[str, Any], cand_ids: list[str]) -> np.ndarray:
    by_id = {m["model_id"]: m for m in card["candidate_models"]}
    return np.array([by_id[c]["validation_metrics"]["mae"] for c in cand_ids], dtype=np.float64)


def _val_horizonwise_mae(card: dict[str, Any], cand_ids: list[str]) -> np.ndarray:
    by_id = {m["model_id"]: m for m in card["candidate_models"]}
    H = card["forecast_horizon"]
    rows = []
    for c in cand_ids:
        v = by_id[c].get("horizonwise_validation_mae")
        if v is None or len(v) != H:
            v = [by_id[c]["validation_metrics"]["mae"]] * H
        rows.append(v)
    return np.asarray(rows, dtype=np.float64)


def validation_best(card: dict[str, Any]) -> dict[str, Any]:
    cand_ids, F = _candidate_matrix(card)
    mae = _val_mae_vector(card, cand_ids)
    j = int(np.argmin(mae))
    return {"forecast": F[j].tolist(), "selected_model_id": cand_ids[j], "confidence": 1.0 / (1.0 + float(mae[j]))}


def horizonwise_validation_best(card: dict[str, Any]) -> dict[str, Any]:
    cand_ids, F = _candidate_matrix(card)
    H = card["forecast_horizon"]
    hw = _val_horizonwise_mae(card, cand_ids)
    pick = np.argmin(hw, axis=0)
    out = np.array([F[pick[h], h] for h in range(H)])
    return {"forecast": out.tolist(), "selected_model_id": "horizonwise_best", "confidence": 0.5}


def simple_mean_ensemble(card: dict[str, Any]) -> dict[str, Any]:
    _, F = _candidate_matrix(card)
    return {"forecast": F.mean(axis=0).tolist(), "selected_model_id": "ensemble_mean", "confidence": 0.5}


def median_ensemble(card: dict[str, Any]) -> dict[str, Any]:
    _, F = _candidate_matrix(card)
    return {"forecast": np.median(F, axis=0).tolist(), "selected_model_id": "ensemble_median", "confidence": 0.5}


def validation_weighted_ensemble(card: dict[str, Any]) -> dict[str, Any]:
    cand_ids, F = _candidate_matrix(card)
    mae = _val_mae_vector(card, cand_ids)
    inv = 1.0 / (mae + 1e-9)
    w = inv / inv.sum()
    out = (w[:, None] * F).sum(axis=0)
    return {"forecast": out.tolist(), "selected_model_id": "ensemble_val_weighted", "confidence": 0.5}
