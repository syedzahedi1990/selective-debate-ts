"""Train + evaluate non-LLM routers.

Strategy:
- Train router on **validation** cards (since failure label depends on
  default error on val). For the smoke test we synthesize a small set of
  val-like cards by reusing the validation forecasts: each val instance gets
  a card whose default-policy error is the val MAE under the default policy.
  This avoids leakage from the test split into router training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sdfts.cards.build_cards import (
    build_default_policy,
    _apply_default_to_val,    # type: ignore[attr-defined]
)
from sdfts.cards.io import read_cards, read_labels
from sdfts.config import run_dir
from sdfts.diagnostics.changepoint import changepoint_score
from sdfts.diagnostics.disagreement import disagreement_features
from sdfts.diagnostics.features import (
    missingness_rate,
    recent_level_shift,
    seasonality_strength,
    trend_strength,
    volatility,
)
from sdfts.diagnostics.ood import OODReference, ood_score
from sdfts.evaluation.metrics import compute_forecast_metrics
from sdfts.experiments.run_forecast_panel import build_or_load_windows, load_panel_outputs
from sdfts.routers.train_router import save_router, train_router
from sdfts.utils.logging import get_logger


log = get_logger(__name__)


def _build_val_cards(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Construct val-instance cards + failure labels (top-quantile)."""
    ws = build_or_load_windows(cfg)
    outs = load_panel_outputs(cfg)
    val_forecasts = outs["val_forecasts"]
    val_Y = outs["val_targets"]
    meta = outs["meta"]["models"]
    cand_ids = list(val_forecasts.keys())
    archs = [meta[c]["architecture"] for c in cand_ids]
    regimes = [meta[c]["training_regime"] for c in cand_ids]

    # Validation metrics for each candidate.
    val_metrics = {c: compute_forecast_metrics(val_Y, val_forecasts[c]) for c in cand_ids}

    # Default policy on val.
    default_forecasts, _ = build_default_policy(val_metrics, val_forecasts)
    default_errs = np.mean(np.abs(val_Y - default_forecasts), axis=1)
    q = float(np.quantile(default_errs, 1.0 - float(cfg["routers"]["top_quantile"])))
    labels = (default_errs >= q).astype(np.int64)

    ood_ref = OODReference.from_windows(ws.train)
    cards: list[dict[str, Any]] = []
    for i, w in enumerate(ws.val):
        forecasts_i = np.stack([val_forecasts[c][i] for c in cand_ids], axis=0)
        diag_d = disagreement_features(forecasts_i, archs, regimes)
        x_orig = w.x * w.scale.std + w.scale.mean
        diag = {
            "architecture_disagreement": diag_d["architecture_disagreement"],
            "training_regime_disagreement": diag_d["training_regime_disagreement"],
            "overall_forecast_dispersion": diag_d["overall_forecast_dispersion"],
            "disagreement_slope_over_horizon": diag_d["disagreement_slope_over_horizon"],
            "horizonwise_disagreement": diag_d["horizonwise_disagreement"],
            "trend_strength": trend_strength(x_orig),
            "seasonality_strength": seasonality_strength(x_orig),
            "changepoint_score": changepoint_score(x_orig),
            "missingness_rate": missingness_rate(x_orig),
            "input_volatility": volatility(x_orig),
            "recent_level_shift_score": recent_level_shift(x_orig),
            "ood_score": ood_score(x_orig, ood_ref, neighbors=int(cfg["cards"]["ood_neighbors"])),
            "foundation_vs_supervised_disagreement": None,
            "context_forecast_conflict_score": None,
        }
        candidate_models = []
        for c in cand_ids:
            candidate_models.append({
                "model_id": c,
                "architecture": meta[c]["architecture"],
                "training_regime": meta[c]["training_regime"],
                "forecast": val_forecasts[c][i].astype(float).tolist(),
                "validation_metrics": val_metrics[c],
                "horizonwise_validation_mae": [
                    float(np.mean(np.abs(val_Y[:, h] - val_forecasts[c][:, h])))
                    for h in range(ws.forecast_horizon)
                ],
                "recent_residual_summary": {"mean": 0.0, "std": 0.0, "autocorr_lag1": 0.0, "n": 0},
            })
        cards.append({
            "instance_id": w.instance_id,
            "dataset_name": cfg["data"]["dataset"],
            "series_id": w.series_id,
            "input_length": ws.input_length,
            "forecast_horizon": ws.forecast_horizon,
            "candidate_models": candidate_models,
            "diagnostics": diag,
            "default_decision": {"policy": "validation_weighted_ensemble",
                                  "forecast": default_forecasts[i].astype(float).tolist()},
            "allowed_evidence_ids": [],
        })
    return cards, labels


def train_routers(cfg: dict[str, Any]) -> dict[str, Any]:
    val_cards, val_labels = _build_val_cards(cfg)
    out_dir = run_dir(cfg) / "routers"
    out_dir.mkdir(parents=True, exist_ok=True)
    metas: dict[str, Any] = {"models": list(cfg["routers"]["models"]),
                              "n_val_instances": int(len(val_cards)),
                              "positive_rate": float(np.mean(val_labels)) if len(val_labels) else 0.0}
    for name in cfg["routers"]["models"]:
        router = train_router(name, val_cards, val_labels, seed=int(cfg["seed"]))
        save_router(router, out_dir / f"{name}.pkl")
        log.info("Trained router '%s' on %d val instances (pos rate=%.2f)",
                 name, len(val_cards), metas["positive_rate"])
    (out_dir / "routers_meta.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")
    return metas
