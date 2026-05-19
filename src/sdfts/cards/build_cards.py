"""Assemble forecast cards and private labels from forecasts + diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np

from sdfts.data.windowing import Window, WindowSet
from sdfts.diagnostics.changepoint import changepoint_score
from sdfts.diagnostics.disagreement import (
    disagreement_features,
    foundation_vs_supervised_disagreement,
)
from sdfts.diagnostics.features import (
    missingness_rate,
    recent_level_shift,
    seasonality_strength,
    trend_strength,
    volatility,
)
from sdfts.diagnostics.ood import OODReference, ood_score
from sdfts.diagnostics.residuals import recent_residual_summary
from sdfts.evaluation.metrics import compute_forecast_metrics


def build_default_policy(
    val_metrics_per_cand: dict[str, dict[str, float]],
    test_forecasts: dict[str, np.ndarray],
    policy: str = "validation_weighted_ensemble",
) -> tuple[np.ndarray, str]:
    """Compute the default-decision forecast over candidates."""
    cand_ids = list(test_forecasts.keys())
    F = np.stack([test_forecasts[c] for c in cand_ids], axis=0)   # (K, N, H)
    if policy == "validation_best":
        best = min(cand_ids, key=lambda c: val_metrics_per_cand[c]["mae"])
        return test_forecasts[best], policy
    if policy == "simple_mean_ensemble":
        return F.mean(axis=0), policy
    if policy == "median_ensemble":
        return np.median(F, axis=0), policy
    if policy == "validation_weighted_ensemble":
        inv = np.array([1.0 / (val_metrics_per_cand[c]["mae"] + 1e-9) for c in cand_ids])
        w = inv / inv.sum()
        return (w[:, None, None] * F).sum(axis=0), policy
    raise ValueError(f"Unknown default policy: {policy}")


def _select_evidence_ids(card: dict[str, Any]) -> list[str]:
    """List of evidence IDs that agents are allowed to cite."""
    eids = [
        "diag.architecture_disagreement",
        "diag.training_regime_disagreement",
        "diag.overall_forecast_dispersion",
        "diag.horizonwise_disagreement",
        "diag.trend_strength",
        "diag.seasonality_strength",
        "diag.changepoint_score",
        "diag.input_volatility",
        "diag.missingness_rate",
        "diag.ood_score",
    ]
    for m in card["candidate_models"]:
        mid = m["model_id"]
        for k in m["validation_metrics"].keys():
            eids.append(f"model.{mid}.validation_metrics.{k}")
        for i in range(len(m.get("horizonwise_validation_mae", []))):
            eids.append(f"model.{mid}.horizonwise_validation_mae[{i}]")
        if "recent_residual_summary" in m:
            for k in m["recent_residual_summary"].keys():
                eids.append(f"model.{mid}.recent_residual_summary.{k}")
    return eids


def build_cards(
    cfg: dict[str, Any],
    ws: WindowSet,
    test_forecasts: dict[str, np.ndarray],
    val_forecasts: dict[str, np.ndarray],
    cand_meta: dict[str, dict[str, str]],
    foundation_test_forecasts: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build forecast cards and private labels for the **test** split."""
    cand_ids = list(test_forecasts.keys())
    archs = [cand_meta[c]["architecture"] for c in cand_ids]
    regimes = [cand_meta[c]["training_regime"] for c in cand_ids]
    horizon = ws.forecast_horizon

    # Per-candidate validation-error metrics (single number + per-horizon).
    val_metrics: dict[str, dict[str, float]] = {}
    val_horizonwise: dict[str, list[float]] = {}
    val_residuals: dict[str, np.ndarray] = {}
    val_Y = _stack_y_orig(ws.val)
    for c in cand_ids:
        yhat = val_forecasts[c]                          # (Nval, H)
        val_metrics[c] = compute_forecast_metrics(val_Y, yhat)
        val_horizonwise[c] = [
            float(np.mean(np.abs(val_Y[:, h] - yhat[:, h]))) for h in range(horizon)
        ]
        val_residuals[c] = (val_Y - yhat).reshape(-1)    # flattened residuals

    # Default policy uses validation-weighted ensemble by default.
    default_forecasts, default_policy = build_default_policy(
        val_metrics, test_forecasts, policy="validation_weighted_ensemble"
    )

    # OOD reference from train windows.
    ood_ref = OODReference.from_windows(ws.train)

    test_Y = _stack_y_orig(ws.test)
    cards: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    for i, w in enumerate(ws.test):
        # Per-candidate forecast on this instance.
        forecasts_i = np.stack([test_forecasts[c][i] for c in cand_ids], axis=0)   # (K, H)
        diag_disag = disagreement_features(forecasts_i, archs, regimes)
        f_vs_s = None
        if foundation_test_forecasts:
            fdn = np.stack([foundation_test_forecasts[c][i] for c in foundation_test_forecasts], axis=0)
            f_vs_s = foundation_vs_supervised_disagreement(fdn, forecasts_i)

        # Input-window diagnostics (use original-scale window for interpretability).
        x_orig = w.x * w.scale.std + w.scale.mean
        diag = {
            "architecture_disagreement": diag_disag["architecture_disagreement"],
            "training_regime_disagreement": diag_disag["training_regime_disagreement"],
            "horizonwise_disagreement": diag_disag["horizonwise_disagreement"],
            "overall_forecast_dispersion": diag_disag["overall_forecast_dispersion"],
            "disagreement_slope_over_horizon": diag_disag["disagreement_slope_over_horizon"],
            "foundation_vs_supervised_disagreement": f_vs_s,
            "trend_strength": trend_strength(x_orig),
            "seasonality_strength": seasonality_strength(x_orig),
            "changepoint_score": changepoint_score(x_orig),
            "missingness_rate": missingness_rate(x_orig),
            "input_volatility": volatility(x_orig),
            "recent_level_shift_score": recent_level_shift(x_orig),
            "ood_score": ood_score(x_orig, ood_ref, neighbors=int(cfg["cards"]["ood_neighbors"])),
            "context_forecast_conflict_score": None,
        }

        cand_entries = []
        for c in cand_ids:
            cand_entries.append({
                "model_id": c,
                "architecture": cand_meta[c]["architecture"],
                "training_regime": cand_meta[c]["training_regime"],
                "forecast": test_forecasts[c][i].astype(float).tolist(),
                "validation_metrics": val_metrics[c],
                "horizonwise_validation_mae": val_horizonwise[c],
                "recent_residual_summary": recent_residual_summary(
                    val_residuals[c], window=int(cfg["cards"]["recent_residual_window"])
                ),
            })

        card = {
            "instance_id": w.instance_id,
            "dataset_name": cfg["data"]["dataset"],
            "series_id": w.series_id,
            "input_length": ws.input_length,
            "forecast_horizon": ws.forecast_horizon,
            "time_index_start": None,
            "time_index_end": None,
            "candidate_models": cand_entries,
            "diagnostics": diag,
            "default_decision": {
                "policy": default_policy,
                "forecast": default_forecasts[i].astype(float).tolist(),
            },
        }
        card["allowed_evidence_ids"] = _select_evidence_ids(card)
        cards.append(card)

        # ------- private label -------
        cand_errs = {c: compute_forecast_metrics(test_Y[i:i+1], test_forecasts[c][i:i+1])
                     for c in cand_ids}
        best_id = min(cand_ids, key=lambda c: cand_errs[c]["mae"])
        default_err = float(np.mean(np.abs(test_Y[i] - default_forecasts[i])))
        labels.append({
            "instance_id": w.instance_id,
            "ground_truth": test_Y[i].astype(float).tolist(),
            "candidate_errors": cand_errs,
            "oracle_best_model_id": best_id,
            "oracle_best_error": cand_errs[best_id]["mae"],
            "default_error": default_err,
            "failure_label_top20": 0,           # filled in below
            "failure_label_threshold": 0,        # filled in below
        })

    # Failure labels are *dataset-relative*, set in a second pass.
    default_errs = np.array([l["default_error"] for l in labels])
    if default_errs.size > 0:
        thr_top = float(np.quantile(default_errs, 1.0 - float(cfg["routers"]["top_quantile"])))
        # Threshold rule: default_err > val_mean + k*val_std on the *default* policy.
        val_default = _apply_default_to_val(val_metrics, val_forecasts, val_Y)
        val_mean = float(np.mean(val_default))
        val_std = float(np.std(val_default))
        k = float(cfg["routers"]["threshold_multiplier"])
        thr_abs = val_mean + k * val_std
        for l, err in zip(labels, default_errs):
            l["failure_label_top20"] = int(err >= thr_top)
            l["failure_label_threshold"] = int(err > thr_abs)
            l["thr_top_quantile"] = thr_top
            l["thr_absolute"] = thr_abs

    return cards, labels


def _stack_y_orig(windows: list[Window]) -> np.ndarray:
    if not windows:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack([w.y * w.scale.std + w.scale.mean for w in windows], axis=0).astype(np.float32)


def _apply_default_to_val(
    val_metrics: dict[str, dict[str, float]],
    val_forecasts: dict[str, np.ndarray],
    val_Y: np.ndarray,
) -> np.ndarray:
    cand_ids = list(val_forecasts.keys())
    F = np.stack([val_forecasts[c] for c in cand_ids], axis=0)
    inv = np.array([1.0 / (val_metrics[c]["mae"] + 1e-9) for c in cand_ids])
    w = inv / inv.sum()
    default = (w[:, None, None] * F).sum(axis=0)
    return np.mean(np.abs(val_Y - default), axis=1)        # per-instance MAE
