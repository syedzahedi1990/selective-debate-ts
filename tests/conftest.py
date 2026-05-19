"""Shared fixtures.

We synthesize a tiny in-memory pipeline state so unit tests don't require
running the full smoke pipeline first. Tests that *do* assert end-to-end
artefacts (no-leakage, schemas) will skip cleanly if those artefacts haven't
been produced yet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sdfts.config import load_config, REPO_ROOT


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory) -> dict[str, Any]:
    cfg = load_config(REPO_ROOT / "configs" / "experiments" / "mvp.yaml")
    # Re-route outputs to a temp dir so tests don't pollute real runs.
    tmp = tmp_path_factory.mktemp("sdfts_run")
    cfg["run_name"] = "pytest_smoke"
    cfg["output_root"] = str(tmp)
    return cfg


@pytest.fixture
def sample_card() -> dict[str, Any]:
    return {
        "instance_id": "synthetic_000__test__00001",
        "dataset_name": "synthetic",
        "series_id": "synthetic_000",
        "input_length": 16,
        "forecast_horizon": 4,
        "candidate_models": [
            {
                "model_id": "lstm_one_step_recursive",
                "architecture": "lstm",
                "training_regime": "one_step_recursive",
                "forecast": [0.1, 0.2, 0.3, 0.4],
                "validation_metrics": {"mae": 0.1, "rmse": 0.15, "smape": 0.1, "mase": 1.0},
                "horizonwise_validation_mae": [0.1, 0.1, 0.1, 0.1],
                "recent_residual_summary": {"mean": 0.0, "std": 0.1, "autocorr_lag1": 0.05, "n": 20},
            },
            {
                "model_id": "tcn_direct_multi_step",
                "architecture": "tcn",
                "training_regime": "direct_multi_step",
                "forecast": [0.12, 0.18, 0.31, 0.39],
                "validation_metrics": {"mae": 0.09, "rmse": 0.14, "smape": 0.09, "mase": 0.9},
                "horizonwise_validation_mae": [0.09, 0.09, 0.09, 0.09],
                "recent_residual_summary": {"mean": 0.0, "std": 0.1, "autocorr_lag1": 0.05, "n": 20},
            },
        ],
        "diagnostics": {
            "architecture_disagreement": 0.01,
            "training_regime_disagreement": 0.02,
            "horizonwise_disagreement": [0.01, 0.02, 0.03, 0.04],
            "overall_forecast_dispersion": 0.025,
            "trend_strength": 0.5, "seasonality_strength": 0.4,
            "changepoint_score": 0.1, "missingness_rate": 0.0,
            "input_volatility": 0.3, "ood_score": 0.5,
            "foundation_vs_supervised_disagreement": None,
            "context_forecast_conflict_score": None,
        },
        "default_decision": {
            "policy": "validation_weighted_ensemble",
            "forecast": [0.11, 0.19, 0.305, 0.395],
        },
        "allowed_evidence_ids": [
            "diag.overall_forecast_dispersion",
            "diag.changepoint_score",
            "model.lstm_one_step_recursive.validation_metrics.mae",
            "model.tcn_direct_multi_step.validation_metrics.mae",
        ],
    }


@pytest.fixture
def sample_label() -> dict[str, Any]:
    return {
        "instance_id": "synthetic_000__test__00001",
        "ground_truth": [0.1, 0.2, 0.3, 0.4],
        "candidate_errors": {
            "lstm_one_step_recursive": {"mae": 0.0, "rmse": 0.0, "smape": 0.0, "mase": 0.0},
            "tcn_direct_multi_step": {"mae": 0.02, "rmse": 0.02, "smape": 0.05, "mase": 0.2},
        },
        "oracle_best_model_id": "lstm_one_step_recursive",
        "oracle_best_error": 0.0,
        "default_error": 0.01,
        "failure_label_top20": 0,
        "failure_label_threshold": 0,
    }
