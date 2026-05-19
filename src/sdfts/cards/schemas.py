"""JSON Schemas for forecast cards and private labels.

The forecast card is the only thing LLM agents are ever shown. The private
label file contains ground-truth horizons and per-candidate test errors, plus
oracle-best annotations; agents must never see it.
"""
from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator


FORECAST_CARD_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ForecastCard",
    "type": "object",
    "required": [
        "instance_id",
        "dataset_name",
        "series_id",
        "input_length",
        "forecast_horizon",
        "candidate_models",
        "diagnostics",
        "default_decision",
        "allowed_evidence_ids",
    ],
    "properties": {
        "instance_id": {"type": "string"},
        "dataset_name": {"type": "string"},
        "series_id": {"type": "string"},
        "input_length": {"type": "integer", "minimum": 1},
        "forecast_horizon": {"type": "integer", "minimum": 1},
        "time_index_start": {"type": ["string", "integer", "null"]},
        "time_index_end": {"type": ["string", "integer", "null"]},
        "candidate_models": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "model_id",
                    "architecture",
                    "training_regime",
                    "forecast",
                    "validation_metrics",
                ],
                "properties": {
                    "model_id": {"type": "string"},
                    "architecture": {"type": "string"},
                    "training_regime": {"type": "string"},
                    "forecast": {"type": "array", "items": {"type": "number"}},
                    "validation_metrics": {
                        "type": "object",
                        "properties": {
                            "mae": {"type": "number"},
                            "rmse": {"type": "number"},
                            "smape": {"type": "number"},
                            "mase": {"type": "number"},
                        },
                    },
                    "horizonwise_validation_mae": {"type": "array", "items": {"type": "number"}},
                    "recent_residual_summary": {
                        "type": "object",
                        "properties": {
                            "mean": {"type": "number"},
                            "std": {"type": "number"},
                            "autocorr_lag1": {"type": "number"},
                            "n": {"type": "integer"},
                        },
                    },
                },
            },
        },
        "diagnostics": {
            "type": "object",
            "properties": {
                "architecture_disagreement": {"type": "number"},
                "training_regime_disagreement": {"type": "number"},
                "horizonwise_disagreement": {"type": "array", "items": {"type": "number"}},
                "overall_forecast_dispersion": {"type": "number"},
                "foundation_vs_supervised_disagreement": {"type": ["number", "null"]},
                "trend_strength": {"type": "number"},
                "seasonality_strength": {"type": "number"},
                "changepoint_score": {"type": "number"},
                "missingness_rate": {"type": "number"},
                "input_volatility": {"type": "number"},
                "ood_score": {"type": "number"},
                "context_forecast_conflict_score": {"type": ["number", "null"]},
            },
        },
        "default_decision": {
            "type": "object",
            "required": ["policy", "forecast"],
            "properties": {
                "policy": {"type": "string"},
                "forecast": {"type": "array", "items": {"type": "number"}},
            },
        },
        "allowed_evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "additionalProperties": True,
}


PRIVATE_LABEL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PrivateLabel",
    "type": "object",
    "required": [
        "instance_id",
        "ground_truth",
        "candidate_errors",
        "oracle_best_model_id",
        "oracle_best_error",
        "default_error",
        "failure_label_top20",
        "failure_label_threshold",
    ],
    "properties": {
        "instance_id": {"type": "string"},
        "ground_truth": {"type": "array", "items": {"type": "number"}},
        "candidate_errors": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "mae": {"type": "number"},
                    "rmse": {"type": "number"},
                    "smape": {"type": "number"},
                    "mase": {"type": "number"},
                },
            },
        },
        "oracle_best_model_id": {"type": "string"},
        "oracle_best_error": {"type": "number"},
        "default_error": {"type": "number"},
        "failure_label_top20": {"type": "integer", "minimum": 0, "maximum": 1},
        "failure_label_threshold": {"type": "integer", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": True,
}


_card_validator = Draft202012Validator(FORECAST_CARD_SCHEMA)
_label_validator = Draft202012Validator(PRIVATE_LABEL_SCHEMA)


def validate_card(obj: dict[str, Any]) -> list[str]:
    return [e.message for e in _card_validator.iter_errors(obj)]


def validate_label(obj: dict[str, Any]) -> list[str]:
    return [e.message for e in _label_validator.iter_errors(obj)]
