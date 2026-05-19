"""Agent and judge JSON output schemas."""
from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator


AGENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AgentOutput",
    "type": "object",
    "required": [
        "agent_id",
        "recommendation",
        "preferred_model_ids",
        "confidence",
        "failure_risk",
        "verified_evidence_ids",
        "unsupported_claims",
    ],
    "properties": {
        "agent_id": {"type": "string"},
        "recommendation": {"enum": ["select_model", "ensemble", "abstain", "flag_failure"]},
        "preferred_model_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "failure_risk": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "verified_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "main_failure_modes": {"type": "array", "items": {"type": "string"}},
        "questions_for_other_agents": {"type": "array", "items": {"type": "string"}},
    },
}


JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "JudgeOutput",
    "type": "object",
    "required": [
        "decision_type",
        "selected_model_ids",
        "ensemble_weights",
        "confidence",
        "failure_probability",
        "verified_evidence_ids",
        "rejected_claims",
        "rationale",
        "should_abstain",
    ],
    "properties": {
        "decision_type": {"enum": ["select_model", "ensemble", "abstain", "flag_failure"]},
        "selected_model_ids": {"type": "array", "items": {"type": "string"}},
        "ensemble_weights": {"type": "object", "additionalProperties": {"type": "number"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "failure_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "verified_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "rejected_claims": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
        "should_abstain": {"type": "boolean"},
    },
}


_agent_validator = Draft202012Validator(AGENT_OUTPUT_SCHEMA)
_judge_validator = Draft202012Validator(JUDGE_OUTPUT_SCHEMA)


def validate_agent_output(obj: dict[str, Any]) -> list[str]:
    return [e.message for e in _agent_validator.iter_errors(obj)]


def validate_judge_output(obj: dict[str, Any]) -> list[str]:
    return [e.message for e in _judge_validator.iter_errors(obj)]
