"""Agent / judge JSON schemas + MockProvider behavior."""
from __future__ import annotations

import json

from sdfts.agents.providers import MockProvider
from sdfts.agents.schemas import (
    AGENT_OUTPUT_SCHEMA,
    JUDGE_OUTPUT_SCHEMA,
    validate_agent_output,
    validate_judge_output,
)


def test_agent_schema_rejects_missing_fields():
    bad = {"agent_id": "x", "recommendation": "select_model"}
    errs = validate_agent_output(bad)
    assert errs


def test_judge_schema_rejects_invalid_enum():
    bad = {
        "decision_type": "invent",
        "selected_model_ids": [],
        "ensemble_weights": {},
        "confidence": 0.5,
        "failure_probability": 0.5,
        "verified_evidence_ids": [],
        "rejected_claims": [],
        "rationale": "",
        "should_abstain": False,
    }
    assert validate_judge_output(bad)


def test_mock_provider_returns_valid_agent_output(sample_card):
    prov = MockProvider(seed=0)
    resp = prov.complete_json(
        system="sys", user="recursive_specialist body",
        schema=AGENT_OUTPUT_SCHEMA, forecast_card=sample_card,
    )
    errs = validate_agent_output(resp.parsed)
    assert not errs, errs


def test_mock_provider_returns_valid_judge_output(sample_card):
    prov = MockProvider(seed=0)
    # Inject one agent output as context.
    agents = [{
        "agent_id": "recursive_specialist",
        "recommendation": "select_model",
        "preferred_model_ids": ["lstm_one_step_recursive"],
        "confidence": 0.7,
        "failure_risk": 0.3,
        "verified_evidence_ids": ["model.lstm_one_step_recursive.validation_metrics.mae"],
        "unsupported_claims": [],
        "main_failure_modes": [],
        "questions_for_other_agents": [],
    }]
    user = (
        "judge body\n"
        f"AGENT_OUTPUTS_JSON_BEGIN\n{json.dumps(agents)}\nAGENT_OUTPUTS_JSON_END"
    )
    resp = prov.complete_json(system="sys", user=user, schema=JUDGE_OUTPUT_SCHEMA, forecast_card=sample_card)
    errs = validate_judge_output(resp.parsed)
    assert not errs, errs
    assert resp.parsed["decision_type"] in {"select_model", "ensemble", "abstain", "flag_failure"}
