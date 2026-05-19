"""Judge step.

The judge consumes a list of validated agent outputs + a verifier report and
returns a JudgeOutput. In ``mock`` mode the MockProvider already implements a
sensible aggregation, so this function is a thin wrapper around the provider
that also enforces evidence-ID validation and schema checking.
"""
from __future__ import annotations

import json
from typing import Any

from sdfts.agents.providers import LLMProvider
from sdfts.agents.schemas import validate_judge_output
from sdfts.agents.prompts import load_prompt
from sdfts.agents.verifier import StatisticalVerifier


def judge_decision(
    provider: LLMProvider,
    card: dict[str, Any],
    agent_outputs: list[dict[str, Any]],
    verifier_report: dict[str, Any] | None = None,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], dict[str, int]]:
    verifier_report = verifier_report or StatisticalVerifier.report(card)
    system = load_prompt("shared_system")
    judge_body = load_prompt("judge")
    user = _build_judge_user_message(card, agent_outputs, verifier_report, judge_body)

    from sdfts.agents.schemas import JUDGE_OUTPUT_SCHEMA
    resp = provider.complete_json(
        system=system,
        user=user,
        schema=JUDGE_OUTPUT_SCHEMA,
        temperature=temperature,
        forecast_card=card,
    )
    parsed = resp.parsed

    errs = validate_judge_output(parsed)
    if errs:
        parsed = _coerce_judge(parsed, card)
        errs = validate_judge_output(parsed)
        if errs:
            raise ValueError(f"Judge output failed schema: {errs[:2]}")

    # Drop unsupported evidence IDs.
    allowed = set(card.get("allowed_evidence_ids", []))
    parsed["verified_evidence_ids"] = [e for e in parsed["verified_evidence_ids"] if e in allowed]

    tokens = {"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens}
    return parsed, tokens


def _build_judge_user_message(
    card: dict[str, Any],
    agent_outputs: list[dict[str, Any]],
    verifier_report: dict[str, Any],
    body: str,
) -> str:
    parts = [body, "", "FORECAST_CARD_JSON_BEGIN", json.dumps(card), "FORECAST_CARD_JSON_END"]
    parts += ["", "VERIFIER_REPORT_JSON_BEGIN", json.dumps(verifier_report), "VERIFIER_REPORT_JSON_END"]
    parts += ["", "AGENT_OUTPUTS_JSON_BEGIN", json.dumps(agent_outputs), "AGENT_OUTPUTS_JSON_END"]
    return "\n".join(parts)


def _coerce_judge(parsed: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    """Heuristic recovery when the judge JSON is malformed.

    We pick the validation-best candidate as a defensible default, then keep
    whatever fields parsed correctly.
    """
    out = dict(parsed or {})
    best = min(card["candidate_models"], key=lambda m: m["validation_metrics"]["mae"])
    out.setdefault("decision_type", "select_model")
    out.setdefault("selected_model_ids", [best["model_id"]])
    out.setdefault("ensemble_weights", {})
    out.setdefault("confidence", 0.5)
    out.setdefault("failure_probability", 0.5)
    out.setdefault("verified_evidence_ids", [])
    out.setdefault("rejected_claims", [])
    out.setdefault("rationale", "auto-coerced")
    out.setdefault("should_abstain", False)
    return out
