"""Agent harnesses: single auditor, independent vote, debate.

Every entry point returns ``(JudgeOutput, AgentRunStats)`` so callers can build
metrics tables and cost reports.

We keep the protocol short by design: at most ``debate_rounds`` of agent
emissions before the judge is invoked. Rounds > 1 feed the previous round's
outputs back as context, but no agent ever sees ground truth.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from sdfts.agents.cache import LLMCache
from sdfts.agents.judge import judge_decision
from sdfts.agents.prompts import load_prompt, stable_hash
from sdfts.agents.providers import LLMProvider
from sdfts.agents.schemas import (
    AGENT_OUTPUT_SCHEMA,
    validate_agent_output,
)
from sdfts.agents.verifier import StatisticalVerifier


@dataclass
class AgentRunStats:
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0
    invalid_responses: int = 0
    debaters: list[str] = field(default_factory=list)


def _agent_call(
    provider: LLMProvider,
    agent_id: str,
    card: dict[str, Any],
    body: str,
    cache: LLMCache,
    cfg: dict[str, Any],
    extra_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int, int]:
    """Run one agent. Returns (parsed_output, prompt_tokens, completion_tokens)."""
    StatisticalVerifier.assert_no_label_leakage(card)
    system = load_prompt("shared_system")
    user_parts = [body, "", "FORECAST_CARD_JSON_BEGIN", json.dumps(card), "FORECAST_CARD_JSON_END"]
    if extra_context:
        user_parts += ["", "CONTEXT_JSON_BEGIN", json.dumps(extra_context), "CONTEXT_JSON_END"]
    user = "\n".join(user_parts)
    key = stable_hash(provider.name, cfg["agents"].get("model"),
                      cfg["agents"].get("prompt_version", "v1"), agent_id, user)
    cached = cache.get(key)
    if cached is not None:
        return cached["parsed"], int(cached.get("prompt_tokens", 0)), int(cached.get("completion_tokens", 0))

    last_errs: list[str] = []
    for attempt in range(int(cfg["agents"].get("max_retries", 2)) + 1):
        resp = provider.complete_json(
            system=system,
            user=user,
            schema=AGENT_OUTPUT_SCHEMA,
            temperature=float(cfg["agents"].get("temperature", 0.0)),
            forecast_card=card,
        )
        # Ensure agent_id is present and matches our expectation.
        if "agent_id" not in resp.parsed:
            resp.parsed["agent_id"] = agent_id
        last_errs = validate_agent_output(resp.parsed)
        if not last_errs:
            # Drop unsupported evidence IDs.
            allowed = set(card.get("allowed_evidence_ids", []))
            resp.parsed["verified_evidence_ids"] = [
                e for e in resp.parsed["verified_evidence_ids"] if e in allowed
            ]
            cache.set(key, {
                "parsed": resp.parsed,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
            })
            return resp.parsed, resp.prompt_tokens, resp.completion_tokens
    raise ValueError(f"Agent {agent_id} produced invalid JSON: {last_errs[:2]}")


def _which_agents(card: dict[str, Any], include_foundation: bool, mode: str) -> list[tuple[str, str]]:
    """Return list of (agent_id, prompt_body)."""
    regimes = {m["training_regime"] for m in card["candidate_models"]}
    archs = {m["architecture"] for m in card["candidate_models"]}
    spec = []
    if "one_step_recursive" in regimes:
        spec.append(("recursive_specialist", load_prompt("recursive_specialist")))
    if "h_step_direct" in regimes:
        spec.append(("h_step_specialist", load_prompt("h_step_specialist")))
    if "direct_multi_step" in regimes:
        spec.append(("direct_multistep_specialist", load_prompt("direct_multistep_specialist")))
    if include_foundation and any(a not in {"lstm", "gru", "transformer", "tcn"} for a in archs):
        spec.append(("foundation_specialist", load_prompt("foundation_specialist")))
    if mode != "single_auditor":
        spec.append(("skeptic", load_prompt("skeptic")))
    return spec


# ---------------------------------------------------------------------------
# Decision systems
# ---------------------------------------------------------------------------

def run_single_auditor(
    provider: LLMProvider,
    cache: LLMCache,
    card: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], AgentRunStats]:
    """LLM with a single prompt; judge aggregates one output."""
    stats = AgentRunStats()
    body = load_prompt("single_auditor")
    t0 = time.time()
    parsed, pt, ct = _agent_call(provider, "single_auditor", card, body, cache, cfg)
    stats.n_calls += 1
    stats.prompt_tokens += pt
    stats.completion_tokens += ct
    stats.debaters.append("single_auditor")
    judge, jtok = judge_decision(provider, card, [parsed], temperature=float(cfg["agents"]["temperature"]))
    stats.n_calls += 1
    stats.prompt_tokens += jtok["prompt_tokens"]
    stats.completion_tokens += jtok["completion_tokens"]
    stats.elapsed_seconds = time.time() - t0
    return judge, stats


def run_independent_vote(
    provider: LLMProvider,
    cache: LLMCache,
    card: dict[str, Any],
    cfg: dict[str, Any],
    include_foundation: bool = False,
) -> tuple[dict[str, Any], AgentRunStats]:
    """Run specialists independently; judge tallies their votes (no debate)."""
    stats = AgentRunStats()
    agents = _which_agents(card, include_foundation, mode="vote")
    t0 = time.time()
    outputs: list[dict[str, Any]] = []
    for agent_id, body in agents:
        parsed, pt, ct = _agent_call(provider, agent_id, card, body, cache, cfg)
        outputs.append(parsed)
        stats.n_calls += 1
        stats.prompt_tokens += pt
        stats.completion_tokens += ct
        stats.debaters.append(agent_id)
    judge, jtok = judge_decision(provider, card, outputs, temperature=float(cfg["agents"]["temperature"]))
    stats.n_calls += 1
    stats.prompt_tokens += jtok["prompt_tokens"]
    stats.completion_tokens += jtok["completion_tokens"]
    stats.elapsed_seconds = time.time() - t0
    return judge, stats


def run_debate(
    provider: LLMProvider,
    cache: LLMCache,
    card: dict[str, Any],
    cfg: dict[str, Any],
    use_tools: bool = True,
    include_foundation: bool = False,
) -> tuple[dict[str, Any], AgentRunStats]:
    """Run specialists with debate rounds; optionally include verifier output.

    Round-2 agents (when ``debate_rounds > 1``) see other agents' round-1
    outputs as ``CONTEXT_JSON_*`` blocks. The verifier report is included only
    when ``use_tools=True``.
    """
    stats = AgentRunStats()
    rounds = max(1, int(cfg["agents"].get("debate_rounds", 1)))
    agents = _which_agents(card, include_foundation, mode="debate")
    t0 = time.time()

    verifier_report = StatisticalVerifier.report(card) if use_tools else None

    last_outputs: list[dict[str, Any]] = []
    for r in range(rounds):
        round_outputs: list[dict[str, Any]] = []
        ctx = None
        if r > 0:
            ctx = {"previous_round_outputs": last_outputs}
        if use_tools:
            ctx = (ctx or {}) | {"verifier_report": verifier_report}
        for agent_id, body in agents:
            parsed, pt, ct = _agent_call(provider, agent_id, card, body, cache, cfg, extra_context=ctx)
            round_outputs.append(parsed)
            stats.n_calls += 1
            stats.prompt_tokens += pt
            stats.completion_tokens += ct
            stats.debaters.append(agent_id)
        last_outputs = round_outputs

    judge, jtok = judge_decision(
        provider, card, last_outputs, verifier_report=verifier_report,
        temperature=float(cfg["agents"]["temperature"]),
    )
    stats.n_calls += 1
    stats.prompt_tokens += jtok["prompt_tokens"]
    stats.completion_tokens += jtok["completion_tokens"]
    stats.elapsed_seconds = time.time() - t0
    return judge, stats


def run_tool_only_verifier(
    card: dict[str, Any],
) -> tuple[dict[str, Any], AgentRunStats]:
    """No LLM. Verifier picks the lowest-disagreement validation-best candidate
    and emits failure-probability proportional to dispersion/changepoint/OOD.

    This is the strongest "non-LLM" arbitration baseline that still uses the
    verifier-style reasoning the judge would do.
    """
    stats = AgentRunStats()
    cands = card["candidate_models"]
    diag = card.get("diagnostics", {})
    best = min(cands, key=lambda m: m["validation_metrics"]["mae"])
    risk = min(1.0, max(0.0,
        0.3 * float(diag.get("overall_forecast_dispersion", 0.0))
        + 0.3 * float(diag.get("changepoint_score", 0.0))
        + 0.2 * float(diag.get("ood_score", 0.0)) / 5.0
        + 0.2 * float(diag.get("training_regime_disagreement", 0.0))
    ))
    decision = "flag_failure" if risk > 0.85 else ("ensemble" if risk > 0.5 else "select_model")
    selected = [best["model_id"]] if decision == "select_model" else []
    weights: dict[str, float] = {}
    if decision == "ensemble":
        inv = [1.0 / (m["validation_metrics"]["mae"] + 1e-9) for m in cands]
        tot = sum(inv)
        weights = {m["model_id"]: inv[i] / tot for i, m in enumerate(cands)}
    return ({
        "decision_type": decision,
        "selected_model_ids": selected,
        "ensemble_weights": weights,
        "confidence": float(1.0 - risk),
        "failure_probability": float(risk),
        "verified_evidence_ids": [
            f"model.{best['model_id']}.validation_metrics.mae",
            "diag.overall_forecast_dispersion",
            "diag.changepoint_score",
        ],
        "rejected_claims": [],
        "rationale": "tool-only verifier",
        "should_abstain": False,
    }, stats)
