"""Run LLM agent decision systems on the test set.

For each enabled decision system in cfg, iterate over the **test** forecast
cards, run the corresponding harness, and store one row per (instance, system)
to ``outputs/<run_name>/agents/decisions.jsonl``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from sdfts.agents.cache import LLMCache
from sdfts.agents.debate import (
    AgentRunStats,
    run_debate,
    run_independent_vote,
    run_single_auditor,
    run_tool_only_verifier,
)
from sdfts.agents.providers import get_provider
from sdfts.cards.io import read_cards
from sdfts.config import REPO_ROOT, run_dir
from sdfts.routers.predict_router import predict_failure_prob
from sdfts.routers.train_router import load_router
from sdfts.utils.io import write_jsonl
from sdfts.utils.logging import get_logger


log = get_logger(__name__)


def _all_decision_systems(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["decision_systems"].get("llm", []))


def run_all(cfg: dict[str, Any], systems: list[str] | None = None, provider_override: str | None = None) -> dict[str, Any]:
    if provider_override:
        cfg = dict(cfg)
        cfg["agents"] = dict(cfg["agents"])
        cfg["agents"]["provider"] = provider_override
    provider = get_provider(cfg)
    cache_dir = REPO_ROOT / cfg["agents"]["cache_dir"]
    cache = LLMCache(cache_dir)

    rd = run_dir(cfg)
    cards = list(read_cards(rd))
    systems = systems or _all_decision_systems(cfg)

    # Load a router for selective-debate systems if available.
    selective_router_path = rd / "routers" / "gradient_boosted.pkl"
    if not selective_router_path.exists():
        # Fall back to whatever router exists.
        for r in cfg["routers"]["models"]:
            p = rd / "routers" / f"{r}.pkl"
            if p.exists():
                selective_router_path = p
                break
    router = load_router(selective_router_path) if selective_router_path.exists() else None
    failure_probs = predict_failure_prob(router, cards) if router is not None else np.full(len(cards), 0.5)

    decisions: list[dict[str, Any]] = []
    cost_summary: dict[str, dict[str, float]] = {}
    for sys_name in systems:
        log.info("Running decision system: %s", sys_name)
        sys_cost = {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "elapsed_seconds": 0.0}
        for i, card in enumerate(cards):
            t0 = time.time()
            jd, stats = _dispatch(sys_name, provider, cache, card, cfg, failure_probs[i])
            elapsed = time.time() - t0
            decisions.append({
                "instance_id": card["instance_id"],
                "decision_system": sys_name,
                "decision": jd,
                "stats": {
                    "n_calls": stats.n_calls,
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.completion_tokens,
                    "elapsed_seconds": stats.elapsed_seconds or elapsed,
                    "debaters": stats.debaters,
                },
                "router_failure_prob": float(failure_probs[i]),
            })
            sys_cost["n_calls"] += stats.n_calls
            sys_cost["prompt_tokens"] += stats.prompt_tokens
            sys_cost["completion_tokens"] += stats.completion_tokens
            sys_cost["elapsed_seconds"] += stats.elapsed_seconds or elapsed
        cost_summary[sys_name] = sys_cost

    out_dir = rd / "agents"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "decisions.jsonl", decisions)
    (out_dir / "cost_summary.json").write_text(json.dumps(cost_summary, indent=2), encoding="utf-8")
    log.info("Wrote %d agent decisions across %d systems", len(decisions), len(systems))
    return {"n_decisions": len(decisions), "systems": systems, "cost": cost_summary}


def _dispatch(
    sys_name: str,
    provider,
    cache: LLMCache,
    card: dict[str, Any],
    cfg: dict[str, Any],
    failure_prob: float,
) -> tuple[dict[str, Any], AgentRunStats]:
    """Map decision-system name to handler."""
    low_thr = 0.3
    high_thr = 0.7

    if sys_name == "single_llm_auditor":
        return run_single_auditor(provider, cache, card, cfg)
    if sys_name == "independent_agents_vote":
        return run_independent_vote(provider, cache, card, cfg)
    if sys_name == "always_on_debate":
        return run_debate(provider, cache, card, cfg, use_tools=False)
    if sys_name == "selective_debate_no_tools":
        if failure_prob < low_thr:
            return _baseline_default(card)
        return run_debate(provider, cache, card, cfg, use_tools=False)
    if sys_name == "tool_only_verifier":
        return run_tool_only_verifier(card)
    if sys_name == "selective_tool_verified_debate":
        if failure_prob < low_thr:
            return _baseline_default(card)
        if failure_prob < high_thr:
            return run_tool_only_verifier(card)
        return run_debate(provider, cache, card, cfg, use_tools=True)
    raise ValueError(f"Unknown LLM decision system: {sys_name}")


def _baseline_default(card: dict[str, Any]) -> tuple[dict[str, Any], AgentRunStats]:
    return ({
        "decision_type": "ensemble",
        "selected_model_ids": [],
        "ensemble_weights": {m["model_id"]: 1.0 / len(card["candidate_models"]) for m in card["candidate_models"]},
        "confidence": 0.7,
        "failure_probability": 0.2,
        "verified_evidence_ids": [],
        "rejected_claims": [],
        "rationale": "low failure-prob from router; default ensemble.",
        "should_abstain": False,
    }, AgentRunStats())
