"""LLM providers.

Common interface (`LLMProvider.complete_json`) and three concrete classes:

* MockProvider: fully deterministic, useful for tests and dry runs.
* OpenAIProvider, AnthropicProvider: real APIs (skipped if SDKs not installed).

Each provider returns a parsed dict matching the requested schema, plus token
counts so callers can roll cost metrics.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

from sdfts.agents.schemas import (
    validate_agent_output,
    validate_judge_output,
)


@dataclass
class LLMResponse:
    parsed: dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_text: str = ""


class LLMProvider:
    name: str = "base"

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
        forecast_card: dict[str, Any] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MockProvider — deterministic, label-free.
# ---------------------------------------------------------------------------

@dataclass
class MockProvider(LLMProvider):
    """Pretends to be an agent. Reads only the forecast card.

    Strategy:
    - Agent: pick the best-validation candidate that matches the agent's regime
      (if "specialist"). Confidence/failure-risk derived from diagnostics.
    - Skeptic: bumps failure_risk if disagreement is high.
    - Judge: tally agent recommendations + verifier-style features.
    """
    name: str = "mock"
    seed: int = 0
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
        forecast_card: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if forecast_card is None:
            # Generic stub: emit empty object satisfying nothing.
            return LLMResponse(parsed={}, prompt_tokens=len(user.split()), completion_tokens=4, raw_text="{}")

        # Decide agent vs judge by looking at the schema title.
        title = schema.get("title", "")
        if title == "JudgeOutput":
            parsed = self._mock_judge(user, forecast_card)
        else:
            parsed = self._mock_agent(user, forecast_card)

        raw = json.dumps(parsed)
        return LLMResponse(
            parsed=parsed,
            prompt_tokens=max(1, len(user.split())),
            completion_tokens=max(1, len(raw.split())),
            raw_text=raw,
        )

    def _agent_id_from_prompt(self, user: str) -> str:
        for tag in [
            "recursive_specialist",
            "h_step_specialist",
            "direct_multistep_specialist",
            "foundation_specialist",
            "skeptic",
            "single_auditor",
        ]:
            if tag in user:
                return tag
        return "agent"

    def _mock_agent(self, user: str, card: dict[str, Any]) -> dict[str, Any]:
        agent_id = self._agent_id_from_prompt(user)
        cands = card["candidate_models"]
        diag = card.get("diagnostics", {})

        # Filter to the regime this specialist cares about.
        regime_map = {
            "recursive_specialist": "one_step_recursive",
            "h_step_specialist": "h_step_direct",
            "direct_multistep_specialist": "direct_multi_step",
        }
        regime = regime_map.get(agent_id)
        if regime is not None:
            scope = [m for m in cands if m["training_regime"] == regime]
        else:
            scope = cands

        if not scope:
            scope = cands

        # Pick best-validation MAE in scope.
        best = min(scope, key=lambda m: m["validation_metrics"]["mae"])
        # Failure-risk: monotone in dispersion, changepoint, ood.
        risk = min(1.0, max(0.0, 0.2 + 0.3 * float(diag.get("overall_forecast_dispersion", 0.0))
                            + 0.2 * float(diag.get("changepoint_score", 0.0))
                            + 0.2 * float(diag.get("ood_score", 0.0)) / 5.0))
        # Skeptic is more conservative.
        if agent_id == "skeptic":
            risk = min(1.0, risk + 0.2)
        conf = 1.0 - risk

        evidence = [
            f"model.{best['model_id']}.validation_metrics.mae",
            "diag.overall_forecast_dispersion",
            "diag.changepoint_score",
        ]
        # Validate evidence IDs.
        allowed = set(card.get("allowed_evidence_ids", []))
        evidence = [e for e in evidence if e in allowed] or list(allowed)[:1]

        rec = "select_model" if risk < 0.5 else ("flag_failure" if risk > 0.85 else "ensemble")
        out = {
            "agent_id": agent_id,
            "recommendation": rec,
            "preferred_model_ids": [best["model_id"]],
            "confidence": float(conf),
            "failure_risk": float(risk),
            "verified_evidence_ids": evidence,
            "unsupported_claims": [],
            "main_failure_modes": ["changepoint" if diag.get("changepoint_score", 0.0) > 0.5 else "noise"],
            "questions_for_other_agents": [],
        }
        return out

    def _mock_judge(self, user: str, card: dict[str, Any]) -> dict[str, Any]:
        # Aggregate by reading the "AGENT OUTPUTS" section of the prompt.
        # The harness injects already-validated agent outputs as JSON; we
        # parse them out by a known marker if present, else fall back.
        agent_outputs: list[dict[str, Any]] = []
        marker = "AGENT_OUTPUTS_JSON_BEGIN"
        end = "AGENT_OUTPUTS_JSON_END"
        if marker in user and end in user:
            blob = user.split(marker, 1)[1].split(end, 1)[0]
            try:
                agent_outputs = json.loads(blob)
            except Exception:  # noqa: BLE001
                agent_outputs = []

        if not agent_outputs:
            # Vanilla auditor decision: pick best-validation MAE.
            best = min(card["candidate_models"], key=lambda m: m["validation_metrics"]["mae"])
            risk = float(card["diagnostics"].get("overall_forecast_dispersion", 0.0))
            return {
                "decision_type": "select_model",
                "selected_model_ids": [best["model_id"]],
                "ensemble_weights": {},
                "confidence": 1.0 - risk,
                "failure_probability": min(1.0, risk),
                "verified_evidence_ids": [f"model.{best['model_id']}.validation_metrics.mae"],
                "rejected_claims": [],
                "rationale": "no agent inputs; defaulted to validation-best.",
                "should_abstain": False,
            }

        # Tally votes
        votes: dict[str, int] = {}
        for a in agent_outputs:
            for mid in a.get("preferred_model_ids", []):
                votes[mid] = votes.get(mid, 0) + 1
        risks = [a.get("failure_risk", 0.0) for a in agent_outputs]
        failure_prob = float(sum(risks) / max(1, len(risks)))
        recs = [a.get("recommendation") for a in agent_outputs]
        if failure_prob > 0.85 or recs.count("flag_failure") >= max(2, len(recs) // 2):
            decision = "flag_failure"
        elif recs.count("abstain") >= max(2, len(recs) // 2):
            decision = "abstain"
        elif recs.count("ensemble") >= max(2, len(recs) // 2):
            decision = "ensemble"
        else:
            decision = "select_model"
        if votes:
            top_id = max(votes.items(), key=lambda kv: kv[1])[0]
        else:
            top_id = min(card["candidate_models"], key=lambda m: m["validation_metrics"]["mae"])["model_id"]
        selected = [top_id] if decision == "select_model" else list(votes.keys()) if decision == "ensemble" else []
        weights: dict[str, float] = {}
        if decision == "ensemble" and votes:
            tot = sum(votes.values())
            weights = {k: v / tot for k, v in votes.items()}
        return {
            "decision_type": decision,
            "selected_model_ids": selected,
            "ensemble_weights": weights,
            "confidence": float(max(0.0, 1.0 - failure_prob)),
            "failure_probability": failure_prob,
            "verified_evidence_ids": sorted({e for a in agent_outputs for e in a.get("verified_evidence_ids", [])}),
            "rejected_claims": [],
            "rationale": f"vote tally={votes}; mean failure_risk={failure_prob:.2f}",
            "should_abstain": decision == "abstain",
        }


# ---------------------------------------------------------------------------
# Real providers (best-effort; raise if SDKs not present).
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("openai SDK not installed: pip install openai") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI()
        self._model = model
        self._temperature = temperature

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
        forecast_card: dict[str, Any] | None = None,
    ) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature or self._temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        txt = resp.choices[0].message.content or "{}"
        parsed = json.loads(txt)
        return LLMResponse(
            parsed=parsed,
            prompt_tokens=int(getattr(resp.usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(resp.usage, "completion_tokens", 0) or 0),
            raw_text=txt,
        )


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-7", temperature: float = 0.0) -> None:
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("anthropic SDK not installed: pip install anthropic") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = Anthropic()
        self._model = model
        self._temperature = temperature

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
        forecast_card: dict[str, Any] | None = None,
    ) -> LLMResponse:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=temperature or self._temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Anthropic returns a list of content blocks.
        txt = "".join(getattr(b, "text", "") for b in resp.content)
        try:
            parsed = json.loads(txt)
        except Exception:  # noqa: BLE001
            # Try to extract a JSON object.
            start = txt.find("{")
            end = txt.rfind("}")
            parsed = json.loads(txt[start:end + 1]) if start >= 0 else {}
        return LLMResponse(
            parsed=parsed,
            prompt_tokens=int(getattr(resp.usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(resp.usage, "output_tokens", 0) or 0),
            raw_text=txt,
        )


_DEFAULT_MODELS = {
    "mock": "mock-1",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}


def get_provider(cfg: dict[str, Any]) -> LLMProvider:
    name = cfg["agents"]["provider"]
    model = cfg["agents"].get("model", "")
    # If the config still has the mock placeholder but the user picked a real
    # provider on the CLI, swap to that provider's sensible default model.
    if name != "mock" and (not model or model.startswith("mock")):
        model = _DEFAULT_MODELS[name]
    if name == "mock":
        return MockProvider(seed=int(cfg["seed"]))
    if name == "openai":
        return OpenAIProvider(model=model, temperature=float(cfg["agents"]["temperature"]))
    if name == "anthropic":
        return AnthropicProvider(model=model, temperature=float(cfg["agents"]["temperature"]))
    raise ValueError(f"Unknown provider: {name}")
