"""End-to-end test of the mock debate harness on a single forecast card."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sdfts.agents.cache import LLMCache
from sdfts.agents.debate import (
    run_debate,
    run_independent_vote,
    run_single_auditor,
    run_tool_only_verifier,
)
from sdfts.agents.providers import MockProvider
from sdfts.agents.schemas import validate_judge_output


@pytest.fixture
def cfg(tmp_path: Path) -> dict[str, Any]:
    return {
        "seed": 0,
        "agents": {
            "provider": "mock",
            "model": "mock-1",
            "temperature": 0.0,
            "debate_rounds": 1,
            "cache_dir": str(tmp_path / "cache"),
            "max_retries": 2,
            "prompt_version": "v1",
        },
    }


def test_single_auditor(cfg, sample_card):
    cache = LLMCache(cfg["agents"]["cache_dir"])
    judge, stats = run_single_auditor(MockProvider(), cache, sample_card, cfg)
    assert not validate_judge_output(judge)
    assert stats.n_calls >= 2  # auditor + judge


def test_vote(cfg, sample_card):
    cache = LLMCache(cfg["agents"]["cache_dir"])
    judge, stats = run_independent_vote(MockProvider(), cache, sample_card, cfg)
    assert not validate_judge_output(judge)
    assert stats.n_calls >= 2


def test_debate_one_round(cfg, sample_card):
    cache = LLMCache(cfg["agents"]["cache_dir"])
    judge, stats = run_debate(MockProvider(), cache, sample_card, cfg, use_tools=True)
    assert not validate_judge_output(judge)
    assert stats.n_calls >= 2


def test_debate_two_rounds(cfg, sample_card):
    cfg["agents"]["debate_rounds"] = 2
    cache = LLMCache(cfg["agents"]["cache_dir"])
    judge, stats = run_debate(MockProvider(), cache, sample_card, cfg, use_tools=True)
    assert not validate_judge_output(judge)


def test_tool_only_verifier(sample_card):
    judge, stats = run_tool_only_verifier(sample_card)
    assert not validate_judge_output(judge)
    assert stats.n_calls == 0
