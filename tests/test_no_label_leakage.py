"""A card must never carry ground-truth or test-error fields."""
from __future__ import annotations

import pytest

from sdfts.agents.verifier import StatisticalVerifier
from sdfts.cards.schemas import validate_card


def test_card_has_no_ground_truth_field(sample_card):
    forbidden = {"ground_truth", "oracle_best_model_id", "default_error",
                 "candidate_errors", "failure_label_top20", "failure_label_threshold"}
    assert not forbidden.intersection(sample_card.keys())


def test_card_is_schema_valid(sample_card):
    errs = validate_card(sample_card)
    assert not errs, errs


def test_verifier_raises_on_leakage(sample_card):
    leaky = dict(sample_card)
    leaky["ground_truth"] = [0.0]
    with pytest.raises(ValueError):
        StatisticalVerifier.assert_no_label_leakage(leaky)


def test_verifier_raises_on_candidate_test_metrics(sample_card):
    leaky = dict(sample_card)
    leaky["candidate_models"] = list(sample_card["candidate_models"])
    leaky["candidate_models"][0] = dict(leaky["candidate_models"][0])
    leaky["candidate_models"][0]["test_metrics"] = {"mae": 0.0}
    with pytest.raises(ValueError):
        StatisticalVerifier.assert_no_label_leakage(leaky)
