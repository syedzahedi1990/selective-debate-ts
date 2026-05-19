"""Forecast card + private label schema tests."""
from __future__ import annotations

from sdfts.cards.schemas import validate_card, validate_label


def test_valid_card(sample_card):
    assert validate_card(sample_card) == []


def test_invalid_card_missing_field(sample_card):
    bad = dict(sample_card)
    bad.pop("candidate_models")
    errs = validate_card(bad)
    assert errs


def test_valid_label(sample_label):
    assert validate_label(sample_label) == []


def test_label_requires_oracle(sample_label):
    bad = dict(sample_label)
    bad.pop("oracle_best_model_id")
    errs = validate_label(bad)
    assert errs
