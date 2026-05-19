"""Router training, feature extraction, and predict."""
from __future__ import annotations

import numpy as np

from sdfts.routers.featurize import card_to_router_features
from sdfts.routers.train_router import train_router
from sdfts.routers.predict_router import predict_failure_prob


def test_featurize_returns_consistent_length(sample_card):
    v, n = card_to_router_features(sample_card)
    assert v.shape[0] == len(n)
    assert v.shape[0] >= 14   # diag + summaries


def _make_cards(sample_card, k: int) -> list[dict]:
    cards = []
    for i in range(k):
        c = {**sample_card, "instance_id": f"test_{i:03d}"}
        c["diagnostics"] = dict(sample_card["diagnostics"])
        # Inject signal: half have high disagreement.
        c["diagnostics"]["overall_forecast_dispersion"] = 1.5 if i % 2 == 0 else 0.05
        c["diagnostics"]["changepoint_score"] = 1.0 if i % 2 == 0 else 0.05
        cards.append(c)
    return cards


def test_train_router_lr_predicts_signal(sample_card):
    cards = _make_cards(sample_card, 20)
    y = np.array([1 if i % 2 == 0 else 0 for i in range(20)])
    r = train_router("logistic_regression", cards, y, seed=0)
    fp = predict_failure_prob(r, cards)
    # Predictions should be higher for label=1 instances on average.
    pos = fp[::2].mean()
    neg = fp[1::2].mean()
    assert pos > neg


def test_disagreement_threshold_router(sample_card):
    cards = _make_cards(sample_card, 10)
    y = np.array([1, 0] * 5)
    r = train_router("disagreement_threshold", cards, y, seed=0)
    fp = predict_failure_prob(r, cards)
    assert len(fp) == 10
    assert ((fp >= 0) & (fp <= 1)).all()


def test_router_degenerate_labels(sample_card):
    cards = _make_cards(sample_card, 6)
    y_all_zero = np.zeros(6, dtype=np.int64)
    r = train_router("logistic_regression", cards, y_all_zero, seed=0)
    fp = predict_failure_prob(r, cards)
    assert np.all(fp <= 0.5)
