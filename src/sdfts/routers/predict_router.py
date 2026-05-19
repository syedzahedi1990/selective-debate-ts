"""Apply a trained router to a list of cards."""
from __future__ import annotations

from typing import Any

import numpy as np

from sdfts.routers.featurize import card_to_router_features
from sdfts.routers.train_router import TrainedRouter


def predict_failure_prob(router: TrainedRouter, cards: list[dict[str, Any]]) -> np.ndarray:
    if not cards:
        return np.array([])
    X = np.stack([card_to_router_features(c)[0] for c in cards], axis=0).astype(np.float32)
    return np.asarray(router.predict_proba(X), dtype=np.float64)
