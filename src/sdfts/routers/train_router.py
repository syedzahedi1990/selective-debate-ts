"""Train a non-LLM failure-prediction router.

The router consumes :func:`sdfts.routers.featurize.card_to_router_features` and
predicts the *failure* label (top-quantile or absolute threshold). We expose
LR, RF, GB, and a degenerate disagreement-threshold rule.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sdfts.routers.featurize import card_to_router_features


@dataclass
class TrainedRouter:
    name: str
    feature_names: list[str]
    scaler: Any | None
    model: Any
    threshold: float | None = None
    feature_index_for_threshold: int | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            Xs = self.scaler.transform(X)
        else:
            Xs = X
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(Xs)
            return p[:, 1] if p.shape[1] == 2 else p.max(axis=1)
        # threshold rule fallback
        idx = self.feature_index_for_threshold or 0
        return (X[:, idx] > self.threshold).astype(np.float32)


def _stack_features(cards: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    feats: list[np.ndarray] = []
    names_ref: list[str] | None = None
    for c in cards:
        v, n = card_to_router_features(c)
        feats.append(v)
        names_ref = n if names_ref is None else names_ref
    return np.stack(feats, axis=0).astype(np.float32), (names_ref or [])


def train_router(
    name: str,
    cards: list[dict[str, Any]],
    failure_labels: np.ndarray,
    seed: int = 0,
) -> TrainedRouter:
    X, names = _stack_features(cards)
    y = np.asarray(failure_labels, dtype=np.int64)
    if name == "disagreement_threshold":
        idx = names.index("diag.overall_forecast_dispersion") if "diag.overall_forecast_dispersion" in names else 0
        # If everyone has the same label, fall back to mean.
        if y.sum() == 0 or y.sum() == y.size:
            thr = float(np.median(X[:, idx]))
        else:
            thr = float(np.median(X[:, idx][y == 1])) if (y == 1).any() else float(np.median(X[:, idx]))
        return TrainedRouter(name=name, feature_names=names, scaler=None, model=None,
                             threshold=thr, feature_index_for_threshold=idx)
    if name == "uncertainty_threshold":
        idx = names.index("cand.val_mae.std") if "cand.val_mae.std" in names else 0
        thr = float(np.median(X[:, idx][y == 1])) if (y == 1).any() else float(np.median(X[:, idx]))
        return TrainedRouter(name=name, feature_names=names, scaler=None, model=None,
                             threshold=thr, feature_index_for_threshold=idx)
    if name == "changepoint_or_ood_threshold":
        idx_cp = names.index("diag.changepoint_score") if "diag.changepoint_score" in names else 0
        idx_oo = names.index("diag.ood_score") if "diag.ood_score" in names else 0
        score = np.maximum(X[:, idx_cp], X[:, idx_oo])
        thr = float(np.median(score[y == 1])) if (y == 1).any() else float(np.median(score))
        # Store as a tiny synthetic model: use first idx (cp) and rely on inference combining; we just use cp as the "trigger" feature for simplicity.
        return TrainedRouter(name=name, feature_names=names, scaler=None, model=None,
                             threshold=thr, feature_index_for_threshold=idx_cp)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if name == "logistic_regression":
        mdl = LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "random_forest":
        mdl = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    elif name == "gradient_boosted":
        mdl = GradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"Unknown router model: {name}")

    # If both classes aren't present, calibrate to majority class.
    if y.sum() == 0:
        # All zeros: a constant predictor at the empirical rate (0).
        mdl = _ConstantClf(0.0)
    elif y.sum() == y.size:
        mdl = _ConstantClf(1.0)
    else:
        mdl.fit(Xs, y)
    return TrainedRouter(name=name, feature_names=names, scaler=scaler, model=mdl)


class _ConstantClf:
    def __init__(self, p: float) -> None:
        self._p = float(p)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        out = np.zeros((n, 2))
        out[:, 1] = self._p
        out[:, 0] = 1.0 - self._p
        return out


def save_router(router: TrainedRouter, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(router, f)


def load_router(path: str | Path) -> TrainedRouter:
    with open(path, "rb") as f:
        return pickle.load(f)
