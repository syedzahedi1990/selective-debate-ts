"""Prediction over the val/test sets for each candidate.

Returns per-instance forecasts in **original (un-scaled)** units so downstream
metrics and forecast cards can compare apples-to-apples across candidates.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sdfts.data.windowing import Window, WindowSet, stack_xy
from sdfts.models.regimes import Candidate, OneStepRecursive


def _device_for(cand: Candidate) -> torch.device:
    return next(cand.module.parameters()).device


def predict_split(cand: Candidate, windows: list[Window]) -> np.ndarray:
    """Return forecasts in original units, shape (N, H)."""
    if not windows:
        return np.empty((0, cand.horizon), dtype=np.float32)
    cand.module.eval()
    device = _device_for(cand)
    X, _ = stack_xy(windows)
    x_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        if isinstance(cand.module, OneStepRecursive):
            y_t = cand.module.rollout(x_t)
        else:
            y_t = cand.module(x_t)
    yhat_std = y_t.cpu().numpy().astype(np.float32)
    # Invert scaling per window
    yhat_orig = np.zeros_like(yhat_std)
    for i, w in enumerate(windows):
        yhat_orig[i] = yhat_std[i] * w.scale.std + w.scale.mean
    return yhat_orig


def target_original(windows: list[Window]) -> np.ndarray:
    """Return the *original-units* targets for a split (for evaluation/labels)."""
    if not windows:
        return np.empty((0, 0), dtype=np.float32)
    Y = np.stack([w.y * w.scale.std + w.scale.mean for w in windows], axis=0).astype(np.float32)
    return Y


def predict_all_splits(cand: Candidate, ws: WindowSet) -> dict[str, np.ndarray]:
    return {
        "val": predict_split(cand, ws.val),
        "test": predict_split(cand, ws.test),
    }
