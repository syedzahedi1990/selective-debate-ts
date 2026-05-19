"""Sliding-window construction for train/val/test instances.

Windowing is the only place where the model sees a (lookback, horizon) pair.
For each split slice we build dense overlapping windows; the test slice uses a
*stride* equal to forecast horizon to avoid heavy overlap between evaluation
instances (this is the convention used by ETT-style benchmarks too).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sdfts.data.splits import SeriesSplit
from sdfts.data.scaling import InstanceZScore, ScaleState


@dataclass
class Window:
    series_id: str
    split: str           # train | val | test
    instance_id: str
    x: np.ndarray        # standardized input,  shape (L,)
    y: np.ndarray        # standardized target, shape (H,)
    scale: ScaleState    # to invert predictions for evaluation


@dataclass
class WindowSet:
    input_length: int
    forecast_horizon: int
    train: list[Window] = field(default_factory=list)
    val: list[Window] = field(default_factory=list)
    test: list[Window] = field(default_factory=list)

    def all(self) -> list[Window]:
        return self.train + self.val + self.test


def make_windows(splits: list[SeriesSplit], cfg: dict[str, Any]) -> WindowSet:
    L = int(cfg["data"]["input_length"])
    H = int(cfg["data"]["forecast_horizon"])
    scaler = InstanceZScore()
    ws = WindowSet(input_length=L, forecast_horizon=H)

    for split in splits:
        for split_name, raw, stride in (
            ("train", split.train, 1),
            ("val", split.val, max(1, H // 2)),
            ("test", split.test, H),
        ):
            if len(raw) < L + H:
                continue
            for i, start in enumerate(range(0, len(raw) - L - H + 1, stride)):
                x = raw[start : start + L]
                y = raw[start + L : start + L + H]
                state = scaler.fit_window(x)
                xs = scaler.transform_window(x, state)
                ys = scaler.transform_window(y, state)
                inst_id = f"{split.series_id}__{split_name}__{i:05d}"
                w = Window(
                    series_id=split.series_id,
                    split=split_name,
                    instance_id=inst_id,
                    x=xs,
                    y=ys,
                    scale=state,
                )
                getattr(ws, split_name).append(w)
    return ws


def stack_xy(windows: list[Window]) -> tuple[np.ndarray, np.ndarray]:
    if not windows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    X = np.stack([w.x for w in windows], axis=0).astype(np.float32)
    Y = np.stack([w.y for w in windows], axis=0).astype(np.float32)
    return X, Y
