"""Temporal train/val/test splits per series. No leakage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SeriesSplit:
    series_id: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def temporal_split(
    series: list[np.ndarray],
    series_ids: list[str],
    cfg: dict[str, Any],
) -> list[SeriesSplit]:
    """Per-series chronological split. test goes at the end."""
    val_f = float(cfg["data"]["val_fraction"])
    test_f = float(cfg["data"]["test_fraction"])
    out: list[SeriesSplit] = []
    for y, sid in zip(series, series_ids):
        n = len(y)
        n_test = int(round(test_f * n))
        n_val = int(round(val_f * n))
        n_train = n - n_val - n_test
        if n_train < int(cfg["data"]["input_length"]) + int(cfg["data"]["forecast_horizon"]):
            raise ValueError(
                f"Series {sid} too short for split: n={n} train={n_train}"
            )
        out.append(
            SeriesSplit(
                series_id=sid,
                train=y[:n_train].astype(np.float32),
                val=y[n_train : n_train + n_val].astype(np.float32),
                test=y[n_train + n_val :].astype(np.float32),
            )
        )
    return out
