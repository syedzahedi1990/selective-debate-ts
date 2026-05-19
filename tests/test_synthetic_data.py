"""Smoke-test the synthetic loader + windowing + splits."""
from __future__ import annotations

from typing import Any

import numpy as np

from sdfts.data import load_dataset, make_windows, temporal_split


def test_synthetic_loader_deterministic(tiny_cfg):
    s1, ids1 = load_dataset(tiny_cfg)
    s2, ids2 = load_dataset(tiny_cfg)
    assert ids1 == ids2
    for a, b in zip(s1, s2):
        assert np.allclose(a, b)


def test_window_shapes(tiny_cfg):
    series, ids = load_dataset(tiny_cfg)
    splits = temporal_split(series, ids, tiny_cfg)
    ws = make_windows(splits, tiny_cfg)
    L = tiny_cfg["data"]["input_length"]
    H = tiny_cfg["data"]["forecast_horizon"]
    assert all(w.x.shape == (L,) for w in ws.train)
    assert all(w.y.shape == (H,) for w in ws.train)
    assert len(ws.train) > 0 and len(ws.val) > 0 and len(ws.test) > 0
