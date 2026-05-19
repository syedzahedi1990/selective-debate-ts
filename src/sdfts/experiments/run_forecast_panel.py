"""Train + predict the full forecast panel.

Stores artefacts under ``outputs/<run_name>/panel/``:
    forecasts_val.npz   { model_id: (Nval, H) }
    forecasts_test.npz  { model_id: (Ntest, H) }
    panel_meta.json     metadata + windowing/dataset info
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from sdfts.config import REPO_ROOT, run_dir
from sdfts.data import load_dataset, temporal_split, make_windows
from sdfts.data.windowing import WindowSet
from sdfts.models.predict import predict_split, target_original
from sdfts.models.regimes import enumerate_candidates
from sdfts.models.train import train_candidate
from sdfts.utils.logging import get_logger
from sdfts.utils.seeds import set_seed


log = get_logger(__name__)


def panel_dir(cfg: dict[str, Any]) -> Path:
    d = run_dir(cfg) / "panel"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _windows_path(cfg: dict[str, Any]) -> Path:
    return panel_dir(cfg) / "windows.pkl"


def build_or_load_windows(cfg: dict[str, Any]) -> WindowSet:
    p = _windows_path(cfg)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    series, ids = load_dataset(cfg)
    splits = temporal_split(series, ids, cfg)
    ws = make_windows(splits, cfg)
    with open(p, "wb") as f:
        pickle.dump(ws, f)
    log.info("Built windows: train=%d val=%d test=%d", len(ws.train), len(ws.val), len(ws.test))
    return ws


def train_panel(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(cfg["seed"]))
    ws = build_or_load_windows(cfg)
    cands = enumerate_candidates(cfg)
    meta: dict[str, Any] = {"models": {}, "input_length": ws.input_length, "horizon": ws.forecast_horizon}
    for c in cands:
        log.info("Training %s", c.model_id)
        info = train_candidate(c, ws, cfg)
        meta["models"][c.model_id] = {
            "architecture": c.architecture,
            "training_regime": c.training_regime,
            "best_val_mse": info["best_val_mse"],
            "history": info["history"],
        }
        # Also stash the candidate so predict can reuse it.
    # Save panel meta + the *trained candidates* (pickled module state).
    # Move modules to CPU before pickling so the artefact loads on machines
    # without CUDA. Re-loaders can move them back to GPU if available.
    for c in cands:
        c.module.cpu()
    state_path = panel_dir(cfg) / "panel_state.pkl"
    with open(state_path, "wb") as f:
        pickle.dump(cands, f)
    (panel_dir(cfg) / "panel_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def predict_panel(cfg: dict[str, Any]) -> dict[str, Any]:
    state_path = panel_dir(cfg) / "panel_state.pkl"
    if not state_path.exists():
        raise FileNotFoundError("Run train-panel before predict-panel.")
    with open(state_path, "rb") as f:
        cands = pickle.load(f)
    ws = build_or_load_windows(cfg)

    # Move models to GPU when available for faster prediction.
    import torch as _torch
    dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    for c in cands:
        c.module.to(dev)

    val_forecasts: dict[str, np.ndarray] = {}
    test_forecasts: dict[str, np.ndarray] = {}
    for c in cands:
        val_forecasts[c.model_id] = predict_split(c, ws.val)
        test_forecasts[c.model_id] = predict_split(c, ws.test)

    np.savez(panel_dir(cfg) / "forecasts_val.npz", **val_forecasts)
    np.savez(panel_dir(cfg) / "forecasts_test.npz", **test_forecasts)

    # Save targets in original units for evaluation convenience.
    np.savez(
        panel_dir(cfg) / "targets.npz",
        val=target_original(ws.val),
        test=target_original(ws.test),
    )
    return {"val_forecasts_keys": list(val_forecasts.keys()), "test_forecasts_keys": list(test_forecasts.keys())}


def load_panel_outputs(cfg: dict[str, Any]) -> dict[str, Any]:
    val = dict(np.load(panel_dir(cfg) / "forecasts_val.npz"))
    test = dict(np.load(panel_dir(cfg) / "forecasts_test.npz"))
    tgt = dict(np.load(panel_dir(cfg) / "targets.npz"))
    meta = json.loads((panel_dir(cfg) / "panel_meta.json").read_text(encoding="utf-8"))
    return {
        "val_forecasts": {k: np.asarray(v) for k, v in val.items()},
        "test_forecasts": {k: np.asarray(v) for k, v in test.items()},
        "val_targets": np.asarray(tgt["val"]),
        "test_targets": np.asarray(tgt["test"]),
        "meta": meta,
    }
