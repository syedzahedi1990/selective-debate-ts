"""Dataset loaders.

The default dataset (`synthetic`) is fully deterministic and runs offline so the
smoke test has no network dependencies. Real loaders for ETT, Electricity,
Weather, Traffic, GIFT-Eval, and fev-bench are stubbed; flip them on once data
is downloaded into ``data_root``.

A loader returns a list of 1-D ``numpy.ndarray`` series plus a parallel list of
``series_id`` strings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sdfts.utils.logging import get_logger


log = get_logger(__name__)


def load_dataset(cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[str]]:
    name = cfg["data"]["dataset"]
    if name == "synthetic":
        return _load_synthetic(cfg)
    if name == "ett":
        return _load_ett(cfg)
    raise NotImplementedError(
        f"Loader for dataset '{name}' is not implemented. Available: synthetic, ett."
    )


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

def _load_synthetic(cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[str]]:
    """Deterministic mixture of sinusoid + trend + AR(1) noise + occasional
    level-shifts. Built so that some series have changepoints / regime shifts
    in their test windows (giving the failure-detection task signal).
    """
    n_series = int(cfg["data"]["series_count"])
    length = int(cfg["data"]["series_length"])
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    series: list[np.ndarray] = []
    ids: list[str] = []
    for i in range(n_series):
        t = np.arange(length, dtype=np.float64)
        period = float(rng.uniform(20, 60))
        amp = float(rng.uniform(0.5, 1.5))
        trend = float(rng.uniform(-0.002, 0.005))
        phase = float(rng.uniform(0, 2 * np.pi))
        # AR(1) noise
        rho = 0.6
        eps = rng.normal(0.0, 0.2, size=length)
        noise = np.zeros_like(eps)
        for k in range(1, length):
            noise[k] = rho * noise[k - 1] + eps[k]
        y = amp * np.sin(2 * np.pi * t / period + phase) + trend * t + noise
        # Inject a level-shift roughly 75% into the series for half of series
        if i % 2 == 0:
            shift_idx = int(0.75 * length) + int(rng.integers(-20, 20))
            y[shift_idx:] += rng.normal(1.0, 0.2)
        series.append(y.astype(np.float32))
        ids.append(f"synthetic_{i:03d}")
    return series, ids


# ---------------------------------------------------------------------------
# ETT (real loader, expects CSVs in ``data_root/ETT/``)
# ---------------------------------------------------------------------------

ETT_FILES = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]


def _load_ett(cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[str]]:
    root = Path(cfg["data"]["data_root"]) / "ETT"
    if not root.exists():
        raise FileNotFoundError(
            f"Expected ETT CSVs under {root}. Download from "
            "https://github.com/zhouhaoyi/ETDataset and place there."
        )
    target_col = "OT"
    series: list[np.ndarray] = []
    ids: list[str] = []
    for fname in ETT_FILES:
        p = root / fname
        if not p.exists():
            log.warning("ETT file missing: %s", p)
            continue
        df = pd.read_csv(p)
        if target_col not in df.columns:
            raise ValueError(f"{fname} missing column {target_col}")
        y = df[target_col].to_numpy(dtype=np.float32)
        series.append(y)
        ids.append(p.stem)
    if not series:
        raise FileNotFoundError("No ETT CSVs found.")
    return series, ids
