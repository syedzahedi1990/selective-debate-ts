"""Recent residual summary.

Given the most-recent block of validation residuals for a candidate, summarize
their mean, std, and lag-1 autocorrelation. We expose this in the forecast card
so agents can reason about whether a model has been *systematically* drifting.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sdfts.diagnostics.features import residual_autocorrelation


def recent_residual_summary(resid: np.ndarray, window: int = 24) -> dict[str, float]:
    r = np.asarray(resid, dtype=np.float64).flatten()
    if r.size == 0:
        return {"mean": 0.0, "std": 0.0, "autocorr_lag1": 0.0, "n": 0}
    r = r[-window:]
    return {
        "mean": float(r.mean()),
        "std": float(r.std()),
        "autocorr_lag1": residual_autocorrelation(r, lag=1),
        "n": int(r.size),
    }
