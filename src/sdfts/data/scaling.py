"""Per-window standardization.

We use instance-level z-scoring (RevIN-style) computed from the input window
*only*, so test-time normalization never peeks at the target. Each call to
``transform`` returns the standardized array along with a callable to invert
predictions back to the original scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScaleState:
    mean: float
    std: float


class InstanceZScore:
    """Per-instance z-score using the input window only.

    Robust to near-constant windows: the std floor scales with the magnitude
    of the mean (so a near-constant signal at level 30 still has std >= 0.03,
    preventing standardized values from exploding to 1e6+).
    """

    abs_floor = 1e-3
    rel_floor = 1e-3      # std floor = max(abs_floor, rel_floor * |mean|)

    def fit_window(self, x: np.ndarray) -> ScaleState:
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        m = float(np.mean(x))
        s = float(np.std(x))
        floor = max(self.abs_floor, self.rel_floor * abs(m))
        if not np.isfinite(s) or s < floor:
            s = floor
        if not np.isfinite(m):
            m = 0.0
        return ScaleState(mean=m, std=s)

    def transform_window(self, x: np.ndarray, state: ScaleState) -> np.ndarray:
        return ((x - state.mean) / state.std).astype(np.float32)

    def inverse(self, y_hat: np.ndarray, state: ScaleState) -> np.ndarray:
        return (y_hat * state.std + state.mean).astype(np.float32)
