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
    """Per-instance z-score using the input window only."""

    eps = 1e-6

    def fit_window(self, x: np.ndarray) -> ScaleState:
        m = float(np.mean(x))
        s = float(np.std(x))
        if s < self.eps:
            s = self.eps
        return ScaleState(mean=m, std=s)

    def transform_window(self, x: np.ndarray, state: ScaleState) -> np.ndarray:
        return ((x - state.mean) / state.std).astype(np.float32)

    def inverse(self, y_hat: np.ndarray, state: ScaleState) -> np.ndarray:
        return (y_hat * state.std + state.mean).astype(np.float32)
