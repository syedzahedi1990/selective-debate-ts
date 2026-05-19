"""Optional wrappers for public foundation forecasters.

Disabled by default. Enable by setting
``foundation_models: {enabled: true, models: [chronos_bolt_tiny, ...]}``
in the experiment config. Each wrapper exposes ``predict(window) -> np.ndarray``
returning a forecast in the original scale.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class FoundationWrapperUnavailable(RuntimeError):
    pass


def build_foundation_candidates(cfg: dict[str, Any]) -> list[Any]:
    """Best-effort loader. Returns an empty list if disabled or imports fail."""
    fm = cfg.get("foundation_models") or {}
    if not fm.get("enabled", False):
        return []
    out: list[Any] = []
    for name in fm.get("models", []):
        try:
            wrap = _build_one(name)
            out.append(wrap)
        except FoundationWrapperUnavailable as exc:
            # Silent skip; user opted in. Surface in logs instead of crashing.
            from sdfts.utils.logging import get_logger
            get_logger(__name__).warning("Foundation model '%s' unavailable: %s", name, exc)
    return out


def _build_one(name: str) -> Any:
    if name.startswith("chronos"):
        try:
            from chronos import ChronosPipeline  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise FoundationWrapperUnavailable(f"chronos not installed: {exc}") from exc
        # Sketch only: real wrapper would memoize the pipeline and adapt the
        # interface to (X, H) -> forecast.
        raise FoundationWrapperUnavailable("Chronos wrapper stub — implement when needed.")
    raise FoundationWrapperUnavailable(f"unknown foundation model name: {name}")
