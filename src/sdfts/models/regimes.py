"""Training regimes.

Three regimes wrap a shared backbone API:

* ``one_step_recursive``: predicts y_{t+1} from a window; rolled out to H.
* ``h_step_direct``: shared backbone + H linear heads (one per horizon step).
* ``direct_multi_step``: shared backbone + single H-dim head.

We use the *shared backbone* variant for h_step_direct by default
(spec calls it out for speed). Setting ``shared_backbone_for_hstep: false`` in
the config switches to one backbone per horizon (heavier).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from sdfts.models.base import BackboneSpec, make_backbone, spec_from_cfg


REGIMES = ("one_step_recursive", "h_step_direct", "direct_multi_step")


def candidate_id(arch: str, regime: str) -> str:
    return f"{arch}_{regime}"


@dataclass
class Candidate:
    model_id: str
    architecture: str
    training_regime: str
    module: nn.Module
    horizon: int


class OneStepRecursive(nn.Module):
    def __init__(self, spec: BackboneSpec, horizon: int) -> None:
        super().__init__()
        self.backbone = make_backbone(spec)
        self.head = nn.Linear(self.backbone.hidden, 1)
        self.horizon = horizon
        self.input_length = spec.input_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L)
        # Training: predict next step only. Eval rollout in predict-time loop.
        h = self.backbone(x)
        return self.head(h)            # (B, 1)

    @torch.no_grad()
    def rollout(self, x: torch.Tensor) -> torch.Tensor:
        """Recursively roll out to ``horizon`` steps."""
        buf = x.clone()
        preds = []
        for _ in range(self.horizon):
            yhat = self.head(self.backbone(buf))   # (B, 1)
            preds.append(yhat)
            buf = torch.cat([buf[:, 1:], yhat], dim=1)
        return torch.cat(preds, dim=1)             # (B, H)


class HStepDirectShared(nn.Module):
    """Shared backbone + per-horizon linear heads."""

    def __init__(self, spec: BackboneSpec, horizon: int) -> None:
        super().__init__()
        self.backbone = make_backbone(spec)
        self.heads = nn.ModuleList([nn.Linear(self.backbone.hidden, 1) for _ in range(horizon)])
        self.horizon = horizon
        self.input_length = spec.input_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        outs = [head(h) for head in self.heads]
        return torch.cat(outs, dim=1)              # (B, H)


class HStepDirectSeparate(nn.Module):
    """One backbone per horizon step (heavy; behind config flag)."""

    def __init__(self, spec: BackboneSpec, horizon: int) -> None:
        super().__init__()
        self.backbones = nn.ModuleList([make_backbone(spec) for _ in range(horizon)])
        self.heads = nn.ModuleList([nn.Linear(self.backbones[0].hidden, 1) for _ in range(horizon)])
        self.horizon = horizon
        self.input_length = spec.input_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [head(bb(x)) for bb, head in zip(self.backbones, self.heads)]
        return torch.cat(outs, dim=1)


class DirectMultiStep(nn.Module):
    def __init__(self, spec: BackboneSpec, horizon: int) -> None:
        super().__init__()
        self.backbone = make_backbone(spec)
        self.head = nn.Linear(self.backbone.hidden, horizon)
        self.horizon = horizon
        self.input_length = spec.input_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))         # (B, H)


def build_candidate(cfg: dict[str, Any], arch: str, regime: str) -> Candidate:
    horizon = int(cfg["data"]["forecast_horizon"])
    spec = spec_from_cfg(cfg, arch)
    if regime == "one_step_recursive":
        mod: nn.Module = OneStepRecursive(spec, horizon)
    elif regime == "h_step_direct":
        if bool(cfg["model_panel"].get("shared_backbone_for_hstep", True)):
            mod = HStepDirectShared(spec, horizon)
        else:
            mod = HStepDirectSeparate(spec, horizon)
    elif regime == "direct_multi_step":
        mod = DirectMultiStep(spec, horizon)
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return Candidate(
        model_id=candidate_id(arch, regime),
        architecture=arch,
        training_regime=regime,
        module=mod,
        horizon=horizon,
    )


def enumerate_candidates(cfg: dict[str, Any]) -> list[Candidate]:
    mp = cfg["model_panel"]
    out: list[Candidate] = []
    for arch in mp["architectures"]:
        for regime in mp["regimes"]:
            out.append(build_candidate(cfg, arch, regime))
    return out
