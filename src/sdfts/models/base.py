"""Shared architecture primitives.

Each backbone consumes a univariate window of shape (B, L) and emits a hidden
representation of shape (B, hidden). Heads are training-regime specific and
live in :mod:`sdfts.models.regimes`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class BackboneSpec:
    arch: str
    input_length: int
    hidden_size: int
    num_layers: int
    dropout: float
    transformer_heads: int
    tcn_kernel_size: int
    tcn_levels: int


def make_backbone(spec: BackboneSpec) -> nn.Module:
    if spec.arch == "lstm":
        from sdfts.models.lstm import LSTMBackbone
        return LSTMBackbone(spec)
    if spec.arch == "gru":
        from sdfts.models.gru import GRUBackbone
        return GRUBackbone(spec)
    if spec.arch == "transformer":
        from sdfts.models.transformer import TransformerBackbone
        return TransformerBackbone(spec)
    if spec.arch == "tcn":
        from sdfts.models.tcn import TCNBackbone
        return TCNBackbone(spec)
    raise ValueError(f"Unknown arch: {spec.arch}")


def spec_from_cfg(cfg: dict[str, Any], arch: str) -> BackboneSpec:
    mp = cfg["model_panel"]
    return BackboneSpec(
        arch=arch,
        input_length=int(cfg["data"]["input_length"]),
        hidden_size=int(mp["hidden_size"]),
        num_layers=int(mp["num_layers"]),
        dropout=float(mp["dropout"]),
        transformer_heads=int(mp["transformer_heads"]),
        tcn_kernel_size=int(mp["tcn_kernel_size"]),
        tcn_levels=int(mp["tcn_levels"]),
    )
