from __future__ import annotations

import math

import torch
import torch.nn as nn

from sdfts.models.base import BackboneSpec


class _SinusoidalPositional(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBackbone(nn.Module):
    """Encoder-only Transformer; uses the last token's representation."""

    def __init__(self, spec: BackboneSpec) -> None:
        super().__init__()
        d_model = spec.hidden_size
        self.in_proj = nn.Linear(1, d_model)
        self.pos = _SinusoidalPositional(d_model, max_len=max(64, spec.input_length + 8))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, spec.transformer_heads),
            dim_feedforward=4 * d_model,
            dropout=spec.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=spec.num_layers)
        self.hidden = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L)
        h = self.in_proj(x.unsqueeze(-1))
        h = self.pos(h)
        h = self.enc(h)
        return h[:, -1, :]
