from __future__ import annotations

import torch
import torch.nn as nn

from sdfts.models.base import BackboneSpec


class GRUBackbone(nn.Module):
    """Final-hidden-state GRU backbone."""

    def __init__(self, spec: BackboneSpec) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=spec.hidden_size,
            num_layers=spec.num_layers,
            dropout=spec.dropout if spec.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.hidden = spec.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L)
        out, _ = self.rnn(x.unsqueeze(-1))
        return out[:, -1, :]
