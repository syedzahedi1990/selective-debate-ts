from __future__ import annotations

import torch
import torch.nn as nn

from sdfts.models.base import BackboneSpec


class _Chomp1d(nn.Module):
    def __init__(self, chomp: int) -> None:
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : -self.chomp] if self.chomp > 0 else x


class _TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x) + self.proj(x))


class TCNBackbone(nn.Module):
    """Causal temporal convolution stack."""

    def __init__(self, spec: BackboneSpec) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for i in range(max(1, spec.tcn_levels)):
            c_out = spec.hidden_size
            layers.append(_TCNBlock(c_in, c_out, k=spec.tcn_kernel_size, dilation=2 ** i, dropout=spec.dropout))
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.hidden = spec.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L)
        h = self.net(x.unsqueeze(1))    # (B, H, L)
        return h[..., -1]               # (B, H)
