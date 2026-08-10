from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("SinusoidalPosEmb dim must be an even integer >= 4")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=torch.float32) * -scale
        )
        values = x.to(dtype=frequencies.dtype)[:, None] * frequencies[None, :]
        return torch.cat((values.sin(), values.cos()), dim=-1)
