from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class BiasFreeConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("bias", False)
        super().__init__(*args, **kwargs)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return normalized * self.weight + self.bias


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        hidden = channels * expansion
        self.depth = BiasFreeConv(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
        )
        self.point = BiasFreeConv(channels, hidden, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point(self.depth(x))


class GlobalGating(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = BiasFreeConv(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = x.mean(dim=(2, 3), keepdim=True)
        gate = torch.sigmoid(self.proj(gate))
        return gate


class GatedResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, kernel_size: int = 5):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.local = DepthwiseSeparableConv(
            channels, expansion=expansion, kernel_size=kernel_size
        )
        self.activation = nn.GELU()
        self.proj = BiasFreeConv(channels * expansion, channels, kernel_size=1)
        self.gate = GlobalGating(channels)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        if cond is not None:
            gamma, beta = cond
            out = out * (1.0 + gamma) + beta
        out = self.local(out)
        out = self.activation(out)
        out = self.proj(out)
        out = out * self.gate(residual)
        return residual + out


class ResidualStage(nn.Module):
    def __init__(self, channels: int, num_blocks: int, expansion: int = 2):
        super().__init__()
        self.channels = channels
        self.blocks = nn.ModuleList(
            GatedResidualBlock(channels, expansion=expansion) for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        block_cond = None
        if cond is not None:
            if cond.dim() == 2:
                cond = cond.unsqueeze(-1).unsqueeze(-1)
            gamma, beta = torch.chunk(cond, 2, dim=1)
            gamma = gamma[:, : self.channels]
            beta = beta[:, : self.channels]
            block_cond = (gamma, beta)
        for block in self.blocks:
            x = block(x, block_cond)
        return x
