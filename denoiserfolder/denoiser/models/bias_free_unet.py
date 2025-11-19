from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BiasFreeConv, ResidualStage


def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


class ResidualPriorExtractor(nn.Module):
    def __init__(self, channels: int, kernel_sizes: Sequence[int]):
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty.")
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        for idx, k in enumerate(self.kernel_sizes):
            sigma = max(0.8, float(k) / 3.0)
            kernel = _gaussian_kernel(k, sigma).view(1, 1, k, k)
            self.register_buffer(f"kernel_{idx}", kernel)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        priors: List[torch.Tensor] = []
        descriptors: List[torch.Tensor] = []
        b, c, _, _ = x.shape
        for idx, k in enumerate(self.kernel_sizes):
            kernel = getattr(self, f"kernel_{idx}").to(dtype=x.dtype, device=x.device)
            kernel = kernel.expand(c, 1, k, k)
            smooth = F.conv2d(x, kernel, padding=k // 2, groups=c)
            residual = x - smooth
            priors.append(residual)
            descriptors.append(residual.abs().mean(dim=(1, 2, 3), keepdim=True))
        prior_map = torch.cat(priors, dim=1)
        descriptor = torch.cat(descriptors, dim=1).view(b, -1)
        return prior_map, descriptor


class NoiseConditioner(nn.Module):
    def __init__(
        self,
        stage_channels: Sequence[int],
        hidden_dim: int = 96,
        descriptor_dim: int = 0,
    ):
        super().__init__()
        self.stage_channels = list(stage_channels)
        in_dim = 1 + max(0, descriptor_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleList(
            nn.Linear(hidden_dim, 2 * channels) for channels in self.stage_channels
        )

    def forward(
        self, sigma: torch.Tensor, descriptor: torch.Tensor | None
    ) -> List[torch.Tensor]:
        sigma = sigma.view(sigma.shape[0], 1)
        sigma = sigma.clamp(min=1e-5)
        embedding = torch.log(sigma * 255.0)
        if descriptor is not None:
            embedding = torch.cat([embedding, descriptor], dim=1)
        base = self.encoder(embedding)
        conds = [head(base).unsqueeze(-1).unsqueeze(-1) for head in self.heads]
        return conds


class BiasFreeResidualDenoiser(nn.Module):
    def __init__(
        self,
        image_channels: int,
        base_channels: int = 48,
        depth: int = 3,
        blocks_per_stage: int = 2,
        residual_kernel_sizes: Sequence[int] = (3, 5, 7),
    ):
        super().__init__()
        self.image_channels = image_channels
        self.prior_extractor = ResidualPriorExtractor(
            image_channels, kernel_sizes=residual_kernel_sizes
        )
        self.prior_scales = len(tuple(residual_kernel_sizes))
        self.prior_channels = self.prior_scales * image_channels
        self.input_channels = image_channels + 1 + self.prior_channels

        encoder_channels = [base_channels * (2**i) for i in range(depth)]
        self.entry = BiasFreeConv(self.input_channels, encoder_channels[0], 3, padding=1)

        self.encoder_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        stage_channels: List[int] = [encoder_channels[0]]

        for idx, channels in enumerate(encoder_channels):
            self.encoder_stages.append(
                ResidualStage(channels, num_blocks=blocks_per_stage)
            )
            stage_channels.append(channels)
            if idx < len(encoder_channels) - 1:
                next_ch = encoder_channels[idx + 1]
                self.downs.append(
                    BiasFreeConv(channels, next_ch, kernel_size=3, stride=2, padding=1)
                )

        bottleneck_channels = encoder_channels[-1]
        self.bottleneck = ResidualStage(
            bottleneck_channels, num_blocks=blocks_per_stage + 1, expansion=3
        )
        stage_channels.append(bottleneck_channels)

        self.up_projs = nn.ModuleList()
        self.merge_convs = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()

        decoder_channels = list(reversed(encoder_channels[:-1]))
        current_channels = bottleneck_channels
        for skip_ch in decoder_channels:
            self.up_projs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    BiasFreeConv(current_channels, skip_ch, kernel_size=3, padding=1),
                )
            )
            self.merge_convs.append(BiasFreeConv(skip_ch * 2, skip_ch, kernel_size=1))
            self.decoder_stages.append(
                ResidualStage(skip_ch, num_blocks=blocks_per_stage)
            )
            stage_channels.append(skip_ch)
            current_channels = skip_ch

        self.exit = BiasFreeConv(current_channels, image_channels, kernel_size=3, padding=1)

        self.conditioner = NoiseConditioner(
            stage_channels=stage_channels, descriptor_dim=self.prior_scales
        )

    def _apply_cond(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is None:
            return x
        if cond.dim() == 2:
            cond = cond.unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(cond, 2, dim=1)
        gamma = gamma[:, : x.shape[1]]
        beta = beta[:, : x.shape[1]]
        return x * (1.0 + gamma) + beta

    def forward(
        self, noisy: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior_map, descriptor = self.prior_extractor(noisy)
        sigma = sigma.view(sigma.shape[0], 1)
        conds = iter(self.conditioner(sigma, descriptor))

        sigma_map = sigma.view(sigma.shape[0], 1, 1, 1).expand(
            -1, 1, noisy.shape[2], noisy.shape[3]
        )
        x = torch.cat([noisy, sigma_map, prior_map], dim=1)
        x = self.entry(x)
        x = self._apply_cond(x, next(conds, None))

        skips: List[torch.Tensor] = []
        for idx, stage in enumerate(self.encoder_stages):
            x = stage(x, next(conds, None))
            if idx < len(self.downs):
                skips.append(x)
                x = self.downs[idx](x)

        x = self.bottleneck(x, next(conds, None))

        for up, merge, stage in zip(
            self.up_projs, self.merge_convs, self.decoder_stages
        ):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = merge(x)
            x = stage(x, next(conds, None))

        residual = self.exit(x)
        denoised = torch.clamp(noisy - residual, 0.0, 1.0)
        return denoised, residual
