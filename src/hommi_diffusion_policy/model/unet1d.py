from __future__ import annotations

from typing import Union

import torch
from torch import nn

from .embeddings import SinusoidalPosEmb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ):
        super().__init__()
        if out_channels % n_groups:
            raise ValueError("out_channels must be divisible by n_groups")
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
            ]
        )
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale, bias = embed[:, 0], embed[:, 1]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        local_cond_dim: int | None = None,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        if not down_dims:
            raise ValueError("down_dims must not be empty")

        all_dims = [input_dim, *down_dims]
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        cond_dim = dsed + (global_cond_dim or 0)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.local_cond_encoder = None
        if local_cond_dim is not None:
            dim_out = in_out[0][1]
            self.local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(
                        local_cond_dim,
                        dim_out,
                        cond_dim,
                        kernel_size,
                        n_groups,
                        cond_predict_scale,
                    ),
                    ConditionalResidualBlock1D(
                        local_cond_dim,
                        dim_out,
                        cond_dim,
                        kernel_size,
                        n_groups,
                        cond_predict_scale,
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim,
                    kernel_size,
                    n_groups,
                    cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim,
                    kernel_size,
                    n_groups,
                    cond_predict_scale,
                ),
            ]
        )

        self.down_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        nn.Identity() if is_last else Downsample1d(dim_out),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()
        reversed_pairs = list(reversed(in_out[1:]))
        for index, (dim_in, dim_out) in enumerate(reversed_pairs):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        Upsample1d(dim_in),
                    ]
                )
            )

        # UMI's final Conv1dBlock uses Conv1dBlock's default n_groups=8,
        # independently of the configurable n_groups used by residual blocks.
        # Keep that behavior for checkpoint/network fidelity.
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        # There is one stride-2 downsample for every level except the last.
        # Horizons that are not divisible by this factor can silently change
        # length (or fail at a skip connection), so reject them explicitly.
        self.downsample_factor = 2 ** max(len(down_dims) - 1, 0)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:
        if sample.shape[1] % self.downsample_factor != 0:
            raise ValueError(
                f"sample horizon {sample.shape[1]} must be divisible by "
                f"the U-Net downsample factor {self.downsample_factor}"
            )

        # B,T,D -> B,D,T
        sample = sample.transpose(1, 2)

        if not torch.is_tensor(timestep):
            timesteps = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timesteps = timestep[None].to(sample.device)
        else:
            timesteps = timestep.to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat((global_feature, global_cond), dim=-1)

        local_features: list[torch.Tensor] = []
        if local_cond is not None:
            local_cond = local_cond.transpose(1, 2)
            if self.local_cond_encoder is None:
                raise ValueError(
                    "local_cond was provided, but local_cond_dim was not configured"
                )
            # UMI registers two local-condition blocks, but due to its original
            # up-path condition only the first block can affect the output. Keep
            # the second registered for strict state-dict compatibility without
            # spending compute on an activation that is never consumed.
            local_features = [
                self.local_cond_encoder[0](local_cond, global_feature)
            ]

        x = sample
        skips: list[torch.Tensor] = []
        for index, (resnet1, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet1(x, global_feature)
            if index == 0 and local_features:
                x = x + local_features[0]
            x = resnet2(x, global_feature)
            skips.append(x)
            x = downsample(x)

        for block in self.mid_modules:
            x = block(x, global_feature)

        for resnet1, resnet2, upsample in self.up_modules:
            skip = skips.pop()
            x = torch.cat((x, skip), dim=1)
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)
