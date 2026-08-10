from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .embeddings import SinusoidalPosEmb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(dim=0)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=self.is_causal,
        )
        x = x.transpose(1, 2).reshape(b, n, d)
        return self.proj_drop(self.proj(x))


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        use_rms_norm: bool,
    ):
        super().__init__()
        norm = (lambda d: RMSNorm(d, eps=1e-6)) if use_rms_norm else (
            lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        )
        self.pre_norm = norm(dim)
        self.post_norm = norm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim)
        self.ada_ln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))

    @property
    def modulation(self) -> nn.Sequential:
        # v0.1.0 compatibility without registering a duplicate module name.
        return self.ada_ln_modulation

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        (
            attn_shift,
            attn_scale,
            attn_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.ada_ln_modulation(condition).chunk(6, dim=1)

        attn_input = self._modulate(self.pre_norm(x), attn_shift, attn_scale)
        x = x + attn_gate.unsqueeze(1) * self.attn(attn_input)

        mlp_input = self._modulate(self.post_norm(x), mlp_shift, mlp_scale)
        x = x + mlp_gate.unsqueeze(1) * self.mlp(mlp_input)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, cond_dim: int, use_rms_norm: bool):
        super().__init__()
        self.norm = (
            RMSNorm(dim, eps=1e-6)
            if use_rms_norm
            else nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        )
        self.ada_ln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        self.final_linear = nn.Linear(dim, dim)

    @property
    def modulation(self) -> nn.Sequential:
        return self.ada_ln_modulation

    @property
    def linear(self) -> nn.Linear:
        return self.final_linear

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln_modulation(cond).chunk(2, dim=1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.final_linear(x)


class ActionDiT(nn.Module):
    def __init__(
        self,
        obs_embed_dim: int,
        action_dim: int,
        action_len: int = 16,
        embed_dim: int = 768,
        timestep_embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rms_norm: bool = False,
    ):
        super().__init__()
        hidden_dim = int(max(action_dim, embed_dim) * mlp_ratio)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.timestep_embedding = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim),
        )
        self.pos_embed = nn.Parameter(torch.empty(1, action_len, embed_dim).normal_(std=0.02))
        cond_dim = obs_embed_dim + timestep_embed_dim
        self.dit_blocks = nn.ModuleList(
            [
                DiTBlock(
                    embed_dim,
                    cond_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    use_rms_norm,
                )
                for _ in range(depth)
            ]
        )
        self.head = FinalLayer(embed_dim, cond_dim, use_rms_norm)
        self._initialize_weights()

    @property
    def blocks(self) -> nn.ModuleList:
        # v0.1.0 source compatibility; state_dict uses HoMMI's dit_blocks.* keys.
        return self.dit_blocks

    def _initialize_weights(self) -> None:
        def init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_linear)
        for module in self.timestep_embedding:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for block in self.dit_blocks:
            linear = block.ada_ln_modulation[-1]
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)

        head_mod = self.head.ada_ln_modulation[-1]
        nn.init.zeros_(head_mod.weight)
        nn.init.zeros_(head_mod.bias)
        nn.init.zeros_(self.head.final_linear.weight)
        nn.init.zeros_(self.head.final_linear.bias)

    def forward(
        self,
        obs_embed: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        x = self.action_encoder(actions)
        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"Expected action length {self.pos_embed.shape[1]}, got {x.shape[1]}"
            )
        x = x + self.pos_embed.to(dtype=x.dtype)

        if timesteps.ndim == 0:
            timesteps = timesteps.expand(actions.shape[0]).to(
                dtype=torch.long, device=actions.device
            )
        else:
            timesteps = timesteps.to(actions.device)
        timestep_embed = self.timestep_embedding(timesteps).to(dtype=obs_embed.dtype)
        cond = torch.cat((obs_embed, timestep_embed), dim=-1)

        for block in self.dit_blocks:
            x = block(x, cond)
        return self.action_decoder(self.head(x, cond))
