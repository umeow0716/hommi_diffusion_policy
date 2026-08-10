from __future__ import annotations

from typing import Union

import torch
from torch import nn

from .embeddings import SinusoidalPosEmb


class TransformerForActionDiffusion(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        action_horizon: int,
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,
        max_cond_tokens: int = 800,
        p_drop_attn: float = 0.1,
    ):
        super().__init__()
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.randn(1, action_horizon, n_emb) * 0.02)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_pos_emb = nn.Parameter(torch.randn(1, max_cond_tokens, n_emb) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
        self.action_horizon = action_horizon
        # Match UMI ModuleAttrMixin state_dict layout for strict checkpoint loading.
        self._dummy_variable = nn.Parameter()

        self.apply(self._init_weights)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Keep this aligned with UMI's TransformerForActionDiffusion.
        # TransformerDecoder clones the prototype layer, so the explicit
        # MultiheadAttention branch is important: it independently reinitializes
        # each layer's packed QKV projection instead of leaving clones identical.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in (
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ):
                weight = getattr(module, name, None)
                if weight is not None:
                    nn.init.normal_(weight, mean=0.0, std=0.02)
            for name in ("in_proj_bias", "bias_k", "bias_v"):
                bias = getattr(module, name, None)
                if bias is not None:
                    nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = []
        no_decay = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if (
                parameter.ndim < 2
                or name.endswith(".bias")
                or name in {"pos_emb", "cond_pos_emb"}
            ):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:
        if cond is None:
            raise ValueError("Transformer diffusion model requires conditioning tokens")

        if not torch.is_tensor(timestep):
            timesteps = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timesteps = timestep[None].to(sample.device)
        else:
            timesteps = timestep.to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        time_emb = self.time_emb(timesteps).to(dtype=sample.dtype).unsqueeze(1)
        cond_emb = torch.cat((cond, time_emb), dim=1)
        if cond_emb.shape[1] > self.cond_pos_emb.shape[1]:
            raise ValueError("Condition token count exceeds max_cond_tokens")
        cond_emb = cond_emb + self.cond_pos_emb[:, : cond_emb.shape[1]].to(cond_emb.dtype)

        input_emb = self.input_emb(sample)
        if input_emb.shape[1] > self.pos_emb.shape[1]:
            raise ValueError("Action sequence is longer than action_horizon")
        input_emb = input_emb + self.pos_emb[:, : input_emb.shape[1]].to(input_emb.dtype)

        if memory_key_padding_mask is not None:
            prefix = torch.zeros(
                (memory_key_padding_mask.shape[0], 1),
                device=memory_key_padding_mask.device,
                dtype=torch.bool,
            )
            memory_key_padding_mask = torch.cat(
                (memory_key_padding_mask, prefix), dim=1
            )

        x = self.decoder(
            tgt=input_emb,
            memory=cond_emb,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.head(self.ln_f(x))
