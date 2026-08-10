
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import torch
import torch.nn.functional as F

from ..common.normalizer import LinearNormalizer
from ..common.utils import encode_features, output_shape
from ..model.transformer import TransformerForActionDiffusion
from .base import BasePolicy

if TYPE_CHECKING:
    from diffusers import DDIMScheduler, DDPMScheduler
else:
    # Keep diffusers out of the runtime import path. Static analyzers still see
    # the concrete scheduler types through the TYPE_CHECKING branch above.
    DDIMScheduler = DDPMScheduler = Any

NoiseScheduler: TypeAlias = DDIMScheduler | DDPMScheduler


class DiffusionTransformerPolicy(BasePolicy):
    def __init__(
        self,
        name: str | None = None,
        *,
        shape_meta: dict[str, Any],
        noise_scheduler: NoiseScheduler,
        obs_encoder: torch.nn.Module,
        num_inference_steps: int | None = None,
        input_pertub: float = 0.1,
        input_perturbation: float | None = None,
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,
        p_drop_attn: float = 0.1,
        max_cond_tokens: int | None = None,
        **scheduler_step_kwargs: Any,
    ) -> None:
        super().__init__(name=name)
        action_shape = tuple(shape_meta["action"]["shape"])
        if len(action_shape) != 1:
            raise ValueError("Only 1D action vectors are supported")
        action_dim = int(action_shape[0])
        action_horizon = int(shape_meta["action"]["horizon"])

        obs_shape = output_shape(obs_encoder)
        if len(obs_shape) < 2:
            raise ValueError(
                "Transformer obs_encoder.output_shape() must end in "
                "(num_tokens, embedding_dim)"
            )
        # UMI's TransformerObsEncoder reports torch.Size([1, N, D]); other
        # encoders commonly report (N, D). The policy only depends on the
        # trailing token and embedding dimensions, exactly as upstream does.
        obs_tokens, obs_emb = obs_shape[-2:]
        if obs_emb != n_emb:
            raise ValueError(
                f"obs encoder embedding dim ({obs_emb}) must equal n_emb ({n_emb})"
            )

        self.obs_encoder = obs_encoder
        self.model = TransformerForActionDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=max_cond_tokens or (obs_tokens + 1),
            p_drop_attn=p_drop_attn,
        )
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.input_pertub = (
            input_pertub if input_perturbation is None else input_perturbation
        )
        self.scheduler_step_kwargs = scheduler_step_kwargs
        self.num_inference_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(noise_scheduler.config.num_train_timesteps)
        )

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self,
        lr: float,
        *,
        weight_decay: float = 1e-3,
        obs_encoder_lr: float | None = None,
        obs_encoder_weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        **kwargs,
    ) -> torch.optim.Optimizer:
        groups = self.model.get_optim_groups(weight_decay=weight_decay)

        # Match UMI: only pretrained backbone parameters (key_model_map.*) get
        # obs_encoder_lr. Projection/aggregation layers stay at the main lr.
        backbone_params = []
        other_obs_params = []
        for key, parameter in self.obs_encoder.named_parameters():
            if not parameter.requires_grad:
                continue
            if key.startswith("key_model_map"):
                backbone_params.append(parameter)
            else:
                other_obs_params.append(parameter)

        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "weight_decay": obs_encoder_weight_decay,
                    "lr": obs_encoder_lr if obs_encoder_lr is not None else lr,
                }
            )
        if other_obs_params:
            groups.append(
                {
                    "params": other_obs_params,
                    "weight_decay": obs_encoder_weight_decay,
                }
            )
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **kwargs)

    def _sample_actions(
        self,
        shape: tuple[int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        cond: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Fast path for the policy's normal (non-inpainting) inference."""
        trajectory = torch.randn(
            shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            model_output = self.model(trajectory, timestep, cond)
            trajectory = self.noise_scheduler.step(
                model_output,
                timestep,
                trajectory,
                generator=generator,
                **self.scheduler_step_kwargs,
            ).prev_sample
        return trajectory

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        *,
        cond: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        trajectory = torch.randn(
            condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(trajectory, timestep, cond)
            trajectory = self.noise_scheduler.step(
                model_output,
                timestep,
                trajectory,
                generator=generator,
                **self.scheduler_step_kwargs,
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if "past_action" in obs_dict:
            raise NotImplementedError("past_action is not implemented")

        nobs = self.normalizer.normalize(obs_dict)
        batch_size = next(iter(nobs.values())).shape[0]
        obs_tokens = encode_features(self.obs_encoder, nobs)

        sample = self._sample_actions(
            (batch_size, self.action_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
            cond=obs_tokens,
        )
        action_pred = self.normalizer["action"].unnormalize(sample)
        return {"action": action_pred, "action_pred": action_pred}

    def compute_loss(self, batch: dict) -> torch.Tensor:
        if "valid_mask" in batch:
            raise ValueError("valid_mask is not supported by this policy")
        nobs = self.normalizer.normalize(batch["obs"])
        actions = self.normalizer["action"].normalize(batch["action"])
        obs_tokens = encode_features(self.obs_encoder, nobs)

        noise = torch.randn_like(actions)
        if self.input_pertub:
            perturbed_noise = noise + self.input_pertub * torch.randn_like(actions)
        else:
            perturbed_noise = noise
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(actions, perturbed_noise, timesteps)
        pred = self.model(noisy, timesteps, cond=obs_tokens)

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = actions
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")
        return F.mse_loss(pred, target)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.compute_loss(batch)
