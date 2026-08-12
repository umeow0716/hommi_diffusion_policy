from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
import torch.nn.functional as F

from ..common.utils import encode_features, output_shape
from ..model.unet1d import ConditionalUnet1D
from .base import BasePolicy

if TYPE_CHECKING:
    from diffusers import DDIMScheduler, DDPMScheduler
else:
    # Keep diffusers out of the runtime import path. Static analyzers still see
    # the concrete scheduler types through the TYPE_CHECKING branch above.
    DDIMScheduler = DDPMScheduler = Any

NoiseScheduler: TypeAlias = DDIMScheduler | DDPMScheduler


class DiffusionUnetPolicy(BasePolicy):
    def __init__(
        self,
        name: str | None = None,
        *,
        shape_meta: dict[str, Any],
        noise_scheduler: NoiseScheduler,
        obs_encoder: torch.nn.Module,
        num_inference_steps: int | None = None,
        obs_as_global_cond: bool = True,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        input_pertub: float = 0.1,
        input_perturbation: float | None = None,
        inpaint_fixed_action_prefix: bool = False,
        train_diffusion_n_samples: int = 1,
        **scheduler_step_kwargs: Any,
    ) -> None:
        super().__init__(name=name)
        action_shape = tuple(shape_meta["action"]["shape"])
        if len(action_shape) != 1:
            raise ValueError("Only 1D action vectors are supported")
        action_dim = int(action_shape[0])
        action_horizon = int(shape_meta["action"]["horizon"])
        obs_feature_dim = math.prod(output_shape(obs_encoder))

        if not obs_as_global_cond:
            raise NotImplementedError("Only global observation conditioning is supported")

        self.obs_encoder = obs_encoder
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        if action_horizon % self.model.downsample_factor != 0:
            raise ValueError(
                f"action horizon {action_horizon} must be divisible by the U-Net "
                f"downsample factor {self.model.downsample_factor}"
            )

        self.noise_scheduler = noise_scheduler
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_feature_dim = obs_feature_dim
        self.input_pertub = (
            input_pertub if input_perturbation is None else input_perturbation
        )
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.scheduler_step_kwargs = scheduler_step_kwargs
        self.num_inference_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(noise_scheduler.config.num_train_timesteps)
        )

    def get_optimizer(
        self,
        lr: float,
        *,
        obs_encoder_lr: float | None = None,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        **kwargs,
    ) -> torch.optim.Optimizer:
        if obs_encoder_lr is None:
            obs_encoder_lr = lr * (0.1 if getattr(self.obs_encoder, "pretrained", False) else 1.0)
        encoder_params = [p for p in self.obs_encoder.parameters() if p.requires_grad]
        groups = [{"params": self.model.parameters(), "weight_decay": weight_decay}]
        if encoder_params:
            groups.append(
                {
                    "params": encoder_params,
                    "lr": obs_encoder_lr,
                    "weight_decay": weight_decay,
                }
            )
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **kwargs)

    def _sample_actions(
        self,
        shape: tuple[int, int, int],
        *,
        global_cond: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Fast unconditional path used by normal HoMMI inference."""
        trajectory = torch.randn(
            shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for timestep in self.noise_scheduler.timesteps:
            model_output = self.model(
                trajectory,
                timestep,
                global_cond=global_cond,
            )
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
        global_cond: torch.Tensor,
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
            model_output = self.model(
                trajectory,
                timestep,
                global_cond=global_cond,
            )
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
        self,
        obs_dict: dict[str, torch.Tensor],
        fixed_action_prefix: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if "past_action" in obs_dict:
            raise NotImplementedError("past_action is not implemented")
        nobs = self.normalizer.normalize(obs_dict)
        batch_size = next(iter(nobs.values())).shape[0]
        global_cond = encode_features(self.obs_encoder, nobs).reshape(batch_size, -1)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed = fixed_action_prefix.shape[1]
            if n_fixed > self.action_horizon:
                raise ValueError(
                    f"fixed action prefix length {n_fixed} exceeds action horizon "
                    f"{self.action_horizon}"
                )
            if fixed_action_prefix.shape[0] != batch_size:
                raise ValueError(
                    "fixed action prefix batch size must match observation batch size"
                )
            if fixed_action_prefix.shape[-1] != self.action_dim:
                raise ValueError(
                    "fixed action prefix action dimension must match shape_meta"
                )

            cond_data = torch.zeros(
                (batch_size, self.action_horizon, self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            normalized_prefix = self.normalizer["action"].normalize(fixed_action_prefix)
            cond_data[:, :n_fixed] = normalized_prefix
            cond_mask[:, :n_fixed] = True
            sample = self.conditional_sample(
                cond_data,
                cond_mask,
                global_cond=global_cond,
            )
        else:
            sample = self._sample_actions(
                (batch_size, self.action_horizon, self.action_dim),
                global_cond=global_cond,
            )

        expected_shape = (batch_size, self.action_horizon, self.action_dim)
        if sample.shape != expected_shape:
            raise RuntimeError(
                f"diffusion sampler returned shape {tuple(sample.shape)}, "
                f"expected {expected_shape}"
            )

        action_pred = self.normalizer["action"].unnormalize(sample)
        return {"action": action_pred, "action_pred": action_pred}

    def compute_loss(self, batch: dict) -> torch.Tensor:
        if "valid_mask" in batch:
            raise ValueError("valid_mask is not supported by this policy")

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        global_cond = encode_features(self.obs_encoder, nobs).reshape(nactions.shape[0], -1)

        if self.train_diffusion_n_samples != 1:
            global_cond = torch.repeat_interleave(
                global_cond, self.train_diffusion_n_samples, dim=0
            )
            nactions = torch.repeat_interleave(
                nactions, self.train_diffusion_n_samples, dim=0
            )

        noise = torch.randn_like(nactions)
        if self.input_pertub:
            perturbed_noise = noise + self.input_pertub * torch.randn_like(nactions)
        else:
            perturbed_noise = noise
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (nactions.shape[0],),
            device=nactions.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(nactions, perturbed_noise, timesteps)
        pred = self.model(noisy, timesteps, global_cond=global_cond)

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = nactions
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        return F.mse_loss(pred, target)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.compute_loss(batch)
