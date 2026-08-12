from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import torch
import torch.nn.functional as F

from ..common.utils import dict_apply, encode_features, output_shape
from ..model.dit import ActionDiT
from .base import BasePolicy

if TYPE_CHECKING:
    from diffusers import DDIMScheduler, DDPMScheduler
else:
    # Keep diffusers out of the runtime import path. Static analyzers still see
    # the concrete scheduler types through the TYPE_CHECKING branch above.
    DDIMScheduler = DDPMScheduler = Any

NoiseScheduler: TypeAlias = DDIMScheduler | DDPMScheduler


class DiffusionDiTImagePolicy(BasePolicy):
    def __init__(
        self,
        name: str | None = None,
        *,
        shape_meta: dict[str, Any],
        noise_scheduler: NoiseScheduler,
        obs_encoder: torch.nn.Module,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        use_flow_matching: bool = False,
        fm_tsampler: str = "uniform",
        num_inference_steps: int | None = None,
        obs_as_global_cond: bool = True,
        train_diffusion_n_samples: int = 1,
        attention_embed_dim: int = 768,
        diffusion_timestep_embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rms_norm: bool = False,
        input_perturbation: float = 0.0,
    ) -> None:
        super().__init__(name=name)
        action_shape = tuple(shape_meta["action"]["shape"])
        if len(action_shape) != 1:
            raise ValueError("Only 1D action vectors are supported")
        action_dim = int(action_shape[0])
        obs_feature_dim = int(output_shape(obs_encoder)[0])

        if not obs_as_global_cond:
            raise NotImplementedError("DiT currently supports global conditioning only")
        if fm_tsampler not in {"uniform", "beta"}:
            raise ValueError("fm_tsampler must be 'uniform' or 'beta'")

        self.obs_encoder = obs_encoder
        self.input_dim = action_dim
        self.obs_as_global_cond = obs_as_global_cond
        self.model = ActionDiT(
            obs_embed_dim=obs_feature_dim * n_obs_steps,
            action_dim=action_dim,
            action_len=horizon,
            embed_dim=attention_embed_dim,
            timestep_embed_dim=diffusion_timestep_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            use_rms_norm=use_rms_norm,
        )
        self.noise_scheduler = noise_scheduler
        # HoMMI keeps a parameterless mask_generator placeholder derived from
        # ModuleAttrMixin. Preserve its zero-sized checkpoint key even though
        # DiT does not use inpainting.
        self.mask_generator = torch.nn.Module()
        self.mask_generator.register_parameter("_dummy_variable", torch.nn.Parameter())
        self.horizon = int(horizon)
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = int(n_action_steps)
        self.obs_history = int(n_obs_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.use_flow_matching = use_flow_matching
        self.fm_tsampler = fm_tsampler
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.input_pertub = float(input_perturbation)
        self.num_inference_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(noise_scheduler.config.num_train_timesteps)
        )
        self.tsampler = (
            torch.distributions.beta.Beta(1.5, 1.0)
            if fm_tsampler == "beta"
            else None
        )

    def get_optimizer(
        self,
        lr: float,
        *,
        weight_decay: float = 0.0,
        obs_encoder_lr: float | None = None,
        obs_encoder_weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        **kwargs,
    ) -> torch.optim.Optimizer:
        groups = [
            {
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "weight_decay": weight_decay,
            }
        ]

        backbone = []
        other = []
        for key, value in self.obs_encoder.named_parameters():
            if not value.requires_grad:
                continue
            if key.startswith("key_model_map"):
                backbone.append(value)
            else:
                other.append(value)

        if backbone:
            groups.append(
                {
                    "params": backbone,
                    "weight_decay": obs_encoder_weight_decay,
                    "lr": obs_encoder_lr if obs_encoder_lr is not None else lr,
                }
            )
        if other:
            groups.append(
                {
                    "params": other,
                    "weight_decay": obs_encoder_weight_decay,
                }
            )
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **kwargs)

    def _encode_history(self, nobs: dict, batch_size: int) -> torch.Tensor:
        reshaped = dict_apply(
            nobs,
            lambda x: x[:, : self.obs_history].reshape(-1, *x.shape[2:]),
        )
        features = encode_features(self.obs_encoder, reshaped)
        return features.reshape(batch_size, -1)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        if "valid_mask" in batch:
            raise ValueError("valid_mask is not supported; use valid_action_mask")

        valid_action_mask = batch.get("valid_action_mask")
        actions = self.normalizer["action"].normalize(batch["action"])
        if actions.shape[1] != self.horizon:
            raise ValueError(
                f"Expected action horizon {self.horizon}, got {actions.shape[1]}"
            )
        batch_size = actions.shape[0]
        nobs = self.normalizer.normalize(batch["obs"])
        global_cond = self._encode_history(nobs, batch_size)
        trajectory = actions

        if self.train_diffusion_n_samples != 1:
            repeat = self.train_diffusion_n_samples
            global_cond = torch.repeat_interleave(global_cond, repeat, dim=0)
            trajectory = torch.repeat_interleave(trajectory, repeat, dim=0)
            if valid_action_mask is not None:
                valid_action_mask = torch.repeat_interleave(
                    valid_action_mask, repeat, dim=0
                )

        noise = torch.randn_like(trajectory)
        if self.input_pertub:
            noise = noise + self.input_pertub * torch.randn_like(trajectory)

        repeated_batch = trajectory.shape[0]

        if self.use_flow_matching:
            if self.fm_tsampler == "uniform":
                timestamps = torch.rand(repeated_batch, device=trajectory.device)
            else:
                assert self.tsampler is not None
                timestamps = self.tsampler.sample((repeated_batch,)).to(trajectory.device)

            continuous_t = timestamps.view(-1, *([1] * (noise.ndim - 1)))
            timesteps = (
                timestamps * self.noise_scheduler.config.num_train_timesteps
            ).long()
            direction = noise - trajectory
            noisy = trajectory + continuous_t * direction
            pred = self.model(global_cond, noisy, timesteps)
            target = direction
        else:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (repeated_batch,),
                device=trajectory.device,
            ).long()
            noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
            pred = self.model(global_cond, noisy, timesteps)
            target = noise

        loss = F.mse_loss(pred, target, reduction="none")
        if valid_action_mask is not None:
            loss = loss * valid_action_mask.to(dtype=loss.dtype)
        return loss.reshape(loss.shape[0], -1).mean(dim=1).mean()

    def _sample_actions(
        self,
        shape: tuple[int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        global_cond: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        trajectory = torch.randn(
            shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        if self.use_flow_matching:
            timesteps = torch.linspace(
                1,
                0,
                self.num_inference_steps + 1,
                device=device,
            )[:-1]
            timesteps = (
                timesteps * self.noise_scheduler.config.num_train_timesteps
            ).long()
            for timestep in timesteps:
                batch_t = timestep.expand(trajectory.shape[0])
                model_output = self.model(global_cond, trajectory, batch_t)
                trajectory = trajectory - model_output / self.num_inference_steps
            return trajectory

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            model_output = self.model(global_cond, trajectory, timestep)
            trajectory = self.noise_scheduler.step(
                model_output,
                timestep,
                trajectory,
                generator=generator,
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
        # Compatibility with HoMMI's generic diffusion-policy sampling API.
        # DiT itself has no inpainting path.
        if condition_mask.any():
            raise ValueError("DiT policy does not implement inpainting conditioning")
        return self._sample_actions(
            tuple(condition_data.shape),
            device=condition_data.device,
            dtype=condition_data.dtype,
            global_cond=global_cond,
            generator=generator,
        )

    def predict_action(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        global_cond = self._encode_history(nobs, batch_size)

        sample = self._sample_actions(
            (batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
            global_cond=global_cond,
        )
        action_pred = self.normalizer["action"].unnormalize(
            sample[..., : self.action_dim]
        )
        start = self.obs_history - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}

    def forward(self, batch: dict) -> torch.Tensor:
        return self.compute_loss(batch)
