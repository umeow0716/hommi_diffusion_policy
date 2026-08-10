from types import SimpleNamespace

import torch
from torch import nn

from hommi_diffusion_policy import (
    DiffusionDiTImagePolicy,
    DiffusionTransformerPolicy,
    DiffusionUnetPolicy,
    LinearNormalizer,
)


class DummyScheduler:
    def __init__(self):
        self.config = SimpleNamespace(
            num_train_timesteps=4,
            prediction_type="epsilon",
        )
        self.timesteps = torch.arange(3, -1, -1)

    def set_timesteps(self, n):
        self.timesteps = torch.arange(n - 1, -1, -1)

    def add_noise(self, original, noise, timesteps):
        del timesteps
        return original + noise * 0.1

    def step(self, model_output, timestep, sample, generator=None, **kwargs):
        del timestep, generator, kwargs
        return SimpleNamespace(prev_sample=sample - model_output * 0.1)


class VectorEncoder(nn.Module):
    pretrained = False

    def __init__(self, dim=8):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def output_shape(self):
        return (8,)

    def forward(self, obs):
        return self.proj(obs["obs"])


class TokenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16)

    def output_shape(self):
        return (4, 16)

    def forward(self, obs):
        return self.proj(obs["tokens"])


class DiTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def output_shape(self):
        return {"features": (8,)}

    def forward(self, obs):
        return {"features": self.proj(obs["obs"])}


def make_normalizer(obs, action):
    normalizer = LinearNormalizer()
    normalizer.fit({**obs, "action": action})
    return normalizer


def test_unet_policy_loss_and_predict():
    scheduler = DummyScheduler()
    policy = DiffusionUnetPolicy(
        shape_meta={"action": {"shape": (3,), "horizon": 8}},
        noise_scheduler=scheduler,
        obs_encoder=VectorEncoder(),
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        n_groups=8,
        num_inference_steps=2,
    )
    obs = {"obs": torch.randn(2, 8)}
    action = torch.randn(2, 8, 3)
    policy.set_normalizer(make_normalizer(obs, action))
    assert policy.compute_loss({"obs": obs, "action": action}).ndim == 0
    assert policy.predict_action(obs)["action"].shape == action.shape


def test_transformer_policy_loss_and_predict():
    scheduler = DummyScheduler()
    policy = DiffusionTransformerPolicy(
        shape_meta={"action": {"shape": (3,), "horizon": 8}},
        noise_scheduler=scheduler,
        obs_encoder=TokenEncoder(),
        n_layer=2,
        n_head=4,
        n_emb=16,
        num_inference_steps=2,
    )
    obs = {"tokens": torch.randn(2, 4, 16)}
    action = torch.randn(2, 8, 3)
    policy.set_normalizer(make_normalizer(obs, action))
    assert policy.compute_loss({"obs": obs, "action": action}).ndim == 0
    assert policy.predict_action(obs)["action"].shape == action.shape


def test_dit_policy_loss_and_predict():
    scheduler = DummyScheduler()
    policy = DiffusionDiTImagePolicy(
        shape_meta={"action": {"shape": (3,)}},
        noise_scheduler=scheduler,
        obs_encoder=DiTEncoder(),
        horizon=8,
        n_action_steps=4,
        n_obs_steps=2,
        attention_embed_dim=16,
        diffusion_timestep_embed_dim=16,
        depth=2,
        num_heads=4,
        num_inference_steps=2,
    )
    obs = {"obs": torch.randn(2, 2, 8)}
    action = torch.randn(2, 8, 3)
    policy.set_normalizer(make_normalizer(obs, action))
    assert policy.compute_loss({"obs": obs, "action": action}).ndim == 0
    result = policy.predict_action(obs)
    assert result["action_pred"].shape == action.shape
    assert result["action"].shape == (2, 4, 3)
    assert "_dummy_variable" in policy.state_dict()
    assert "mask_generator._dummy_variable" in policy.state_dict()


class UmiTokenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_model_map = nn.ModuleDict({"cam": nn.Linear(16, 16)})
        self.proj = nn.Linear(16, 16)

    def output_shape(self):
        # UMI TransformerObsEncoder includes a singleton sample dimension.
        return torch.Size((1, 4, 16))

    def forward(self, obs):
        x = self.key_model_map["cam"](obs["tokens"])
        return self.proj(x)


def test_transformer_policy_accepts_umi_output_shape_and_optimizer_groups():
    scheduler = DummyScheduler()
    encoder = UmiTokenEncoder()
    policy = DiffusionTransformerPolicy(
        shape_meta={"action": {"shape": (3,), "horizon": 8}},
        noise_scheduler=scheduler,
        obs_encoder=encoder,
        n_layer=2,
        n_head=4,
        n_emb=16,
        num_inference_steps=2,
    )

    optimizer = policy.get_optimizer(
        1e-3,
        obs_encoder_lr=1e-4,
        obs_encoder_weight_decay=1e-2,
    )
    lr_by_param_id = {
        id(parameter): group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert lr_by_param_id[id(encoder.key_model_map["cam"].weight)] == 1e-4
    assert lr_by_param_id[id(encoder.proj.weight)] == 1e-3

    obs = {"tokens": torch.randn(2, 4, 16)}
    action = torch.randn(2, 8, 3)
    policy.set_normalizer(make_normalizer(obs, action))
    assert policy.compute_loss({"obs": obs, "action": action}).ndim == 0
    assert policy.predict_action(obs)["action"].shape == action.shape

    keys = set(policy.state_dict())
    assert "_dummy_variable" in keys
    assert "model._dummy_variable" in keys


def test_unet_policy_rejects_invalid_action_horizon():
    try:
        DiffusionUnetPolicy(
            shape_meta={"action": {"shape": (3,), "horizon": 7}},
            noise_scheduler=DummyScheduler(),
            obs_encoder=VectorEncoder(),
            diffusion_step_embed_dim=16,
            down_dims=(16, 32),
            n_groups=8,
        )
    except ValueError as exc:
        assert "downsample factor 2" in str(exc)
    else:
        raise AssertionError("expected invalid U-Net action horizon to be rejected")


def test_unet_policy_normal_predict_uses_fast_unconditional_path():
    scheduler = DummyScheduler()
    policy = DiffusionUnetPolicy(
        shape_meta={"action": {"shape": (3,), "horizon": 8}},
        noise_scheduler=scheduler,
        obs_encoder=VectorEncoder(),
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        n_groups=8,
        num_inference_steps=2,
    )
    obs = {"obs": torch.randn(2, 8)}
    action = torch.randn(2, 8, 3)
    policy.set_normalizer(make_normalizer(obs, action))

    def unexpected_conditional_sample(*args, **kwargs):
        raise AssertionError("normal inference should not allocate/use an empty mask")

    policy.conditional_sample = unexpected_conditional_sample
    assert policy.predict_action(obs)["action"].shape == action.shape
