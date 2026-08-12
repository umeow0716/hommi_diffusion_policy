import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from hommi_diffusion_policy import (
    DiTObsEncoderLite,
    available_encoders,
    create_encoder,
    register_encoder,
)


class FakeViT(nn.Module):
    def __init__(self, feature_dim: int = 4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.feature_dim = feature_dim
        self.pretrained_cfg = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(-2, -1)).mean(dim=1, keepdim=True)
        cls = pooled.repeat(1, self.feature_dim) * self.scale
        patch = torch.zeros_like(cls)
        return torch.stack((cls, patch), dim=1)


def install_fake_timm(monkeypatch) -> None:
    timm = ModuleType("timm")
    timm.create_model = lambda **kwargs: FakeViT()
    timm.data = SimpleNamespace(
        resolve_data_config=lambda cfg: {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.25, 0.25, 0.25),
        }
    )
    monkeypatch.setitem(sys.modules, "timm", timm)


def make_shape_meta():
    return {
        "obs": {
            "camera0_main_rgb": {"shape": (3, 32, 32), "type": "rgb"},
            "camera1_main_rgb": {"shape": (3, 32, 32), "type": "rgb"},
            "robot0_eef_pos": {"shape": (3,), "type": "low_dim"},
            "ignored": {
                "shape": (2,),
                "type": "low_dim",
                "ignore_by_policy": True,
            },
        }
    }


def test_dit_obs_encoder_lite_output_shape_and_forward(monkeypatch):
    install_fake_timm(monkeypatch)
    encoder = DiTObsEncoderLite(
        make_shape_meta(),
        model_name="vit_fake",
        pretrained=False,
    )
    encoder.eval()

    obs = {
        "camera0_main_rgb": torch.rand(2, 3, 32, 32),
        "camera1_main_rgb": torch.rand(2, 3, 32, 32),
        "robot0_eef_pos": torch.randn(2, 3),
    }
    output = encoder(obs)

    # Two RGB streams * 4-D CLS token + 3-D low-dimensional state.
    assert output["features"].shape == (2, 11)
    assert encoder.output_shape() == {"features": (11,)}
    assert encoder.training is False
    assert "key_model_map.rgb.scale" in encoder.state_dict()
    assert "image_mean" in encoder.state_dict()
    assert "image_std" in encoder.state_dict()


def test_encoder_factory_builds_builtin_lazily(monkeypatch):
    install_fake_timm(monkeypatch)
    assert "dit_obs_lite" in available_encoders()
    encoder = create_encoder(
        "dit_obs_lite",
        shape_meta=make_shape_meta(),
        model_name="vit_fake",
        pretrained=False,
    )
    assert isinstance(encoder, DiTObsEncoderLite)


def test_encoder_registry_supports_custom_extensions():
    class CustomEncoder(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.width = width

        def forward(self, obs):
            return obs["x"]

        def output_shape(self):
            return (self.width,)

    name = "test_custom_encoder"
    register_encoder(name, CustomEncoder, overwrite=True)
    encoder = create_encoder(name, width=7)
    assert isinstance(encoder, CustomEncoder)
    assert encoder.output_shape() == (7,)


def test_dit_obs_encoder_lite_integrates_with_dit_policy(monkeypatch):
    from types import SimpleNamespace

    from hommi_diffusion_policy import (
        DiffusionDiTImagePolicy,
        LinearNormalizer,
        SingleFieldLinearNormalizer,
    )

    class Scheduler:
        def __init__(self):
            self.config = SimpleNamespace(num_train_timesteps=4)

        def add_noise(self, original, noise, timesteps):
            del timesteps
            return original + 0.1 * noise

    install_fake_timm(monkeypatch)
    shape_meta = make_shape_meta()
    shape_meta["action"] = {"shape": (3,)}
    encoder = DiTObsEncoderLite(
        shape_meta,
        model_name="vit_fake",
        pretrained=False,
    )
    policy = DiffusionDiTImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=Scheduler(),
        obs_encoder=encoder,
        horizon=4,
        n_action_steps=2,
        n_obs_steps=2,
        attention_embed_dim=16,
        diffusion_timestep_embed_dim=16,
        depth=1,
        num_heads=4,
    )

    obs = {
        "camera0_main_rgb": torch.rand(2, 2, 3, 32, 32),
        "camera1_main_rgb": torch.rand(2, 2, 3, 32, 32),
        "robot0_eef_pos": torch.randn(2, 2, 3),
    }
    action = torch.randn(2, 4, 3)

    normalizer = LinearNormalizer()
    identity = SingleFieldLinearNormalizer.create_identity()
    for key in shape_meta["obs"]:
        if not shape_meta["obs"][key].get("ignore_by_policy", False):
            normalizer[key] = identity
    normalizer["action"] = identity
    policy.set_normalizer(normalizer)

    loss = policy.compute_loss({"obs": obs, "action": action})
    assert loss.ndim == 0
    assert policy.obs_feature_dim == 11

    optimizer = policy.get_optimizer(1e-3, obs_encoder_lr=1e-4)
    lr_by_param_id = {
        id(parameter): group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert lr_by_param_id[id(encoder.key_model_map["rgb"].scale)] == 1e-4
