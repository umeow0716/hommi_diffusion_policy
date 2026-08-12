from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import BaseObsEncoder


class DiTObsEncoderLite(BaseObsEncoder):
    """Standalone equivalent of HoMMI's 2D ``DiTObsEncoder`` for this dataset.

    The DiT policy itself flattens ``[B, T, ...]`` observations to
    ``[B*T, ...]`` before calling the encoder. This class therefore consumes
    one observation timestep per leading batch element and returns
    ``{"features": [B*T, D]}``, matching HoMMI's encoder contract.

    HoMMI ``diffusion_dit.yaml`` settings mirrored here::

        model_name='vit_base_patch16_clip_224.openai'
        pretrained=True, frozen=False, global_pool=''
        feature_aggregation='cls'
        use_group_norm=True (no effect for pretrained ViT)
        share_rgb_model=True
        use_vision_norm=True

    ``timm`` and ``torchvision`` are package dependencies used by this encoder.
    They are imported lazily when the encoder is instantiated so importing the
    package does not eagerly load the vision stack.
    """

    def __init__(
        self,
        shape_meta: dict[str, Any],
        *,
        model_name: str = "vit_base_patch16_clip_224.openai",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Keep heavyweight vision imports lazy even though they are required
        # package dependencies. This also makes encoder tests/custom factories
        # able to substitute a timm implementation before instantiation.
        import timm
        import torchvision

        self.shape_meta = shape_meta
        self.model_name = model_name
        self.pretrained = bool(pretrained)
        self.feature_aggregation = "cls"
        self.share_rgb_model = True
        self.use_vision_norm = True

        obs_meta = shape_meta.get("obs")
        if not isinstance(obs_meta, dict):
            raise ValueError("shape_meta must contain an 'obs' mapping")

        self.rgb_keys = sorted(
            key
            for key, value in obs_meta.items()
            if value.get("type") == "rgb"
            and not value.get("ignore_by_policy", False)
        )
        self.low_dim_keys = sorted(
            key
            for key, value in obs_meta.items()
            if value.get("type", "low_dim") == "low_dim"
            and not value.get("ignore_by_policy", False)
        )
        if not self.rgb_keys:
            raise ValueError("DiTObsEncoderLite requires at least one RGB observation")

        rgb_shapes = [tuple(obs_meta[key]["shape"]) for key in self.rgb_keys]
        for key, shape in zip(self.rgb_keys, rgb_shapes, strict=True):
            if len(shape) != 3:
                raise ValueError(f"{key} RGB shape must be [C,H,W], got {shape}")
            if shape[0] != 3:
                raise ValueError(f"{key} must have 3 RGB channels, got shape {shape}")
        if len(set(rgb_shapes)) != 1:
            raise ValueError(
                "share_rgb_model=True requires all RGB observations to have the same shape"
            )

        # Keep the same registered name used by HoMMI. The standalone policy's
        # optimizer recognizes key_model_map.* as the low-LR vision backbone.
        backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool="",
            num_classes=0,
        )
        data_config = timm.data.resolve_data_config(backbone.pretrained_cfg)
        self.key_model_map = nn.ModuleDict({"rgb": backbone})

        mean = torch.tensor(data_config["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(data_config["std"], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=True)
        self.register_buffer("image_std", std, persistent=True)

        image_height = int(rgb_shapes[0][1])
        image_width = int(rgb_shapes[0][2])
        if image_height != image_width:
            raise ValueError(
                "DiTObsEncoderLite currently mirrors HoMMI's square-image transform; "
                f"got HxW={image_height}x{image_width}"
            )
        image_size = image_height
        crop_size = int(image_size * 0.95)

        # Matches singletask_umi_policy.yaml augmentation.
        self.train_transform = nn.Sequential(
            torchvision.transforms.RandomCrop(crop_size),
            torchvision.transforms.Resize(image_size, antialias=True),
            torchvision.transforms.ColorJitter(
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
        )
        self.eval_transform = nn.Sequential(
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.Resize(image_size, antialias=True),
        )

    def _aggregate_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if self.model_name.startswith("vit"):
            if feature.ndim != 3:
                raise RuntimeError(
                    "expected ViT token tensor [N,tokens,D], "
                    f"got {tuple(feature.shape)}"
                )
            # HoMMI feature_aggregation='cls' keeps a singleton token dimension,
            # then flattens it when assembling all observation embeddings.
            return feature[:, [0], :]
        if feature.ndim == 4:
            # Not used by the exact default, retained only for a useful
            # error-safe fallback for spatial CNN-like outputs.
            return feature.flatten(start_dim=2).transpose(1, 2)
        if feature.ndim == 2:
            return feature[:, None, :]
        raise RuntimeError(f"unsupported timm feature shape {tuple(feature.shape)}")

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings: list[torch.Tensor] = []
        batch_size: int | None = None

        # share_rgb_model=True: concatenate all RGB streams before one backbone
        # forward. FastUMI commonly has one stream, camera0_main_rgb.
        images: list[torch.Tensor] = []
        for key in self.rgb_keys:
            if key not in obs:
                raise KeyError(f"missing RGB observation {key!r}")
            image = obs[key]
            if image.ndim != 4:
                raise ValueError(
                    f"{key} must be [B*T,C,H,W] inside DiT encoder, "
                    f"got {tuple(image.shape)}"
                )
            if batch_size is None:
                batch_size = int(image.shape[0])
            elif image.shape[0] != batch_size:
                raise ValueError("observation batch sizes do not match")

            image = (
                self.train_transform(image)
                if self.training
                else self.eval_transform(image)
            )
            image = (image - self.image_mean) / self.image_std
            images.append(image)

        assert batch_size is not None
        merged = torch.cat(images, dim=0)
        raw_feature = self.key_model_map["rgb"](merged)
        feature = self._aggregate_feature(raw_feature)

        # HoMMI: (N_rgb * B*T, token, D) -> (B*T, N_rgb * token * D)
        feature = feature.reshape(-1, batch_size, *feature.shape[1:])
        feature = torch.moveaxis(feature, 0, 1)
        feature = feature.reshape(batch_size, -1)
        embeddings.append(feature)

        for key in self.low_dim_keys:
            if key not in obs:
                raise KeyError(f"missing low-dimensional observation {key!r}")
            value = obs[key]
            if value.shape[0] != batch_size:
                raise ValueError("observation batch sizes do not match")
            expected_shape = tuple(self.shape_meta["obs"][key]["shape"])
            if tuple(value.shape[1:]) != expected_shape:
                raise ValueError(
                    f"{key} expected [B*T,{expected_shape}], got {tuple(value.shape)}"
                )
            embeddings.append(value)

        return {"features": torch.cat(embeddings, dim=-1)}

    @torch.no_grad()
    def output_shape(self) -> dict[str, tuple[int, ...]]:
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            example: dict[str, torch.Tensor] = {}
            for key, attr in self.shape_meta["obs"].items():
                if attr.get("ignore_by_policy", False):
                    continue
                if attr.get("type", "low_dim") not in {"rgb", "low_dim"}:
                    continue
                example[key] = torch.zeros(
                    (1, *tuple(attr["shape"])),
                    dtype=torch.float32,
                    device=device,
                )
            output = self.forward(example)
            return {key: tuple(value.shape[1:]) for key, value in output.items()}
        finally:
            self.train(was_training)
