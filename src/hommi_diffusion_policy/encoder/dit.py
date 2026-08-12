from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from .base import BaseObsEncoder


@dataclass(frozen=True, slots=True)
class DiTObsEncoderConfig:
    """Configuration for :class:`DiTObsEncoderLite`.

    Defaults mirror HoMMI's ``diffusion_dit.yaml`` and
    ``singletask_umi_policy.yaml``. Image augmentation belongs to the encoder
    rather than the dataset/training package, so downstream applications can
    configure it without forking the model implementation.
    """

    model_name: str = "vit_base_patch16_clip_224.openai"
    pretrained: bool = True
    frozen: bool = False
    global_pool: str = ""
    feature_aggregation: Literal["cls", "avg", "max"] = "cls"
    use_group_norm: bool = True
    share_rgb_model: bool = True
    use_vision_norm: bool = True

    train_crop_ratio: float = 0.95
    eval_crop_ratio: float = 0.95
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.5
    color_jitter_hue: float = 0.08

    def validate(self) -> None:
        for name, value in (
            ("train_crop_ratio", self.train_crop_ratio),
            ("eval_crop_ratio", self.eval_crop_ratio),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must satisfy 0 < ratio <= 1, got {value}")
        if self.feature_aggregation not in {"cls", "avg", "max"}:
            raise ValueError(
                "feature_aggregation must be one of 'cls', 'avg', or 'max'"
            )
        for name, value in (
            ("color_jitter_brightness", self.color_jitter_brightness),
            ("color_jitter_contrast", self.color_jitter_contrast),
            ("color_jitter_saturation", self.color_jitter_saturation),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if not 0.0 <= float(self.color_jitter_hue) <= 0.5:
            raise ValueError(
                "color_jitter_hue must satisfy 0 <= hue <= 0.5, "
                f"got {self.color_jitter_hue}"
            )
        if not self.model_name:
            raise ValueError("model_name cannot be empty")


class DiTObsEncoderLite(BaseObsEncoder):
    """Standalone equivalent of HoMMI's 2D ``DiTObsEncoder``.

    The DiT policy flattens ``[B, T, ...]`` observations to ``[B*T, ...]``
    before calling the encoder. This class consumes one observation timestep
    per leading batch element and returns ``{"features": [B*T, D]}``.

    ``timm`` and ``torchvision`` are package dependencies used by this encoder.
    They are imported lazily when the encoder is instantiated so importing the
    package does not eagerly load the vision stack.
    """

    def __init__(
        self,
        shape_meta: dict[str, Any],
        *,
        config: DiTObsEncoderConfig | None = None,
        # Backwards-compatible overrides kept for 0.1.x callers.
        model_name: str | None = None,
        pretrained: bool | None = None,
    ) -> None:
        super().__init__()

        import timm
        import torchvision

        cfg = config or DiTObsEncoderConfig()
        if model_name is not None or pretrained is not None:
            cfg = DiTObsEncoderConfig(
                **{
                    **{field: getattr(cfg, field) for field in cfg.__dataclass_fields__},
                    **({"model_name": model_name} if model_name is not None else {}),
                    **({"pretrained": bool(pretrained)} if pretrained is not None else {}),
                }
            )
        cfg.validate()

        self.config = cfg
        self.shape_meta = shape_meta
        self.model_name = cfg.model_name
        self.pretrained = bool(cfg.pretrained)
        self.feature_aggregation = cfg.feature_aggregation
        self.share_rgb_model = bool(cfg.share_rgb_model)
        self.use_vision_norm = bool(cfg.use_vision_norm)
        self.use_group_norm = bool(cfg.use_group_norm)

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
                "all RGB observations must have the same shape for shared preprocessing"
            )

        def create_backbone() -> nn.Module:
            backbone = timm.create_model(
                model_name=cfg.model_name,
                pretrained=cfg.pretrained,
                global_pool=cfg.global_pool,
                num_classes=0,
            )
            if cfg.frozen:
                backbone.requires_grad_(False)
            return backbone

        if cfg.share_rgb_model:
            backbone = create_backbone()
            self.key_model_map = nn.ModuleDict({"rgb": backbone})
            vision_cfg_source = backbone
        else:
            self.key_model_map = nn.ModuleDict(
                {key: create_backbone() for key in self.rgb_keys}
            )
            vision_cfg_source = self.key_model_map[self.rgb_keys[0]]

        data_config = timm.data.resolve_data_config(vision_cfg_source.pretrained_cfg)
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
        train_crop_size = max(1, int(image_size * cfg.train_crop_ratio))
        eval_crop_size = max(1, int(image_size * cfg.eval_crop_ratio))

        self.train_transform = nn.Sequential(
            torchvision.transforms.RandomCrop(train_crop_size),
            torchvision.transforms.Resize(image_size, antialias=True),
            torchvision.transforms.ColorJitter(
                brightness=cfg.color_jitter_brightness,
                contrast=cfg.color_jitter_contrast,
                saturation=cfg.color_jitter_saturation,
                hue=cfg.color_jitter_hue,
            ),
        )
        self.eval_transform = nn.Sequential(
            torchvision.transforms.CenterCrop(eval_crop_size),
            torchvision.transforms.Resize(image_size, antialias=True),
        )

    def _aggregate_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 3:
            if self.feature_aggregation == "cls":
                return feature[:, [0], :]
            tokens = feature[:, 1:, :] if feature.shape[1] > 1 else feature
            if self.feature_aggregation == "avg":
                return tokens.mean(dim=1, keepdim=True)
            if self.feature_aggregation == "max":
                return tokens.amax(dim=1, keepdim=True)
        if feature.ndim == 4:
            tokens = feature.flatten(start_dim=2).transpose(1, 2)
            if self.feature_aggregation == "avg":
                return tokens.mean(dim=1, keepdim=True)
            if self.feature_aggregation == "max":
                return tokens.amax(dim=1, keepdim=True)
            # CNNs do not have a semantic CLS token. Retain all spatial tokens
            # for backwards-compatible fallback behavior.
            return tokens
        if feature.ndim == 2:
            return feature[:, None, :]
        raise RuntimeError(f"unsupported timm feature shape {tuple(feature.shape)}")

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self.train_transform(image) if self.training else self.eval_transform(image)
        if self.use_vision_norm:
            image = (image - self.image_mean) / self.image_std
        return image

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings: list[torch.Tensor] = []
        batch_size: int | None = None

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
            images.append(self._preprocess_image(image))

        assert batch_size is not None
        if self.share_rgb_model:
            merged = torch.cat(images, dim=0)
            raw_feature = self.key_model_map["rgb"](merged)
            feature = self._aggregate_feature(raw_feature)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            feature = torch.moveaxis(feature, 0, 1).reshape(batch_size, -1)
            embeddings.append(feature)
        else:
            for key, image in zip(self.rgb_keys, images, strict=True):
                feature = self._aggregate_feature(self.key_model_map[key](image))
                embeddings.append(feature.reshape(batch_size, -1))

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
