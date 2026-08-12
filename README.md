# hommi-diffusion-policy

A small standalone Python package containing the three diffusion-policy families used by
[HoMMI](https://github.com/xxm19/hommi), with the dependency on UMI's
`diffusion_policy.*` package removed.

Included policies:

- `DiffusionUnetPolicy`
- `DiffusionTransformerPolicy`
- `DiffusionDiTImagePolicy`

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- torchvision >= 0.15
- timm >= 0.9, < 2
- diffusers >= 0.18.2, < 1

`timm` and `torchvision` are normal package dependencies because the bundled
`DiTObsEncoderLite` is part of the supported public API. They are imported lazily by
the encoder so importing `hommi_diffusion_policy` does not eagerly load the vision
stack. Observation encoders are still injected as normal `torch.nn.Module` instances,
so policies remain independent of any specific backbone.

## Install with uv

From this directory:

```bash
uv sync
```

From another project:

```bash
uv add /path/to/hommi_diffusion_policy
```

Or after publishing / pushing to Git:

```bash
uv add git+https://github.com/OWNER/hommi_diffusion_policy.git
```

## Public API

```python
from hommi_diffusion_policy import (
    DiffusionUnetPolicy,
    DiffusionTransformerPolicy,
    DiffusionDiTImagePolicy,
    DiTObsEncoderConfig,
    DiTObsEncoderLite,
    LinearNormalizer,
)
```

## Observation encoder contracts

The policy owns no vision backbone. Pass an encoder to the constructor.

### U-Net policy

The encoder must implement:

```python
class Encoder(torch.nn.Module):
    def output_shape(self) -> tuple[int, ...]:
        ...

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # [B, feature_dim] (or another shape that flattens to output_shape())
        ...
```

### Transformer policy

The encoder must return token features:

```python
class Encoder(torch.nn.Module):
    def output_shape(self):
        # Either (num_tokens, embedding_dim) or UMI-style
        # (1, num_tokens, embedding_dim).
        ...

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # [B, num_tokens, embedding_dim]
        ...
```

### DiT policy

For HoMMI compatibility, the DiT encoder may return a dict containing `"features"`:

```python
class Encoder(torch.nn.Module):
    def output_shape(self):
        return {"features": (feature_dim,)}

    def forward(self, obs):
        return {"features": feature_tensor}
```

Returning the feature tensor directly is also accepted.

### Built-in DiT observation encoder

`DiTObsEncoderLite` mirrors HoMMI's 2D observation encoder. Encoder architecture
and image augmentation are configured by `DiTObsEncoderConfig`, so downstream
training packages do not need to fork or hard-code model behavior:

```python
from hommi_diffusion_policy import (
    DiTObsEncoderConfig,
    DiTObsEncoderLite,
    DiffusionDiTImagePolicy,
)

encoder_config = DiTObsEncoderConfig(
    model_name="vit_base_patch16_clip_224.openai",
    pretrained=True,
    train_crop_ratio=0.95,
    eval_crop_ratio=0.95,
    color_jitter_brightness=0.3,
    color_jitter_contrast=0.4,
    color_jitter_saturation=0.5,
    color_jitter_hue=0.08,
)
encoder = DiTObsEncoderLite(shape_meta=shape_meta, config=encoder_config)
policy = DiffusionDiTImagePolicy(
    shape_meta=shape_meta,
    noise_scheduler=scheduler,
    obs_encoder=encoder,
    horizon=32,
    n_action_steps=16,
    n_obs_steps=2,
)
```

The defaults match HoMMI's `diffusion_dit.yaml` plus the single-task train/eval
image transforms. `feature_aggregation` supports `cls`, `avg`, and `max`;
`share_rgb_model=False` constructs one timm backbone per RGB observation. The
legacy `model_name=` and `pretrained=` constructor keywords remain accepted for
0.1.x callers.

The encoder can also be constructed by name:

```python
from hommi_diffusion_policy import create_encoder

encoder = create_encoder("dit_obs_lite", shape_meta=shape_meta)
```

### Adding another encoder

Policies do not depend on the registry, so a custom encoder can always be passed
directly. For configuration-driven projects, register a factory once and construct it
by name:

```python
from hommi_diffusion_policy import create_encoder, register_encoder

register_encoder("my_encoder", MyEncoder)
encoder = create_encoder("my_encoder", shape_meta=shape_meta, width=512)
```

A custom encoder only needs to be an `nn.Module` implementing `forward()` and
`output_shape()` with the contract required by the target policy. Subclassing
`BaseObsEncoder` is optional.

## Scheduler

Schedulers are passed into the policies, so the policy code is not tied to a specific
DDPM implementation:

```python
from diffusers import DDPMScheduler

scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="epsilon",
)
```

## Normalization

`LinearNormalizer` is a runtime policy utility implemented locally using PyTorch only.
The policy owns **application** of normalization during both training and inference,
while the training/data package owns **fitting/building** the normalization statistics.

Typical training setup:

```python
normalizer = build_hommi_normalizer(train_dataset)  # lives in hommi_train
policy.set_normalizer(normalizer)
loss = policy.compute_loss(batch)                   # policy normalizes internally
```

Datasets should return the canonical/raw training representation rather than applying
the normalizer inside `Dataset.__getitem__()`. This keeps train, validation, and
inference on the same policy-side normalization path and ensures validation never fits
its own statistics.

`LinearNormalizer` supports:

- per-key dict normalization
- `limits` normalization
- `gaussian` normalization
- `normalize` / `unnormalize`
- checkpointing with `state_dict`
- `.to(device)` / `.cuda()` through `nn.Module`

The normalizer is owned by `BasePolicy`, so all policy families expose the same
`set_normalizer()` behavior and keep the existing `normalizer.params_dict.*` checkpoint
namespace.

## Attribution

The policy/model implementations are adapted from HoMMI and the Universal Manipulation
Interface (UMI), both MIT licensed. See `THIRD_PARTY_NOTICES.md`.
