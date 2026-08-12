# hommi-diffusion-policy

A small standalone Python package containing the three diffusion-policy families used by
[HoMMI](https://github.com/xxm19/hommi), with the dependency on UMI's
`diffusion_policy.*` package removed.

Included policies:

- `DiffusionUnetPolicy` (`DiffusionUnetTimmPolicy` compatibility alias)
- `DiffusionTransformerPolicy` (`DiffusionTransformerTimmPolicy` compatibility alias)
- `DiffusionDiTImagePolicy` (`DiffusionDiTPolicy` alias)

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- diffusers >= 0.18.2, < 1

The core policy/model package still keeps vision dependencies optional. The bundled
`DiTObsEncoderLite` additionally uses `timm` and `torchvision`; install the `vision`
extra when you want that encoder. Observation encoders are still injected as normal
`torch.nn.Module` instances, so policies remain independent of any specific backbone.

## Install with uv

From this directory:

```bash
uv sync
```

With the bundled HoMMI 2D DiT vision encoder:

```bash
uv sync --extra vision
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
    DiffusionUnetTimmPolicy,
    DiffusionTransformerPolicy,
    DiffusionTransformerTimmPolicy,
    DiffusionDiTImagePolicy,
    DiffusionDiTPolicy,
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

`DiTObsEncoderLite` mirrors the 2D HoMMI DiT observation encoder configuration and
keeps `key_model_map.rgb` as the timm backbone name used by the policy optimizer:

```python
from hommi_diffusion_policy import DiTObsEncoderLite, DiffusionDiTImagePolicy

encoder = DiTObsEncoderLite(shape_meta=shape_meta)
policy = DiffusionDiTImagePolicy(
    shape_meta=shape_meta,
    noise_scheduler=scheduler,
    obs_encoder=encoder,
    horizon=32,
    n_action_steps=16,
    n_obs_steps=2,
)
```

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

`LinearNormalizer` is implemented locally using PyTorch only. It supports:

- per-key dict normalization
- `limits` normalization
- `gaussian` normalization
- `normalize` / `unnormalize`
- checkpointing with `state_dict`
- `.to(device)` / `.cuda()` through `nn.Module`

## Attribution

The policy/model implementations are adapted from HoMMI and the Universal Manipulation
Interface (UMI), both MIT licensed. See `THIRD_PARTY_NOTICES.md`.
