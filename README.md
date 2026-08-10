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

The core package intentionally does **not** depend on `timm`, `torchvision`, `numpy`,
`einops`, `accelerate`, HoMMI, or UMI. Observation encoders are injected as normal
`torch.nn.Module` instances.

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

## HoMMI / UMI compatibility

Version 0.1.2 keeps the registered module names used by HoMMI/UMI checkpoints for
U-Net, Transformer, and DiT. The DiT model in particular uses HoMMI's
`dit_blocks.*`, `ada_ln_modulation.*`, and `final_linear.*` state-dict keys.

The Transformer policy accepts both `(num_tokens, embedding_dim)` and UMI-style
`(1, num_tokens, embedding_dim)` encoder output shapes. Its attention initialization
and observation-encoder optimizer grouping also follow the UMI implementation.

Normal U-Net, Transformer, and DiT prediction use a direct unconditional sampling
path to avoid allocating unused zero condition/mask tensors. Their public
`conditional_sample(...)` compatibility methods remain available. U-Net action
horizons are validated against the model's downsample factor instead of silently
interpolating mismatched skip-connection lengths.

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
