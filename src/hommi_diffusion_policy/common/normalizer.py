from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def _to_tensor(data: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        # Handles NumPy arrays and most array-like inputs without taking a NumPy dependency.
        try:
            tensor = torch.as_tensor(data)
        except Exception:
            # Handles lazy array objects such as zarr.Array if the caller has zarr installed.
            tensor = torch.as_tensor(data[:])
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


class _DynamicParameterDictModule(nn.Module):
    """nn.Module containing a dynamically shaped nested ParameterDict.

    The custom state-dict loader lets a freshly constructed normalizer restore
    arbitrary observation keys from a checkpoint.
    """

    def __init__(self, params_dict: nn.ParameterDict | None = None):
        super().__init__()
        self.params_dict = params_dict if params_dict is not None else nn.ParameterDict()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        param_prefix = prefix + "params_dict."

        def insert(dest: nn.ParameterDict, keys: list[str], value: torch.Tensor) -> None:
            if len(keys) == 1:
                dest[keys[0]] = nn.Parameter(value.clone(), requires_grad=False)
                return
            key = keys[0]
            if key not in dest:
                dest[key] = nn.ParameterDict()
            insert(dest[key], keys[1:], value)

        rebuilt = nn.ParameterDict()
        found = False
        for key, value in state_dict.items():
            if key.startswith(param_prefix):
                found = True
                insert(rebuilt, key[len(param_prefix) :].split("."), value)

        if found:
            self.params_dict = rebuilt
            self.params_dict.requires_grad_(False)
        elif strict:
            missing_keys.append(param_prefix.rstrip("."))


class SingleFieldLinearNormalizer(_DynamicParameterDictModule):
    @classmethod
    def create_identity(cls, dtype: torch.dtype = torch.float32):
        return cls.create_manual(
            scale=torch.tensor([1.0], dtype=dtype),
            offset=torch.tensor([0.0], dtype=dtype),
            input_stats={
                "min": torch.tensor([-1.0], dtype=dtype),
                "max": torch.tensor([1.0], dtype=dtype),
                "mean": torch.tensor([0.0], dtype=dtype),
                "std": torch.tensor([1.0], dtype=dtype),
            },
        )

    @classmethod
    def create_manual(cls, scale, offset, input_stats):
        scale = _to_tensor(scale).flatten()
        offset = _to_tensor(offset, dtype=scale.dtype).flatten()
        if scale.shape != offset.shape:
            raise ValueError("scale and offset must have identical shapes")

        stats = nn.ParameterDict()
        for name, value in input_stats.items():
            tensor = _to_tensor(value, dtype=scale.dtype).flatten()
            if tensor.shape != scale.shape:
                raise ValueError(f"input_stats[{name!r}] shape does not match scale")
            stats[name] = nn.Parameter(tensor, requires_grad=False)

        params = nn.ParameterDict(
            {
                "scale": nn.Parameter(scale, requires_grad=False),
                "offset": nn.Parameter(offset, requires_grad=False),
                "input_stats": stats,
            }
        )
        return cls(params)

    @torch.no_grad()
    def fit(
        self,
        data,
        *,
        last_n_dims: int = 1,
        dtype: torch.dtype = torch.float32,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ) -> None:
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )

    def normalize(self, x):
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x):
        return _normalize(x, self.params_dict, forward=False)

    def forward(self, x):
        return self.normalize(x)

    def get_input_stats(self):
        return self.params_dict["input_stats"]


class LinearNormalizer(_DynamicParameterDictModule):
    @torch.no_grad()
    def fit(
        self,
        data,
        *,
        last_n_dims: int = 1,
        dtype: torch.dtype = torch.float32,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ) -> None:
        if isinstance(data, Mapping):
            params = nn.ParameterDict()
            for key, value in data.items():
                params[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                )
            self.params_dict = params
        else:
            self.params_dict = nn.ParameterDict(
                {
                    "_default": _fit(
                        data,
                        last_n_dims=last_n_dims,
                        dtype=dtype,
                        mode=mode,
                        output_max=output_max,
                        output_min=output_min,
                        range_eps=range_eps,
                        fit_offset=fit_offset,
                    )
                }
            )

    def __getitem__(self, key: str) -> SingleFieldLinearNormalizer:
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: SingleFieldLinearNormalizer) -> None:
        self.params_dict[key] = value.params_dict

    def _apply_impl(self, x, *, forward: bool):
        if isinstance(x, Mapping):
            result = {}
            for key, value in x.items():
                if key not in self.params_dict:
                    raise KeyError(f"Normalizer has no statistics for key {key!r}")
                result[key] = _normalize(value, self.params_dict[key], forward=forward)
            return result

        if "_default" not in self.params_dict:
            raise RuntimeError("Normalizer is not initialized for a single tensor")
        return _normalize(x, self.params_dict["_default"], forward=forward)

    def normalize(self, x):
        return self._apply_impl(x, forward=True)

    def unnormalize(self, x):
        return self._apply_impl(x, forward=False)

    def forward(self, x):
        return self.normalize(x)

    def get_input_stats(self):
        if len(self.params_dict) == 0:
            raise RuntimeError("Normalizer is not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]
        return {
            key: value["input_stats"]
            for key, value in self.params_dict.items()
            if key != "_default"
        }


def _fit(
    data,
    *,
    last_n_dims: int,
    dtype: torch.dtype,
    mode: str,
    output_max: float,
    output_min: float,
    range_eps: float,
    fit_offset: bool,
) -> nn.ParameterDict:
    if mode not in {"limits", "gaussian"}:
        raise ValueError("mode must be 'limits' or 'gaussian'")
    if last_n_dims < 0:
        raise ValueError("last_n_dims must be >= 0")
    if output_max <= output_min:
        raise ValueError("output_max must be larger than output_min")

    tensor = _to_tensor(data, dtype=dtype)
    dim = math.prod(tensor.shape[-last_n_dims:]) if last_n_dims > 0 else 1
    tensor = tensor.reshape(-1, dim)

    input_min = tensor.amin(dim=0)
    input_max = tensor.amax(dim=0)
    input_mean = tensor.mean(dim=0)
    input_std = tensor.std(dim=0)

    if mode == "limits":
        if fit_offset:
            input_range = input_max - input_min
            ignore = input_range < range_eps
            safe_range = torch.where(
                ignore,
                torch.full_like(input_range, output_max - output_min),
                input_range,
            )
            scale = (output_max - output_min) / safe_range
            offset = output_min - scale * input_min
            center_offset = (output_max + output_min) / 2 - input_min
            offset = torch.where(ignore, center_offset, offset)
        else:
            if not (output_min < 0 < output_max):
                raise ValueError("fit_offset=False requires output_min < 0 < output_max")
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(input_min.abs(), input_max.abs())
            ignore = input_abs < range_eps
            safe_abs = torch.where(ignore, torch.full_like(input_abs, output_abs), input_abs)
            scale = output_abs / safe_abs
            offset = torch.zeros_like(input_mean)
    else:
        ignore = input_std < range_eps
        safe_std = torch.where(ignore, torch.ones_like(input_std), input_std)
        scale = safe_std.reciprocal()
        offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)

    stats = nn.ParameterDict(
        {
            "min": nn.Parameter(input_min, requires_grad=False),
            "max": nn.Parameter(input_max, requires_grad=False),
            "mean": nn.Parameter(input_mean, requires_grad=False),
            "std": nn.Parameter(input_std, requires_grad=False),
        }
    )
    return nn.ParameterDict(
        {
            "scale": nn.Parameter(scale, requires_grad=False),
            "offset": nn.Parameter(offset, requires_grad=False),
            "input_stats": stats,
        }
    )


def _normalize(x, params: nn.ParameterDict, *, forward: bool):
    is_tensor = isinstance(x, torch.Tensor)
    tensor = _to_tensor(x)
    scale = params["scale"]
    offset = params["offset"]
    tensor = tensor.to(device=scale.device, dtype=scale.dtype)
    source_shape = tensor.shape
    tensor = tensor.reshape(-1, scale.numel())
    if forward:
        tensor = tensor * scale + offset
    else:
        tensor = (tensor - offset) / scale
    tensor = tensor.reshape(source_shape)
    return tensor if is_tensor else tensor
