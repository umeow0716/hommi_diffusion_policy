from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch


def dict_apply(data: Mapping[str, Any], fn: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            result[key] = dict_apply(value, fn)
        else:
            result[key] = fn(value)
    return result


def output_shape(encoder) -> tuple[int, ...]:
    shape = encoder.output_shape()
    if isinstance(shape, Mapping):
        shape = shape["features"]
    return tuple(int(x) for x in shape)


def encode_features(encoder, obs) -> torch.Tensor:
    features = encoder(obs)
    if isinstance(features, Mapping):
        features = features["features"]
    if not isinstance(features, torch.Tensor):
        raise TypeError(
            "Observation encoder must return a Tensor or a mapping containing "
            "a Tensor under the 'features' key."
        )
    return features
