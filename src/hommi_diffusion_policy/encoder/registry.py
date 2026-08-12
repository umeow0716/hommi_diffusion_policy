from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

EncoderFactory = Callable[..., nn.Module]

# Built-ins are imported lazily so the core package remains usable without the
# optional vision dependencies required by a particular encoder.
_BUILTIN_ENCODERS: dict[str, tuple[str, str]] = {
    "dit_obs_lite": ("hommi_diffusion_policy.encoder.dit", "DiTObsEncoderLite"),
}
_REGISTERED_ENCODERS: dict[str, EncoderFactory] = {}


def available_encoders() -> tuple[str, ...]:
    """Return the names understood by :func:`create_encoder`."""
    return tuple(sorted(_BUILTIN_ENCODERS.keys() | _REGISTERED_ENCODERS.keys()))


def register_encoder(
    name: str,
    factory: EncoderFactory,
    *,
    overwrite: bool = False,
) -> None:
    """Register a custom observation encoder factory.

    ``factory`` may be an ``nn.Module`` class or any callable returning one.
    Registered factories take the same keyword arguments passed to
    :func:`create_encoder`.
    """
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("encoder name must not be empty")
    if not callable(factory):
        raise TypeError("encoder factory must be callable")
    if not overwrite and (
        normalized in _BUILTIN_ENCODERS or normalized in _REGISTERED_ENCODERS
    ):
        raise ValueError(f"encoder {normalized!r} is already registered")
    _REGISTERED_ENCODERS[normalized] = factory


def _load_builtin_factory(name: str) -> EncoderFactory:
    import importlib

    module_name, attribute = _BUILTIN_ENCODERS[name]
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"built-in encoder factory {name!r} is not callable")
    return factory


def create_encoder(name: str, /, **kwargs: Any) -> nn.Module:
    """Create a built-in or registered observation encoder by name."""
    normalized = name.strip().lower()
    factory = _REGISTERED_ENCODERS.get(normalized)
    if factory is None:
        if normalized not in _BUILTIN_ENCODERS:
            choices = ", ".join(available_encoders()) or "<none>"
            raise KeyError(f"unknown encoder {name!r}; available encoders: {choices}")
        factory = _load_builtin_factory(normalized)

    encoder = factory(**kwargs)
    if not isinstance(encoder, nn.Module):
        raise TypeError(
            f"encoder factory {normalized!r} returned {type(encoder).__name__}, "
            "expected torch.nn.Module"
        )
    return encoder
