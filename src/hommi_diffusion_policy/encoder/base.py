from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn


class BaseObsEncoder(nn.Module, ABC):
    """Base class for observation encoders consumed by HoMMI policies.

    Policies intentionally accept any compatible ``nn.Module``; subclassing this
    class is optional. It mainly documents the small encoder contract and gives
    built-in encoders a common extension point.
    """

    @abstractmethod
    def forward(
        self, obs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | Mapping[str, torch.Tensor]:
        """Encode one policy observation batch."""
        raise NotImplementedError

    @abstractmethod
    def output_shape(self) -> tuple[int, ...] | Mapping[str, tuple[int, ...]]:
        """Return the feature shape excluding the leading batch dimension."""
        raise NotImplementedError
