from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from ..common.normalizer import LinearNormalizer


class BasePolicy(nn.Module, ABC):
    def __init__(self, name: str | None = None):
        super().__init__()
        # UMI ModuleAttrMixin compatibility. This zero-sized parameter is used
        # only to make device/dtype queries work on otherwise parameterless modules
        # and is present in upstream HoMMI/UMI checkpoints.
        self._dummy_variable = nn.Parameter()
        self.name = name

    @property
    def device(self) -> torch.device:
        parameter = next(self.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        parameter = next(self.parameters(), None)
        if parameter is not None:
            return parameter.dtype
        buffer = next(self.buffers(), None)
        if buffer is not None:
            return buffer.dtype
        return torch.get_default_dtype()

    @abstractmethod
    def predict_action(self, obs_dict: dict[str, torch.Tensor]):
        raise NotImplementedError

    def predict_action_training(self, obs_dict: dict[str, torch.Tensor]):
        return self.predict_action(obs_dict)

    def num_available_actions(self) -> Optional[int]:
        return None

    def reset(self, action_exec_horizon=None) -> None:
        del action_exec_horizon

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        raise NotImplementedError

    def get_optimizer(self, *args, **kwargs) -> torch.optim.Optimizer:
        raise NotImplementedError
