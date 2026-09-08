"""Explicit 21-dimensional critic network for ``critic-h1-v2``."""

from __future__ import annotations

import torch
from torch import nn

from instinctlab.task.critic_h1 import CRITIC_H1_DIMENSION
from .networks import NetworkConfig, PrivilegedCritic


def validate_critic_h1_input(privileged: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(privileged, torch.Tensor) or privileged.ndim != 2
        or privileged.shape[0] <= 0 or privileged.shape[1] != CRITIC_H1_DIMENSION
        or privileged.dtype != torch.float32 or not bool(torch.isfinite(privileged).all().item())
    ):
        raise ValueError(f"privileged must be finite float32 [N,{CRITIC_H1_DIMENSION}]")
    return privileged


class PrivilegedCriticH1(PrivilegedCritic):
    critic_dimension = CRITIC_H1_DIMENSION

    def __init__(self, config: NetworkConfig = NetworkConfig()) -> None:
        nn.Module.__init__(self)
        if not isinstance(config, NetworkConfig):
            raise TypeError("config must be NetworkConfig")
        self.config = config
        first, second = config.critic_hidden
        self.network = nn.Sequential(
            nn.Linear(CRITIC_H1_DIMENSION, first), nn.ELU(), nn.LayerNorm(first),
            nn.Linear(first, second), nn.ELU(), nn.LayerNorm(second), nn.Linear(second, 1),
        )

    def forward(self, privileged: torch.Tensor) -> torch.Tensor:
        result = self.network(validate_critic_h1_input(privileged))
        if result.shape != (privileged.shape[0], 1) or not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("critic-h1 produced invalid value")
        return result

    def parameter_manifest(self) -> dict[str, object]:
        result = super().parameter_manifest()
        result["class"] = type(self).__name__
        result["critic_dimension"] = CRITIC_H1_DIMENSION
        result["critic_version"] = "critic-h1-v2"
        return result


__all__ = ["PrivilegedCriticH1", "validate_critic_h1_input"]
