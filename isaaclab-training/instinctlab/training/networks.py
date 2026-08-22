"""Exact-schema feedforward actor and separate privileged critic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

ACTOR_KEYS = ("lidar_grid", "state_vec")
LIDAR_SHAPE = (12, 59, 360)
STATE_DIM = 52
CRITIC_DIM = 18
ACTION_DIM = 4


def _real(name: str, value: float, *, lower: float, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < lower or (upper is not None and result > upper):
        bound = f"[{lower}, {upper}]" if upper is not None else f"at least {lower}"
        raise ValueError(f"{name} must be finite and {bound}")
    return result


@dataclass(frozen=True)
class NetworkConfig:
    """Immutable bounded network and Beta-distribution configuration."""

    lidar_channels: tuple[int, int, int] = (8, 16, 16)
    state_hidden: int = 32
    actor_hidden: int = 128
    critic_hidden: tuple[int, int] = (64, 64)
    concentration_min: float = 1.0
    concentration_max: float = 30.0
    action_log_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if (
            not isinstance(self.lidar_channels, tuple)
            or len(self.lidar_channels) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.lidar_channels)
        ):
            raise ValueError("lidar_channels must be a three-item tuple of positive integers")
        if isinstance(self.state_hidden, bool) or not isinstance(self.state_hidden, int) or self.state_hidden <= 0:
            raise ValueError("state_hidden must be a positive integer")
        if isinstance(self.actor_hidden, bool) or not isinstance(self.actor_hidden, int) or self.actor_hidden <= 0:
            raise ValueError("actor_hidden must be a positive integer")
        if (
            not isinstance(self.critic_hidden, tuple)
            or len(self.critic_hidden) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.critic_hidden)
        ):
            raise ValueError("critic_hidden must be a two-item tuple of positive integers")
        minimum = _real("concentration_min", self.concentration_min, lower=1.0e-6)
        maximum = _real("concentration_max", self.concentration_max, lower=minimum)
        epsilon = _real("action_log_epsilon", self.action_log_epsilon, lower=1.0e-12, upper=0.499999)
        if maximum <= minimum:
            raise ValueError("concentration_max must be greater than concentration_min")
        object.__setattr__(self, "concentration_min", minimum)
        object.__setattr__(self, "concentration_max", maximum)
        object.__setattr__(self, "action_log_epsilon", epsilon)

    def manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyOutput:
    alpha: torch.Tensor
    beta: torch.Tensor
    mean_action: torch.Tensor


def validate_policy(policy: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Require the ordered exact M4 policy pair, float32, one device, and finite values."""
    if not isinstance(policy, Mapping):
        raise TypeError("policy must be a mapping containing only the exact M4 pair")
    if tuple(policy.keys()) != ACTOR_KEYS:
        raise ValueError(f"policy keys and order must be exactly {ACTOR_KEYS}")
    lidar = policy["lidar_grid"]
    state = policy["state_vec"]
    if not isinstance(lidar, torch.Tensor) or not isinstance(state, torch.Tensor):
        raise TypeError("both policy values must be torch.Tensor instances")
    if lidar.ndim != 4 or tuple(lidar.shape[1:]) != LIDAR_SHAPE:
        raise ValueError(f"lidar_grid must have shape (N, {LIDAR_SHAPE[0]}, {LIDAR_SHAPE[1]}, {LIDAR_SHAPE[2]})")
    batch = lidar.shape[0]
    if batch <= 0 or state.shape != (batch, STATE_DIM):
        raise ValueError(f"state_vec must have shape (N, {STATE_DIM}) with the same nonempty batch")
    if lidar.dtype != torch.float32 or state.dtype != torch.float32:
        raise TypeError("actor inputs must use exactly torch.float32")
    if lidar.device != state.device:
        raise ValueError("actor inputs must share one device")
    if not bool(torch.isfinite(lidar).all().item()) or not bool(torch.isfinite(state).all().item()):
        raise ValueError("actor inputs must be finite")
    return lidar, state


def validate_critic_input(privileged: torch.Tensor) -> torch.Tensor:
    """Require the exact separate M6 critic tensor."""
    if not isinstance(privileged, torch.Tensor):
        raise TypeError("privileged must be a torch.Tensor")
    if privileged.ndim != 2 or privileged.shape[0] <= 0 or privileged.shape[1] != CRITIC_DIM:
        raise ValueError(f"privileged must have shape (N, {CRITIC_DIM})")
    if privileged.dtype != torch.float32:
        raise TypeError("privileged must use exactly torch.float32")
    if not bool(torch.isfinite(privileged).all().item()):
        raise ValueError("privileged must be finite")
    return privileged


def _validate_action(action: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    if not isinstance(action, torch.Tensor) or action.shape != (batch, ACTION_DIM):
        raise ValueError(f"action must have shape (N, {ACTION_DIM})")
    if action.dtype != torch.float32 or action.device != device:
        raise TypeError("action must use float32 on the policy device")
    if not bool(torch.isfinite(action).all().item()):
        raise ValueError("action must be finite")
    if bool(((action < 0.0) | (action > 1.0)).any().item()):
        raise ValueError("action must remain in the closed [0, 1] environment domain")
    return action


class HistoryActor(nn.Module):
    """Feedforward four-frame actor that can receive only the exact policy mapping."""

    def __init__(self, config: NetworkConfig = NetworkConfig()) -> None:
        super().__init__()
        if not isinstance(config, NetworkConfig):
            raise TypeError("config must be NetworkConfig")
        self.config = config
        c1, c2, c3 = config.lidar_channels
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(12, c1, kernel_size=5, stride=(2, 4), padding=2),
            nn.ELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=(2, 3), padding=1),
            nn.ELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=3, padding=1),
            nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            lidar_features = int(self.lidar_encoder(torch.zeros(1, *LIDAR_SHAPE)).shape[1])
        self.lidar_feature_dimension = lidar_features
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, config.state_hidden), nn.ELU(), nn.LayerNorm(config.state_hidden)
        )
        self.trunk = nn.Sequential(
            nn.Linear(lidar_features + config.state_hidden, config.actor_hidden),
            nn.ELU(),
            nn.LayerNorm(config.actor_hidden),
            nn.Linear(config.actor_hidden, ACTION_DIM * 2),
        )

    def forward(self, policy: Mapping[str, torch.Tensor]) -> PolicyOutput:
        lidar, state = validate_policy(policy)
        # Only physical range channels are fixed-scale normalized; validity and reliability are already in [0,1].
        normalized_lidar = torch.cat(
            (lidar[:, 0::3] / 40.0, lidar[:, 1::3], lidar[:, 2::3]), dim=1
        )
        raw = self.trunk(torch.cat((self.lidar_encoder(normalized_lidar), self.state_encoder(state)), dim=-1))
        alpha_raw, beta_raw = raw.chunk(2, dim=-1)
        span = self.config.concentration_max - self.config.concentration_min
        alpha = self.config.concentration_min + span * torch.sigmoid(alpha_raw)
        beta = self.config.concentration_min + span * torch.sigmoid(beta_raw)
        if not bool(torch.isfinite(alpha).all().item()) or not bool(torch.isfinite(beta).all().item()):
            raise FloatingPointError("actor produced nonfinite Beta concentrations")
        if bool(((alpha <= 0.0) | (beta <= 0.0)).any().item()):
            raise FloatingPointError("actor produced nonpositive Beta concentrations")
        mean = alpha / (alpha + beta)
        if mean.shape != (lidar.shape[0], ACTION_DIM) or not bool(torch.isfinite(mean).all().item()):
            raise FloatingPointError("actor produced an invalid deterministic action")
        return PolicyOutput(alpha=alpha, beta=beta, mean_action=mean)

    def distribution(self, policy: Mapping[str, torch.Tensor]) -> tuple[torch.distributions.Beta, PolicyOutput]:
        output = self(policy)
        return torch.distributions.Beta(output.alpha, output.beta, validate_args=True), output

    def sample(self, policy: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, output = self.distribution(policy)
        action = distribution.sample()
        action = _validate_action(action, output.alpha.shape[0], output.alpha.device)
        log_probability = self.log_probability_from_output(output, action)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        if entropy.shape != (action.shape[0], 1) or not bool(torch.isfinite(entropy).all().item()):
            raise FloatingPointError("actor produced invalid entropy")
        return action, log_probability, entropy

    def log_probability(self, policy: Mapping[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        output = self(policy)
        return self.log_probability_from_output(output, action)

    def log_probability_from_output(self, output: PolicyOutput, action: torch.Tensor) -> torch.Tensor:
        action = _validate_action(action, output.alpha.shape[0], output.alpha.device)
        # The sole declared clamp is an open-boundary epsilon for Beta log-probability evaluation.
        evaluation_action = action.clamp(self.config.action_log_epsilon, 1.0 - self.config.action_log_epsilon)
        result = torch.distributions.Beta(output.alpha, output.beta, validate_args=True).log_prob(
            evaluation_action
        ).sum(dim=-1, keepdim=True)
        if result.shape != (action.shape[0], 1) or not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("actor produced invalid log probability")
        return result

    def entropy_from_output(self, output: PolicyOutput) -> torch.Tensor:
        result = torch.distributions.Beta(output.alpha, output.beta, validate_args=True).entropy().sum(
            dim=-1, keepdim=True
        )
        if result.shape != (output.alpha.shape[0], 1) or not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("actor produced invalid entropy")
        return result

    def parameter_manifest(self) -> dict[str, object]:
        return {
            "class": type(self).__name__,
            "actor_keys": ACTOR_KEYS,
            "lidar_shape": LIDAR_SHAPE,
            "state_dimension": STATE_DIM,
            "action_dimension": ACTION_DIM,
            "lidar_feature_dimension": self.lidar_feature_dimension,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "layers": [(name, tuple(parameter.shape)) for name, parameter in self.named_parameters()],
            "config": self.config.manifest(),
        }


class PrivilegedCritic(nn.Module):
    """One value critic consuming only the exact M6 privileged branch."""

    def __init__(self, config: NetworkConfig = NetworkConfig()) -> None:
        super().__init__()
        if not isinstance(config, NetworkConfig):
            raise TypeError("config must be NetworkConfig")
        self.config = config
        first, second = config.critic_hidden
        self.network = nn.Sequential(
            nn.Linear(CRITIC_DIM, first), nn.ELU(), nn.LayerNorm(first),
            nn.Linear(first, second), nn.ELU(), nn.LayerNorm(second),
            nn.Linear(second, 1),
        )

    def forward(self, privileged: torch.Tensor) -> torch.Tensor:
        result = self.network(validate_critic_input(privileged))
        if result.shape != (privileged.shape[0], 1) or not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("critic produced invalid value")
        return result

    def parameter_manifest(self) -> dict[str, object]:
        return {
            "class": type(self).__name__,
            "critic_dimension": CRITIC_DIM,
            "output_dimension": 1,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "layers": [(name, tuple(parameter.shape)) for name, parameter in self.named_parameters()],
            "config": self.config.manifest(),
        }


__all__ = [
    "ACTION_DIM", "ACTOR_KEYS", "CRITIC_DIM", "HistoryActor", "LIDAR_SHAPE",
    "NetworkConfig", "PolicyOutput", "PrivilegedCritic", "STATE_DIM",
    "validate_critic_input", "validate_policy",
]
