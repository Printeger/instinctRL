"""Pure-tensor M6 task contracts."""

from .critic import CRITIC_DIMENSION, CRITIC_FIELD_ORDER, CriticState, build_critic_state
from .rewards import REWARD_COMPONENT_KEYS, REWARD_DIAGNOSTIC_KEYS, RewardConfig, RewardResult, compute_reward
from .termination import (
    TERMINATION_ABOVE_HEIGHT,
    TERMINATION_BELOW_HEIGHT,
    TERMINATION_COLLISION,
    TERMINATION_MASK_KEYS,
    TERMINATION_NONE,
    TERMINATION_TIMEOUT,
    TerminationConfig,
    TerminationResult,
    compute_termination,
)

__all__ = [
    "CRITIC_DIMENSION", "CRITIC_FIELD_ORDER", "CriticState", "REWARD_COMPONENT_KEYS",
    "REWARD_DIAGNOSTIC_KEYS", "RewardConfig", "RewardResult", "TERMINATION_ABOVE_HEIGHT",
    "TERMINATION_BELOW_HEIGHT", "TERMINATION_COLLISION", "TERMINATION_MASK_KEYS",
    "TERMINATION_NONE", "TERMINATION_TIMEOUT", "TerminationConfig", "TerminationResult",
    "build_critic_state", "compute_reward", "compute_termination",
]
