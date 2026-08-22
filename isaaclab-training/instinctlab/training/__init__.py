"""Bounded pure-tensor actor, critic, PPO, GAE, and checkpoint core."""

from .checkpoint import CHECKPOINT_VERSION, CheckpointError, load_checkpoint, save_checkpoint
from .networks import HistoryActor, NetworkConfig, PolicyOutput, PrivilegedCritic
from .ppo import GAEOutput, PPOBatch, PPOConfig, PPOTrainer, clipped_surrogate, compute_gae
from .rollout import (
    RolloutConfig, RolloutResult, RolloutSegment, RolloutTransition,
    assemble_rollout, collect_rollout, collect_transition,
)

__all__ = [
    "CHECKPOINT_VERSION", "CheckpointError", "GAEOutput", "HistoryActor", "NetworkConfig",
    "PPOBatch", "PPOConfig", "PPOTrainer", "PolicyOutput", "PrivilegedCritic",
    "RolloutConfig", "RolloutResult", "RolloutSegment", "RolloutTransition",
    "assemble_rollout", "clipped_surrogate", "collect_rollout", "collect_transition",
    "compute_gae", "load_checkpoint", "save_checkpoint",
]
