"""Versioned strict and atomic PPO tensor-core checkpoints."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from .networks import ACTION_DIM, ACTOR_KEYS, CRITIC_DIM, LIDAR_SHAPE, STATE_DIM
from .ppo import PPOTrainer

CHECKPOINT_VERSION = "instinctlab-m6-ppo-core-1"
CHECKPOINT_KEYS = frozenset((
    "version", "schema", "network_config", "ppo_config", "actor", "critic",
    "actor_optimizer", "critic_optimizer", "update_counter", "rng",
))
SCHEMA = {
    "actor_keys": ACTOR_KEYS,
    "lidar_shape": LIDAR_SHAPE,
    "state_dimension": STATE_DIM,
    "critic_dimension": CRITIC_DIM,
    "action_dimension": ACTION_DIM,
}


class CheckpointError(ValueError):
    """Raised when a checkpoint fails any strict schema or tensor check."""


def _finite_tree(value: Any, path: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all().item()):
                raise CheckpointError(f"nonfinite checkpoint tensor at {path}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")


def _validate_model_state(label: str, saved: object, expected: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(saved, dict) or set(saved) != set(expected):
        raise CheckpointError(f"{label} state keys do not match exactly")
    for name, expected_tensor in expected.items():
        saved_tensor = saved[name]
        if not isinstance(saved_tensor, torch.Tensor):
            raise CheckpointError(f"{label}.{name} must be a tensor")
        if saved_tensor.shape != expected_tensor.shape or saved_tensor.dtype != expected_tensor.dtype:
            raise CheckpointError(f"{label}.{name} shape/dtype mismatch")
        if not bool(torch.isfinite(saved_tensor).all().item()):
            raise CheckpointError(f"{label}.{name} is nonfinite")
    return saved


def _validate_optimizer_state(label: str, saved: object, trainer_optimizer: torch.optim.Optimizer) -> dict[str, object]:
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise CheckpointError(f"{label} optimizer must have exact state and param_groups keys")
    state = saved["state"]
    groups = saved["param_groups"]
    if not isinstance(state, dict) or not isinstance(groups, list):
        raise CheckpointError(f"{label} optimizer structure is invalid")
    expected = trainer_optimizer.state_dict()
    if len(groups) != len(expected["param_groups"]):
        raise CheckpointError(f"{label} optimizer parameter-group count mismatch")
    saved_indices: list[int] = []
    expected_indices: list[int] = []
    for saved_group, expected_group in zip(groups, expected["param_groups"], strict=True):
        if not isinstance(saved_group, dict) or set(saved_group) != set(expected_group):
            raise CheckpointError(f"{label} optimizer parameter-group schema mismatch")
        if not isinstance(saved_group["params"], list):
            raise CheckpointError(f"{label} optimizer params must be a list")
        saved_indices.extend(saved_group["params"])
        expected_indices.extend(expected_group["params"])
        for key in expected_group:
            if key != "params" and saved_group[key] != expected_group[key]:
                raise CheckpointError(f"{label} optimizer hyperparameter mismatch at {key}")
    if saved_indices != expected_indices or set(state) != set(saved_indices):
        raise CheckpointError(f"{label} optimizer state coverage/order mismatch")
    parameters = [parameter for group in trainer_optimizer.param_groups for parameter in group["params"]]
    if len(parameters) != len(saved_indices):
        raise CheckpointError(f"{label} optimizer parameter count mismatch")
    for index, parameter in zip(saved_indices, parameters, strict=True):
        item = state[index]
        if not isinstance(item, dict) or set(item) != {"step", "exp_avg", "exp_avg_sq"}:
            raise CheckpointError(f"{label} optimizer Adam state schema mismatch")
        for moment in ("exp_avg", "exp_avg_sq"):
            tensor = item[moment]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise CheckpointError(f"{label} optimizer {moment} shape/dtype mismatch")
        step = item["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not step.is_floating_point():
            raise CheckpointError(f"{label} optimizer step must be one floating tensor")
    _finite_tree(saved, label)
    return saved


def _rng_payload() -> dict[str, object]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def save_checkpoint(path: str | Path, trainer: PPOTrainer) -> Path:
    """Atomically save complete model/optimizer/config/schema/counter/RNG state."""
    if not isinstance(trainer, PPOTrainer):
        raise TypeError("trainer must be PPOTrainer")
    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("checkpoint path must be absolute and explicit")
    if destination.suffix != ".pt" or not destination.parent.is_dir():
        raise ValueError("checkpoint parent must exist and checkpoint suffix must be .pt")
    if destination.exists() and not destination.is_file():
        raise ValueError("checkpoint destination must not be a directory")
    payload = {
        "version": CHECKPOINT_VERSION,
        "schema": SCHEMA,
        "network_config": trainer.actor.config.manifest(),
        "ppo_config": trainer.config.manifest(),
        "actor": trainer.actor.state_dict(),
        "critic": trainer.critic.state_dict(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "update_counter": trainer.update_counter,
        "rng": _rng_payload(),
    }
    _finite_tree(payload, "checkpoint")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as stream:
            temporary_name = stream.name
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return destination


def load_checkpoint(path: str | Path, trainer: PPOTrainer) -> None:
    """Load a schema-exact checkpoint with restricted PyTorch deserialization."""
    if not isinstance(trainer, PPOTrainer):
        raise TypeError("trainer must be PPOTrainer")
    source = Path(path)
    if not source.is_absolute() or not source.is_file():
        raise CheckpointError("checkpoint path must name an existing absolute file")
    device = next(trainer.actor.parameters()).device
    try:
        payload = torch.load(source, map_location=device, weights_only=True)
    except Exception as error:
        raise CheckpointError("checkpoint could not be safely deserialized") from error
    if not isinstance(payload, dict) or set(payload) != CHECKPOINT_KEYS:
        raise CheckpointError("checkpoint top-level keys do not match exactly")
    if payload["version"] != CHECKPOINT_VERSION or payload["schema"] != SCHEMA:
        raise CheckpointError("checkpoint version/schema mismatch")
    if payload["network_config"] != asdict(trainer.actor.config) or trainer.actor.config != trainer.critic.config:
        raise CheckpointError("checkpoint network config mismatch")
    if payload["ppo_config"] != asdict(trainer.config):
        raise CheckpointError("checkpoint PPO config mismatch")
    counter = payload["update_counter"]
    if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
        raise CheckpointError("checkpoint update_counter is invalid")
    actor_state = _validate_model_state("actor", payload["actor"], trainer.actor.state_dict())
    critic_state = _validate_model_state("critic", payload["critic"], trainer.critic.state_dict())
    actor_optimizer = _validate_optimizer_state("actor", payload["actor_optimizer"], trainer.actor_optimizer)
    critic_optimizer = _validate_optimizer_state("critic", payload["critic_optimizer"], trainer.critic_optimizer)
    rng = payload["rng"]
    if not isinstance(rng, dict) or set(rng) != {"cpu", "cuda"}:
        raise CheckpointError("checkpoint RNG schema mismatch")
    if not isinstance(rng["cpu"], torch.Tensor) or rng["cpu"].dtype != torch.uint8 or rng["cpu"].ndim != 1:
        raise CheckpointError("checkpoint CPU RNG state is invalid")
    if not isinstance(rng["cuda"], list) or any(
        not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1 for state in rng["cuda"]
    ):
        raise CheckpointError("checkpoint CUDA RNG state is invalid")
    if rng["cuda"] and (not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count()):
        raise CheckpointError("checkpoint CUDA RNG device count mismatch")
    _finite_tree(payload, "checkpoint")

    trainer.actor.load_state_dict(actor_state, strict=True)
    trainer.critic.load_state_dict(critic_state, strict=True)
    trainer.actor_optimizer.load_state_dict(actor_optimizer)
    trainer.critic_optimizer.load_state_dict(critic_optimizer)
    trainer.update_counter = counter
    torch.set_rng_state(rng["cpu"].cpu())
    if rng["cuda"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])


__all__ = ["CHECKPOINT_KEYS", "CHECKPOINT_VERSION", "CheckpointError", "SCHEMA", "load_checkpoint", "save_checkpoint"]
