"""Strict, atomic and transactional ``checkpoint-h1-v2`` persistence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from instinctlab.task.critic_h1 import CRITIC_H1_DIMENSION, CRITIC_H1_FIELD_ORDER, CRITIC_H1_SCALES
from .checkpoint import CheckpointError, _finite_tree, _validate_model_state, _validate_optimizer_state
from .networks import ACTION_DIM, ACTOR_KEYS, LIDAR_SHAPE, STATE_DIM, HistoryActor
from .networks_h1 import PrivilegedCriticH1
from .ppo import PPOTrainer

CHECKPOINT_H1_VERSION = "checkpoint-h1-v2"
CHECKPOINT_H1_KEYS = frozenset((
    "version", "schema", "network_config", "ppo_config", "method_hashes", "source_identity",
    "actor", "critic", "actor_optimizer", "critic_optimizer", "counters", "curriculum",
    "training_seed", "reset_number", "rng",
))
CHECKPOINT_H1_SCHEMA = {
    "actor_keys": ACTOR_KEYS, "lidar_shape": LIDAR_SHAPE, "state_dimension": STATE_DIM,
    "critic_dimension": CRITIC_H1_DIMENSION, "critic_field_order": CRITIC_H1_FIELD_ORDER,
    "critic_scales": CRITIC_H1_SCALES, "action_dimension": ACTION_DIM,
}


def _optimizer_step(optimizer: torch.optim.Optimizer) -> int | None:
    states = optimizer.state_dict()["state"]
    if not states:
        return None
    steps = [int(item["step"].item()) for item in states.values()]
    if len(steps) != len(states) or len(set(steps)) != 1:
        raise CheckpointError("Adam optimizer parameters do not share one exact step")
    return steps[0]


def _json_tree(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_tree(item) for item in value]
    if isinstance(value, list):
        return [_json_tree(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, list) else value


def _rng_payload(curriculum_generator: torch.Generator) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [item.clone() for item in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "python": _json_tree(random.getstate()),
        "numpy": {
            "algorithm": numpy_state[0], "keys": numpy_state[1].tolist(), "position": numpy_state[2],
            "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4],
        },
        "curriculum": curriculum_generator.get_state().clone(),
    }


def _validate_metadata(mapping: object, label: str) -> dict[str, str]:
    if not isinstance(mapping, dict) or not mapping:
        raise CheckpointError(f"{label} must be a nonempty exact string mapping")
    if any(not isinstance(key, str) or not key or not isinstance(value, str) or not value for key, value in mapping.items()):
        raise CheckpointError(f"{label} must contain nonempty strings")
    return mapping


def _validate_counter(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointError(f"{label} must be a nonnegative integer")
    return value


def save_checkpoint_h1(
    path: str | Path, trainer: PPOTrainer, *, method_hashes: Mapping[str, str],
    source_identity: Mapping[str, str], transition_counter: int, global_policy_step: int,
    curriculum: Mapping[str, Any], training_seed: int, reset_number: int,
    curriculum_generator: torch.Generator,
) -> Path:
    if not isinstance(trainer, PPOTrainer) or not isinstance(trainer.actor, HistoryActor) or not isinstance(trainer.critic, PrivilegedCriticH1):
        raise TypeError("checkpoint-h1 requires HistoryActor/PPOTrainer/PrivilegedCriticH1")
    destination = Path(path)
    if not destination.is_absolute() or destination.suffix != ".pt" or not destination.parent.is_dir():
        raise ValueError("checkpoint path must be absolute .pt with an existing parent")
    if destination.exists():
        raise FileExistsError("checkpoint-h1 never overwrites an existing checkpoint")
    update = _validate_counter(trainer.update_counter, "update_counter")
    transition = _validate_counter(transition_counter, "transition_counter")
    policy_step = _validate_counter(global_policy_step, "global_policy_step")
    seed = _validate_counter(training_seed, "training_seed")
    reset = _validate_counter(reset_number, "reset_number")
    actor_step = _optimizer_step(trainer.actor_optimizer)
    critic_step = _optimizer_step(trainer.critic_optimizer)
    if update == 0:
        if actor_step is not None or critic_step is not None:
            raise CheckpointError("cold update-0 checkpoint requires both Adam states to be empty")
    elif actor_step != critic_step or actor_step != 8 * update:
        raise CheckpointError("updated checkpoint requires both Adam counters to equal 8*update")
    payload = {
        "version": CHECKPOINT_H1_VERSION, "schema": CHECKPOINT_H1_SCHEMA,
        "network_config": trainer.actor.config.manifest(), "ppo_config": trainer.config.manifest(),
        "method_hashes": dict(method_hashes), "source_identity": dict(source_identity),
        "actor": trainer.actor.state_dict(), "critic": trainer.critic.state_dict(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(), "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "counters": {"update": update, "transition": transition, "global_policy_step": policy_step,
                     "actor_optimizer": actor_step or 0, "critic_optimizer": critic_step or 0},
        "curriculum": dict(curriculum), "training_seed": seed, "reset_number": reset,
        "rng": _rng_payload(curriculum_generator),
    }
    _validate_metadata(payload["method_hashes"], "method_hashes")
    _validate_metadata(payload["source_identity"], "source_identity")
    _finite_tree(payload, "checkpoint-h1")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as stream:
            temporary = stream.name
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, destination); temporary = None
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return destination


def load_checkpoint_h1(
    path: str | Path, trainer: PPOTrainer, *, expected_method_hashes: Mapping[str, str],
    expected_source_identity: Mapping[str, str], curriculum_generator: torch.Generator,
) -> dict[str, Any]:
    """Validate everything before mutation; rollback the trainer and all RNG on any load error."""
    if not isinstance(trainer, PPOTrainer) or not isinstance(trainer.critic, PrivilegedCriticH1):
        raise TypeError("checkpoint-h1 requires a PrivilegedCriticH1 trainer")
    source = Path(path)
    if not source.is_absolute() or not source.is_file():
        raise CheckpointError("checkpoint path must name an existing absolute file")
    try:
        payload = torch.load(source, map_location=next(trainer.actor.parameters()).device, weights_only=True)
    except Exception as error:
        raise CheckpointError("checkpoint-h1 could not be safely deserialized") from error
    if not isinstance(payload, dict) or set(payload) != CHECKPOINT_H1_KEYS:
        raise CheckpointError("checkpoint-h1 top-level keys mismatch")
    if payload["version"] != CHECKPOINT_H1_VERSION or payload["schema"] != CHECKPOINT_H1_SCHEMA:
        raise CheckpointError("checkpoint-h1 version/schema mismatch")
    if payload["network_config"] != asdict(trainer.actor.config) or trainer.actor.config != trainer.critic.config:
        raise CheckpointError("checkpoint-h1 network config mismatch")
    if payload["ppo_config"] != asdict(trainer.config):
        raise CheckpointError("checkpoint-h1 PPO config mismatch")
    if payload["method_hashes"] != dict(expected_method_hashes) or payload["source_identity"] != dict(expected_source_identity):
        raise CheckpointError("checkpoint-h1 method/source identity mismatch")
    actor = _validate_model_state("actor", payload["actor"], trainer.actor.state_dict())
    critic = _validate_model_state("critic", payload["critic"], trainer.critic.state_dict())
    counters = payload["counters"]
    if not isinstance(counters, dict) or set(counters) != {"update", "transition", "global_policy_step", "actor_optimizer", "critic_optimizer"}:
        raise CheckpointError("checkpoint-h1 counter schema mismatch")
    for key in counters:
        _validate_counter(counters[key], key)
    update = counters["update"]
    if update == 0:
        for label in ("actor_optimizer", "critic_optimizer"):
            saved = payload[label]
            expected_groups = trainer.actor_optimizer.state_dict()["param_groups"] if label == "actor_optimizer" else trainer.critic_optimizer.state_dict()["param_groups"]
            if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"} or saved["state"] != {} or saved["param_groups"] != expected_groups:
                raise CheckpointError("cold checkpoint Adam state is not exact and empty")
        actor_optimizer = payload["actor_optimizer"]; critic_optimizer = payload["critic_optimizer"]
        if counters["actor_optimizer"] != 0 or counters["critic_optimizer"] != 0:
            raise CheckpointError("cold checkpoint optimizer counters must be zero")
    else:
        actor_optimizer = _validate_optimizer_state("actor", payload["actor_optimizer"], trainer.actor_optimizer)
        critic_optimizer = _validate_optimizer_state("critic", payload["critic_optimizer"], trainer.critic_optimizer)
        if counters["actor_optimizer"] != counters["critic_optimizer"] or counters["actor_optimizer"] != 8 * update:
            raise CheckpointError("checkpoint-h1 Adam counters must equal 8*update")
    rng = payload["rng"]
    if not isinstance(rng, dict) or set(rng) != {"torch_cpu", "torch_cuda", "python", "numpy", "curriculum"}:
        raise CheckpointError("checkpoint-h1 RNG schema mismatch")
    if not isinstance(rng["torch_cpu"], torch.Tensor) or rng["torch_cpu"].dtype != torch.uint8 or not isinstance(rng["curriculum"], torch.Tensor) or rng["curriculum"].dtype != torch.uint8:
        raise CheckpointError("checkpoint-h1 CPU/curriculum RNG is invalid")
    if rng["torch_cuda"] and (not torch.cuda.is_available() or len(rng["torch_cuda"]) != torch.cuda.device_count()):
        raise CheckpointError("checkpoint-h1 CUDA RNG device mismatch")
    numpy_state = rng["numpy"]
    if not isinstance(numpy_state, dict) or set(numpy_state) != {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}:
        raise CheckpointError("checkpoint-h1 NumPy RNG schema mismatch")
    _finite_tree(payload, "checkpoint-h1")

    before = {
        "actor": deepcopy(trainer.actor.state_dict()), "critic": deepcopy(trainer.critic.state_dict()),
        "actor_optimizer": deepcopy(trainer.actor_optimizer.state_dict()),
        "critic_optimizer": deepcopy(trainer.critic_optimizer.state_dict()), "update": trainer.update_counter,
        "torch": torch.get_rng_state().clone(), "python": random.getstate(), "numpy": np.random.get_state(),
        "curriculum": curriculum_generator.get_state().clone(),
    }
    try:
        trainer.actor.load_state_dict(actor, strict=True); trainer.critic.load_state_dict(critic, strict=True)
        trainer.actor_optimizer.load_state_dict(actor_optimizer); trainer.critic_optimizer.load_state_dict(critic_optimizer)
        trainer.update_counter = update
        torch.set_rng_state(rng["torch_cpu"].cpu())
        if rng["torch_cuda"]:
            torch.cuda.set_rng_state_all([item.cpu() for item in rng["torch_cuda"]])
        random.setstate(_tuple_tree(rng["python"]))
        np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32), numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
        curriculum_generator.set_state(rng["curriculum"].cpu())
    except BaseException as error:
        trainer.actor.load_state_dict(before["actor"]); trainer.critic.load_state_dict(before["critic"])
        trainer.actor_optimizer.load_state_dict(before["actor_optimizer"]); trainer.critic_optimizer.load_state_dict(before["critic_optimizer"])
        trainer.update_counter = before["update"]; torch.set_rng_state(before["torch"])
        random.setstate(before["python"]); np.random.set_state(before["numpy"]); curriculum_generator.set_state(before["curriculum"])
        raise CheckpointError("checkpoint-h1 transactional load failed without partial mutation") from error
    return {"counters": dict(counters), "curriculum": dict(payload["curriculum"]), "training_seed": payload["training_seed"], "reset_number": payload["reset_number"]}


__all__ = [
    "CHECKPOINT_H1_KEYS", "CHECKPOINT_H1_SCHEMA", "CHECKPOINT_H1_VERSION",
    "load_checkpoint_h1", "save_checkpoint_h1",
]
