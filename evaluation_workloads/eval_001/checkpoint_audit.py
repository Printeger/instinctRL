"""Read-only CPU audit of the accepted update-250 checkpoint."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from attempt_binding import binding_from_environment, bound_path, path_from_environment
from constants import ACTOR_SHA256, CHECKPOINT_PATH, CHECKPOINT_SHA256
from io_utils import atomic_canonical_replace, finite_tree


EXPECTED_KEYS = {
    "version", "schema", "network_config", "ppo_config", "actor", "critic",
    "actor_optimizer", "critic_optimizer", "update_counter", "rng",
}


def module_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def optimizer_steps(state: dict[str, Any]) -> list[int]:
    return [int(item["step"].item()) for item in state["state"].values()]


def audit_checkpoint(path: Path) -> OrderedDict[str, Any]:
    if path != CHECKPOINT_PATH or not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint path/type mismatch")
    checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("checkpoint byte hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != EXPECTED_KEYS:
        raise ValueError("checkpoint schema keys mismatch")
    expected_schema = {
        "actor_keys": ("lidar_grid", "state_vec"), "lidar_shape": (12, 59, 360),
        "state_dimension": 52, "critic_dimension": 18, "action_dimension": 4,
    }
    if payload["version"] != "instinctlab-m6-ppo-core-1" or payload["schema"] != expected_schema:
        raise ValueError("checkpoint version/schema mismatch")
    if payload["update_counter"] != 250:
        raise ValueError("checkpoint counter mismatch")
    finite_tree(payload)
    actor_digest = module_hash(payload["actor"])
    if actor_digest != ACTOR_SHA256:
        raise ValueError("checkpoint actor hash mismatch")
    actor_steps = optimizer_steps(payload["actor_optimizer"])
    critic_steps = optimizer_steps(payload["critic_optimizer"])
    if not actor_steps or not critic_steps or set(actor_steps) != {2000} or set(critic_steps) != {2000}:
        raise ValueError("optimizer steps mismatch")
    return OrderedDict((
        ("schema_version", "eval-001-checkpoint-audit.v1"),
        ("status", "PASS"),
        ("checkpoint_sha256", checkpoint_hash),
        ("schema", payload["schema"]),
        ("update_counter", payload["update_counter"]),
        ("actor_sha256", actor_digest),
        ("actor_tensor_count", len(payload["actor"])),
        ("critic_tensor_count", len(payload["critic"])),
        ("actor_optimizer_state_count", len(actor_steps)),
        ("critic_optimizer_state_count", len(critic_steps)),
        ("optimizer_steps", 2000),
        ("all_finite", True),
        ("map_location", "cpu"),
        ("weights_only", True),
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    binding = binding_from_environment()
    output = bound_path(
        binding, path_from_environment("EVAL_CHECKPOINT_AUDIT_OUTPUT"),
        "results/checkpoint_audit.json", must_exist=False,
    )
    try:
        result = audit_checkpoint(args.checkpoint)
        atomic_canonical_replace(output, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as error:
        atomic_canonical_replace(output, {
            "failure_class": "FROZEN_IDENTITY_LOSS",
            "message": f"{type(error).__name__}: {error}",
            "semantic_exit": 1,
            "status": "FAIL",
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
