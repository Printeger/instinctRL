"""Strict source-only configuration loading for SCALE-003."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


CONFIG_PATH = Path(__file__).with_name("candidate_config.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def loads_strict_json(text: str, *, source: str = "JSON") -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ConfigError(f"non-standard JSON constant {token!r}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConfigError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{source}: root must be an object")
    return value


def load_strict_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(str(exc)) from exc
    return loads_strict_json(text, source=str(path))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate_config(value: dict[str, Any]) -> None:
    expected_top = {"schema_version", "status", "order", "proposal", "paths", "pins", "training", "scene", "commands", "evaluation", "actor_boundary", "outputs"}
    _require(set(value) == expected_top, "configuration fields differ from schema")
    _require(value["schema_version"] == "scale-003-workload-config.v1", "configuration schema mismatch")
    _require(value["status"] == "SOURCE_CANDIDATE_NOT_AUTHORIZED_FOR_LIVE_EXECUTION", "configuration overclaims live status")
    _require(value["order"] == {
        "order_id": "SCALE-003", "order_revision": 0, "training_authorized": False,
        "evaluation_authorized": False, "runtime_process_starts": 0,
        "environment_transitions": 0, "optimizer_updates": 0, "real_checkpoint_writes": False,
    }, "current zero-runtime Order is misstated")
    proposal = value["proposal"]
    _require(sha256_file(Path(proposal["path"])) == proposal["sha256"], "approved proposal identity mismatch")
    pins = value["pins"]
    _require(
        pins["raycaster_geometry_comparison_standard"] == "forward-reconstruction-v1",
        "raycaster geometry comparison standard mismatch",
    )
    product_source_root = Path(value["paths"]["product_repository"]) / "isaaclab-training"
    _require(
        all(sha256_file(product_source_root / relative) == digest for relative, digest in pins["product_source_sha256"].items()),
        "pinned product source identity mismatch",
    )
    _require(sha256_file(Path(pins["eval_002_scene_path"])) == pins["eval_002_scene_sha256"], "EVAL-002 scene identity mismatch")
    _require(sha256_file(Path(pins["eval_002_gpu_gate_path"])) == pins["eval_002_gpu_gate_sha256"], "EVAL-002 coexistence gate identity mismatch")
    nvidia_smi = Path(pins["nvidia_smi_path"])
    _require(nvidia_smi.is_file() and not nvidia_smi.is_symlink() and sha256_file(nvidia_smi) == pins["nvidia_smi_sha256"], "nvidia-smi identity/type mismatch")
    training = value["training"]
    _require(training["num_envs"] == 32 and training["full_rollout_horizon"] == 16, "full rollout configuration mismatch")
    _require(training["full_rollout_count"] * 32 * 16 + training["tail_samples"] == training["additional_transitions"] == 872000, "transition accounting mismatch")
    _require(training["initial_transitions"] + training["additional_transitions"] == training["target_cumulative_transitions"] == 1000000, "cumulative target mismatch")
    _require(training["new_updates"] == training["full_rollout_count"] + 1 == 1704, "update count mismatch")
    _require(training["initial_update_counter"] + training["new_updates"] == training["target_update_counter"] == 1954, "update counter mismatch")
    _require(training["initial_optimizer_step"] + training["new_optimizer_steps"] == training["target_optimizer_step"] == 15625, "optimizer accounting mismatch")
    _require(training["policy_steps_per_environment"] * training["decimation"] == training["physics_steps_per_environment"] == 545000, "physics accounting mismatch")
    checkpoints = training["checkpoint_counters"]
    regular = list(range(checkpoints["regular_first"], checkpoints["regular_last"] + 1, checkpoints["regular_interval"]))
    _require(len(regular) == 68 and checkpoints["pre_tail"] == 1953 and checkpoints["final"] == 1954 and checkpoints["total"] == 70, "checkpoint schedule mismatch")
    scene = value["scene"]
    _require(scene["mesh_prim_path"] == "/World/MID360StaticScene/Scale003CombinedMesh", "mesh prim must be unique and explicit")
    _require(scene["wall_colliders"] == 64 and scene["sensed_mesh_collision"] is False, "collision boundary mismatch")
    _require(scene["normal_jacobian_rank_required"] == 3 and scene["point_cloud_affine_rank_is_gate"] is False, "observability method mismatch")
    _require(scene["mid360_max_range_m"] == 40.0 and training["env_spacing_m"] == 100.0, "range/spacing mismatch")
    commands = value["commands"]
    _require(commands["phase_1_policy_steps"] + commands["phase_2_complete_cycles"] * 8 * commands["hold_policy_steps"] + commands["phase_2_final_partial_steps"] == training["policy_steps_per_environment"], "global command clock mismatch")
    _require(len(commands["phase_2_cycle"]) == 8 and commands["reset_restarts_clock"] is False and commands["domain_randomization"] == "none", "command method mismatch")
    evaluation = value["evaluation"]
    _require(evaluation["cases"] == ["station", "tracking"] and evaluation["process_starts"] == 2, "evaluation cases mismatch")
    _require(evaluation["policy_steps_per_process"] * evaluation["num_envs"] == evaluation["transitions_per_process"] == 512, "evaluation process budget mismatch")
    _require(evaluation["transitions_per_process"] * evaluation["process_starts"] == evaluation["total_transitions"] == 1024, "evaluation total mismatch")
    _require(evaluation["target_checkpoint_counter"] == 1954 and evaluation["performance_gate"] is False, "evaluation checkpoint/method mismatch")
    for case in evaluation["cases"]:
        _require(Path(value["paths"][f"{case}_evaluation_entry"]).is_file(), f"{case} evaluation entry is absent")
        _require(Path(value["paths"][f"{case}_evaluation_config"]).is_file(), f"{case} evaluation config is absent")
    _require(value["actor_boundary"]["keys"] == ["lidar_grid", "state_vec"], "actor key boundary mismatch")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    value = load_strict_json(path)
    validate_config(value)
    return value


def load_evaluation_case_config(case: str) -> dict[str, Any]:
    if case not in {"station", "tracking"}:
        raise ConfigError("unknown evaluation case")
    value = load_strict_json(CONFIG_PATH.with_name(f"evaluation_{case}.json"))
    expected = {
        "schema_version": "scale-003-evaluation-case.v1", "case": case,
        "fixture": "EVAL-002 global floor-plus-corner at 4m spacing",
        "environment_seed": 20260905, "inference_seed": 20260905,
        "num_envs": 2, "policy_steps": 256, "environment_transitions": 512,
        "action_mode": "deterministic_mean", "checkpoint_counter": 1954,
        "performance_gate": False,
    }
    if value != expected:
        raise ConfigError(f"{case} evaluation config differs from the reviewed exact case")
    return value
