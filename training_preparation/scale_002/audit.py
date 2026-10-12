"""Read-only CPU source, checkpoint, and public-interface audit for SCALE-002."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .budget import BudgetError, load_and_validate_proposal


COORDINATION = Path("/home/mint/rl_dev/instinctRL_coordination")
OLD_ATTEMPT = COORDINATION / "attempts/development/SCALE-001/SCALE-001-D0002"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _optimizer_steps(payload: dict[str, Any], key: str) -> tuple[int, ...]:
    steps: list[int] = []
    for state in payload[key]["state"].values():
        value = state.get("step")
        if value is not None:
            steps.append(int(value.item() if hasattr(value, "item") else value))
    if not steps:
        raise BudgetError(f"{key} contains no optimizer steps")
    return tuple(steps)


def run_audit() -> dict[str, Any]:
    """Run only read-only checks; CUDA masking is mandatory and fail-closed."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise BudgetError("set CUDA_VISIBLE_DEVICES to the empty string for the CPU-only audit")
    proposal, budgets = load_and_validate_proposal()
    pins = proposal["pins"]
    product = Path(pins["product_repository"])
    if Path(sys.executable).resolve() != Path(pins["python_path"]).resolve():
        raise BudgetError("audit must use the pinned Python interpreter")
    if _sha256(Path(sys.executable).resolve()) != pins["python_sha256"]:
        raise BudgetError("Python interpreter hash mismatch")

    product_identity = {
        "branch": _git(product, "branch", "--show-current"),
        "commit": _git(product, "rev-parse", "HEAD"),
        "tree": _git(product, "rev-parse", "HEAD^{tree}"),
        "status": _git(product, "status", "--porcelain"),
    }
    expected_identity = {
        "branch": pins["product_branch"], "commit": pins["product_commit"],
        "tree": pins["product_tree"], "status": "",
    }
    if product_identity != expected_identity:
        raise BudgetError(f"product identity mismatch: {product_identity!r}")

    actual_sources = {relative: _sha256(product / relative) for relative in pins["source_sha256"]}
    if actual_sources != pins["source_sha256"]:
        raise BudgetError("protected product source hash mismatch")
    environment_source = (product / "isaaclab-training/instinctlab/envs/taslab_direct_env.py").read_text(encoding="utf-8")
    environment_contract_tokens = (
        'InteractiveSceneCfg(num_envs=2, env_spacing=4.0, replicate_physics=True)',
        'mesh_prim_paths=["/World/MID360StaticScene"]',
        'commanded_body_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)',
        'self._ray_caster.update(UPDATE_PERIOD_S, force_recompute=True)',
        'commanded_body_velocity_m_s=self._command_b',
        'root[:, :3] += self.scene.env_origins[indices]',
    )
    for token in environment_contract_tokens:
        if token not in environment_source:
            raise BudgetError(f"frozen environment contract token is absent: {token}")

    checkpoint_path = Path(proposal["initialization"]["checkpoint_path"])
    if _sha256(checkpoint_path) != proposal["initialization"]["checkpoint_sha256"]:
        raise BudgetError("checkpoint hash mismatch")

    import torch
    from instinctlab.training.ppo import PPOConfig, deterministic_minibatches
    from instinctlab.training.rollout import RolloutConfig

    if torch.cuda.is_available():
        raise BudgetError("CUDA unexpectedly remained visible during CPU audit")
    full_config = RolloutConfig(horizon=16, batch_size=32)
    tail_config = RolloutConfig(horizon=2, batch_size=32)
    ppo_config = PPOConfig(**proposal["ppo_config"])
    if ppo_config.manifest() != proposal["ppo_config"]:
        raise BudgetError("public PPOConfig normalized to different values")
    full_batches = deterministic_minibatches(512, 64, 1, 1701 + 250)
    tail_batches = deterministic_minibatches(64, 64, 1, 1701 + 1953)
    full_coverage = torch.cat(full_batches).sort().values.tolist()
    tail_coverage = torch.cat(tail_batches).sort().values.tolist()
    if full_coverage != list(range(512)) or tail_coverage != list(range(64)):
        raise BudgetError("public minibatch API did not provide lossless sample coverage")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required_checkpoint_keys = {
        "version", "schema", "network_config", "ppo_config", "actor", "critic",
        "actor_optimizer", "critic_optimizer", "update_counter", "rng",
    }
    if set(payload) != required_checkpoint_keys:
        raise BudgetError("checkpoint key set mismatch")
    normalized_network = json.loads(json.dumps(payload["network_config"]))
    if payload["update_counter"] != 250 or payload["ppo_config"] != proposal["ppo_config"] or normalized_network != proposal["network_config"]:
        raise BudgetError("checkpoint configuration/counter mismatch")
    actor_steps = _optimizer_steps(payload, "actor_optimizer")
    critic_steps = _optimizer_steps(payload, "critic_optimizer")
    if set(actor_steps) != {2000} or set(critic_steps) != {2000}:
        raise BudgetError("checkpoint optimizer step mismatch")
    if set(payload["rng"]) != {"cpu", "cuda"} or len(payload["rng"]["cuda"]) != 1:
        raise BudgetError("checkpoint RNG state mismatch")

    old_entry = OLD_ATTEMPT / "formal_scale_training.py"
    old_manifest = OLD_ATTEMPT / "training_manifest.json"
    old_postrun = OLD_ATTEMPT / "POSTRUN_AUDIT.json"
    if _sha256(old_entry) != pins["scale_001_entry_sha256"] or _sha256(old_manifest) != pins["scale_001_manifest_sha256"]:
        raise BudgetError("SCALE-001 source identity mismatch")
    if _sha256(old_postrun) != pins["scale_001_postrun_audit_sha256"]:
        raise BudgetError("SCALE-001 postrun audit identity mismatch")
    postrun = json.loads(old_postrun.read_text(encoding="utf-8"))
    estimate = proposal["resource_estimate"]
    if postrun["peak_cuda_memory_allocated_bytes"] != estimate["source_peak_cuda_allocated_bytes"]:
        raise BudgetError("source peak CUDA memory fact mismatch")
    if postrun["minimum_free_cuda_memory_bytes"] != estimate["source_minimum_free_cuda_memory_bytes"]:
        raise BudgetError("source minimum free CUDA memory fact mismatch")
    old_text = old_entry.read_text(encoding="utf-8")
    for binding in ("AppLauncher", "128000", 'f"update_{update:04d}.pt"', "attempt != Path(__file__).resolve().parent"):
        if binding not in old_text:
            raise BudgetError(f"expected old-entry binding is absent: {binding}")

    eval_scene = COORDINATION / "evaluation_workloads/eval_002/scene.py"
    eval_terminal = COORDINATION / "attempts/development/EVAL-002/EVAL-002-D0001/TERMINAL_MANIFEST.json"
    eval_analysis = COORDINATION / "analyses/EVAL-002/EVAL-002-A0001/ANALYSIS.json"
    eval_approval = COORDINATION / "commander/reviews/EVAL-002_A0001_APPROVAL.md"
    eval_hashes = {
        "scene": _sha256(eval_scene), "terminal_manifest": _sha256(eval_terminal),
        "analysis": _sha256(eval_analysis), "approval": _sha256(eval_approval),
    }
    expected_eval_hashes = {
        "scene": pins["eval_002_scene_sha256"],
        "terminal_manifest": pins["eval_002_terminal_manifest_sha256"],
        "analysis": pins["eval_002_analysis_sha256"],
        "approval": pins["eval_002_approval_sha256"],
    }
    if eval_hashes != expected_eval_hashes:
        raise BudgetError("EVAL-002 evidence identity mismatch")
    eval_scene_text = eval_scene.read_text(encoding="utf-8")
    for token in ("cfg.scene.num_envs = 2", "result['numerical_rank'] != [3,3]", "UsdPhysics.CollisionAPI.Apply"):
        if token not in eval_scene_text:
            raise BudgetError(f"EVAL-002 two-environment geometry token is absent: {token}")

    cli_path = COORDINATION / "evaluation_tooling/cli.py"
    cli_at_certified = _git(COORDINATION, "show", f"{pins['certified_tooling_commit']}:evaluation_tooling/cli.py")
    if cli_path.read_text(encoding="utf-8").rstrip("\n") != cli_at_certified.rstrip("\n"):
        raise BudgetError("current CLI bytes differ from the certified commit")
    for command in ('"preflight"', '"run"', '"run-command"', '"seal"', '"inspect"'):
        if command not in cli_at_certified:
            raise BudgetError(f"certified CLI command missing: {command}")
    future_entry = Path(proposal["future_execution"]["required_entry_path"])
    future_attempt = Path(proposal["future_execution"]["output_attempt_path"])
    if future_entry.exists() or future_attempt.exists():
        raise BudgetError("proposal incorrectly describes a future entry/attempt that already exists")

    return {
        "schema_version": "scale-002-cpu-audit.v1",
        "status": "PASS",
        "cuda_visible_devices": "EMPTY",
        "cuda_available": False,
        "product_identity": product_identity,
        "source_sha256": actual_sources,
        "frozen_environment_source": {
            "training_scene": "one z=0 plane mesh",
            "default_num_envs": 2,
            "default_env_spacing_m": 4.0,
            "default_commanded_body_velocity_m_s": [0.0, 0.0, 0.0],
            "observation_source": "one forced MID360 update in _get_observations",
            "reset_translation_source": "InteractiveScene.env_origins",
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "update_counter": payload["update_counter"],
            "actor_optimizer_parameter_states": len(actor_steps),
            "critic_optimizer_parameter_states": len(critic_steps),
            "optimizer_step": 2000,
            "restored_by_product_api": ["actor", "critic", "actor_optimizer", "critic_optimizer", "update_counter", "cpu_rng", "cuda_rng"],
            "environment_state_present": False,
        },
        "public_interface": {
            "full_rollout": {"horizon": full_config.horizon, "batch_size": full_config.batch_size, "samples": 512, "minibatches": len(full_batches)},
            "tail_rollout": {"horizon": tail_config.horizon, "batch_size": tail_config.batch_size, "samples": 64, "minibatches": len(tail_batches)},
            "sample_drop_or_padding": False,
            "optimizer_update_called": False,
        },
        "old_entry": {
            "sha256": _sha256(old_entry),
            "directly_reusable": False,
            "bindings": ["import-time AppLauncher", "128000-transition manifest", "attempt-directory self-binding", "fixed four-digit update checkpoint scheme"],
        },
        "scale_001_resource_source": {
            "postrun_audit_sha256": _sha256(old_postrun),
            "peak_cuda_memory_allocated_bytes": postrun["peak_cuda_memory_allocated_bytes"],
            "minimum_free_cuda_memory_bytes": postrun["minimum_free_cuda_memory_bytes"],
        },
        "certified_cli": {
            "commit": pins["certified_tooling_commit"],
            "current_bytes_match_certified_commit": True,
            "commands": ["preflight", "run", "run-command", "seal", "inspect"],
            "generic_command_lifecycle_available": True,
            "training_specific_entry_available": False,
            "reviewed_successor_training_manifest_available": False,
        },
        "eval_002_evidence": {
            "sha256": eval_hashes,
            "proves": "two environments each had initial floor/x-wall/y-wall hits and affine rank 3",
            "does_not_prove": "the proposed 32-environment translated geometry or visibility",
        },
        "budget": budgets,
        "runtime_claims": {
            "thirty_two_environment_corner_visibility": "NOT_PROVEN_REQUIRES_FUTURE_RUNTIME_PREFLIGHT",
            "proposed_dynamic_command_interface": "NOT_IMPLEMENTED",
            "training_performed": False,
            "simulator_process_starts": 0,
            "environment_transitions": 0,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
        },
    }


def main() -> int:
    print(json.dumps(run_audit(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
