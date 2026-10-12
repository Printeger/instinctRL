"""Fail-closed integer accounting for the SCALE-002 proposal.

This module performs no simulation, training, optimizer update, or checkpoint write.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


class BudgetError(ValueError):
    """Raised when a proposal would overrun, round, or mislabel its budget."""


@dataclass(frozen=True)
class Budget:
    additional_transitions: int
    cumulative_transitions: int
    full_samples: int
    full_updates: int
    tail_samples: int
    tail_horizon: int
    new_updates: int
    target_update_counter: int
    full_optimizer_steps_per_update: int
    tail_optimizer_steps: int
    new_optimizer_steps_per_actor_and_critic: int
    target_optimizer_steps_per_actor_and_critic: int
    policy_steps_per_environment: int
    physics_steps_per_environment: int
    aggregate_policy_driven_physics_transitions: int
    regular_checkpoint_count: int
    new_checkpoint_count: int
    checkpoint_counters: tuple[int, ...]

    def manifest(self) -> dict[str, Any]:
        result = asdict(self)
        result["checkpoint_counters"] = list(self.checkpoint_counters)
        return result


def _positive_integer(name: str, value: Any, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BudgetError(f"{name} must be an integer")
    if (allow_zero and value < 0) or (not allow_zero and value <= 0):
        raise BudgetError(f"{name} is outside its allowed range")
    return value


def derive_budget(
    *,
    additional_transitions: int,
    completed_transitions: int = 128_000,
    num_envs: int = 32,
    full_horizon: int = 16,
    minibatch_size: int = 64,
    epoch_count: int = 1,
    decimation: int = 20,
    initial_update_counter: int = 250,
    initial_optimizer_steps: int = 2_000,
    checkpoint_interval: int = 25,
) -> Budget:
    """Derive exact full and tail segments without dropping or adding samples."""

    additional_transitions = _positive_integer("additional_transitions", additional_transitions)
    completed_transitions = _positive_integer("completed_transitions", completed_transitions, allow_zero=True)
    num_envs = _positive_integer("num_envs", num_envs)
    full_horizon = _positive_integer("full_horizon", full_horizon)
    minibatch_size = _positive_integer("minibatch_size", minibatch_size)
    epoch_count = _positive_integer("epoch_count", epoch_count)
    decimation = _positive_integer("decimation", decimation)
    initial_update_counter = _positive_integer("initial_update_counter", initial_update_counter, allow_zero=True)
    initial_optimizer_steps = _positive_integer("initial_optimizer_steps", initial_optimizer_steps, allow_zero=True)
    checkpoint_interval = _positive_integer("checkpoint_interval", checkpoint_interval)

    full_samples = num_envs * full_horizon
    full_updates, tail_samples = divmod(additional_transitions, full_samples)
    if tail_samples % num_envs:
        raise BudgetError("exact tail cannot be expressed as whole policy steps for every environment")
    if full_samples % minibatch_size:
        raise BudgetError("full rollout is not exactly divisible by the minibatch size")
    if tail_samples and tail_samples % minibatch_size:
        raise BudgetError("tail rollout is not exactly divisible by the minibatch size")

    tail_horizon = tail_samples // num_envs
    new_updates = full_updates + int(tail_samples > 0)
    full_optimizer_steps = full_samples // minibatch_size * epoch_count
    tail_optimizer_steps = tail_samples // minibatch_size * epoch_count
    new_optimizer_steps = full_updates * full_optimizer_steps + tail_optimizer_steps
    policy_steps = full_updates * full_horizon + tail_horizon
    if policy_steps * num_envs != additional_transitions:
        raise AssertionError("lossless transition accounting failed")

    regular_new_update_ordinals = tuple(range(checkpoint_interval, full_updates + 1, checkpoint_interval))
    extra_new_update_ordinals: tuple[int, ...] = ()
    if tail_samples:
        if full_updates not in regular_new_update_ordinals:
            extra_new_update_ordinals += (full_updates,)
        extra_new_update_ordinals += (new_updates,)
    elif new_updates not in regular_new_update_ordinals:
        extra_new_update_ordinals += (new_updates,)
    checkpoint_ordinals = tuple(sorted(set(regular_new_update_ordinals + extra_new_update_ordinals)))
    checkpoint_counters = tuple(initial_update_counter + value for value in checkpoint_ordinals)

    return Budget(
        additional_transitions=additional_transitions,
        cumulative_transitions=completed_transitions + additional_transitions,
        full_samples=full_samples,
        full_updates=full_updates,
        tail_samples=tail_samples,
        tail_horizon=tail_horizon,
        new_updates=new_updates,
        target_update_counter=initial_update_counter + new_updates,
        full_optimizer_steps_per_update=full_optimizer_steps,
        tail_optimizer_steps=tail_optimizer_steps,
        new_optimizer_steps_per_actor_and_critic=new_optimizer_steps,
        target_optimizer_steps_per_actor_and_critic=initial_optimizer_steps + new_optimizer_steps,
        policy_steps_per_environment=policy_steps,
        physics_steps_per_environment=policy_steps * decimation,
        aggregate_policy_driven_physics_transitions=policy_steps * decimation * num_envs,
        regular_checkpoint_count=len(regular_new_update_ordinals),
        new_checkpoint_count=len(checkpoint_counters),
        checkpoint_counters=checkpoint_counters,
    )


def _load_strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise BudgetError(f"non-standard JSON constant {token!r}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BudgetError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant, object_pairs_hook=reject_duplicate)
    except (OSError, json.JSONDecodeError) as exc:
        raise BudgetError(f"cannot load proposal: {exc}") from exc
    if not isinstance(value, dict):
        raise BudgetError("proposal root must be an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BudgetError(message)


def validate_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    """Validate the fixed recommendation and return both exact budget definitions."""

    required_top = {
        "schema_version", "status", "authorization", "target", "initialization", "environment",
        "command_curriculum", "rollout", "network_config", "ppo_config", "checkpoint_plan",
        "end_condition", "future_execution", "future_evaluation", "resource_estimate", "pins",
        "required_successor_implementation", "user_decisions_required",
    }
    _require(set(proposal) == required_top, "proposal top-level keys differ from the fixed schema")
    _require(proposal["schema_version"] == "scale-002-training-proposal.v1", "unexpected schema version")
    _require(proposal["status"] == "PROPOSAL_REQUIRES_USER_APPROVAL", "proposal status overclaims approval")

    auth = proposal["authorization"]
    _require(auth == {
        "training_authorized": False,
        "simulator_process_starts_authorized": 0,
        "environment_transitions_authorized": 0,
        "optimizer_updates_authorized": 0,
        "checkpoint_writes_authorized": False,
    }, "current zero-runtime authorization is misstated")

    target = proposal["target"]
    _require(target == {
        "recommended_definition": "cumulative_total_1000000",
        "completed_transitions": 128000,
        "additional_transitions": 872000,
        "target_cumulative_transitions": 1000000,
        "alternative_new_transitions": 1000000,
        "alternative_cumulative_transitions": 1128000,
    }, "target definitions are incomplete, rounded, or mislabeled")

    initialization = proposal["initialization"]
    _require(initialization["mode"] == "resume_exact_checkpoint", "unexpected initialization mode")
    _require(initialization["initial_update_counter"] == 250, "initial counter must be 250")
    _require(initialization["initial_optimizer_steps_per_actor_and_critic"] == 2000, "initial optimizer steps must be 2000")
    _require(initialization["environment_state_restored"] is False, "environment state is not checkpointed")
    _require(initialization["training_rng_source"] == "checkpoint_after_environment_reset", "RNG/reset order is ambiguous")

    environment = proposal["environment"]
    _require(environment["num_envs"] == 32 and environment["decimation"] == 20, "environment size or decimation drifted")
    _require(environment["env_spacing_m"] == 100.0, "scene isolation spacing drifted")
    _require(environment["actor_keys"] == ["lidar_grid", "state_vec"], "actor measurement boundary drifted")
    _require(environment["raycaster_geometry"]["runtime_requirement"].endswith("affine rank 3"), "32-env runtime proof gate is missing")

    rollout = proposal["rollout"]
    _require(rollout == {
        "full_horizon": 16,
        "full_samples": 512,
        "tail_horizon": 2,
        "tail_samples": 64,
        "sample_rounding": "forbidden",
    }, "rollout plan would drop, add, or round samples")

    ppo = proposal["ppo_config"]
    expected_ppo = {
        "gamma": 0.99, "gae_lambda": 0.95, "clip_ratio": 0.2,
        "value_loss_coefficient": 0.5, "entropy_coefficient": 0.01,
        "actor_learning_rate": 0.0003, "critic_learning_rate": 0.0003,
        "maximum_gradient_norm": 0.5, "advantage_epsilon": 1e-08,
        "minibatch_size": 64, "epoch_count": 1, "permutation_seed": 1701,
    }
    _require(ppo == expected_ppo, "PPO configuration is not checkpoint-compatible")

    recommended = derive_budget(additional_transitions=target["additional_transitions"])
    alternative = derive_budget(additional_transitions=target["alternative_new_transitions"])
    _require(recommended.cumulative_transitions == target["target_cumulative_transitions"], "recommended cumulative target mismatch")
    _require(alternative.cumulative_transitions == target["alternative_cumulative_transitions"], "alternative cumulative target mismatch")

    curriculum = proposal["command_curriculum"]
    phase_1, phase_2 = curriculum["phase_1"], curriculum["phase_2"]
    _require(phase_1["additional_transitions"] + phase_2["additional_transitions"] == recommended.additional_transitions, "curriculum phases do not cover the exact budget")
    _require(phase_1["full_rollouts"] + phase_2["full_rollouts"] == recommended.full_updates, "curriculum full-rollout count mismatch")
    _require(phase_1["policy_steps_per_environment"] + phase_2["policy_steps_per_environment"] == recommended.policy_steps_per_environment, "curriculum policy-step count mismatch")
    _require(phase_2["complete_cycles"] * 8 * phase_2["hold_policy_steps"] + phase_2["final_partial_cycle_policy_steps"] == phase_2["policy_steps_per_environment"], "phase-2 command clock mismatch")
    _require(sum(part["policy_steps"] for part in phase_2["final_partial_cycle"]) == phase_2["final_partial_cycle_policy_steps"], "partial command cycle mismatch")
    _require(curriculum["domain_randomization"] == "none", "unapproved randomization was introduced")

    checkpoint = proposal["checkpoint_plan"]
    _require(checkpoint["expected_new_checkpoint_count"] == recommended.new_checkpoint_count == 70, "checkpoint count mismatch")
    _require(checkpoint["regular_last_new_update"] == 1700 and checkpoint["save_pre_tail_new_update"] == 1703 and checkpoint["save_final_new_update"] == 1704, "checkpoint timing mismatch")

    end = proposal["end_condition"]
    _require(end["exact_additional_transitions"] == recommended.additional_transitions, "end transition count mismatch")
    _require(end["exact_cumulative_transitions"] == recommended.cumulative_transitions, "end cumulative count mismatch")
    _require(end["target_update_counter"] == recommended.target_update_counter, "end update counter mismatch")
    _require(end["target_optimizer_steps_per_actor_and_critic"] == recommended.target_optimizer_steps_per_actor_and_critic, "end optimizer count mismatch")
    _require(end["performance_early_stop"] is False, "performance selection was introduced")

    future = proposal["future_execution"]
    _require(future["entry_status"] == "NOT_IMPLEMENTED_OR_AUTHORIZED", "future entry is falsely marked available")
    _require(future["command_status"] == "NOT_RUNNABLE_UNTIL_SUCCESSOR_ORDER", "future command overclaims authority")
    _require(future["maximum_process_starts"] == 1 and future["automatic_retries"] == 0, "training start/retry scope is not minimal")
    _require(proposal["future_evaluation"]["total_environment_transitions"] == 1024, "future evaluation budget mismatch")
    _require(len(proposal["user_decisions_required"]) == 5, "user decision list is incomplete")

    estimate = proposal["resource_estimate"]
    _require(estimate["source_attempt"] == "SCALE-001-D0002", "resource estimate source attempt mismatch")
    _require(estimate["source_peak_cuda_allocated_bytes"] == 2_978_465_280, "source peak CUDA allocation mismatch")
    _require(estimate["source_minimum_free_cuda_memory_bytes"] == 5_419_106_304, "source minimum free CUDA memory mismatch")
    _require(estimate["source_checkpoint_size_bytes"] == 1_492_651, "source checkpoint size mismatch")
    expected_seconds = estimate["source_elapsed_seconds"] * recommended.additional_transitions / estimate["source_transitions"]
    _require(abs(estimate["estimated_elapsed_seconds"] - expected_seconds) < 1e-9, "resource time extrapolation mismatch")
    _require(abs(estimate["estimated_elapsed_minutes"] - expected_seconds / 60.0) < 1e-9, "resource minute extrapolation mismatch")
    _require(estimate["estimated_new_checkpoint_bytes"] == estimate["source_checkpoint_size_bytes"] * recommended.new_checkpoint_count, "checkpoint storage estimate mismatch")

    return {"recommended": recommended.manifest(), "alternative_new_1000000": alternative.manifest()}


def load_and_validate_proposal(path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    proposal_path = Path(path) if path is not None else Path(__file__).with_name("proposal.json")
    proposal = _load_strict_json(proposal_path)
    return proposal, validate_proposal(proposal)
