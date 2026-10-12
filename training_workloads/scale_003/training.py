"""Dependency-injected exact training orchestration; CPU tests use inert fakes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from .config import load_config
from .schedule import Segment, build_segments, checkpoint_counters
from .semantic import require_sha256, validate_training_preflight


class TrainingBackend(Protocol):
    def bootstrap(self, seed: int) -> None: ...
    def create_environment(self) -> None: ...
    def reset_environment(self, seed: int) -> None: ...
    def create_trainer(self) -> None: ...
    def load_checkpoint(self, path: Path, sha256: str) -> None: ...
    def run_initial_preflight(self) -> Mapping[str, Any]: ...
    def prepare_segment(self, segment: Segment) -> None: ...
    def collect_segment(self, segment: Segment) -> Any: ...
    def update(self, batch: Any) -> Mapping[str, Any]: ...
    def optimizer_steps(self) -> tuple[int, int]: ...
    def save_checkpoint(self, counter: int, kind: str) -> Mapping[str, Any]: ...
    def final_reload_audit(self, counter: int, optimizer_step: int) -> Mapping[str, Any]: ...
    def preserve_partial(self, error: BaseException) -> None: ...


class Progress(Protocol):
    def write(self, **fields: Any) -> None: ...
    def fail(self, error: BaseException, **fields: Any) -> None: ...


def _coverage(metrics: Mapping[str, Any], samples: int) -> None:
    if metrics.get("sample_count") != samples or metrics.get("coverage_min") != 1 or metrics.get("coverage_max") != 1:
        raise ValueError("PPO update did not cover every segment sample exactly once")


def run_training(backend: TrainingBackend, progress: Progress) -> dict[str, Any]:
    """Run the reviewed schedule through a live or inert backend exactly once."""

    config = load_config()
    training = config["training"]
    segments = build_segments()
    completed_transitions = 0
    completed_updates = 0
    checkpoints: list[dict[str, Any]] = []
    initial_preflight: Mapping[str, Any] = {}
    failure_stage = "bootstrap"
    try:
        backend.bootstrap(training["bootstrap_seed"])
        failure_stage = "environment_creation"
        backend.create_environment()
        failure_stage = "environment_reset"
        backend.reset_environment(training["environment_reset_seed"])
        failure_stage = "trainer_creation"
        backend.create_trainer()
        failure_stage = "checkpoint_load"
        backend.load_checkpoint(Path(config["pins"]["checkpoint_path"]), config["pins"]["checkpoint_sha256"])
        failure_stage = "initial_preflight"
        initial_preflight = backend.run_initial_preflight()
        progress.write(
            status="RUNNING", stage="initialized", attempted_transitions=0,
            completed_transitions=0, completed_updates=0,
            update_counter=training["initial_update_counter"], optimizer_step=training["initial_optimizer_step"],
            initialization_or_reset_policy_steps=0,
        )
        for segment in segments:
            failure_stage = "segment_prepare"
            backend.prepare_segment(segment)
            progress.write(
                status="RUNNING", stage="rollout", segment_ordinal=segment.ordinal,
                attempted_transitions=completed_transitions,
                completed_transitions=completed_transitions, completed_updates=completed_updates,
                update_counter=training["initial_update_counter"] + completed_updates,
                optimizer_step=segments[completed_updates - 1].target_optimizer_step if completed_updates else training["initial_optimizer_step"],
                initialization_or_reset_policy_steps=0,
            )
            failure_stage = "rollout_collection"
            collected = backend.collect_segment(segment)
            if not isinstance(collected, Mapping) or collected.get("sample_count") != segment.samples or "batch" not in collected:
                raise ValueError("rollout backend returned a wrong exact sample count")
            progress.write(
                status="RUNNING", stage="collected", segment_ordinal=segment.ordinal,
                attempted_transitions=completed_transitions + segment.samples,
                completed_transitions=completed_transitions + segment.samples,
                completed_updates=completed_updates,
                update_counter=training["initial_update_counter"] + completed_updates,
                optimizer_step=segments[completed_updates - 1].target_optimizer_step if completed_updates else training["initial_optimizer_step"],
                initialization_or_reset_policy_steps=0,
            )
            failure_stage = "ppo_update"
            metrics = backend.update(collected["batch"])
            _coverage(metrics, segment.samples)
            completed_transitions += segment.samples
            completed_updates += 1
            if metrics.get("update_counter") != segment.update_counter:
                raise ValueError("PPO update counter differs from the segment schedule")
            actor_step, critic_step = backend.optimizer_steps()
            if (actor_step, critic_step) != (segment.target_optimizer_step, segment.target_optimizer_step):
                raise ValueError("optimizer step differs from the segment schedule")
            if segment.checkpoint_kind is not None:
                failure_stage = "checkpoint_save"
                row = dict(backend.save_checkpoint(segment.update_counter, segment.checkpoint_kind))
                if row.get("counter") != segment.update_counter or row.get("kind") != segment.checkpoint_kind:
                    raise ValueError("checkpoint receipt differs from schedule")
                checkpoints.append(row)
            progress.write(
                status="RUNNING", stage="updated", segment_ordinal=segment.ordinal,
                attempted_transitions=completed_transitions, completed_transitions=completed_transitions,
                completed_updates=completed_updates, update_counter=segment.update_counter,
                optimizer_step=segment.target_optimizer_step, initialization_or_reset_policy_steps=0,
            )
        if tuple(row["counter"] for row in checkpoints) != checkpoint_counters():
            raise ValueError("checkpoint receipts do not cover the exact schedule")
        failure_stage = "final_reload_audit"
        reload_audit = dict(backend.final_reload_audit(1954, 15625))
        result = {
            "schema_version": "scale-003-training-semantic.v1", "status": "PASS",
            "initial_checkpoint_counter": 250, "initial_optimizer_step": 2000,
            "additional_transitions": completed_transitions,
            "cumulative_transitions": 128000 + completed_transitions,
            "full_rollouts": 1703, "tail_horizon": 2, "tail_samples": 64,
            "new_updates": completed_updates, "target_update_counter": 250 + completed_updates,
            "new_optimizer_steps_per_actor_and_critic": 13625,
            "target_optimizer_step_per_actor_and_critic": 15625,
            "policy_steps_per_environment": 27250, "physics_steps_per_environment": 545000,
            "checkpoint_count": len(checkpoints), "checkpoint_counters": [row["counter"] for row in checkpoints],
            "initial_preflight": dict(initial_preflight), "final_reload_audit": reload_audit,
            "environment_state_restored": False, "sample_drop_padding_or_roundup": False,
            "performance_early_stop": False, "checkpoints": checkpoints,
        }
        validate_training_semantic(result)
        progress.write(
            status="PASS", stage="complete", attempted_transitions=completed_transitions,
            completed_transitions=completed_transitions,
            completed_updates=completed_updates,
            update_counter=1954, optimizer_step=15625, initialization_or_reset_policy_steps=0,
        )
        return result
    except BaseException as error:
        latest = progress.latest if hasattr(progress, "latest") and isinstance(progress.latest, Mapping) else {}
        accounting: dict[str, Any] = {}
        failure_accounting = getattr(backend, "failure_accounting", None)
        if callable(failure_accounting):
            try:
                observed = failure_accounting()
                if isinstance(observed, Mapping):
                    accounting = dict(observed)
            except BaseException:
                accounting = {"optimizer_state_status": "UNAVAILABLE_DURING_FAILURE_AUDIT"}
        attempted = max(completed_transitions, int(latest.get("attempted_transitions", completed_transitions)))
        completed = max(completed_transitions, int(latest.get("completed_transitions", completed_transitions)))
        confirmed_updates = max(completed_updates, int(accounting.get("completed_updates", completed_updates)))
        last_checkpoint = checkpoints[-1] if checkpoints else None
        accounted_counter = accounting.get("last_checkpoint_counter")
        if type(accounted_counter) is int and (last_checkpoint is None or accounted_counter >= last_checkpoint["counter"]):
            accounted_path = accounting.get("last_checkpoint_path")
            accounted_sha256 = accounting.get("last_checkpoint_sha256")
            if isinstance(accounted_path, str) and isinstance(accounted_sha256, str):
                last_checkpoint = {"counter": accounted_counter, "path": accounted_path, "sha256": accounted_sha256}
        optimizer_status = accounting.get("optimizer_state_status")
        if optimizer_status is None:
            optimizer_status = "UNKNOWN_AFTER_UPDATE_FAILURE" if failure_stage == "ppo_update" else "LAST_CONFIRMED_BOUNDARY"
        progress.fail(
            error, failure_stage=failure_stage, attempted_transitions=attempted,
            completed_transitions=completed, completed_updates=confirmed_updates,
            update_counter=accounting.get("update_counter", latest.get("update_counter", 250 + confirmed_updates)),
            actor_optimizer_step=accounting.get("actor_optimizer_step"),
            critic_optimizer_step=accounting.get("critic_optimizer_step"),
            optimizer_state_status=optimizer_status,
            last_checkpoint_counter=last_checkpoint["counter"] if last_checkpoint else 250,
            last_checkpoint_path=last_checkpoint.get("path") if last_checkpoint else str(config["pins"]["checkpoint_path"]),
            last_checkpoint_sha256=last_checkpoint.get("sha256") if last_checkpoint else config["pins"]["checkpoint_sha256"],
        )
        try:
            backend.preserve_partial(error)
        except BaseException:
            pass
        raise


def validate_training_semantic(value: Mapping[str, Any], *, result_root: Path | None = None) -> None:
    required = {
        "schema_version", "status", "initial_checkpoint_counter", "initial_optimizer_step",
        "additional_transitions", "cumulative_transitions", "full_rollouts", "tail_horizon", "tail_samples",
        "new_updates", "target_update_counter", "new_optimizer_steps_per_actor_and_critic",
        "target_optimizer_step_per_actor_and_critic", "policy_steps_per_environment",
        "physics_steps_per_environment", "checkpoint_count", "checkpoint_counters", "initial_preflight",
        "final_reload_audit", "environment_state_restored", "sample_drop_padding_or_roundup",
        "performance_early_stop", "checkpoints",
    }
    if set(value) != required or value.get("schema_version") != "scale-003-training-semantic.v1" or value.get("status") != "PASS":
        raise ValueError("training semantic schema/status mismatch")
    exact = {
        "initial_checkpoint_counter": 250, "initial_optimizer_step": 2000,
        "additional_transitions": 872000, "cumulative_transitions": 1000000,
        "full_rollouts": 1703, "tail_horizon": 2, "tail_samples": 64,
        "new_updates": 1704, "target_update_counter": 1954,
        "new_optimizer_steps_per_actor_and_critic": 13625,
        "target_optimizer_step_per_actor_and_critic": 15625,
        "policy_steps_per_environment": 27250, "physics_steps_per_environment": 545000,
        "checkpoint_count": 70, "environment_state_restored": False,
        "sample_drop_padding_or_roundup": False, "performance_early_stop": False,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ValueError("training semantic exact accounting mismatch")
    if tuple(value["checkpoint_counters"]) != checkpoint_counters() or len(value["checkpoints"]) != 70:
        raise ValueError("training semantic checkpoint schedule mismatch")
    validate_training_preflight(value["initial_preflight"])
    expected_segments = [segment for segment in build_segments() if segment.checkpoint_kind is not None]
    seen_paths: set[Path] = set()
    for receipt, segment in zip(value["checkpoints"], expected_segments, strict=True):
        if not isinstance(receipt, Mapping) or set(receipt) != {"counter", "kind", "path", "sha256", "bytes"}:
            raise ValueError("checkpoint receipt schema mismatch")
        expected_name = f"update_{segment.update_counter:04d}_{segment.checkpoint_kind}.pt"
        path = Path(receipt["path"]) if isinstance(receipt["path"], str) else Path("")
        if (receipt["counter"], receipt["kind"], path.name) != (segment.update_counter, segment.checkpoint_kind, expected_name):
            raise ValueError("checkpoint receipt schedule/path mismatch")
        require_sha256(receipt["sha256"], "checkpoint receipt sha256")
        if type(receipt["bytes"]) is not int or receipt["bytes"] <= 0 or path in seen_paths:
            raise ValueError("checkpoint receipt size/path uniqueness mismatch")
        seen_paths.add(path)
        if result_root is not None:
            expected_path = (result_root / "checkpoints" / expected_name).resolve()
            if path.resolve() != expected_path or path.is_symlink() or not path.is_file():
                raise ValueError("checkpoint receipt does not bind the result checkpoint file")
            from .config import sha256_file
            if path.stat().st_size != receipt["bytes"] or sha256_file(path) != receipt["sha256"]:
                raise ValueError("checkpoint receipt file size/hash mismatch")
    reload_audit = value["final_reload_audit"]
    if not isinstance(reload_audit, Mapping) or set(reload_audit) != {"counter", "optimizer_step", "checkpoint_path", "checkpoint_sha256", "exact"}:
        raise ValueError("final checkpoint reload audit schema mismatch")
    final_receipt = value["checkpoints"][-1]
    if (reload_audit.get("counter"), reload_audit.get("optimizer_step"), reload_audit.get("checkpoint_path"), reload_audit.get("checkpoint_sha256"), reload_audit.get("exact")) != (
        1954, 15625, final_receipt["path"], final_receipt["sha256"], True,
    ):
        raise ValueError("final checkpoint reload audit mismatch")
