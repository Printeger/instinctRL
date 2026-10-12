"""Live PPO backend. The direct entry imports this module only after authorization."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Mapping

from .config import load_config, sha256_file
from .live_scene import live_scene_readback, make_training_environment, run_initial_gate
from .lifecycle import (
    LifecycleOutcome, persist_failure_semantic, persist_shutdown_boundary, run_lifecycle,
)
from .persistence import ProgressStore, atomic_json_replace
from .resources import (
    RuntimeResourceGuard, start_after_coexistence_gate, validate_graceful_shutdown_configuration,
)
from .runtime_support import prepare_runtime_support
from .schedule import build_segments
from .training import run_training, validate_training_semantic


def _optimizer_step(optimizer: Any) -> int:
    steps = [int(item["step"].item()) for item in optimizer.state_dict()["state"].values()]
    if not steps or len(set(steps)) != 1:
        raise ValueError("Adam optimizer state does not have one exact common step")
    return steps[0]


def _state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    import torch

    return set(left) == set(right) and all(
        isinstance(left[key], torch.Tensor) and isinstance(right[key], torch.Tensor)
        and torch.equal(left[key], right[key]) for key in left
    )


class _LiveProgress:
    def __init__(self, store: ProgressStore) -> None:
        self.store = store
        self.latest = store.latest

    def write(self, **fields: Any) -> None:
        self.store.write(**fields)
        self.latest = self.store.latest

    def fail(self, error: BaseException, **fields: Any) -> None:
        self.store.fail(error, **fields)
        self.latest = self.store.latest

    def write_step_attempt(self, policy_step: int) -> None:
        current = dict(self.latest)
        current.update(
            status="RUNNING", stage="environment_step",
            attempted_policy_steps=policy_step, attempted_transitions=policy_step * 32,
            completed_policy_steps=policy_step - 1,
            completed_transitions=(policy_step - 1) * 32,
            initialization_or_reset_policy_steps=0,
        )
        current.pop("schema_version", None); current.pop("mode", None)
        self.write(**current)

    def write_step_complete(self, policy_step: int) -> None:
        current = dict(self.latest)
        current.update(
            status="RUNNING", stage="environment_step_complete",
            attempted_policy_steps=policy_step, attempted_transitions=policy_step * 32,
            completed_policy_steps=policy_step, completed_transitions=policy_step * 32,
            initialization_or_reset_policy_steps=0,
        )
        current.pop("schema_version", None); current.pop("mode", None)
        self.write(**current)


class LiveTrainingBackend:
    def __init__(self, output_root: Path, progress: _LiveProgress, resource_guard: RuntimeResourceGuard,
                 prelaunch_gate: Mapping[str, Any]) -> None:
        self.output_root = output_root
        self.progress = progress
        self.env: Any = None
        self.observation: Any = None
        self.trainer: Any = None
        self.rollout: Any = None
        self.resource_guard = resource_guard
        self.prelaunch_gate = dict(prelaunch_gate)
        self.initial_runtime_resource: Mapping[str, Any] | None = None
        self.last_checkpoint_receipt: Mapping[str, Any] | None = None

    def bootstrap(self, seed: int) -> None:
        import torch

        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def create_environment(self) -> None:
        self.initial_runtime_resource = self.resource_guard.check(stage="before_environment", ordinal=0)
        self.env = make_training_environment(self.progress, self.output_root.parent.parent)

    def reset_environment(self, seed: int) -> None:
        if self.env.schedule_index != 0:
            raise RuntimeError("global command clock moved before the explicit reset")
        self.observation, _ = self.env.reset(seed=seed)
        if self.env.schedule_index != 0:
            raise RuntimeError("explicit reset consumed policy steps")

    def create_trainer(self) -> None:
        from instinctlab.training import HistoryActor, NetworkConfig, PPOConfig, PPOTrainer, PrivilegedCritic

        network = NetworkConfig()
        ppo = PPOConfig(minibatch_size=64, epoch_count=1)
        actor = HistoryActor(network).to(self.env.device)
        critic = PrivilegedCritic(network).to(self.env.device)
        self.trainer = PPOTrainer(actor, critic, ppo)

    def load_checkpoint(self, path: Path, sha256: str) -> None:
        from instinctlab.training import load_checkpoint

        if sha256_file(path) != sha256:
            raise ValueError("input checkpoint hash mismatch")
        load_checkpoint(path, self.trainer)
        if self.trainer.update_counter != 250 or self.optimizer_steps() != (2000, 2000):
            raise ValueError("input checkpoint counter/optimizer state mismatch")

    def run_initial_preflight(self) -> Mapping[str, Any]:
        geometry = live_scene_readback(self.env)
        atomic_json_replace(self.output_root, "initial_preflight/geometry.json", geometry)
        observability = run_initial_gate(self.env, self.output_root)
        result = {"geometry": geometry, "observability": observability,
                  "prelaunch_resource_gate": self.prelaunch_gate,
                  "runtime_resource_initial": dict(self.initial_runtime_resource or {})}
        atomic_json_replace(self.output_root, "initial_preflight/complete.json", result)
        return result

    def prepare_segment(self, segment: Any) -> None:
        self.resource_guard.check(stage="before_rollout", ordinal=segment.ordinal)
        self.env.prepare_schedule_segment(segment, self.observation)

    def collect_segment(self, segment: Any) -> Mapping[str, Any]:
        from instinctlab.training import RolloutConfig, collect_rollout

        self.rollout = collect_rollout(self.env, self.trainer, self.observation, RolloutConfig(segment.horizon, 32))
        self.observation = self.rollout.next_observation
        if self.env.schedule_index != segment.policy_step_end:
            raise ValueError("rollout consumed a wrong number of global policy steps")
        return {"sample_count": segment.samples, "batch": self.rollout.segment.batch}

    def update(self, batch: Any) -> Mapping[str, Any]:
        self.resource_guard.check(stage="before_update", ordinal=self.trainer.update_counter + 1)
        result = self.trainer.update(batch)
        coverage = result["coverage"]
        return {
            "sample_count": result["sample_count"], "update_counter": result["update_counter"],
            "coverage_min": int(coverage.min().item()), "coverage_max": int(coverage.max().item()),
        }

    def optimizer_steps(self) -> tuple[int, int]:
        return (_optimizer_step(self.trainer.actor_optimizer), _optimizer_step(self.trainer.critic_optimizer))

    def save_checkpoint(self, counter: int, kind: str) -> Mapping[str, Any]:
        from instinctlab.training import save_checkpoint

        path = self.output_root / "checkpoints" / f"update_{counter:04d}_{kind}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError("checkpoint output already exists")
        save_checkpoint(path.resolve(), self.trainer)
        receipt = {"counter": counter, "kind": kind, "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        self.last_checkpoint_receipt = dict(receipt)
        return receipt

    def final_reload_audit(self, counter: int, optimizer_step: int) -> Mapping[str, Any]:
        from instinctlab.training import HistoryActor, NetworkConfig, PPOConfig, PPOTrainer, PrivilegedCritic, load_checkpoint

        final_path = self.output_root / "checkpoints" / "update_1954_final.pt"
        network = NetworkConfig(); ppo = PPOConfig(minibatch_size=64, epoch_count=1)
        loaded = PPOTrainer(HistoryActor(network).to(self.env.device), PrivilegedCritic(network).to(self.env.device), ppo)
        load_checkpoint(final_path.resolve(), loaded)
        exact = (
            loaded.update_counter == counter
            and (_optimizer_step(loaded.actor_optimizer), _optimizer_step(loaded.critic_optimizer)) == (optimizer_step, optimizer_step)
            and _state_equal(self.trainer.actor.state_dict(), loaded.actor.state_dict())
            and _state_equal(self.trainer.critic.state_dict(), loaded.critic.state_dict())
        )
        return {"counter": loaded.update_counter, "optimizer_step": _optimizer_step(loaded.actor_optimizer),
                "checkpoint_path": str(final_path), "checkpoint_sha256": sha256_file(final_path), "exact": exact}

    def preserve_partial(self, error: BaseException) -> None:
        return None

    def failure_accounting(self) -> Mapping[str, Any]:
        if self.trainer is None:
            return {"completed_updates": 0, "update_counter": 250,
                    "actor_optimizer_step": None, "critic_optimizer_step": None,
                    "optimizer_state_status": "NOT_CREATED_OR_LOADED"}
        actor_step, critic_step = self.optimizer_steps()
        completed = max(0, self.trainer.update_counter - 250)
        expected_step = 2000 + sum(segment.optimizer_step_delta for segment in build_segments()[:completed])
        status = "CONFIRMED_UPDATE_BOUNDARY" if actor_step == critic_step == expected_step else "PARTIAL_UPDATE_STATE"
        evidence = {"completed_updates": completed, "update_counter": self.trainer.update_counter,
                "actor_optimizer_step": actor_step, "critic_optimizer_step": critic_step,
                "optimizer_state_status": status}
        if self.last_checkpoint_receipt is not None:
            evidence.update(
                last_checkpoint_counter=self.last_checkpoint_receipt["counter"],
                last_checkpoint_path=self.last_checkpoint_receipt["path"],
                last_checkpoint_sha256=self.last_checkpoint_receipt["sha256"],
            )
        return evidence

    def close(self) -> None:
        if self.env is not None:
            self.env.close()


def run_live(authorization: Any, attempt_dir: Path) -> int:
    """Consume one already-validated training authorization and persist exact evidence."""
    if authorization.mode != "training" or attempt_dir.resolve() != authorization.attempt_path:
        raise PermissionError("validated training authorization binding changed")
    results = attempt_dir / "results"
    output = results / "training"
    if not results.is_dir() or output.exists():
        raise ValueError("attempt results root is absent or training output already exists")
    output.mkdir(mode=0o755)
    if list(attempt_dir.glob("results/training/process_start.json")):
        raise RuntimeError("one-start training budget exhausted")
    atomic_json_replace(output, "process_start.json", {
        "schema_version": "scale-003-process-start.v1", "mode": "training", "pid": os.getpid(),
        "maximum_environment_transitions": 872000, "maximum_optimizer_updates": 1704,
        "maximum_checkpoint_writes": 70, "automatic_retries": 0,
    })
    resources: dict[str, Any] = {"simulation_app": None, "backend": None}

    def operation() -> Mapping[str, Any]:
        prepare_runtime_support(attempt_dir, mode="training")
        prelaunch_gate, _launcher, simulation_app = start_after_coexistence_gate(
            output, mode="training", attempt_id=authorization.attempt_id,
            authorization_sha256=sha256_file(authorization.document_path),
        )
        resources["simulation_app"] = simulation_app
        shutdown = validate_graceful_shutdown_configuration(simulation_app)
        atomic_json_replace(output, "launcher_shutdown_configuration.json", {
            "schema_version": "scale-003-launcher-shutdown.v1", **shutdown,
        })
        store = ProgressStore(output, "training")
        progress = _LiveProgress(store)
        backend = LiveTrainingBackend(output, progress, RuntimeResourceGuard(output), prelaunch_gate)
        resources["backend"] = backend
        semantic = run_training(backend, progress)
        validate_training_semantic(semantic, result_root=output)
        return semantic

    def close_backend() -> None:
        if resources["backend"] is not None:
            resources["backend"].close()

    def close_app() -> None:
        if resources["simulation_app"] is not None:
            resources["simulation_app"].close()

    def success(semantic: Mapping[str, Any]) -> None:
        atomic_json_replace(output, "semantic.json", semantic)
        atomic_json_replace(output, "status.json", {"schema_version": "scale-003-status.v1", "status": "PASS", "mode": "training"})

    def failure(outcome: LifecycleOutcome) -> None:
        persist_failure_semantic(output, "training", outcome)

    def before_cleanup(name: str, outcome: LifecycleOutcome) -> None:
        if name == "simulation_app.close":
            if not outcome.failed:
                atomic_json_replace(output, "semantic_pending_shutdown.json", outcome.value)
            persist_shutdown_boundary(output, "training", outcome)

    return run_lifecycle(
        operation,
        (("backend.close", close_backend), ("simulation_app.close", close_app)),
        success,
        failure,
        before_cleanup,
    )
