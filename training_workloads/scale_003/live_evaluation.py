"""Live final-checkpoint evaluation backend, imported only after authorization."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .config import load_config, sha256_file
from .evaluation import run_evaluation
from .lifecycle import (
    LifecycleOutcome, persist_failure_semantic, persist_shutdown_boundary, run_lifecycle,
)
from .live_scene import persist_initial_acquisition, persist_initial_gate_failure, persist_initial_source
from .persistence import ProgressStore, atomic_json_replace
from .resources import (
    RuntimeResourceGuard, start_after_coexistence_gate, validate_graceful_shutdown_configuration,
)
from .runtime_support import prepare_runtime_support


def _optimizer_step(optimizer: Any) -> int:
    steps = [int(item["step"].item()) for item in optimizer.state_dict()["state"].values()]
    if not steps or len(set(steps)) != 1:
        raise ValueError("Adam optimizer state does not have one exact common step")
    return steps[0]


def _clone_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _tree_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right) and all(_tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _jsonable(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


class LiveEvaluationBackend:
    def __init__(self, output: Path, resource_guard: RuntimeResourceGuard,
                 prelaunch_gate: Mapping[str, Any]) -> None:
        self.output = output
        self.env: Any = None
        self.observation: Any = None
        self.trainer: Any = None
        self.collector: Any = None
        self.case = ""
        self.rows: list[dict[str, Any]] = []
        self.before: Any = None
        self.geometry: Any = None
        self.initial: Any = None
        self.resource_guard = resource_guard
        self.prelaunch_gate = dict(prelaunch_gate)
        self.initial_runtime_resource: Mapping[str, Any] | None = None
        self.completed_transitions = 0

    def bootstrap(self, seed: int) -> None:
        import torch

        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def create_environment(self, case: str) -> None:
        from evaluation_workloads.eval_002.scene import make_environment

        self.case = case
        self.initial_runtime_resource = self.resource_guard.check(stage="before_environment", ordinal=0)
        self.env = make_environment(case)

    def reset_environment(self, seed: int) -> None:
        self.observation, _ = self.env.reset(seed=seed)

    def load_checkpoint(self, path: Path, counter: int) -> None:
        from instinctlab.training import HistoryActor, NetworkConfig, PPOConfig, PPOTrainer, PrivilegedCritic, load_checkpoint

        network = NetworkConfig(); ppo = PPOConfig(minibatch_size=64, epoch_count=1)
        self.trainer = PPOTrainer(HistoryActor(network).to(self.env.device), PrivilegedCritic(network).to(self.env.device), ppo)
        load_checkpoint(path.resolve(), self.trainer)
        if self.trainer.update_counter != counter or (_optimizer_step(self.trainer.actor_optimizer), _optimizer_step(self.trainer.critic_optimizer)) != (15625, 15625):
            raise ValueError("final checkpoint counter/optimizer state mismatch")
        self.before = _clone_tree({
            "actor": self.trainer.actor.state_dict(), "critic": self.trainer.critic.state_dict(),
            "actor_optimizer": self.trainer.actor_optimizer.state_dict(),
            "critic_optimizer": self.trainer.critic_optimizer.state_dict(),
        })

    def run_initial_preflight(self) -> Mapping[str, Any]:
        from evaluation_workloads.eval_002.scene import initial_gate, initial_source, live_geometry
        from instinctlab.evaluation.raycaster_observability import acquire_exact_observability_from_source

        self.geometry = live_geometry(self.env)
        atomic_json_replace(self.output, "initial_preflight/geometry.json", self.geometry)
        source = initial_source(self.env)
        persist_initial_source(self.output, source)
        try:
            acquisition = acquire_exact_observability_from_source(source)
        except BaseException as error:
            persist_initial_gate_failure(self.output, "raw_ray_source", error)
            raise
        persist_initial_acquisition(self.output, acquisition)
        try:
            self.initial = initial_gate(source, acquisition)
        except BaseException as error:
            persist_initial_gate_failure(self.output, "raw_acquisition", error)
            raise
        atomic_json_replace(self.output, "initial_preflight/observability.json", self.initial)
        result = {"geometry": self.geometry, "observability": self.initial,
                  "prelaunch_resource_gate": self.prelaunch_gate,
                  "runtime_resource_initial": dict(self.initial_runtime_resource or {})}
        atomic_json_replace(self.output, "initial_preflight/complete.json", result)
        return result

    def disable_gradients(self) -> None:
        import torch
        from instinctlab.evaluation import RuntimeEvaluationCollector, RuntimeEvaluationConfig

        self.trainer.actor.eval(); self.trainer.critic.eval()
        torch.manual_seed(20260905); torch.cuda.manual_seed_all(20260905)
        torch.set_grad_enabled(False)
        self.collector = RuntimeEvaluationCollector(
            self.observation["critic"].detach().clone(),
            RuntimeEvaluationConfig(num_envs=2, expected_policy_steps=256),
        )

    def evaluate_step(self, case: str, index: int, command: tuple[float, float, float]) -> Mapping[str, Any]:
        import torch
        from evaluation_workloads.eval_002.scene import visibility
        from instinctlab.evaluation.raycaster_observability import acquire_exact_observability_from_source
        from instinctlab.evaluation.runtime import distill_exact_runtime_observability

        self.resource_guard.check(stage="before_evaluation_step", ordinal=index)
        audited = self.collector.prepare_policy(self.observation["policy"])
        expected = torch.tensor(command, device=self.env.device, dtype=self.env._command_b.dtype).repeat(2, 1)
        if tuple(audited) != ("lidar_grid", "state_vec") or not torch.equal(audited["state_vec"][:, -7:-4], expected) or not torch.equal(self.env._command_b, expected):
            raise ValueError("pre-action command/actor alignment mismatch")
        output = self.trainer.actor(audited)
        action = output.mean_action.detach()
        unchanged = action.clone()
        observation, _, terminated, truncated, _ = self.env.step(action)
        self.completed_transitions += 2
        if not torch.equal(action, unchanged):
            raise RuntimeError("environment mutated deterministic mean action")
        pre_reset = self.env.extras["pre_reset"]
        source = self.env.extras["evaluation_ray_source"]
        acquisition = acquire_exact_observability_from_source(source)
        exact = distill_exact_runtime_observability(source, acquisition)
        self.collector.record_transition(
            pre_action_policy=audited, action=action, pre_reset=pre_reset,
            exact_observability=exact, returned_policy=observation["policy"],
            returned_critic=observation["critic"].detach().clone(),
            returned_episode_length=self.env.episode_length_buf.detach().clone(),
        )
        row = {
            "global_index": index, "command_b_m_s": list(command),
            "actor_keys": ["lidar_grid", "state_vec"],
            "terminal_environment_count": int((terminated | truncated).sum().item()),
            "visibility": visibility(source, acquisition),
        }
        self.rows.append(row)
        atomic_json_replace(self.output, "partial_rows.json", self.rows)
        self.observation = observation
        return row

    def finalize(self) -> Mapping[str, Any]:
        runtime = self.collector.finalize({
            "product_commit": load_config()["pins"]["product_commit"],
            "fixture_id": f"scale-003-eval-002-{self.case}",
            "generator_id": "training_workloads/scale_003/formal_evaluation.py", "seed": 20260905,
        })
        after = {
            "actor": self.trainer.actor.state_dict(), "critic": self.trainer.critic.state_dict(),
            "actor_optimizer": self.trainer.actor_optimizer.state_dict(),
            "critic_optimizer": self.trainer.critic_optimizer.state_dict(),
        }
        if not _tree_equal(self.before, after):
            raise RuntimeError("evaluation mutated checkpoint model or optimizer state")
        canonical = json.loads(runtime["canonical_record"])
        return {
            "canonical_metrics_available": True,
            "terminal_samples_preserved": True,
            "zero_denominator_is_unavailable": True,
            "runtime_evaluation": _jsonable(runtime), "canonical_record": canonical,
            "reset_count": runtime["gates"]["reset_count"],
            "checkpoint_state_unchanged": True,
        }

    def preserve_partial(self, error: BaseException) -> None:
        return None

    def failure_accounting(self) -> Mapping[str, Any]:
        return {"completed_transitions": self.completed_transitions,
                "completed_policy_steps": self.completed_transitions // 2}

    def close(self) -> None:
        if self.env is not None:
            self.env.close()


def run_live(authorization: Any, attempt_dir: Path, case: str, checkpoint: Path) -> int:
    if authorization.mode != "evaluation" or attempt_dir.resolve() != authorization.attempt_path:
        raise PermissionError("validated evaluation authorization binding changed")
    if authorization.input_handoff is None:
        raise PermissionError("validated evaluation authorization has no sealed training handoff")
    config = load_config()
    scene = Path(config["pins"]["eval_002_scene_path"])
    if sha256_file(scene) != config["pins"]["eval_002_scene_sha256"]:
        raise ValueError("EVAL-002 scene source identity mismatch")
    expected_checkpoint = authorization.input_handoff.final_checkpoint_path.resolve()
    if checkpoint.resolve() != expected_checkpoint or not expected_checkpoint.is_file() or expected_checkpoint.is_symlink():
        raise ValueError("evaluation checkpoint must match the immutable training handoff")
    results = attempt_dir / "results"
    output = results / case
    if not results.is_dir() or output.exists():
        raise ValueError("attempt results root is absent or case output already exists")
    starts = list(results.glob("*/process_start.json"))
    evaluation_starts = [path for path in starts if path.parent.name in {"station", "tracking"}]
    if len(evaluation_starts) >= 2:
        raise RuntimeError("two-start evaluation budget exhausted")
    output.mkdir(mode=0o755)
    atomic_json_replace(output, "process_start.json", {
        "schema_version": "scale-003-process-start.v1", "mode": "evaluation", "case": case,
        "pid": os.getpid(), "maximum_environment_transitions": 512,
        "maximum_policy_steps": 256, "automatic_retries": 0,
    })
    resources: dict[str, Any] = {"simulation_app": None, "backend": None}

    def operation() -> Mapping[str, Any]:
        prepare_runtime_support(attempt_dir, mode="evaluation")
        prelaunch_gate, _launcher, simulation_app = start_after_coexistence_gate(
            output, mode="evaluation", attempt_id=authorization.attempt_id,
            authorization_sha256=sha256_file(authorization.document_path),
        )
        resources["simulation_app"] = simulation_app
        shutdown = validate_graceful_shutdown_configuration(simulation_app)
        atomic_json_replace(output, "launcher_shutdown_configuration.json", {
            "schema_version": "scale-003-launcher-shutdown.v1", **shutdown,
        })
        progress = ProgressStore(output, case)
        backend = LiveEvaluationBackend(output, RuntimeResourceGuard(output), prelaunch_gate)
        resources["backend"] = backend
        semantic = run_evaluation(case, checkpoint, backend, progress)
        return semantic

    def close_backend() -> None:
        if resources["backend"] is not None:
            resources["backend"].close()

    def close_app() -> None:
        if resources["simulation_app"] is not None:
            resources["simulation_app"].close()

    def success(semantic: Mapping[str, Any]) -> None:
        atomic_json_replace(output, "semantic.json", semantic)
        atomic_json_replace(output, "status.json", {"schema_version": "scale-003-status.v1", "status": "PASS", "mode": "evaluation", "case": case})

    def failure(outcome: LifecycleOutcome) -> None:
        persist_failure_semantic(output, "evaluation", outcome, case=case)

    def before_cleanup(name: str, outcome: LifecycleOutcome) -> None:
        if name == "simulation_app.close":
            if not outcome.failed:
                atomic_json_replace(output, "semantic_pending_shutdown.json", outcome.value)
            persist_shutdown_boundary(output, "evaluation", outcome, case=case)

    return run_lifecycle(
        operation,
        (("backend.close", close_backend), ("simulation_app.close", close_app)),
        success,
        failure,
        before_cleanup,
    )
