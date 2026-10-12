"""R3 orchestration through the approved failure-preserving lifecycle."""

from __future__ import annotations

from typing import Any, Protocol

from training_workloads.scale_003.lifecycle import LifecycleOutcome, run_lifecycle

from .evaluation import CASE_ORDER


class R3Backend(Protocol):
    def resource_check(self, stage: str, ordinal: int) -> None: ...
    def initial_preflight(self) -> None: ...
    def evaluate_b0(self, case: str) -> None: ...
    def cold_start(self) -> None: ...
    def save_checkpoint(self, label: str) -> None: ...
    def smoke_update(self, profile: str) -> None: ...
    def reload_final(self) -> None: ...
    def mixed_reload_rollout(self, steps: int) -> None: ...
    def initial_raw_rays(self) -> None: ...
    def preserve(self, status: str, outcome: LifecycleOutcome | None = None) -> None: ...
    def close_bounded(self) -> None: ...


class _ControlledFailure(RuntimeError):
    pass


def _preserve(backend: R3Backend, status: str, outcome: LifecycleOutcome | None = None) -> None:
    backend.preserve(status, outcome)


def run_r3_main(backend: R3Backend) -> int:
    """One process: preflight, B0 cases, four smoke updates, reload and audit."""
    def before_cleanup(name: str, outcome: LifecycleOutcome) -> None:
        _preserve(backend, "FAIL_PENDING_CLOSE" if outcome.failed else "PASS_PENDING_CLOSE", outcome)

    return run_lifecycle(lambda: execute_r3_main(backend), (("backend.close_bounded", backend.close_bounded),),
                         lambda value: _preserve(backend, "PASS"),
                         lambda outcome: _preserve(backend, "FAIL", outcome), before_cleanup)


def run_r3_controlled_failure(backend: R3Backend) -> int:
    """Classify expected failure only after the exact reviewed injection point is durable."""
    def failure(outcome: LifecycleOutcome) -> None:
        expected = isinstance(outcome.primary_error, _ControlledFailure)
        _preserve(backend, "EXPECTED_CONTROLLED_FAILURE" if expected else "FAIL", outcome)

    def before_cleanup(name: str, outcome: LifecycleOutcome) -> None:
        expected = isinstance(outcome.primary_error, _ControlledFailure)
        _preserve(backend, "EXPECTED_FAILURE_PENDING_CLOSE" if expected else "FAIL_PENDING_CLOSE", outcome)

    return run_lifecycle(lambda: execute_controlled_failure(backend), (("backend.close_bounded", backend.close_bounded),),
                         lambda value: _preserve(backend, "FAIL"), failure, before_cleanup)


def run_training_phase(backend: Any, *, resume_checkpoint: Any | None, target_update: int) -> int:
    """Lifecycle-owned R4 cold start or R5 strict resume through the same backend."""
    return run_lifecycle(lambda: execute_training_phase(backend, resume_checkpoint=resume_checkpoint, target_update=target_update), (("backend.close_bounded", backend.close_bounded),),
        lambda value: _preserve(backend, "PASS"), lambda outcome: _preserve(backend, "FAIL", outcome),
        lambda name, outcome: _preserve(backend, "FAIL_PENDING_CLOSE" if outcome.failed else "PASS_PENDING_CLOSE", outcome))


def run_evaluation_phase(backend: Any, *, checkpoints: list[dict[str, Any]]) -> int:
    """Lifecycle-owned, independent-process deterministic metric-h1 evaluation."""
    return run_lifecycle(lambda: execute_evaluation_phase(backend, checkpoints=checkpoints), (("backend.close_bounded", backend.close_bounded),),
        lambda value: _preserve(backend, "PASS"), lambda outcome: _preserve(backend, "FAIL", outcome),
        lambda name, outcome: _preserve(backend, "FAIL_PENDING_CLOSE" if outcome.failed else "PASS_PENDING_CLOSE", outcome))


def execute_r3_main(backend: R3Backend) -> None:
    backend.resource_check("before_preflight", 0); backend.initial_preflight()
    for ordinal, case in enumerate(CASE_ORDER, 1):
        backend.resource_check("before_b0_case", ordinal); backend.evaluate_b0(case)
    backend.resource_check("before_cold_start", 0); backend.cold_start(); backend.save_checkpoint("smoke_init")
    for ordinal, profile in enumerate(("zero", "mixed", "role-swap-a", "role-swap-b"), 1):
        backend.resource_check("before_smoke_update", ordinal); backend.smoke_update(profile)
    backend.save_checkpoint("smoke_final"); backend.reload_final()
    backend.resource_check("before_reload_rollout", 0); backend.mixed_reload_rollout(16)


def execute_controlled_failure(backend: R3Backend) -> None:
    backend.resource_check("before_preflight", 0); backend.initial_raw_rays()
    _preserve(backend, "CONTROLLED_FAILURE_INJECTION_REACHED")
    raise _ControlledFailure("CONTROLLED_FAILURE_AFTER_INITIAL_RAYS_BEFORE_POLICY_STEP")


def execute_training_phase(backend: Any, *, resume_checkpoint: Any | None, target_update: int) -> None:
    backend.resource_check("before_training_phase", target_update)
    if resume_checkpoint is None:
        backend.cold_start(); backend.save_checkpoint("init0000")
    else: backend.resume_from_checkpoint(resume_checkpoint)
    backend.train_to_update(target_update)


def execute_evaluation_phase(backend: Any, *, checkpoints: list[dict[str, Any]]) -> None:
    backend.resource_check("before_evaluation", len(checkpoints))
    backend.evaluate_policy_checkpoints(checkpoints)


def is_controlled_failure(error: BaseException | None) -> bool:
    return isinstance(error, _ControlledFailure)


__all__ = ["R3Backend", "execute_controlled_failure", "execute_evaluation_phase", "execute_r3_main",
           "execute_training_phase", "is_controlled_failure", "run_evaluation_phase",
           "run_r3_controlled_failure", "run_r3_main", "run_training_phase"]
