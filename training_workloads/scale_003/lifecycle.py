"""Process lifecycle helpers that retain the first failure through cleanup."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class CleanupFailure:
    name: str
    error: BaseException


@dataclass(frozen=True)
class LifecycleOutcome:
    value: Any = None
    primary_error: BaseException | None = None
    cleanup_failures: tuple[CleanupFailure, ...] = ()

    @property
    def failed(self) -> bool:
        return self.primary_error is not None or bool(self.cleanup_failures)


def exception_record(error: BaseException) -> dict[str, str]:
    return {
        "error_module": type(error).__module__,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def persist_failure_semantic(output: Path, mode: str, outcome: LifecycleOutcome, *, case: str | None = None) -> None:
    """Write a runner-readable FAIL semantic while preserving an existing primary failure."""
    from .persistence import atomic_json_replace

    selected = outcome.primary_error or outcome.cleanup_failures[0].error
    failure_path = output / "failure.json"
    if not failure_path.exists():
        atomic_json_replace(output, "failure.json", {
            "schema_version": "scale-003-failure.v1",
            "mode": mode,
            "failure_class": "RAW_RUNTIME_FAILURE",
            **exception_record(selected),
        })
    failure_bytes = failure_path.read_bytes()
    failure_class = "PRODUCT_SEMANTIC_FAILURE" if outcome.primary_error is not None else "PROCESS_CRASH"
    semantic = {
        "schema_version": "scale-003-failure-semantic.v1",
        "status": "FAIL",
        "semantic_exit": 1,
        "failure_class": failure_class,
        "mode": mode,
        "primary_error": exception_record(outcome.primary_error) if outcome.primary_error is not None else None,
        "cleanup_errors": [
            {"cleanup": row.name, **exception_record(row.error)} for row in outcome.cleanup_failures
        ],
        "failure_evidence": {
            "path": "failure.json",
            "bytes": len(failure_bytes),
            "sha256": hashlib.sha256(failure_bytes).hexdigest(),
        },
    }
    if case is not None:
        semantic["case"] = case
    atomic_json_replace(output, "semantic.json", semantic)
    status = {"schema_version": "scale-003-status.v1", "status": "FAIL", "mode": mode}
    if case is not None:
        status["case"] = case
    atomic_json_replace(output, "status.json", status)


def persist_shutdown_boundary(output: Path, mode: str, outcome: LifecycleOutcome, *, case: str | None = None) -> None:
    """Preserve known failure evidence before entering SimulationApp.close()."""
    from .persistence import atomic_json_replace

    value = {
        "schema_version": "scale-003-shutdown-boundary.v1",
        "mode": mode,
        "fast_shutdown": False,
        "status_before_app_close": "FAIL" if outcome.failed else "PENDING_SUCCESS",
        "primary_error": exception_record(outcome.primary_error) if outcome.primary_error is not None else None,
        "cleanup_errors_before_app_close": [
            {"cleanup": row.name, **exception_record(row.error)} for row in outcome.cleanup_failures
        ],
    }
    if case is not None:
        value["case"] = case
    atomic_json_replace(output, "shutdown_boundary.json", value)
    if outcome.failed:
        persist_failure_semantic(output, mode, outcome, case=case)


def run_lifecycle(
    operation: Callable[[], Any],
    cleanup_steps: Iterable[tuple[str, Callable[[], None]]],
    on_success: Callable[[Any], None],
    on_failure: Callable[[LifecycleOutcome], None],
    before_cleanup: Callable[[str, LifecycleOutcome], None] | None = None,
) -> int:
    """Run one operation, collect every cleanup failure, and return a shell exit code."""
    value: Any = None
    primary: BaseException | None = None
    try:
        value = operation()
    except BaseException as error:
        primary = error

    cleanup_failures: list[CleanupFailure] = []
    for name, cleanup in cleanup_steps:
        if before_cleanup is not None:
            current = LifecycleOutcome(value=value, primary_error=primary, cleanup_failures=tuple(cleanup_failures))
            try:
                before_cleanup(name, current)
            except BaseException as error:
                cleanup_failures.append(CleanupFailure(name=f"before:{name}", error=error))
        try:
            cleanup()
        except BaseException as error:
            cleanup_failures.append(CleanupFailure(name=name, error=error))

    outcome = LifecycleOutcome(value=value, primary_error=primary, cleanup_failures=tuple(cleanup_failures))
    if not outcome.failed:
        try:
            on_success(value)
            return 0
        except BaseException as error:
            outcome = LifecycleOutcome(value=value, primary_error=error)

    try:
        on_failure(outcome)
    except BaseException as error:
        # Persistence failure cannot turn a failed workload into a successful process.
        print(f"failure evidence persistence also failed: {type(error).__name__}: {error}", file=sys.stderr)
    return 1
