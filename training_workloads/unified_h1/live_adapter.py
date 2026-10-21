"""Lifecycle-owned future live construction, dispatch, and independent cleanup."""

from __future__ import annotations

import ctypes
import math
import json
import os
from pathlib import Path
import select
import signal
import threading
import time
from typing import Any, Callable, Mapping

from training_workloads.scale_003.lifecycle import LifecycleOutcome, exception_record, run_lifecycle
from training_workloads.scale_003.persistence import atomic_json_replace

from .budget import ExecutionCounters


def environment_class():
    from instinctlab.envs.unified_h1_direct_env import UnifiedH1DirectEnv, UnifiedH1DirectEnvCfg
    return UnifiedH1DirectEnv, UnifiedH1DirectEnvCfg


def b0_direct_adapter():
    from instinctlab.control.baseline_b0 import execute_b0_direct_velocity
    return execute_b0_direct_velocity


def _process_identity() -> dict[str, Any]:
    """Capture this entry process and its kernel process-group ownership."""
    pid = os.getpid()
    process_group_id = os.getpgid(0)
    session_id = os.getsid(0)
    return {
        "schema_version": "unified-h1-process-identity.v1",
        "pid": pid,
        "parent_pid": os.getppid(),
        "process_group_id": process_group_id,
        "session_id": session_id,
        "is_process_group_leader": pid == process_group_id,
        "captured_at_monotonic_ns": time.monotonic_ns(),
        "scope": "current Python entry process; no authority to signal unrelated processes",
    }


def _durable_hard_timeout_marker(path: Path, *, name: str, owner_pid: int,
                                 timeout_seconds: float) -> None:
    """Persist the last trustworthy fact before a cleanup watchdog kills its owner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    value = {
        "schema_version":"unified-h1-hard-cleanup-timeout.v1",
        "status":"HARD_TIMEOUT",
        "cleanup":name,
        "owner_pid":owner_pid,
        "watchdog_pid":os.getpid(),
        "timeout_seconds":float(timeout_seconds),
        "action":"SIGKILL_OWNER_PROCESS",
    }
    data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, data); os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)


def _open_owner_pidfd(owner_pid: int) -> int:
    """Open an identity-stable handle for the cleanup owner (Linux pidfd)."""
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(owner_pid, 0)
    # The frozen Python runtime omits os.pidfd_open even though its Linux
    # kernel exposes pidfd_open and signal.pidfd_send_signal.  syscall 434 is
    # pidfd_open on the pinned x86_64 runtime.
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = int(libc.syscall(434, owner_pid, 0))
    if descriptor < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return descriptor


def _kill_exact_parent_after_marker(*, owner_pid: int, owner_pidfd: int, name: str,
                                    timeout_seconds: float, marker_path: Path | None,
                                    process_api: Any = os, signal_api: Any = signal,
                                    marker_writer: Callable[..., None] = _durable_hard_timeout_marker) -> bool:
    """Write timeout evidence, then revalidate the direct parent before kill."""
    if process_api.getppid() != owner_pid:
        return False
    if marker_path is not None:
        marker_writer(marker_path, name=name, owner_pid=owner_pid,
                      timeout_seconds=timeout_seconds)
    # Marker persistence may block on storage.  Recheck immediately before
    # signalling so external owner reclamation during that interval cannot
    # turn PID reuse into an unrelated kill.
    if process_api.getppid() != owner_pid:
        return False
    # pidfd binds the original process identity, so even a PID reuse between
    # the final getppid check and signal delivery cannot hit another process.
    signal_api.pidfd_send_signal(owner_pidfd, signal.SIGKILL, None, 0)
    return True


def _start_hard_cleanup_watchdog(timeout_seconds: float, name: str,
                                  marker_path: Path | None) -> tuple[int, int]:
    """Fork a kill boundary that does not depend on Python signal delivery."""
    owner_pid = os.getpid(); owner_pidfd = _open_owner_pidfd(owner_pid)
    read_fd, write_fd = os.pipe(); watchdog_pid = os.fork()
    if watchdog_pid == 0:
        try:
            os.close(write_fd)
            ready, _, _ = select.select([read_fd], [], [], float(timeout_seconds))
            if ready:
                os.read(read_fd, 1)
                os._exit(0)
            _kill_exact_parent_after_marker(owner_pid=owner_pid, name=name,
                                            owner_pidfd=owner_pidfd,
                                            timeout_seconds=timeout_seconds,
                                            marker_path=marker_path)
        except BaseException:
            # A marker failure must not disable the safety boundary.
            if os.getppid() == owner_pid:
                try: signal.pidfd_send_signal(owner_pidfd, signal.SIGKILL, None, 0)
                except ProcessLookupError: pass
        finally:
            os.close(owner_pidfd)
            os._exit(124)
    os.close(read_fd); os.close(owner_pidfd)
    return watchdog_pid, write_fd


def _disarm_hard_cleanup_watchdog(watchdog_pid: int, write_fd: int) -> None:
    try:
        os.write(write_fd, b"1")
    except BrokenPipeError:
        pass
    finally:
        os.close(write_fd)
    while True:
        try:
            waited, _ = os.waitpid(watchdog_pid, 0)
            if waited == watchdog_pid: return
        except InterruptedError:
            continue
        except ChildProcessError:
            return


def _bounded_owner_call(function: Callable[[], None], timeout_seconds: float, name: str,
                        *, hard_timeout_marker: Path | None = None) -> None:
    """Run a Kit-owned cleanup on the main thread with a bounded alarm.

    SimulationApp.close drives Kit/Usd shutdown directly.  Moving that call to
    a daemon worker lets the lifecycle continue while native shutdown is still
    mutating process-global state.  A POSIX timer keeps the independent cleanup
    opportunity bounded without changing the resource's owning thread.  The
    When a hard-timeout marker is supplied, an independent forked watchdog is
    the final boundary for native code that cannot be interrupted by Python
    signal delivery.  It is disarmed and reaped synchronously when the
    owner-thread call returns.  Marker-free calls retain the CPU-testable soft
    alarm behavior without authorizing a process kill.
    """
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError("cleanup timeout must be a positive finite number")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(f"{name} cleanup must run on the process main thread")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    started = time.monotonic()
    deadline_expired = False
    watchdog = (None if hard_timeout_marker is None else
                _start_hard_cleanup_watchdog(float(timeout_seconds), name, hard_timeout_marker))

    def timeout_error() -> TimeoutError:
        return TimeoutError(f"{name} did not return within {timeout_seconds:g}s")

    def expired(signum, frame) -> None:
        nonlocal deadline_expired
        deadline_expired = True
        raise timeout_error()

    try:
        signal.signal(signal.SIGALRM, expired)
        # Repeat until the call returns or the exception escapes.  A dependency
        # may catch Exception internally (SimulationApp does so around optional
        # Replicator cleanup); a one-shot alarm could otherwise be swallowed.
        call_started = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds), float(timeout_seconds))
        function()
        # A dependency can catch the signal exception and return before the
        # repeating timer fires again.  Preserve that expired deadline as a
        # cleanup failure; also reject a return that crossed the monotonic
        # boundary before the process delivered SIGALRM.
        if deadline_expired or time.monotonic() - call_started >= float(timeout_seconds):
            raise timeout_error()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if watchdog is not None:
            _disarm_hard_cleanup_watchdog(*watchdog)
        if previous_timer[0] > 0:
            elapsed = time.monotonic() - started
            remaining = max(previous_timer[0] - elapsed, 1e-6)
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def _persist(output: Path, counters: ExecutionCounters, status: str,
             initialization_measurement: Mapping[str, Any],
             outcome: LifecycleOutcome | None = None) -> None:
    atomic_json_replace(output, "status.json", {"schema_version":"unified-h1-status.v3", "status":status,
        "primary_error":None if outcome is None or outcome.primary_error is None else exception_record(outcome.primary_error),
        "cleanup_errors":[] if outcome is None else [{"cleanup":row.name, **exception_record(row.error)} for row in outcome.cleanup_failures],
        "initialization_measurement":dict(initialization_measurement),
        "counters":counters.snapshot()})


def _dispatch(backend: Any, authorization: Mapping[str, Any]) -> None:
    from .r3_sequence import execute_controlled_failure, execute_evaluation_phase, execute_r3_main, execute_training_phase
    mode = authorization["mode"]; parameters = authorization["parameters"]
    if mode == "main": execute_r3_main(backend)
    elif mode == "controlled-failure": execute_controlled_failure(backend)
    elif mode == "r4-training": execute_training_phase(backend, resume_checkpoint=None, target_update=256)
    elif mode == "r5-resume": execute_training_phase(backend, resume_checkpoint=Path(parameters["checkpoint"]), target_update=parameters["target_update"])
    elif mode == "evaluation": execute_evaluation_phase(backend, checkpoints=parameters["checkpoints"])
    elif mode == "control-diagnostic": backend.run_control_diagnostic(parameters)
    elif mode == "research-r6-training": backend.train_r6()
    elif mode == "research-r6-evaluation":
        from .robustness_contract import evaluate_registered
        evaluate_registered(backend)
    elif mode == "research-training": backend.train_research()
    elif mode == "research-evaluation": backend.evaluate_research()
    elif mode == "research-transition": backend.run_research_transition()
    elif mode == "research-formal-training":
        from .formal_segment import train_segment
        train_segment(backend)
    elif mode == "research-formal-evaluation":
        from .formal_segment import evaluate_segment
        evaluate_segment(backend)
    elif mode == "pilot-training": backend.train_pilot(parameters["group"])
    elif mode == "pilot-evaluation": backend.evaluate_pilot_z(parameters["group"], parameters["checkpoints"])
    elif mode == "enhanced-b0-diagnostic": backend.run_enhanced_b0_diagnostic(parameters)
    else: raise PermissionError("authorization contains an unsupported mode")


def run_main_sequence(*, authorization: Mapping[str, Any], attempt: Path,
                      backend_factory: Callable[..., Any] | None = None,
                      cleanup_timeout_seconds: float | None = None,
                      process_started_monotonic: float | None = None) -> int:
    """Own construction and every acquired resource inside one failure boundary."""
    # Keep CPU failure-path tests and caller-provided factories import-safe:
    # the live backend imports torch and must only be loaded when it is used.
    entered = time.monotonic()
    started = entered if process_started_monotonic is None else float(process_started_monotonic)
    if not 0 < started <= entered: raise ValueError("process wall-clock boundary is invalid")
    if cleanup_timeout_seconds is None:
        # Keep the existing ten-second outer cleanup allowance while assigning
        # most of it to Kit shutdown, which releases the renderer/framework.
        environment_cleanup_seconds = 0.5
        app_cleanup_seconds = 9.5
    else:
        environment_cleanup_seconds = float(cleanup_timeout_seconds)
        app_cleanup_seconds = float(cleanup_timeout_seconds)
    cleanup_margin_seconds = environment_cleanup_seconds + app_cleanup_seconds
    attempt = attempt.resolve(); results = attempt / "results"; mode = str(authorization["mode"])
    output = results / mode
    if not attempt.is_dir() or not results.is_dir() or os.path.lexists(output):
        raise ValueError("attempt results root is absent or mode output already exists")
    starts = list(results.glob("*/process_start.json"))
    if starts:
        training = results / "r4-training"
        status_path = training / "status.json"
        # The formal entry already validates the independent mode authority and
        # exact checkpoint hashes. Only successful R4 training may precede its
        # separately authorized evaluation in this attempt; this is not a retry.
        allowed = (mode == "evaluation" and authorization.get("order_id") == "UNIFIED-H1-R4"
                   and authorization.get("parameters", {}).get("stage") == "R4"
                   and starts == [training / "process_start.json"]
                   and not training.is_symlink() and not starts[0].is_symlink()
                   and status_path.is_file() and not status_path.is_symlink())
        if not allowed or json.loads(status_path.read_text()).get("status") != "PASS":
            raise RuntimeError("authorized attempt live-process-start budget is already consumed")
    if backend_factory is None:
        from .backend import create_live_backend
        backend_factory = create_live_backend
    factory = backend_factory
    output.mkdir(mode=0o755)
    research_mode = mode.startswith("research-")
    if research_mode or authorization.get("parameters", {}).get("group") in {"C0", "C1"}:
        from .pilot_contract import PILOT_OUTPUT_CAP_BYTES, PILOT_WORKLOAD_WRITE_CAP_BYTES
        from training_workloads.scale_003.persistence import install_output_write_quota
        import resource
        if research_mode:
            PILOT_OUTPUT_CAP_BYTES = authorization["parameters"]["output_cap_bytes"]
            PILOT_WORKLOAD_WRITE_CAP_BYTES = PILOT_OUTPUT_CAP_BYTES - 3 * 1024**3
            if mode in ("research-r6-evaluation","research-r6-training") and "native_volume" in authorization["parameters"]:
                from .result_volume import workload_write_cap
                PILOT_WORKLOAD_WRITE_CAP_BYTES = workload_write_cap(
                    authorization["parameters"],expected_target=output.parent)
        if authorization["parameters"]["output_cap_bytes"] != PILOT_OUTPUT_CAP_BYTES:
            raise PermissionError("new pilot output cap differs from exact authorization")
        install_output_write_quota(output, PILOT_WORKLOAD_WRITE_CAP_BYTES)
        # Inherited stdout/native files have a kernel before-write ceiling.
        # Legacy modes retain the 3-GiB allowance. Exact fixed-volume R6 jobs
        # separately bind native image, control allowance and filesystem space.
        soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        ceiling = 2 * 1024**3
        resource.setrlimit(resource.RLIMIT_FSIZE, (min(ceiling, soft) if soft >= 0 else ceiling,
                                                 min(ceiling, hard) if hard >= 0 else ceiling))
        atomic_json_replace(output, "output_quota_binding.json", {
            "entry_output_cap_bytes": PILOT_OUTPUT_CAP_BYTES,
            "workload_cumulative_write_cap_bytes": PILOT_WORKLOAD_WRITE_CAP_BYTES,
            "native_control_archive_reserved_bytes": PILOT_OUTPUT_CAP_BYTES - PILOT_WORKLOAD_WRITE_CAP_BYTES,
            "per_file_kernel_ceiling_bytes": ceiling,
            "accounting": "charge_before_write; no overwrite or failure refund"})
    # Persist synchronously from inside the actual entry process.  An external
    # ps sample can miss short-lived failures; getpgid/getsid record the
    # kernel identities before configuration or backend construction begins.
    atomic_json_replace(output, "process_identity.json", _process_identity())
    counters = ExecutionCounters(); counters.reserve_live_start(authorization["budgets"]["live_process_starts"])
    atomic_json_replace(output, "process_start.json", {"schema_version":"unified-h1-process-start.v2",
        "mode":mode, "budgets":authorization["budgets"], "automatic_retries":0,
        "authorization_sha256":authorization["authorization_sha256"], "counters":counters.snapshot(),
        "wall_clock_boundary":"formal entry before argument/config/authorization validation",
        "action_wall_seconds":authorization["budgets"]["wall_seconds"],
        "cleanup_margin_seconds":cleanup_margin_seconds,
        "cleanup_timeouts_seconds":{"environment.close":environment_cleanup_seconds,
                                    "app.close":app_cleanup_seconds},
        "minimum_outer_watchdog_seconds":authorization["budgets"]["wall_seconds"] + cleanup_margin_seconds,
        "outer_watchdog":"formal command timeout must include the action budget plus independent cleanup margin"})
    resources: dict[str, Any] = {"launcher":None, "app":None, "environment":None, "backend":None,
                                 "process_started_monotonic":started}
    initialization_measurement: dict[str, Any] = {
        "status":"NOT_REACHED", "observed_physics_ticks":None,
        "measurement_boundary":"after_environment_construction_before_explicit_reset_or_policy_step",
        "reserved_simulator_initializations":1,
    }

    def controlled_failure(error: BaseException | None) -> bool:
        # Ordinary construction/cleanup failures must remain testable without
        # importing the torch-dependent sequence module. Controlled failures
        # originate in that module, so only that exact case needs the import.
        if error is None or error.__class__.__module__ != f"{__package__}.r3_sequence":
            return False
        from .r3_sequence import is_controlled_failure
        return is_controlled_failure(error)

    def operation() -> None:
        counters.reserve_simulator_initialization(authorization["budgets"]["simulator_initializations"])
        backend = factory(authorization=authorization, attempt=attempt, resources=resources, counters=counters)
        resources["backend"] = backend
        backend.initialization_measurement = dict(initialization_measurement)
        backend.resource_check("after_initialization_before_reset", 0)
        ticks = getattr(backend.environment, "h1_initialization_physics_ticks", None)
        source = getattr(backend.environment, "h1_initialization_tick_source", None)
        initialization_measurement.update(
            status="OBSERVED" if type(ticks) is int else "UNAVAILABLE_AT_MEASUREMENT_BOUNDARY",
            counter_source=source,
            observed_physics_ticks=ticks,
        )
        backend.initialization_measurement = dict(initialization_measurement)
        atomic_json_replace(output, "initialization_accounting.json", {
            "schema_version":"unified-h1-initialization-accounting.v1",
            "counter_source":source, "observed_physics_ticks":ticks,
            "maximum_physics_ticks":authorization["budgets"]["initialization_ticks"],
            "measurement_boundary":"after_environment_construction_before_explicit_reset_or_policy_step"})
        if type(ticks) is not int:
            raise RuntimeError("live environment did not expose measured initialization physics ticks")
        counters.record_initialization_ticks(ticks, authorization["budgets"]["initialization_ticks"])
        _dispatch(backend, authorization)

    def close_environment() -> None:
        resource = resources["environment"]
        if resource is not None:
            _bounded_owner_call(resource.close, environment_cleanup_seconds, "environment.close",
                                hard_timeout_marker=output / "cleanup_timeout.json")

    def close_app() -> None:
        resource = resources["app"]
        if resource is not None:
            # Installed skip_cleanup=True calls framework shutdown immediately;
            # it can still enter later cleanup and native calls may hang. These
            # kwargs do not guarantee graceful close. Keep the bounded owner
            # alarm and independent pidfd watchdog; forced termination remains
            # failed cleanup, followed by external descendant settlement.
            _bounded_owner_call(
                lambda: resource.close(wait_for_replicator=False, skip_cleanup=True),
                app_cleanup_seconds,
                "app.close",
                hard_timeout_marker=output / "cleanup_timeout.json",
            )

    def before_cleanup(name: str, outcome: LifecycleOutcome) -> None:
        backend = resources["backend"]
        status = "EXPECTED_FAILURE_PENDING_CLOSE" if controlled_failure(outcome.primary_error) else ("FAIL_PENDING_CLOSE" if outcome.failed else "PASS_PENDING_CLOSE")
        if backend is not None: backend.preserve(status, outcome)
        else: _persist(output, counters, status, initialization_measurement, outcome)

    def final_evidence(status: str, outcome: LifecycleOutcome | None) -> None:
        if mode != "enhanced-b0-diagnostic": return
        cleanup_errors=[] if outcome is None else [{"cleanup":row.name, **exception_record(row.error)} for row in outcome.cleanup_failures]
        atomic_json_replace(output, "budget_ledger.json", {"schema_version":"instinctrl-enhanced-b0-budget-ledger/1","status":status,"counters":counters.snapshot(),"automatic_retries":0})
        atomic_json_replace(output, "cleanup_evidence.json", {"schema_version":"instinctrl-enhanced-b0-cleanup/1","status":status,"environment_close":"PASS" if not cleanup_errors else "FAIL","app_close":"PASS" if not cleanup_errors else "FAIL","cleanup_errors":cleanup_errors,"automatic_retry":False})

    def success(value: Any) -> None:
        backend = resources["backend"]
        if backend is not None: backend.preserve("PASS")
        else: _persist(output, counters, "PASS", initialization_measurement)
        final_evidence("PASS", None)

    def failure(outcome: LifecycleOutcome) -> None:
        backend = resources["backend"]
        status = "EXPECTED_CONTROLLED_FAILURE" if controlled_failure(outcome.primary_error) else "FAIL"
        if backend is not None: backend.preserve(status, outcome)
        else: _persist(output, counters, status, initialization_measurement, outcome)
        final_evidence(status, outcome)

    return run_lifecycle(operation,
        (("environment.close", close_environment), ("app.close", close_app)),
        success, failure, before_cleanup)


__all__ = ["b0_direct_adapter", "environment_class", "run_main_sequence"]
