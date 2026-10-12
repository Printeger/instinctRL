from __future__ import annotations

import json
from pathlib import Path
import signal
import threading
import time

import pytest

from training_workloads.unified_h1.budget import required_mode_budget
from training_workloads.unified_h1.live_adapter import _bounded_owner_call, run_main_sequence


def _authorization():
    return {
        "mode": "main",
        "parameters": {},
        "budgets": required_mode_budget("main", {}),
        "authorization_sha256": "d" * 64,
    }


def test_owner_thread_cleanup_success_exception_timeout_and_timer_restoration():
    calls = []

    def success():
        calls.append(("success", threading.get_ident()))

    _bounded_owner_call(success, 0.2, "success.close")
    assert calls == [("success", threading.main_thread().ident)]

    def failure():
        calls.append(("failure", threading.get_ident()))
        raise OSError("close failed")

    with pytest.raises(OSError, match="close failed"):
        _bounded_owner_call(failure, 0.2, "failure.close")

    entered = time.monotonic()
    with pytest.raises(TimeoutError, match="hung.close did not return"):
        _bounded_owner_call(lambda: time.sleep(5), 0.02, "hung.close")
    assert time.monotonic() - entered < 0.5
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)

    def swallows_first_alarm():
        try:
            time.sleep(5)
        except Exception:
            pass
        time.sleep(5)

    entered = time.monotonic()
    with pytest.raises(TimeoutError, match="swallowing.close did not return"):
        _bounded_owner_call(swallows_first_alarm, 0.02, "swallowing.close")
    assert time.monotonic() - entered < 0.5


def test_cleanup_rejects_secondary_thread_instead_of_moving_kit_resource():
    captured = []

    def invoke():
        try:
            _bounded_owner_call(lambda: None, 0.1, "app.close")
        except BaseException as error:
            captured.append(error)

    worker = threading.Thread(target=invoke)
    worker.start()
    worker.join(1)
    assert not worker.is_alive()
    assert len(captured) == 1
    assert isinstance(captured[0], RuntimeError)
    assert str(captured[0]) == "app.close cleanup must run on the process main thread"


def test_main_sequence_preserves_primary_and_independent_cleanup_failures(tmp_path: Path):
    calls = []
    app_durations = []

    class Environment:
        def close(self):
            calls.append(("environment", threading.get_ident()))
            raise OSError("environment close failed")

    class App:
        def close(self, *, wait_for_replicator, skip_cleanup):
            calls.append(("app", wait_for_replicator, skip_cleanup, threading.get_ident()))
            entered = time.monotonic()
            try:
                time.sleep(1)
            finally:
                app_durations.append(time.monotonic() - entered)

    def constructor_failure(**kwargs):
        kwargs["resources"].update(environment=Environment(), app=App())
        raise RuntimeError("constructor failed")

    attempt = tmp_path / "attempt"
    (attempt / "results").mkdir(parents=True)
    assert run_main_sequence(
        authorization=_authorization(),
        attempt=attempt,
        backend_factory=constructor_failure,
        cleanup_timeout_seconds=0.02,
    ) == 1
    assert len(app_durations) == 1 and app_durations[0] < 0.5
    assert calls == [
        ("environment", threading.main_thread().ident),
        ("app", False, True, threading.main_thread().ident),
    ]
    status = json.loads((attempt / "results/main/status.json").read_text())
    assert status["status"] == "FAIL"
    assert status["primary_error"]["message"] == "constructor failed"
    assert [(row["cleanup"], row["error_type"]) for row in status["cleanup_errors"]] == [
        ("environment.close", "OSError"),
        ("app.close", "TimeoutError"),
    ]
    assert status["counters"]["live_process_starts"] == 1
    assert status["counters"]["simulator_initializations"] == 1
    assert status["counters"]["initialization_physics_ticks"] == 0
    assert status["initialization_measurement"] == {
        "status": "NOT_REACHED",
        "observed_physics_ticks": None,
        "measurement_boundary": "after_environment_construction_before_explicit_reset_or_policy_step",
        "reserved_simulator_initializations": 1,
    }
    assert not (attempt / "results/main/initialization_accounting.json").exists()
