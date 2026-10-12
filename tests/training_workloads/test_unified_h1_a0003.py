from __future__ import annotations

import json
from pathlib import Path
import time
import threading

import pytest

from training_workloads.unified_h1 import live_adapter
from training_workloads.unified_h1.formal_unified_h1 import finalize_process_exit
from training_workloads.unified_h1.budget import required_mode_budget


def test_swallowed_deadline_then_immediate_return_remains_timeout_failure():
    caught = []

    def close():
        try:
            time.sleep(1)
        except Exception as error:
            caught.append(type(error).__name__)
            return

    with pytest.raises(TimeoutError, match="app.close did not return within 0.02s"):
        live_adapter._bounded_owner_call(close, 0.02, "app.close")
    assert caught == ["TimeoutError"]


def test_live_entry_uses_durable_exit_code_without_python_native_finalizers():
    observed = []
    assert finalize_process_exit(0, ["--validate-only"], observed.append) == 0
    assert observed == []
    assert finalize_process_exit(0, ["--mode", "main"], observed.append) == 0
    assert finalize_process_exit(1, ["--mode", "main"], observed.append) == 1
    assert observed == [0, 1]


def test_swallowed_deadline_cannot_promote_lifecycle_to_pass(tmp_path: Path, monkeypatch):
    events = []

    class Environment:
        h1_initialization_physics_ticks = 0
        h1_initialization_tick_source = "CPU_FAKE_NO_PHYSICS"

        def close(self):
            events.append("environment.close")

    class App:
        def close(self, *, wait_for_replicator, skip_cleanup):
            events.append(("app.close", wait_for_replicator, skip_cleanup, threading.get_ident()))
            try:
                time.sleep(1)
            except Exception as error:
                events.append(type(error).__name__)
                return

    class Backend:
        def __init__(self, counters, output):
            self.environment = Environment()
            self.counters = counters
            self.output = output

        def resource_check(self, *args):
            pass

        def preserve(self, status, outcome=None):
            live_adapter._persist(
                self.output, self.counters, status, self.initialization_measurement, outcome
            )

    attempt = tmp_path / "CPU_SYNTHETIC_ONLY"
    (attempt / "results").mkdir(parents=True)

    def factory(**kwargs):
        backend = Backend(kwargs["counters"], attempt / "results/main")
        kwargs["resources"].update(environment=backend.environment, app=App())
        return backend

    monkeypatch.setattr(live_adapter, "_dispatch", lambda *args: None)
    authorization = {
        "mode": "main",
        "parameters": {},
        "budgets": required_mode_budget("main", {}),
        "authorization_sha256": "0" * 64,
    }
    result = live_adapter.run_main_sequence(
        authorization=authorization,
        attempt=attempt,
        backend_factory=factory,
        cleanup_timeout_seconds=0.02,
    )
    status = json.loads((attempt / "results/main/status.json").read_text())
    assert events == [
        "environment.close",
        ("app.close", False, True, threading.main_thread().ident),
        "TimeoutError",
    ]
    assert result == 1
    assert status["status"] == "FAIL"
    assert [(row["cleanup"], row["error_type"]) for row in status["cleanup_errors"]] == [
        ("app.close", "TimeoutError")
    ]


def test_default_cleanup_split_preserves_ten_second_outer_margin(tmp_path: Path, monkeypatch):
    class Resource:
        h1_initialization_physics_ticks = 0
        h1_initialization_tick_source = "CPU_FAKE_NO_PHYSICS"
        def close(self, *, wait_for_replicator=None, skip_cleanup=None): pass

    class Backend:
        def __init__(self, counters, output, environment):
            self.counters = counters; self.output = output; self.environment = environment
        def resource_check(self, *args): pass
        def preserve(self, status, outcome=None):
            live_adapter._persist(self.output, self.counters, status, self.initialization_measurement, outcome)

    attempt = tmp_path / "CPU_CLEANUP_SPLIT"; (attempt / "results").mkdir(parents=True)
    def factory(**kwargs):
        environment = Resource(); app = Resource()
        kwargs["resources"].update(environment=environment, app=app)
        return Backend(kwargs["counters"], attempt / "results/main", environment)
    monkeypatch.setattr(live_adapter, "_dispatch", lambda *args: None)
    authorization = {"mode":"main", "parameters":{}, "budgets":required_mode_budget("main", {}),
                     "authorization_sha256":"1" * 64}
    assert live_adapter.run_main_sequence(authorization=authorization, attempt=attempt,
                                          backend_factory=factory) == 0
    start = json.loads((attempt / "results/main/process_start.json").read_text())
    assert start["cleanup_margin_seconds"] == 10.0
    assert start["cleanup_timeouts_seconds"] == {"environment.close":0.5, "app.close":9.5}
    assert start["minimum_outer_watchdog_seconds"] == authorization["budgets"]["wall_seconds"] + 10.0
