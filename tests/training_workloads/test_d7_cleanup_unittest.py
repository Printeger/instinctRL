from __future__ import annotations

import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from training_workloads.unified_h1.budget import required_mode_budget
from training_workloads.unified_h1 import live_adapter


def authorization():
    return {"mode": "main", "parameters": {}, "budgets": required_mode_budget("main", {}),
            "authorization_sha256": "d" * 64}


class D7CleanupTests(unittest.TestCase):
    def test_hard_watchdog_bounds_native_style_hang_and_persists_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "cleanup_timeout.json"
            code = """
import signal, time
from pathlib import Path
from training_workloads.unified_h1 import live_adapter
live_adapter.signal.setitimer = lambda *args: (0.0, 0.0)
live_adapter.signal.signal = lambda *args: signal.SIG_DFL
live_adapter._bounded_owner_call(lambda: time.sleep(30), 0.05, 'app.close', hard_timeout_marker=Path(%r))
""" % str(marker)
            started = time.monotonic()
            completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
            self.assertEqual(completed.returncode, -signal.SIGKILL)
            self.assertLess(time.monotonic() - started, 1.0)
            value = json.loads(marker.read_text())
            self.assertEqual(value["status"], "HARD_TIMEOUT")
            self.assertEqual(value["cleanup"], "app.close")
            self.assertEqual(value["action"], "SIGKILL_OWNER_PROCESS")

    def test_hard_watchdog_is_disarmed_and_reaped_after_quick_return(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "cleanup_timeout.json"
            watchdog_pids = []
            actual_start = live_adapter._start_hard_cleanup_watchdog
            def capture_start(*args, **kwargs):
                value = actual_start(*args, **kwargs); watchdog_pids.append(value[0]); return value
            with patch.object(live_adapter, "_start_hard_cleanup_watchdog", side_effect=capture_start):
                live_adapter._bounded_owner_call(lambda: None, 0.05, "app.close",
                                                 hard_timeout_marker=marker)
            time.sleep(0.08)
            self.assertFalse(marker.exists())
            self.assertEqual(len(watchdog_pids), 1)
            with self.assertRaises(ChildProcessError):
                live_adapter.os.waitpid(watchdog_pids[0], live_adapter.os.WNOHANG)

    def test_reparenting_check_is_immediately_before_kill(self):
        class ProcessApi:
            def __init__(self, parents): self.parents=iter(parents)
            def getppid(self): return next(self.parents)
        class SignalApi:
            def __init__(self): self.signals=[]
            def pidfd_send_signal(self, pidfd, signum, info, flags): self.signals.append((pidfd,signum,info,flags))
        markers=[];owner=12345;path=Path('/cpu-only/marker.json')
        reparented=ProcessApi([owner,1]);reparented_signals=SignalApi()
        result=live_adapter._kill_exact_parent_after_marker(owner_pid=owner,owner_pidfd=99,name='app.close',
            timeout_seconds=.05,marker_path=path,process_api=reparented,signal_api=reparented_signals,
            marker_writer=lambda *args,**kwargs:markers.append((args,kwargs)))
        self.assertFalse(result);self.assertEqual(reparented_signals.signals,[]);self.assertEqual(len(markers),1)
        exact=ProcessApi([owner,owner]);exact_signals=SignalApi();markers.clear()
        result=live_adapter._kill_exact_parent_after_marker(owner_pid=owner,owner_pidfd=99,name='app.close',
            timeout_seconds=.05,marker_path=path,process_api=exact,signal_api=exact_signals,
            marker_writer=lambda *args,**kwargs:markers.append((args,kwargs)))
        self.assertTrue(result);self.assertEqual(exact_signals.signals,[(99,signal.SIGKILL,None,0)]);self.assertEqual(len(markers),1)

    def test_owner_thread_success_exception_deadlines_and_timer_restoration(self):
        calls = []
        live_adapter._bounded_owner_call(lambda: calls.append(threading.get_ident()), 0.2, "success.close")
        self.assertEqual(calls, [threading.main_thread().ident])
        with self.assertRaisesRegex(OSError, "close failed"):
            live_adapter._bounded_owner_call(lambda: (_ for _ in ()).throw(OSError("close failed")), 0.2, "failure.close")
        entered = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "hung.close did not return"):
            live_adapter._bounded_owner_call(lambda: time.sleep(5), 0.02, "hung.close")
        self.assertLess(time.monotonic() - entered, 0.5)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

        caught = []
        def swallows_alarm():
            try:
                time.sleep(1)
            except Exception as error:
                caught.append(type(error).__name__)
                return
        with self.assertRaisesRegex(TimeoutError, "app.close did not return within 0.02s"):
            live_adapter._bounded_owner_call(swallows_alarm, 0.02, "app.close")
        self.assertEqual(caught, ["TimeoutError"])

    def test_cleanup_rejects_secondary_thread(self):
        captured = []
        def invoke():
            try:
                live_adapter._bounded_owner_call(lambda: None, 0.1, "app.close")
            except BaseException as error:
                captured.append(error)
        worker = threading.Thread(target=invoke)
        worker.start(); worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(captured), 1)
        self.assertIsInstance(captured[0], RuntimeError)
        self.assertEqual(str(captured[0]), "app.close cleanup must run on the process main thread")

    def test_close_keywords_and_fail_closed_status(self):
        calls = []
        class Environment:
            def close(self):
                calls.append(("environment", threading.get_ident()))
                raise OSError("environment close failed")
        class App:
            def close(self, *, wait_for_replicator, skip_cleanup):
                calls.append(("app", wait_for_replicator, skip_cleanup, threading.get_ident()))
                time.sleep(1)
        def constructor_failure(**kwargs):
            kwargs["resources"].update(environment=Environment(), app=App())
            raise RuntimeError("constructor failed")
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt"; (attempt / "results").mkdir(parents=True)
            result = live_adapter.run_main_sequence(authorization=authorization(), attempt=attempt,
                backend_factory=constructor_failure, cleanup_timeout_seconds=0.02)
            status = json.loads((attempt / "results/main/status.json").read_text())
        self.assertEqual(result, 1)
        self.assertEqual(calls, [("environment", threading.main_thread().ident),
                                 ("app", False, True, threading.main_thread().ident)])
        self.assertEqual(status["status"], "FAIL")
        self.assertEqual(status["primary_error"]["message"], "constructor failed")
        self.assertEqual([(row["cleanup"], row["error_type"]) for row in status["cleanup_errors"]],
                         [("environment.close", "OSError"), ("app.close", "TimeoutError")])

    def test_success_path_cleanup_failure_cannot_become_pass_and_outer_margin_is_ten_seconds(self):
        events = []
        class Resource:
            h1_initialization_physics_ticks = 0
            h1_initialization_tick_source = "CPU_FAKE_NO_PHYSICS"
            def close(self, *, wait_for_replicator=None, skip_cleanup=None):
                if wait_for_replicator is not None:
                    events.append((wait_for_replicator, skip_cleanup))
                    try: time.sleep(1)
                    except Exception: return
        class Backend:
            def __init__(self, counters, output, environment):
                self.counters=counters; self.output=output; self.environment=environment
            def resource_check(self, *args): pass
            def preserve(self, status, outcome=None):
                live_adapter._persist(self.output, self.counters, status, self.initialization_measurement, outcome)
        with tempfile.TemporaryDirectory() as directory:
            attempt=Path(directory)/"attempt"; (attempt/"results").mkdir(parents=True)
            def factory(**kwargs):
                environment=Resource(); app=Resource(); kwargs["resources"].update(environment=environment,app=app)
                return Backend(kwargs["counters"], attempt/"results/main", environment)
            with patch.object(live_adapter, "_dispatch", lambda *args: None):
                result=live_adapter.run_main_sequence(authorization=authorization(),attempt=attempt,
                    backend_factory=factory,cleanup_timeout_seconds=0.02)
            status=json.loads((attempt/"results/main/status.json").read_text())
        self.assertEqual(events, [(False, True)])
        self.assertEqual(result, 1)
        self.assertEqual(status["status"], "FAIL")
        self.assertEqual([(row["cleanup"],row["error_type"]) for row in status["cleanup_errors"]],
                         [("app.close","TimeoutError")])

        class QuickResource(Resource):
            def close(self, *, wait_for_replicator=None, skip_cleanup=None): pass
        with tempfile.TemporaryDirectory() as directory:
            attempt=Path(directory)/"attempt"; (attempt/"results").mkdir(parents=True)
            def quick_factory(**kwargs):
                environment=QuickResource(); app=QuickResource(); kwargs["resources"].update(environment=environment,app=app)
                return Backend(kwargs["counters"],attempt/"results/main",environment)
            with patch.object(live_adapter,"_dispatch",lambda *args: None):
                self.assertEqual(live_adapter.run_main_sequence(authorization=authorization(),attempt=attempt,
                    backend_factory=quick_factory),0)
            start=json.loads((attempt/"results/main/process_start.json").read_text())
        self.assertEqual(start["cleanup_margin_seconds"],10.0)
        self.assertEqual(start["cleanup_timeouts_seconds"],{"environment.close":0.5,"app.close":9.5})
        self.assertEqual(start["minimum_outer_watchdog_seconds"],authorization()["budgets"]["wall_seconds"]+10.0)


if __name__ == "__main__":
    unittest.main()
