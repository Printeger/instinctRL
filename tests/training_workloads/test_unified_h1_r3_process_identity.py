from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from training_workloads.unified_h1.budget import required_mode_budget
from training_workloads.unified_h1.live_adapter import _process_identity, run_main_sequence


WORKLOAD_ROOT = Path(__file__).resolve().parents[2]


def _authorization():
    return {
        "mode": "main",
        "parameters": {},
        "budgets": required_mode_budget("main", {}),
        "authorization_sha256": "e" * 64,
    }


def test_process_identity_uses_kernel_values_in_current_process():
    identity = _process_identity()
    assert identity["schema_version"] == "unified-h1-process-identity.v1"
    assert identity["pid"] == os.getpid()
    assert identity["parent_pid"] == os.getppid()
    assert identity["process_group_id"] == os.getpgid(0)
    assert identity["session_id"] == os.getsid(0)
    assert identity["is_process_group_leader"] is (os.getpid() == os.getpgid(0))
    assert identity["captured_at_monotonic_ns"] > 0


def test_short_lived_new_session_process_reports_own_pid_and_pgid():
    code = (
        "import json; from training_workloads.unified_h1.live_adapter import _process_identity; "
        "print(json.dumps(_process_identity(), sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=WORKLOAD_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(WORKLOAD_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        start_new_session=True,
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    identity = json.loads(completed.stdout)
    assert completed.stderr == ""
    assert identity["pid"] == identity["process_group_id"] == identity["session_id"]
    assert identity["is_process_group_leader"] is True


def test_failure_path_persists_identity_before_backend_construction(tmp_path: Path):
    attempt = tmp_path / "attempt"
    (attempt / "results").mkdir(parents=True)

    def fail_before_backend(**kwargs):
        raise RuntimeError("short-lived construction failure")

    assert run_main_sequence(
        authorization=_authorization(),
        attempt=attempt,
        backend_factory=fail_before_backend,
    ) == 1
    output = attempt / "results/main"
    identity = json.loads((output / "process_identity.json").read_text())
    status = json.loads((output / "status.json").read_text())
    assert identity["pid"] == os.getpid()
    assert identity["process_group_id"] == os.getpgid(0)
    assert identity["session_id"] == os.getsid(0)
    assert status["status"] == "FAIL"
    assert status["primary_error"]["message"] == "short-lived construction failure"
