import copy
import unittest
from pathlib import Path

from training_workloads.unified_h1.host_cpu_evidence_successor import (
    HOST_ARGV, load_failed_attempt, load_manifest, validate_executor_identity,
)


class HostCpuEvidenceSuccessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.manifest = load_manifest(cls.root)

    def test_exact_successor_identity_and_single_external_host_command(self):
        self.assertEqual(self.manifest["package_id"], "INSTINCTRL-R5-STAGE1-HOST-EXECUTOR-SUCCESSOR-20260929")
        self.assertEqual(self.manifest["task_id"], "T-93e0afe401775fb0")
        self.assertEqual(self.manifest["host_argv"], HOST_ARGV)
        self.assertEqual(HOST_ARGV, ["/home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python", "-m", "training_workloads.unified_h1.host_cpu_evidence_successor"])
        readme = (self.root / "review_evidence/README.md").read_text(encoding="utf-8")
        self.assertIn('["/home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python", "-m", "training_workloads.unified_h1.host_cpu_evidence_successor"]', readme)
        self.assertNotIn('["/usr/bin/python3", "-m", "training_workloads.unified_h1.host_cpu_evidence_v2"]', readme)

    def test_interpreter_identity_drift_fails_closed(self):
        with self.assertRaisesRegex(PermissionError, "interpreter identity drift"):
            validate_executor_identity(self.manifest, executable="/usr/bin/python3", pytest_version="8.3.5", importer=lambda name: object())

    def test_pytest_identity_drift_fails_closed(self):
        with self.assertRaisesRegex(PermissionError, "pytest identity drift"):
            validate_executor_identity(self.manifest, executable=HOST_ARGV[0], pytest_version="7.4.4", importer=lambda name: object())

    def test_required_dependency_drift_fails_closed(self):
        class Pytest:
            __version__ = "8.3.5"
        def importer(name):
            if name == "referencing":
                raise ModuleNotFoundError(name)
            return Pytest()
        with self.assertRaisesRegex(PermissionError, "dependency drift: referencing"):
            validate_executor_identity(self.manifest, executable=HOST_ARGV[0], importer=importer)

    def test_interpreter_manifest_drift_fails_closed(self):
        changed = copy.deepcopy(self.manifest)
        changed["interpreter"]["required_imports"] = ["pytest"]
        with self.assertRaisesRegex(PermissionError, "interpreter manifest drift"):
            validate_executor_identity(changed, executable=HOST_ARGV[0], pytest_version="8.3.5", importer=lambda name: object())

    def test_failed_predecessor_attempt_is_immutable_and_exact(self):
        bound = load_failed_attempt(self.root)
        self.assertEqual(bound["record"]["exit_code"], 1)
        self.assertEqual(bound["record"]["setup_errors"], 8)
        self.assertFalse(bound["record"]["receipt_created"])
        self.assertEqual(bound["record"]["failure"], "ModuleNotFoundError: No module named 'referencing'")


if __name__ == "__main__":
    unittest.main()
