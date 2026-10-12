import json
import tempfile
import unittest
from pathlib import Path

from training_workloads.unified_h1.host_cpu_evidence_source_closure_successor import (
    MANIFEST_PATH,
    load_manifest,
    repository_source_closure,
    validate_source_inventory,
)


class SourceClosureSuccessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.manifest = load_manifest(cls.root)

    def test_named_minimum_omissions_and_runner_manifest_are_hash_bound(self):
        required = {
            "control_snapshot/rlflow/__init__.py",
            "control_snapshot/rlflow/authorizations.py",
            "control_snapshot/rlflow/jobs.py",
            "control_snapshot/rlflow/policy.py",
            "control_snapshot/rlflow/store.py",
            "control_snapshot/tests/test_packages.py",
            "training_workloads/unified_h1/__init__.py",
            "training_workloads/unified_h1/local_dispatch.py",
            "training_workloads/unified_h1/host_cpu_evidence_source_closure_successor.py",
            MANIFEST_PATH,
        }
        self.assertTrue(required.issubset(self.manifest["source_paths"]))
        self.assertEqual(
            self.manifest["source_closure_paths"],
            repository_source_closure(self.root, [row["path"] for row in self.manifest["modules"]]),
        )

    def test_omitted_imported_source_fails_closed(self):
        closure = self.manifest["source_closure_paths"]
        omitted = "control_snapshot/rlflow/jobs.py"
        with self.assertRaisesRegex(PermissionError, "not hash-bound: " + omitted):
            validate_source_inventory(closure, [path for path in self.manifest["source_paths"] if path != omitted])

    def test_duplicate_declared_root_and_inventory_fail_closed(self):
        seed = self.manifest["modules"][0]["path"]
        with self.assertRaisesRegex(PermissionError, "duplicated"):
            repository_source_closure(self.root, [seed, seed])
        with self.assertRaisesRegex(PermissionError, "duplicated"):
            validate_source_inventory([seed], [seed, seed])

    def test_path_escape_fails_closed(self):
        with self.assertRaisesRegex(PermissionError, "unsafe host evidence path"):
            repository_source_closure(self.root, ["../outside.py"])

    def test_ambiguous_local_import_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "control_snapshot/tests").mkdir(parents=True)
            (root / "control_snapshot/ambiguous.py").write_text("VALUE = 1\n")
            (root / "ambiguous.py").write_text("VALUE = 2\n")
            (root / "seed.py").write_text("import ambiguous\n")
            with self.assertRaisesRegex(PermissionError, "ambiguous repository-local import"):
                repository_source_closure(root, ["seed.py"])


if __name__ == "__main__":
    unittest.main()
