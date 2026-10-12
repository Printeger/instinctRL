import hashlib, json, os, tempfile, unittest
from pathlib import Path
from training_workloads.unified_h1.host_cpu_evidence import BASE_COMMIT, PACKAGE_ID, PREDECESSOR, SCHEMA, TASK_ID, load_manifest, source_inventory, verify_host_cpu_evidence


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(data)
        while view: view = view[os.write(fd, view):]
    finally: os.close(fd)


class HostCpuEvidenceTests(unittest.TestCase):
    def fixture(self, root: Path, receipt_change=None, mutable_receipt=False, bad_digest=False) -> Path:
        (root / "review_evidence").mkdir()
        (root / "src.py").write_text("frozen source\n", encoding="utf-8")
        sealed(root / "review_evidence/host_cpu.stdout", b"ok\n")
        sealed(root / "review_evidence/host_cpu.stderr", b"Ran 1 test\nOK\n")
        argv = ["/usr/bin/python3", "-m", "unittest", "-v", "example"]
        manifest = {"schema_version": SCHEMA, "package_id": PACKAGE_ID, "task_id": TASK_ID, "base_commit": BASE_COMMIT, "test_argv": argv, "source_paths": ["src.py"]}
        (root / "review_evidence/host_cpu_test_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        rows = lambda names: [{"path": name, "bytes": (root/name).stat().st_size, "sha256": sha(root/name)} for name in names]
        receipt = {"schema_version": SCHEMA, "package_id": PACKAGE_ID, "task_id": TASK_ID, "base_commit": BASE_COMMIT,
                   "executor": {"kind": "external-host-cpu", "python": "3.x", "platform": "host", "fork": True, "signals": True},
                   "predecessor": PREDECESSOR, "test_argv": argv,
                   "result": {"exit_code": 0, "tests_run": 1, "failures": 0, "errors": 0, "skipped": 0},
                   "outputs": rows(["review_evidence/host_cpu.stdout", "review_evidence/host_cpu.stderr"]), "sources": rows(["src.py"])}
        if receipt_change: receipt.update(receipt_change)
        receipt_path = root / "review_evidence/HOST_CPU_EVIDENCE.json"
        data = json.dumps(receipt).encode()
        receipt_path.write_bytes(data) if mutable_receipt else sealed(receipt_path, data)
        digest = "0" * 64 if bad_digest else sha(receipt_path)
        sealed(root / "review_evidence/HOST_CPU_EVIDENCE.sha256", (digest + "\n").encode("ascii"))
        return receipt_path

    def test_current_manifest_is_exact_and_every_test_module_is_hash_bound(self):
        root = Path(__file__).resolve().parents[2]
        manifest = load_manifest(root)
        prefix = ["/usr/bin/env", "PYTHONPATH=control_snapshot:control_snapshot/tests",
                  "/usr/bin/python3", "-m", "unittest", "-v"]
        self.assertEqual(manifest["test_argv"][:len(prefix)], prefix)
        bound = set(manifest["source_paths"])
        for module in manifest["test_argv"][len(prefix):]:
            self.assertIn(module.replace(".", "/") + ".py", bound)
        self.assertEqual([row["path"] for row in source_inventory(root, manifest["source_paths"])],
                         manifest["source_paths"])

    def test_absent_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as td, self.assertRaisesRegex(PermissionError, "absent"):
            verify_host_cpu_evidence(Path(td))

    def test_regular_immutable_hash_bound_receipt_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.fixture(root)
            self.assertEqual(verify_host_cpu_evidence(root)["result"]["tests_run"], 1)

    def test_mutable_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.fixture(root, mutable_receipt=True)
            with self.assertRaisesRegex(PermissionError, "mutable"):
                verify_host_cpu_evidence(root)

    def test_tampered_receipt_digest_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.fixture(root, bad_digest=True)
            with self.assertRaisesRegex(PermissionError, "hash mismatch"):
                verify_host_cpu_evidence(root)

    def test_stale_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.fixture(root); (root / "src.py").write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "source drift"):
                verify_host_cpu_evidence(root)

    def test_argv_and_predecessor_drift_fail_closed(self):
        for field, value in (("test_argv", []), ("predecessor", {})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                root = Path(td); self.fixture(root, {field: value})
                with self.assertRaisesRegex(PermissionError, "argv drift|predecessor provenance mismatch"):
                    verify_host_cpu_evidence(root)


if __name__ == "__main__":
    unittest.main()
