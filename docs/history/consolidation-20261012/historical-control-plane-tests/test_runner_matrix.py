from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qualification.runner import runner


COORD = Path("/home/mint/rl_dev/instinctRL_coordination")
PRODUCT = Path("/home/mint/rl_dev/worktrees/instinctRL-blackwell")
FIXTURES = COORD / "tests/control_plane_v2/fixtures"
PRODUCT_COMMIT = "6eb521f2174a64030e30ea5e8ce8db1f89006619"


def git_text(*args):
    return subprocess.run(
        ["git", "-C", str(COORD), *args], text=True, capture_output=True, check=True
    ).stdout.strip()


def thaw(root: Path):
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        try:
            path.chmod(0o755 if path.is_dir() else 0o644)
        except FileNotFoundError:
            pass


class RunnerMatrixTests(unittest.TestCase):
    def setUp(self):
        self.verifier_commit = git_text("rev-parse", "HEAD")
        self.original_state_path = runner.COMMANDER_STATE_PATH
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        commander = self.root / "commander"
        commander.mkdir()
        record = self.root / "certification_record.md"
        record.write_text("test-only exact certification record\n", encoding="utf-8")
        state = {
            "verifier": {
                "verifier_id": "M1A-VERIFIER",
                "verifier_commit": self.verifier_commit,
                "certification_status": "CERTIFIED",
                "certification_record": "certification_record.md",
                "certification_record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
            }
        }
        self.state_path = commander / "COMMANDER_STATE.json"
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        runner.COMMANDER_STATE_PATH = self.state_path
        self.attempts = []

    def tearDown(self):
        runner.COMMANDER_STATE_PATH = self.original_state_path
        for attempt in self.attempts:
            thaw(attempt)
        self.temp.cleanup()

    def manifest_data(self, fixture="validator_pass.py", mapping="FAIL", timeout=10, extra_inputs=None):
        relpath = f"tests/control_plane_v2/fixtures/{fixture}"
        blob = git_text("rev-parse", f"{self.verifier_commit}:{relpath}")
        content = subprocess.run(
            ["git", "-C", str(COORD), "cat-file", "blob", blob],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        ).stdout
        fixture_path = COORD / relpath
        inputs = list(extra_inputs or [])
        repositories = {
            "product": {
                "kind": "git", "path": str(PRODUCT), "commit": PRODUCT_COMMIT,
                "require_head": True, "require_clean": True,
            },
            "coordination": {"kind": "path", "path": str(COORD)},
            "package_inventory": {
                "kind": "file", "path": str(FIXTURES / "certification_record.md"),
                "sha256": hashlib.sha256((FIXTURES / "certification_record.md").read_bytes()).hexdigest(),
            },
        }
        for index, item in enumerate(inputs):
            repositories[f"test_input_{index}"] = {"kind": "path", "path": str(item)}
        argv = [sys.executable, str(fixture_path)]
        if fixture == "validator_mutate_input.py" and inputs:
            argv.append(str(inputs[0]))
        return {
            "schema_version": "control-plane-v2/qualification-manifest-1",
            "product_order_id": "M1A-001",
            "product_commit": PRODUCT_COMMIT,
            "verifier_commit": self.verifier_commit,
            "runner_commit": self.verifier_commit,
            "manifest_version": "development-test-1",
            "repositories": repositories,
            "validators": [{
                "validator_id": "fixture", "path": relpath, "git_blob": blob,
                "sha256": hashlib.sha256(content).hexdigest(), "argv": argv,
                "cwd": str(COORD), "timeout": timeout,
                "allowed_read_only_inputs": [str(item) for item in inputs], "gates": ["FIXTURE_GATE"],
            }],
            "gates": [{
                "gate_id": "FIXTURE_GATE", "track": "PRODUCT" if mapping == "FAIL" else "VERIFIER",
                "validator_id": "fixture", "expected_exit": 0,
                "required_evidence": ["stdout", "stderr", "exit_code"], "failure_mapping": mapping,
            }],
            "forbidden_command_forms": ["bash -c", "python3 -"],
            "required_artifacts": [
                "ATTEMPT.json", "INPUTS.json", "qualification_manifest", "verifier_inventory.json",
                "command_ledger.jsonl", "INDEX.md", "RESULT.json",
            ],
            "result_policy": {"valid_gate_failure": "FAIL", "tooling_defect": "INVALID"},
        }

    def invoke(self, data, attempt_id="CP2-VAL-001-D9001", attempt_name="attempt"):
        manifest_path = self.root / f"{attempt_name}.manifest.json"
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        attempt = self.root / attempt_name
        self.attempts.append(attempt)
        argv = [
            "run", "--order-id", "M1A-001", "--product-repo", str(PRODUCT),
            "--product-commit", PRODUCT_COMMIT, "--verifier-repo", str(COORD),
            "--verifier-commit", self.verifier_commit, "--manifest", str(manifest_path),
            "--manifest-sha256", hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "--attempt-id", attempt_id, "--attempt-root", str(attempt),
        ]
        return runner.main(argv), attempt, manifest_path, argv

    def result(self, attempt):
        return json.loads((attempt / "RESULT.json").read_text(encoding="utf-8"))

    def test_runner_interface_required_arguments(self):
        with self.assertRaises(SystemExit):
            runner.main(["run", "--order-id", "M1A-001"])

    def test_runner_end_to_end_pass(self):
        exit_code, attempt, _, _ = self.invoke(self.manifest_data(), attempt_name="pass")
        self.assertEqual(exit_code, 0)
        self.assertEqual(self.result(attempt)["classification"], "PASS")

    def test_runner_end_to_end_product_fail(self):
        exit_code, attempt, _, _ = self.invoke(self.manifest_data("validator_fail.py"), attempt_name="fail")
        self.assertEqual(exit_code, 1)
        self.assertEqual(self.result(attempt)["attempt_state"], "ATTEMPT_FAILED")

    def test_runner_end_to_end_verifier_invalid(self):
        exit_code, attempt, _, _ = self.invoke(
            self.manifest_data("validator_fail.py", mapping="INVALID"), attempt_name="invalid"
        )
        self.assertEqual(exit_code, 2)
        self.assertEqual(self.result(attempt)["attempt_state"], "ATTEMPT_INVALID")

    def test_runner_timeout_routes_invalid(self):
        exit_code, attempt, _, _ = self.invoke(
            self.manifest_data("validator_sleep.py", mapping="INVALID", timeout=0.05), attempt_name="timeout"
        )
        self.assertEqual(exit_code, 2)
        ledger = json.loads((attempt / "command_ledger.jsonl").read_text().strip())
        self.assertEqual(ledger["tooling_error"], "timeout")

    def test_runner_signal_routes_invalid(self):
        exit_code, attempt, _, _ = self.invoke(
            self.manifest_data("validator_signal.py", mapping="INVALID"), attempt_name="signal"
        )
        self.assertEqual(exit_code, 2)
        ledger = json.loads((attempt / "command_ledger.jsonl").read_text().strip())
        self.assertEqual(ledger["tooling_error"], "signal:15")

    def test_runner_manifest_mutation_routes_invalid(self):
        data = self.manifest_data("validator_mutate_input.py", mapping="INVALID")
        # The manifest path is known only after invoke setup; build the run explicitly.
        manifest_path = self.root / "mutation.manifest.json"
        data["repositories"]["test_input_0"] = {"kind": "path", "path": str(manifest_path)}
        data["validators"][0]["allowed_read_only_inputs"] = [str(manifest_path)]
        data["validators"][0]["argv"].append(str(manifest_path))
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        attempt = self.root / "mutation"
        self.attempts.append(attempt)
        argv = [
            "run", "--order-id", "M1A-001", "--product-repo", str(PRODUCT), "--product-commit", PRODUCT_COMMIT,
            "--verifier-repo", str(COORD), "--verifier-commit", self.verifier_commit,
            "--manifest", str(manifest_path), "--manifest-sha256", hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "--attempt-id", "CP2-VAL-001-D9002", "--attempt-root", str(attempt),
        ]
        self.assertEqual(runner.main(argv), 2)
        self.assertEqual(self.result(attempt)["reason_code"], "MANIFEST_MUTATED")

    def test_runner_exact_certification_preflight(self):
        state = json.loads(self.state_path.read_text())
        state["verifier"]["certification_status"] = "UNCERTIFIED"
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        exit_code, attempt, _, _ = self.invoke(self.manifest_data(), attempt_name="uncertified")
        self.assertEqual(exit_code, 2)
        self.assertEqual(self.result(attempt)["reason_code"], "VERIFIER_NOT_EXACTLY_CERTIFIED")

    def test_runner_certification_record_hash_preflight(self):
        state = json.loads(self.state_path.read_text())
        state["verifier"]["certification_record_sha256"] = "0" * 64
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        exit_code, attempt, _, _ = self.invoke(self.manifest_data(), attempt_name="cert-hash")
        self.assertEqual(exit_code, 2)
        self.assertEqual(self.result(attempt)["reason_code"], "VERIFIER_CERTIFICATION_RECORD_HASH_MISMATCH")

    def test_runner_manifest_hash_preflight(self):
        data = self.manifest_data()
        manifest_path = self.root / "hash.manifest.json"
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        attempt = self.root / "hash"
        self.attempts.append(attempt)
        argv = [
            "run", "--order-id", "M1A-001", "--product-repo", str(PRODUCT), "--product-commit", PRODUCT_COMMIT,
            "--verifier-repo", str(COORD), "--verifier-commit", self.verifier_commit,
            "--manifest", str(manifest_path), "--manifest-sha256", "0" * 64,
            "--attempt-id", "CP2-VAL-001-D9003", "--attempt-root", str(attempt),
        ]
        self.assertEqual(runner.main(argv), 2)
        self.assertEqual(self.result(attempt)["reason_code"], "MANIFEST_HASH_MISMATCH")

    def test_runner_product_commit_preflight(self):
        data = self.manifest_data()
        data["product_commit"] = "0" * 40
        exit_code, attempt, _, _ = self.invoke(data, attempt_name="product-pin")
        self.assertEqual(exit_code, 2)
        self.assertEqual(self.result(attempt)["reason_code"], "MANIFEST_PRODUCT_PIN_MISMATCH")

    def test_runner_validator_hash_preflight(self):
        data = self.manifest_data()
        data["validators"][0]["sha256"] = "0" * 64
        exit_code, attempt, _, _ = self.invoke(data, attempt_name="validator-hash")
        self.assertEqual(exit_code, 2)
        self.assertEqual(self.result(attempt)["reason_code"], "PROVENANCE_VALIDATOR_PIN_MISMATCH")

    def test_runner_cwd_allowlist_preflight(self):
        data = self.manifest_data()
        data["validators"][0]["cwd"] = "/tmp"
        exit_code, attempt, _, _ = self.invoke(data, attempt_name="cwd")
        self.assertEqual(exit_code, 2)
        self.assertEqual(self.result(attempt)["reason_code"], "MANIFEST_CWD_NOT_ALLOWED")

    def test_runner_input_allowlist_preflight(self):
        data = self.manifest_data()
        data["validators"][0]["argv"].append("/etc/passwd")
        exit_code, attempt, _, _ = self.invoke(data, attempt_name="input")
        self.assertEqual(exit_code, 2)
        self.assertEqual(self.result(attempt)["reason_code"], "MANIFEST_ARG_NOT_ALLOWED")

    def test_runner_frozen_package_hash_preflight(self):
        data = self.manifest_data()
        data["repositories"]["package_inventory"]["sha256"] = "0" * 64
        exit_code, attempt, _, _ = self.invoke(data, attempt_name="package")
        self.assertEqual(exit_code, 2)
        self.assertEqual(self.result(attempt)["reason_code"], "PROVENANCE_FROZEN_HASH_MISMATCH")

    def test_runner_unexpected_frozen_status_preflight(self):
        data = self.manifest_data()
        data["repositories"]["coordination"] = {
            "kind": "git", "path": str(COORD), "commit": self.verifier_commit,
            "require_head": True, "require_clean": True,
        }
        exit_code, attempt, _, _ = self.invoke(data, attempt_name="frozen-status")
        self.assertEqual(exit_code, 2)
        self.assertEqual(self.result(attempt)["reason_code"], "PROVENANCE_WORKTREE_DIRTY")

    def test_runner_failed_inner_git_command_preflight(self):
        with mock.patch.object(runner, "_run_git", return_value=subprocess.CompletedProcess([], 1, "", "forced failure")):
            exit_code, attempt, _, _ = self.invoke(self.manifest_data(), attempt_name="inner-git")
        self.assertEqual(exit_code, 2)
        self.assertEqual(self.result(attempt)["reason_code"], "PROVENANCE_COMMIT_MISSING")

    def test_runner_working_validator_byte_mutation(self):
        fixture = FIXTURES / "validator_pass.py"
        original = fixture.read_bytes()
        try:
            fixture.write_bytes(original + b"# test-only mutation\n")
            exit_code, attempt, _, _ = self.invoke(self.manifest_data(), attempt_name="validator-mutation")
            self.assertEqual(exit_code, 2)
            self.assertIn("WORKTREE_MISMATCH", self.result(attempt)["reason_code"])
        finally:
            fixture.write_bytes(original)

    def test_runner_working_runner_byte_mutation(self):
        source = COORD / "qualification/runner/runner.py"
        original = source.read_bytes()
        try:
            source.write_bytes(original + b"# test-only runner mutation\n")
            exit_code, attempt, _, _ = self.invoke(self.manifest_data(), attempt_name="runner-mutation")
            self.assertEqual(exit_code, 2)
            self.assertEqual(self.result(attempt)["reason_code"], "PROVENANCE_RUNNER_WORKTREE_MISMATCH")
        finally:
            source.write_bytes(original)

    def test_runner_preexisting_and_terminal_attempt_reuse(self):
        data = self.manifest_data()
        exit_code, attempt, _, argv = self.invoke(data, attempt_name="reuse")
        self.assertEqual(exit_code, 0)
        before = hashlib.sha256((attempt / "ATTEMPT.json").read_bytes()).hexdigest()
        self.assertEqual(runner.main(argv), 2)
        after = hashlib.sha256((attempt / "ATTEMPT.json").read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_runner_atomic_terminal_immutability(self):
        exit_code, attempt, _, _ = self.invoke(self.manifest_data(), attempt_name="immutable")
        self.assertEqual(exit_code, 0)
        self.assertEqual(stat.S_IMODE(attempt.stat().st_mode), 0o555)
        for path in attempt.rglob("*"):
            expected = 0o555 if path.is_dir() else 0o444
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected)

    def test_runner_attempt_records_exact_provenance_and_evidence(self):
        exit_code, attempt, _, _ = self.invoke(self.manifest_data(), attempt_name="evidence")
        self.assertEqual(exit_code, 0)
        record = json.loads((attempt / "ATTEMPT.json").read_text())
        inventory = json.loads((attempt / "verifier_inventory.json").read_text())
        ledger = [json.loads(line) for line in (attempt / "command_ledger.jsonl").read_text().splitlines()]
        index = (attempt / "INDEX.md").read_text()
        self.assertEqual(record["product"]["parent"], "ebb2bd7894e6b225651f0d2e4d174a39701a0159")
        self.assertTrue(record["product"]["clean_at_start"] and record["product"]["clean_at_end"])
        self.assertEqual(record["verifier"]["certification_status"], "CERTIFIED")
        self.assertEqual(record["runner"]["commit"], self.verifier_commit)
        self.assertEqual(inventory["verifier_commit"], self.verifier_commit)
        self.assertEqual(len(ledger), 1)
        for field in ("argv", "cwd", "start", "end", "exit_code", "stdout_sha256", "stderr_sha256", "validator_blob", "validator_sha256"):
            self.assertIn(field, ledger[0])
        self.assertIn("ledger rows covered by protocol audit: 1", index)
        self.assertIn("FIXTURE_GATE", index)


if __name__ == "__main__":
    unittest.main()
