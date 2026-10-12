from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from evaluation_tooling.canonical import canonical_load
from evaluation_tooling.runner import EvaluationRunner
from evaluation_tooling.sealer import seal_attempt
import evaluation_tooling.sealer as sealer_module
from evaluation_tooling.validation import validate_sealed
from evaluation_tooling.validation import collect_draft_findings
from tests.evaluation_tooling.helpers import (
    PRODUCT_COMMIT,
    TOOLING_COMMIT,
    make_attempt,
    python_source_argv,
    write_canonical,
)


def command(root: Path, argv: list[str], *, label: str = "command", semantic: str | None = None, cleanup: list[str] | None = None, timeout: float = 5, expected: int = 0):
    return {
        "label": label,
        "argv": argv,
        "cwd": str(root),
        "environment": {},
        "expected_exit": expected,
        "timeout_seconds": timeout,
        "semantic_result_path": semantic,
        "cleanup_argv": cleanup,
    }


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def runner(self, attempt: Path) -> EvaluationRunner:
        return EvaluationRunner(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)

    def run_positive(self) -> Path:
        attempt = make_attempt(self.root)
        runner = self.runner(attempt)
        self.assertEqual(runner.run_all(), 0)
        sealed = seal_attempt(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertEqual(sealed["status"], "PASS")
        return attempt

    def test_positive_lifecycle_and_offline_validation(self) -> None:
        attempt = self.run_positive()
        self.assertEqual(validate_sealed(attempt)["status"], "PASS")

    def test_separate_streams_and_hashes(self) -> None:
        attempt = self.run_positive()
        metadata = canonical_load(next((attempt / "commands").glob("*/metadata.json")))
        self.assertEqual((attempt / metadata["stdout_path"]).read_bytes(), b"fixture-output")
        self.assertEqual((attempt / metadata["stderr_path"]).read_bytes(), b"")
        self.assertEqual(len(metadata["stdout_sha256"]), 64)

    def test_exact_argv_cwd_environment_and_timestamps(self) -> None:
        attempt = self.run_positive()
        metadata = canonical_load(next((attempt / "commands").glob("*/metadata.json")))
        self.assertEqual(metadata["argv"][0:2], [str(Path(sys.executable).resolve()), "-B"])
        self.assertEqual(metadata["argv"][2], str((self.root / "fixture_0000_argv.py").resolve()))
        self.assertEqual(metadata["cwd"], str(self.root))
        self.assertEqual(metadata["environment"], {})
        self.assertLessEqual(metadata["started_at"], metadata["ended_at"])

    def test_raw_exit_failure(self) -> None:
        attempt = make_attempt(self.root, commands=[command(self.root, ["/usr/bin/false"])])
        runner = self.runner(attempt)
        self.assertEqual(runner.run_all(), 1)
        receipt = canonical_load(attempt / "launcher_receipts" / "0001.json")
        self.assertEqual((receipt["raw_exit"], receipt["receipt_status"]), (1, "FAIL"))

    def test_expected_nonzero_is_recorded_but_cannot_pass(self) -> None:
        attempt = make_attempt(self.root, commands=[command(self.root, ["/usr/bin/false"], expected=1)])
        self.assertEqual(self.runner(attempt).run_all(), 1)
        metadata = canonical_load(next((attempt / "commands").glob("*/metadata.json")))
        self.assertEqual((metadata["expected_exit"], metadata["raw_exit"], metadata["receipt_status"]), (1, 1, "FAIL"))

    def test_semantic_exit_failure(self) -> None:
        attempt = make_attempt(self.root, commands=[command(self.root, ["/usr/bin/true"], semantic="results/result.json")], required_results=["results/result.json"])
        runner = self.runner(attempt)
        self.assertEqual(runner.prepare()["status"], "PASS")
        write_canonical(attempt / "results" / "result.json", {"status": "FAIL", "semantic_exit": 8, "failure_class": "PRODUCT_SEMANTIC_FAILURE"})
        receipt, exit_code = runner.run_command(0)
        self.assertEqual(exit_code, 1)
        self.assertIn("PRODUCT_SEMANTIC_FAILURE", receipt.primary_failure)

    def test_missing_pass_receipt_is_nonzero(self) -> None:
        attempt = make_attempt(self.root, commands=[command(self.root, ["/usr/bin/true"], semantic="results/missing.json")], required_results=["results/missing.json"])
        runner = self.runner(attempt)
        runner.prepare()
        receipt, exit_code = runner.run_command(0)
        self.assertEqual((receipt.primary_failure, exit_code), ("MISSING_PASS_EVIDENCE", 1))

    def test_raw_nonzero_prevents_pass_with_semantic_pass(self) -> None:
        result_path = self.root / "attempt" / "results" / "result.json"
        source = (
            "from pathlib import Path\n"
            f"path = Path({str(result_path)!r})\n"
            "path.parent.mkdir(parents=True, exist_ok=True)\n"
            "path.write_bytes(b'{\"semantic_exit\":0,\"status\":\"PASS\"}\\n')\n"
            "raise SystemExit(7)\n"
        )
        argv = python_source_argv(self.root, "semantic_pass_raw_fail", source)
        attempt = make_attempt(
            self.root,
            commands=[command(self.root, argv, semantic="results/result.json")],
            required_results=["results/result.json"],
        )
        runner = self.runner(attempt)
        self.assertEqual(runner.prepare()["status"], "PASS")
        receipt, exit_code = runner.run_command(0)
        self.assertEqual((receipt.raw_exit, receipt.semantic_exit), (7, 0))
        self.assertEqual((receipt.receipt_status, exit_code), ("FAIL", 1))

    def test_runtime_start_rejects_mutated_frozen_manifest(self) -> None:
        attempt = make_attempt(self.root)
        runner = self.runner(attempt)
        runner.prepare()
        manifest = canonical_load(attempt / "ATTEMPT_MANIFEST.json")
        manifest["attempt_id"] = "MUTATED-D0001"
        write_canonical(attempt / "ATTEMPT_MANIFEST.json", manifest)
        receipt, exit_code = runner.run_command(0)
        self.assertEqual((receipt.receipt_status, exit_code), ("FAIL", 1))
        self.assertIn("FROZEN_IDENTITY_LOSS", receipt.primary_failure)

    def test_unknown_launcher_receipt_field_is_rejected(self) -> None:
        attempt = make_attempt(self.root)
        self.runner(attempt).run_all()
        receipt_path = attempt / "launcher_receipts" / "0001.json"
        receipt = canonical_load(receipt_path)
        receipt["unknown"] = True
        write_canonical(receipt_path, receipt)
        findings = collect_draft_findings(
            attempt,
            tooling_commit=TOOLING_COMMIT,
            product_commit=PRODUCT_COMMIT,
            require_complete=True,
        )
        self.assertIn("UNKNOWN_FIELD", {finding.code for finding in findings})

    def test_process_crash_receipt(self) -> None:
        attempt = make_attempt(self.root, commands=[command(self.root, ["/usr/bin/evaluation-tooling-does-not-exist"])])
        runner = self.runner(attempt)
        runner.prepare()
        receipt, _ = runner.run_command(0)
        self.assertTrue(receipt.primary_failure.startswith("PROCESS_CRASH"))

    def test_timeout_receipt(self) -> None:
        attempt = make_attempt(self.root, commands=[command(self.root, ["/usr/bin/sleep", "1"], timeout=0.01)])
        runner = self.runner(attempt)
        self.assertEqual(runner.run_all(), 1)
        receipt = canonical_load(attempt / "launcher_receipts" / "0001.json")
        self.assertEqual(receipt["primary_failure"], "TIMEOUT")

    def test_runtime_failure_stops_before_next_command(self) -> None:
        marker = self.root / "should-not-exist"
        commands = [
            command(self.root, ["/usr/bin/false"], label="fail"),
            command(self.root, ["/usr/bin/touch", str(marker)], label="must_not_run"),
        ]
        attempt = make_attempt(self.root, commands=commands)
        self.assertEqual(self.runner(attempt).run_all(), 1)
        self.assertFalse(marker.exists())

    def test_timeout_cleanup_still_runs(self) -> None:
        marker = self.root / "cleanup-ran"
        attempt = make_attempt(self.root, commands=[command(self.root, ["/usr/bin/sleep", "1"], timeout=0.2, cleanup=["/usr/bin/touch", str(marker)])])
        self.assertEqual(self.runner(attempt).run_all(), 1)
        self.assertTrue(marker.exists())
        receipt = canonical_load(attempt / "launcher_receipts" / "0001.json")
        self.assertEqual((receipt["primary_failure"], receipt["cleanup_status"]), ("TIMEOUT", "PASS"))

    def test_offline_validation_does_not_require_live_inputs(self) -> None:
        attempt = self.run_positive()
        (self.root / "input.dat").unlink()
        self.assertEqual(validate_sealed(attempt)["status"], "PASS")

    def test_cleanup_error_does_not_mask_raw_primary(self) -> None:
        attempt = make_attempt(self.root, commands=[command(self.root, ["/usr/bin/false"], cleanup=["/usr/bin/false"])])
        runner = self.runner(attempt)
        runner.run_all()
        receipt = canonical_load(attempt / "launcher_receipts" / "0001.json")
        self.assertTrue(receipt["primary_failure"].startswith("RAW_EXIT"))
        self.assertEqual(receipt["cleanup_status"], "FAIL")

    def test_draft_repair_same_namespace(self) -> None:
        attempt = make_attempt(self.root, commands=[command(self.root, ["/usr/bin/true"], semantic="results/result.json")], required_results=["results/result.json"])
        runner = self.runner(attempt)
        runner.prepare()
        first, first_exit = runner.run_command(0)
        self.assertEqual(first_exit, 1)
        write_canonical(attempt / "results" / "result.json", {"status": "PASS", "semantic_exit": 0})
        second, second_exit = runner.run_command(0, repair_of=1)
        self.assertEqual((second.receipt_status, second_exit), ("PASS", 0))
        result = seal_attempt(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertEqual(result["status"], "PASS")

    def test_repair_must_reference_matching_failed_command(self) -> None:
        attempt = make_attempt(self.root)
        runner = self.runner(attempt)
        runner.prepare()
        runner.run_command(0)
        with self.assertRaisesRegex(ValueError, "failed command"):
            runner.run_command(0, repair_of=1)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.run_command(0, repair_of=99)

    def test_failed_seal_leaves_draft_without_manifest(self) -> None:
        attempt = make_attempt(self.root, required_results=["results/missing.json"])
        self.runner(attempt).run_all()
        result = seal_attempt(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertEqual((result["status"], result["lifecycle"]), ("FAIL", "DRAFT"))
        self.assertFalse((attempt / "TERMINAL_MANIFEST.json").exists())

    def test_atomic_terminal_creation_failure_restores_draft(self) -> None:
        attempt = make_attempt(self.root)
        self.runner(attempt).run_all()
        original_create = sealer_module.atomic_canonical_create

        def fail_terminal(path, value):
            if path.name == "TERMINAL_MANIFEST.json":
                raise OSError("injected terminal create failure")
            return original_create(path, value)

        with mock.patch.object(sealer_module, "atomic_canonical_create", side_effect=fail_terminal):
            with self.assertRaisesRegex(OSError, "injected"):
                seal_attempt(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertFalse((attempt / "TERMINAL_MANIFEST.json").exists())
        self.assertEqual(canonical_load(attempt / "TOOL_STATE.json")["lifecycle"], "DRAFT")

    def test_preexisting_terminal_rejected(self) -> None:
        attempt = make_attempt(self.root)
        (attempt / "TERMINAL_MANIFEST.json").write_text("occupied")
        with self.assertRaisesRegex(RuntimeError, "pre-existing"):
            seal_attempt(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)

    def test_double_seal_rejected(self) -> None:
        attempt = self.run_positive()
        with self.assertRaisesRegex(RuntimeError, "pre-existing"):
            seal_attempt(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)

    def test_post_seal_mutation_detected(self) -> None:
        attempt = self.run_positive()
        stream = next((attempt / "commands").glob("*/stdout.bin"))
        stream.write_bytes(b"mutated")
        result = validate_sealed(attempt)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("FILE_HASH", {item["code"] for item in result["findings"]})

    def test_complete_inventory_and_sole_exclusion(self) -> None:
        attempt = self.run_positive()
        terminal = canonical_load(attempt / "TERMINAL_MANIFEST.json")
        actual = sorted(path.relative_to(attempt).as_posix() for path in attempt.rglob("*") if path.is_file() and path.name != "TERMINAL_MANIFEST.json")
        self.assertEqual([row["path"] for row in terminal["files"]], actual)
        self.assertEqual(terminal["self_exclusion"], ["TERMINAL_MANIFEST.json"])
        self.assertEqual(terminal["schema_version"], "evaluation-terminal-manifest.v2")
        self.assertTrue(all(row["type"] == "file" for row in terminal["files"]))
        self.assertTrue(all(isinstance(row["mode"], int) for row in terminal["files"]))

    def test_terminal_inventory_detects_mode_mutation(self) -> None:
        attempt = self.run_positive()
        stream = next((attempt / "commands").glob("*/stdout.bin"))
        stream.chmod(stream.stat().st_mode ^ 0o100)
        result = validate_sealed(attempt)
        self.assertIn("FILE_MODE", {item["code"] for item in result["findings"]})

    def test_candidate_terminal_cannot_downgrade_to_v1_inventory(self) -> None:
        attempt = self.run_positive()
        terminal_path = attempt / "TERMINAL_MANIFEST.json"
        terminal = canonical_load(terminal_path)
        terminal["schema_version"] = "evaluation-terminal-manifest.v1"
        for row in terminal["files"]:
            row.pop("type")
            row.pop("mode")
        write_canonical(terminal_path, terminal)
        result = validate_sealed(attempt)
        self.assertIn("SCHEMA_VERSION", {item["code"] for item in result["findings"]})

    def test_terminal_is_last_file_write(self) -> None:
        attempt = self.run_positive()
        terminal = attempt / "TERMINAL_MANIFEST.json"
        others = [path.stat().st_mtime_ns for path in attempt.rglob("*") if path.is_file() and path != terminal]
        self.assertGreaterEqual(terminal.stat().st_mtime_ns, max(others))

    def test_newer_post_seal_timestamp_is_rejected(self) -> None:
        attempt = self.run_positive()
        terminal = attempt / "TERMINAL_MANIFEST.json"
        stream = next((attempt / "commands").glob("*/stdout.bin"))
        newer = terminal.stat().st_mtime_ns + 1_000_000_000
        os.utime(stream, ns=(newer, newer))
        result = validate_sealed(attempt)
        self.assertIn("POST_SEAL_MUTATION", {item["code"] for item in result["findings"]})

    def test_exact_binding_categories(self) -> None:
        attempt = self.run_positive()
        terminal = canonical_load(attempt / "TERMINAL_MANIFEST.json")
        bound = set().union(*(set(paths) for paths in terminal["bindings"].values()))
        self.assertEqual(bound, {row["path"] for row in terminal["files"]})
        self.assertTrue(terminal["bindings"]["streams"])
        self.assertTrue(terminal["bindings"]["metadata"])
        self.assertTrue(terminal["bindings"]["command_ledger"])
        self.assertTrue(terminal["bindings"]["command_index"])
        self.assertTrue(terminal["bindings"]["launcher_receipts"])
        self.assertTrue(terminal["bindings"]["runner_receipts"])

    def test_offline_inspection_never_executes_added_source(self) -> None:
        attempt = self.run_positive()
        marker = self.root / "marker"
        (attempt / "results" / "evil.py").write_text(f"open({str(marker)!r}, 'w').write('bad')\n")
        result = validate_sealed(attempt)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(marker.exists())
