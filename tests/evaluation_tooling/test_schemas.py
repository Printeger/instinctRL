from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import sys
import unittest

from evaluation_tooling.canonical import canonical_dumps
from evaluation_tooling.runner import load_submission, preflight
from evaluation_tooling.schemas import validate_input_hashes, validate_manifest
from tests.evaluation_tooling.helpers import PRODUCT_COMMIT, TOOLING_COMMIT, make_attempt


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.attempt = make_attempt(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifest(self):
        import json
        return json.loads((self.attempt / "ATTEMPT_MANIFEST.json").read_text())

    def input_hashes(self):
        import json
        return json.loads((self.attempt / "INPUT_HASHES.json").read_text())

    def codes(self, findings):
        return {finding.code for finding in findings}

    def test_valid_schemas(self) -> None:
        manifest = self.manifest()
        self.assertEqual(validate_manifest(manifest, self.attempt), ())
        self.assertEqual(validate_input_hashes(self.input_hashes(), manifest), ())

    def test_unknown_field(self) -> None:
        value = self.manifest()
        value["surprise"] = True
        self.assertIn("UNKNOWN_FIELD", self.codes(validate_manifest(value)))

    def test_missing_field(self) -> None:
        value = self.manifest()
        del value["float_policy"]
        self.assertIn("MISSING_FIELD", self.codes(validate_manifest(value)))

    def test_bad_schema_version(self) -> None:
        value = self.manifest()
        value["schema_version"] = "future"
        self.assertIn("SCHEMA_VERSION", self.codes(validate_manifest(value)))

    def test_relative_executable_rejected(self) -> None:
        value = self.manifest()
        value["commands"][0]["argv"][0] = "python"
        self.assertIn("ARGV", self.codes(validate_manifest(value)))

    def test_dynamic_code_rejected(self) -> None:
        value = self.manifest()
        value["commands"][0]["argv"] = [str(Path(sys.executable).resolve()), "-c", "print(1)"]
        self.assertIn("DYNAMIC_CODE", self.codes(validate_manifest(value)))

    def test_unpinned_module_execution_rejected(self) -> None:
        value = self.manifest()
        value["commands"][0]["argv"] = [str(Path(sys.executable).resolve()), "-m", "unittest"]
        self.assertIn("UNPINNED_SOURCE", self.codes(validate_manifest(value)))

    def test_attempt_local_source_rejected(self) -> None:
        local = self.attempt / "runner.py"
        local.write_text("pass\n")
        value = self.manifest()
        value["commands"][0]["argv"] = [str(Path(sys.executable).resolve()), str(local)]
        self.assertIn("ATTEMPT_LOCAL_EXECUTABLE", self.codes(validate_manifest(value, self.attempt)))

    def test_wrong_interpreters_wrappers_and_loaders_rejected(self) -> None:
        value = self.manifest()
        for executable in ("/usr/bin/env", "/usr/bin/timeout", "/usr/bin/xargs", "/lib64/ld-linux-x86-64.so.2"):
            with self.subTest(executable=executable):
                value["commands"][0]["argv"] = [executable, "/usr/bin/true"]
                self.assertIn("LAUNCH_PROFILE", self.codes(validate_manifest(value)))

    def test_wrapper_hidden_dynamic_relative_generated_and_nested_dispatch_rejected(self) -> None:
        value = self.manifest()
        cases = [
            ["/usr/bin/env", "python3", "-c", "print(1)"],
            ["/usr/bin/env", "python3", "relative_payload.py"],
            ["/usr/bin/env", "python3", str(self.root / "generated.py")],
            ["/usr/bin/timeout", "5", "/usr/bin/env", "python3", "payload.py"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                value["commands"][0]["argv"] = argv
                self.assertIn("LAUNCH_PROFILE", self.codes(validate_manifest(value)))

    def test_interpreter_without_absolute_source_rejected(self) -> None:
        value = self.manifest()
        value["commands"][0]["argv"] = [str(Path(sys.executable).resolve())]
        self.assertIn("LAUNCH_PROFILE", self.codes(validate_manifest(value)))

    def test_cleanup_dispatch_wrapper_rejected(self) -> None:
        value = self.manifest()
        value["commands"][0]["cleanup_argv"] = ["/usr/bin/env", "python3", "payload.py"]
        self.assertIn("LAUNCH_PROFILE", self.codes(validate_manifest(value)))

    def test_cwd_confinement(self) -> None:
        value = self.manifest()
        value["commands"][0]["cwd"] = "/"
        self.assertIn("PATH", self.codes(validate_manifest(value)))

    def test_input_confinement(self) -> None:
        manifest = self.manifest()
        value = self.input_hashes()
        value["files"][0]["path"] = "/etc/hosts"
        self.assertIn("PATH", self.codes(validate_input_hashes(value, manifest)))

    def test_duplicate_input(self) -> None:
        manifest = self.manifest()
        value = self.input_hashes()
        value["files"].append(copy.deepcopy(value["files"][0]))
        self.assertIn("DUPLICATE", self.codes(validate_input_hashes(value, manifest)))

    def test_missing_input_collects(self) -> None:
        (self.root / "input.dat").unlink()
        result = preflight(self.attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertIn("MISSING_INPUT", {item["code"] for item in result["findings"]})

    def test_hash_mismatch(self) -> None:
        (self.root / "input.dat").write_text("changed")
        result = preflight(self.attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertIn("INPUT_HASH", {item["code"] for item in result["findings"]})

    def test_extra_builder_file(self) -> None:
        (self.attempt / "notes.txt").write_text("not allowed")
        _, _, findings = load_submission(self.attempt)
        self.assertIn("EXTRA_SUBMISSION", self.codes(findings))

    def test_attempt_local_program_file(self) -> None:
        (self.attempt / "sealer.py").write_text("pass")
        _, _, findings = load_submission(self.attempt)
        self.assertIn("ATTEMPT_LOCAL_EXECUTABLE", self.codes(findings))

    def test_multiple_simultaneous_defects_stable(self) -> None:
        (self.attempt / "ATTEMPT_MANIFEST.json").write_bytes(b'{"schema_version":"bad","schema_version":"again"}\n')
        (self.attempt / "INPUT_HASHES.json").unlink()
        (self.attempt / "extra.txt").write_text("x")
        first = preflight(self.attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        second = preflight(self.attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertEqual(first["findings"], second["findings"])
        self.assertGreaterEqual(len(first["findings"]), 3)

    def test_noncanonical_submission(self) -> None:
        value = self.manifest()
        (self.attempt / "ATTEMPT_MANIFEST.json").write_text(__import__("json").dumps(value, indent=2))
        _, _, findings = load_submission(self.attempt)
        self.assertIn("CANONICAL_JSON", self.codes(findings))

    def test_malformed_top_level_collects_without_crash(self) -> None:
        (self.attempt / "ATTEMPT_MANIFEST.json").write_bytes(canonical_dumps([]))
        result = preflight(self.attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("TYPE", {item["code"] for item in result["findings"]})

    def test_persisted_schema_documents_are_canonical(self) -> None:
        from evaluation_tooling.canonical import canonical_load
        schema_root = Path(__file__).resolve().parents[2] / "evaluation_tooling" / "schema_documents"
        versions = {canonical_load(path)["schema_version"] for path in sorted(schema_root.glob("*.json"))}
        self.assertEqual(versions, {
            "evaluation-attempt-manifest.v1",
            "evaluation-input-hashes.v1",
            "evaluation-terminal-manifest.v1",
            "evaluation-terminal-manifest.v2",
        })
