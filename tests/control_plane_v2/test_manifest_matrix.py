from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from qualification.runner import manifest as manifest_mod


COORD = Path("/home/mint/rl_dev/instinctRL_coordination")
EXAMPLE = COORD / "qualification/manifests/examples/valid_manifest.json"


class ManifestSchemaMatrixTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def load(self, data):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return manifest_mod.load_manifest(path)

    def assert_invalid(self, data):
        with self.assertRaises(ValueError):
            self.load(data)

    def test_manifest_schema_positive(self):
        self.assertEqual(self.load(self.data)["product_order_id"], "M1A-001")

    def test_manifest_schema_malformed_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            path.write_text("{malformed", encoding="utf-8")
            with self.assertRaises(ValueError):
                manifest_mod.load_manifest(path)

    def test_manifest_schema_missing_input(self):
        with self.assertRaises(ValueError):
            manifest_mod.load_manifest(Path("/tmp/cp2-missing-manifest.json"))

    def test_manifest_schema_missing_each_top_level_field(self):
        for field in sorted(manifest_mod.REQUIRED_TOP_LEVEL):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data.pop(field)
                self.assert_invalid(data)

    def test_manifest_schema_unknown_top_level_field(self):
        data = copy.deepcopy(self.data)
        data["unexpected"] = True
        self.assert_invalid(data)

    def test_manifest_schema_unknown_validator_and_gate_fields(self):
        for target, field in (("validators", "body"), ("gates", "script")):
            with self.subTest(target=target):
                data = copy.deepcopy(self.data)
                data[target][0][field] = "print('embedded')"
                self.assert_invalid(data)

    def test_manifest_schema_rejects_embedded_executable_body(self):
        data = copy.deepcopy(self.data)
        data["validators"][0]["argv"].append("\nprint('embedded')")
        self.assert_invalid(data)

    def test_manifest_schema_rejects_duplicate_ids(self):
        for key in ("validators", "gates"):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data[key].append(copy.deepcopy(data[key][0]))
                self.assert_invalid(data)

    def test_manifest_schema_rejects_gate_mapping_mismatch(self):
        data = copy.deepcopy(self.data)
        data["validators"][0]["gates"] = ["OTHER"]
        self.assert_invalid(data)

    def test_manifest_schema_rejects_incomplete_evidence(self):
        data = copy.deepcopy(self.data)
        data["gates"][0]["required_evidence"] = ["exit_code"]
        self.assert_invalid(data)

    def test_manifest_schema_rejects_nonproduct_fail_mapping(self):
        data = copy.deepcopy(self.data)
        data["gates"][0]["track"] = "VERIFIER"
        self.assert_invalid(data)

    def test_manifest_schema_valid_repository_integrity_specs(self):
        data = copy.deepcopy(self.data)
        data["repositories"]["product"] = {
            "kind": "git",
            "path": data["repositories"]["product"],
            "commit": data["product_commit"],
            "require_head": True,
            "require_clean": True,
        }
        self.assertEqual(self.load(data)["repositories"]["product"]["kind"], "git")

    def test_manifest_hash_mismatch_is_detectable(self):
        original = EXAMPLE.read_bytes()
        declared = hashlib.sha256(original).hexdigest()
        self.assertNotEqual(hashlib.sha256(original + b"\n").hexdigest(), declared)


class CommandPolicyMatrixTests(unittest.TestCase):
    def test_ledger_policy_allows_persisted_module_and_file(self):
        self.assertFalse(manifest_mod.is_forbidden_argv(["python3", "-m", "module", "/tmp/input.json"]))
        self.assertFalse(manifest_mod.is_forbidden_argv(["python3", "/tmp/persisted.py", "/tmp/input.json"]))

    def test_ledger_policy_rejects_each_forbidden_form(self):
        cases = {
            "shell-c": ["bash", "-c", "true"],
            "interpreter-c": ["python3", "-c", "print(1)"],
            "stdin-body": ["python3", "-"],
            "heredoc": ["bash", "heredoc"],
            "here-string": ["bash", "<<<", "x"],
            "process-substitution": ["tool", "<(writer)"],
            "writer": ["rm", "/tmp/input"],
            "package-mutation": ["pip", "install", "x"],
            "network": ["curl", "https://example.invalid"],
            "git-mutation": ["git", "reset", "--hard"],
            "runtime": ["python3", "/tmp/isaaclab-runtime.py"],
            "multiline-body": ["python3", "/tmp/persisted.py", "\nprint(1)"],
        }
        for name, argv in cases.items():
            with self.subTest(name=name):
                self.assertTrue(manifest_mod.is_forbidden_argv(argv))


if __name__ == "__main__":
    unittest.main()
