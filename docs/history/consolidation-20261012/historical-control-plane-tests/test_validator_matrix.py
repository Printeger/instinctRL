from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from verifiers.m1a import validators


PRODUCT = Path("/home/mint/rl_dev/worktrees/instinctRL-blackwell")
MANIFEST = PRODUCT / "docs/blackwell_devlog/m1a/TASLAB_SOURCE_MANIFEST.md"
DIFF_DOC = PRODUCT / "docs/blackwell_devlog/m1a/TASLAB_ORIGINAL_VS_MODERN_DIFF.md"
MODERN = Path("/home/mint/rl_blackwell_test/repos/OmniDrones/omni_drones")


class ManifestFixtureMixin:
    def mutated_manifest(self, transform):
        text = MANIFEST.read_text(encoding="utf-8")
        changed = transform(text)
        self.assertNotEqual(changed, text)
        temporary = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
        try:
            temporary.write(changed)
            temporary.close()
            self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
            return Path(temporary.name)
        except Exception:
            temporary.close()
            raise


class SemanticValidatorMatrixTests(ManifestFixtureMixin, unittest.TestCase):
    def test_semantic_representative_positive(self):
        self.assertEqual(validators.validate_semantic(MANIFEST), 0)

    def test_semantic_missing_input(self):
        self.assertNotEqual(validators.validate_semantic(Path("/tmp/cp2-semantic-missing.md")), 0)

    def test_semantic_missing_field(self):
        path = self.mutated_manifest(lambda text: re.sub(r"^10\..*$", "", text, count=1, flags=re.M))
        self.assertNotEqual(validators.validate_semantic(path), 0)

    def test_semantic_duplicate_field(self):
        path = self.mutated_manifest(lambda text: re.sub(r"^(1\..*)$", r"\1\n\1", text, count=1, flags=re.M))
        self.assertNotEqual(validators.validate_semantic(path), 0)

    def test_semantic_incorrect_section(self):
        path = self.mutated_manifest(lambda text: text.replace("## 2. ", "## 99. ", 1))
        self.assertNotEqual(validators.validate_semantic(path), 0)

    def test_semantic_malformed_field(self):
        path = self.mutated_manifest(lambda text: re.sub(r"^5\. ", "5: ", text, count=1, flags=re.M))
        self.assertNotEqual(validators.validate_semantic(path), 0)

    def test_semantic_valid_multiline_and_whitespace(self):
        def transform(text):
            return re.sub(r"^(9\. .{30})(.+)$", r"\1\n   \2", text, count=1, flags=re.M)
        path = self.mutated_manifest(transform)
        self.assertEqual(validators.validate_semantic(path), 0)


class ObjectValidatorMatrixTests(ManifestFixtureMixin, unittest.TestCase):
    def field_mutation(self, field: int, replacement: str):
        return self.mutated_manifest(
            lambda text: re.sub(rf"^{field}\. .*$", f"{field}. {replacement}", text, count=1, flags=re.M)
        )

    def test_object_exact_pin_positive(self):
        self.assertEqual(validators.validate_objects(MANIFEST), 0)

    def test_object_wrong_blob(self):
        self.assertNotEqual(validators.validate_objects(self.field_mutation(4, "`0000000000000000000000000000000000000000`")), 0)

    def test_object_wrong_sha256(self):
        self.assertNotEqual(validators.validate_objects(self.field_mutation(5, "`" + "0" * 64 + "`")), 0)

    def test_object_wrong_size(self):
        self.assertNotEqual(validators.validate_objects(self.field_mutation(6, "1 bytes; `ASCII text`")), 0)

    def test_object_wrong_type(self):
        section = validators.parse_manifest_sections(MANIFEST.read_text(encoding="utf-8"))[0][1]
        size = re.search(r"(\d+)", section[6]).group(1)
        self.assertNotEqual(validators.validate_objects(self.field_mutation(6, f"{size} bytes; `wrong type`")), 0)

    def test_object_missing_path(self):
        self.assertNotEqual(validators.validate_objects(self.field_mutation(3, "`missing/path.py`")), 0)

    def test_object_missing_commit(self):
        self.assertNotEqual(validators.validate_objects(self.field_mutation(2, "`0000000000000000000000000000000000000000`")), 0)

    def test_object_missing_repository(self):
        with mock.patch.object(validators, "_repo_for_path", return_value="/tmp/cp2-repository-missing"):
            self.assertNotEqual(validators.validate_objects(MANIFEST), 0)

    def test_object_missing_input(self):
        self.assertNotEqual(validators.validate_objects(Path("/tmp/cp2-object-input-missing.md")), 0)

    def test_object_proven_absence_positive(self):
        self.assertEqual(validators.validate_objects(MANIFEST), 0)

    def test_object_false_absence_negative(self):
        path = self.mutated_manifest(lambda text: text.replace("proven absent", "claimed absent", 1))
        self.assertNotEqual(validators.validate_objects(path), 0)


class HistoryValidatorMatrixTests(ManifestFixtureMixin, unittest.TestCase):
    def test_history_exact_pin_positive(self):
        self.assertEqual(validators.validate_history(MANIFEST), 0)

    def test_history_false_claimed_commit(self):
        path = self.mutated_manifest(
            lambda text: re.sub(r"^7\. (.*)$", r"7. \1 `0000000000000000000000000000000000000000`", text, count=1, flags=re.M)
        )
        self.assertNotEqual(validators.validate_history(path), 0)

    def test_history_moving_head_substitution(self):
        original = MANIFEST.read_text(encoding="utf-8")
        path = self.mutated_manifest(lambda text: re.sub(r"^2\. .*$", "2. `HEAD`", text, count=1, flags=re.M))
        self.assertNotEqual(validators.validate_history(path), 0)
        self.assertIn("6f6dee3da72d0b0202846b330adcaeeec8ee7bc3", original)

    def test_history_modern_shallow_boundary(self):
        proc = subprocess.run(
            ["git", "-C", "/home/mint/rl_blackwell_test/repos/OmniDrones", "rev-list", "--parents", "-n", "1", "f8f65d3c6ea3e243335f3444d58dfe241a7655b5"],
            text=True, capture_output=True, check=True,
        )
        self.assertEqual(len(proc.stdout.split()), 1)
        self.assertEqual(validators.validate_history(MANIFEST), 0)

    def test_history_isaac_one_commit_shallow(self):
        proc = subprocess.run(
            ["git", "-C", "/home/mint/rl_blackwell_test/repos/IsaacLab", "rev-list", "--parents", "-n", "1", "37ddf626871758333d6ed89cf64ad702aef127d0"],
            text=True, capture_output=True, check=True,
        )
        self.assertEqual(len(proc.stdout.split()), 1)
        self.assertEqual(validators.validate_history(MANIFEST), 0)

    def test_history_malformed_input(self):
        path = self.mutated_manifest(lambda text: text.replace("## 1. ", "## bad. ", 1))
        self.assertNotEqual(validators.validate_history(path), 0)

    def test_history_missing_input(self):
        self.assertNotEqual(validators.validate_history(Path("/tmp/cp2-history-input-missing.md")), 0)


class StaticContractMatrixTests(unittest.TestCase):
    def build_fixture(self, root: Path):
        legacy_source = PRODUCT / "isaac-training/third_party/OmniDrones/omni_drones"
        legacy_target = root / "isaac-training/third_party/OmniDrones/omni_drones"
        modern_target = root / "modern/omni_drones"
        doc_target = root / "docs/blackwell_devlog/m1a/TASLAB_ORIGINAL_VS_MODERN_DIFF.md"
        doc_target.parent.mkdir(parents=True)
        shutil.copy2(DIFF_DOC, doc_target)
        for relative in (
            "utils/torchrl/transforms.py", "controllers/lee_position_controller.py",
            "actuators/rotor_group.py", "robots/drone/multirotor.py",
        ):
            target = legacy_target / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_source / relative, target)
        for relative in ("utils/torchrl/transforms.py", "robots/drone/multirotor.py"):
            target = modern_target / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(MODERN / relative, target)
        return doc_target, legacy_target, modern_target

    def test_static_contract_positive_and_multiline_whitespace(self):
        self.assertEqual(validators.validate_static(PRODUCT), 0)

    def test_static_contract_each_required_fact_removed(self):
        facts = [
            ("doc", "affine-normalized"), ("doc", "not a guaranteed [-1, 1] value"),
            ("doc", "Per-rotor thrust forces are applied"), ("doc", "aggregate reaction torque and body force are applied"),
            ("doc", "controller/adapter boundary"), ("doc", "not an authorized actor observation field"),
            ("doc", "actor-contract compliance"),
            ("legacy/utils/torchrl/transforms.py", 'action_key: str = ("agents", "action")'),
            ("legacy/utils/torchrl/transforms.py", 'in_keys_inv=[("info", "drone_state")]'),
            ("legacy/utils/torchrl/transforms.py", "torch.nan_to_num_(cmds, 0.)"),
            ("legacy/controllers/lee_position_controller.py", "cmd = (cmd / self.max_thrusts) * 2 - 1"),
            ("legacy/actuators/rotor_group.py", "target_throttle = self.f_inv(torch.clamp((cmds + 1) / 2, 0, 1))"),
            ("legacy/robots/drone/multirotor.py", "self.rotors_view.apply_forces_and_torques_at_pos("),
            ("legacy/robots/drone/multirotor.py", "self.base_link.apply_forces_and_torques_at_pos("),
            ("modern/utils/torchrl/transforms.py", 'action_key: str = ("agents", "action")'),
            ("modern/utils/torchrl/transforms.py", 'in_keys_inv=[("info", "drone_state")]'),
            ("modern/utils/torchrl/transforms.py", "torch.nan_to_num_(cmds, 0.)"),
            ("modern/utils/torchrl/transforms.py", "cmds = self.controller.compute("),
            ("modern/robots/drone/multirotor.py", "self.rotors_view.apply_forces_and_torques_at_pos("),
            ("modern/robots/drone/multirotor.py", "self.base_link.apply_forces_and_torques_at_pos("),
        ]
        for location, needle in facts:
            with self.subTest(location=location, needle=needle), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                doc, legacy, modern = self.build_fixture(root)
                if location == "doc":
                    target = doc
                elif location.startswith("legacy/"):
                    target = legacy / location.removeprefix("legacy/")
                else:
                    target = modern / location.removeprefix("modern/")
                content = re.sub(r"\s+", " ", target.read_text(encoding="utf-8")).replace("`", "")
                self.assertIn(needle, content)
                target.write_text(content.replace(needle, "REMOVED_FACT"), encoding="utf-8")
                self.assertNotEqual(validators.validate_static(root, modern), 0)

    def test_static_contract_actor_boundary_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            doc, _, modern = self.build_fixture(root)
            with doc.open("a", encoding="utf-8") as stream:
                stream.write("\nDEPLOYED_ACTOR_INPUT: pose\n")
            self.assertNotEqual(validators.validate_static(root, modern), 0)


class LedgerValidatorMatrixTests(unittest.TestCase):
    def validate_line(self, line):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as stream:
            stream.write(line + "\n")
            path = Path(stream.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return validators.validate_ledger(path)

    def test_ledger_validator_allows_persisted_file(self):
        self.assertEqual(self.validate_line("python3 -m module explicit-file.json"), 0)

    def test_ledger_validator_missing_input(self):
        self.assertNotEqual(validators.validate_ledger(Path("/tmp/cp2-ledger-missing.jsonl")), 0)

    def test_ledger_validator_rejects_policy_matrix(self):
        commands = [
            "bash -c true", "python3 -c pass", "python3 -", "bash heredoc", "bash <<< x",
            "tool <(writer)", "git add file", "git commit", "git push", "git merge",
            "git rebase main", "git config x y", "pip install x", "conda remove x",
            "apt install x", "curl example.invalid", "wget example.invalid", "runtime smoke",
            "training run", "evaluation run", "mid360 run", "omni_drones run",
        ]
        for command in commands:
            with self.subTest(command=command):
                self.assertNotEqual(self.validate_line(command), 0)


if __name__ == "__main__":
    unittest.main()
