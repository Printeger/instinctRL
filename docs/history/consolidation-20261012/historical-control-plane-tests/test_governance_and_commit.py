from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path


COORD = Path("/home/mint/rl_dev/instinctRL_coordination")
PRODUCT = Path("/home/mint/rl_dev/worktrees/instinctRL-blackwell")
PRODUCT_COMMIT = "6eb521f2174a64030e30ea5e8ce8db1f89006619"
PRODUCT_PARENT = "ebb2bd7894e6b225651f0d2e4d174a39701a0159"
D0006_PARENT = "d4d20f23252844c910d349ca48cc4afe4daa623b"
ALLOWLIST = (
    "verifiers/m1a/", "qualification/runner/", "qualification/manifests/examples/",
    "tests/control_plane_v2/",
)


def git(repo: Path, *argv: str, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *argv], text=True, capture_output=True, check=check
    )


class GovernanceIdentityTests(unittest.TestCase):
    def test_canonical_commander_executor_json_parse(self):
        commander = json.loads((COORD / "commander/COMMANDER_STATE.json").read_text(encoding="utf-8"))
        executor = json.loads((COORD / "executor/EXECUTOR_STATE.json").read_text(encoding="utf-8"))
        self.assertEqual(commander["current_order_id"], "CP2-VAL-001")
        self.assertEqual(executor["current_order_id"], "CP2-VAL-001")
        self.assertEqual(commander["control_plane_version"], 2)
        self.assertEqual(executor["control_plane_version"], 2)

    def test_archived_current_order_byte_identity(self):
        current = (COORD / "commander/CURRENT_ORDER.md").read_bytes()
        archived = (COORD / "commander/orders/CP2-VAL-001.md").read_bytes()
        self.assertEqual(current, archived)


class FrozenProductIdentityTests(unittest.TestCase):
    def test_product_preflight_exact_branch_head_parent_clean(self):
        self.assertEqual(git(PRODUCT, "branch", "--show-current").stdout.strip(), "blackwell-migration")
        self.assertEqual(git(PRODUCT, "rev-parse", "HEAD").stdout.strip(), PRODUCT_COMMIT)
        self.assertEqual(git(PRODUCT, "rev-parse", "HEAD^").stdout.strip(), PRODUCT_PARENT)
        self.assertEqual(git(PRODUCT, "status", "--porcelain=v1").stdout, "")

    def test_product_zero_changes_and_zero_new_commits(self):
        self.assertEqual(git(PRODUCT, "diff", "--name-status", PRODUCT_COMMIT).stdout, "")
        self.assertEqual(git(PRODUCT, "rev-list", "--count", f"{PRODUCT_COMMIT}..HEAD").stdout.strip(), "0")


class CandidateCommitTests(unittest.TestCase):
    def test_final_candidate_parent_and_exact_path_allowlist(self):
        head = git(COORD, "rev-parse", "HEAD").stdout.strip()
        parent = git(COORD, "rev-parse", "HEAD^").stdout.strip()
        self.assertNotEqual(head, D0006_PARENT, "D0006 candidate commit has not been created")
        self.assertEqual(parent, D0006_PARENT)
        changed = git(COORD, "diff-tree", "--no-commit-id", "--name-only", "-r", head).stdout.splitlines()
        self.assertTrue(changed)
        self.assertTrue(all(path.startswith(ALLOWLIST) for path in changed), changed)
        self.assertEqual(git(COORD, "diff", "--check", f"{parent}..{head}").returncode, 0)

    def test_final_candidate_per_commit_show_excludes_governance_history_product(self):
        head = git(COORD, "rev-parse", "HEAD").stdout.strip()
        shown = git(COORD, "show", "--format=", "--name-only", head).stdout.splitlines()
        self.assertTrue(all(path.startswith(ALLOWLIST) for path in shown if path))
        self.assertFalse(any(path.startswith(("commander/", "protocols/", "schemas/", "attempts/", "reports/", "executor/")) for path in shown))
        self.assertEqual(git(PRODUCT, "status", "--porcelain=v1").stdout, "")

    def test_certification_suite_source_inventory_repeatable(self):
        head = git(COORD, "rev-parse", "HEAD").stdout.strip()
        paths = sorted(
            path for path in git(COORD, "ls-tree", "-r", "--name-only", head).stdout.splitlines()
            if path.startswith(ALLOWLIST) and not path.endswith((".pyc", "/__pycache__"))
        )
        first = []
        second = []
        for target in (first, second):
            for path in paths:
                data = subprocess.run(
                    ["git", "-C", str(COORD), "show", f"{head}:{path}"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                ).stdout
                target.append((path, hashlib.sha256(data).hexdigest()))
        self.assertEqual(first, second)
        self.assertTrue(first)


if __name__ == "__main__":
    unittest.main()
