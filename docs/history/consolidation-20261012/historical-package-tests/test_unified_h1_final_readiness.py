from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "a0816149ced88bd46c1e0104413b73050626a7e9"
HISTORICAL_BASE_COMMIT = "524e738a65dc6954fec797899b5e9e21d27d0324"
EXPECTED_REPOSITORY = str(ROOT)
EXPECTED_PRODUCT_REPOSITORY = "/home/mint/rl_dev/instinctRL_orchestrator/implementation_records/instinctrl-first-use-20260915/state/workspaces/instinctrl-r5-stage1-review-evidence-process-successor-20260929/product_snapshot"
PROTECTED_HASHES = {
    "LONG_GOAL_CONTRACT.md": "90db666b6f29a54e259f833ec2b7c252c51903844f600d457f88405f22c56b46",
    "READINESS_INPUTS.json": "ffa454857db84f54339958ef7548c33f198fec71ce0efc4ee995b4a589d0ef72",
    "review_bundle/evidence/a0009/EVIDENCE.json": "c9f1ed80093556ac53f010b185e64886a0c4e7ece33a6e07df325b7ad97d0cb4",
    "review_bundle/evidence/d0005/EVIDENCE.json": "c87d7725c5047dbd5080fabf410e8852869c6ed7445aeac5e6e7ee8856500a26",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalReadinessIntegrationTests(unittest.TestCase):
    def test_protected_inputs_are_byte_exact(self):
        self.assertEqual({name: digest(ROOT / name) for name in PROTECTED_HASHES}, PROTECTED_HASHES)

    def test_config_changes_only_workload_identity_and_local_asset_binding(self):
        relative = "training_workloads/unified_h1/candidate_config.json"
        baseline = json.loads(subprocess.run(
            ["git", "show", f"{HISTORICAL_BASE_COMMIT}:{relative}"], cwd=ROOT,
            check=True, capture_output=True, text=True).stdout)
        current = json.loads((ROOT / relative).read_text())
        baseline_workload = baseline.pop("workload")
        current_workload = current.pop("workload")
        baseline_product = baseline.pop("product")
        current_product = current.pop("product")
        baseline_execution = baseline.pop("execution")
        current_execution = current.pop("execution")
        # The isolated D7 preparation extends only the declared future mode;
        # all other frozen method/runtime fields must remain identical.
        baseline_modes = baseline["future_entry"].pop("modes")
        current_modes = current["future_entry"].pop("modes")
        baseline_sequences = baseline["future_entry"].pop("sequences")
        current_sequences = current["future_entry"].pop("sequences")
        self.assertEqual(current, baseline)
        self.assertEqual(current_modes, [*baseline_modes, "control-diagnostic", "pilot-training", "pilot-evaluation", "enhanced-b0-diagnostic"])
        self.assertEqual({key:current_sequences[key] for key in baseline_sequences},baseline_sequences)
        self.assertEqual(current_sequences["pilot-training"],["cold-start","init0000","u1","u2","u8","u16","u32","u64"])
        self.assertEqual(current_sequences["pilot-evaluation"],["development-only","Z","u16","u64"])
        self.assertNotIn("control-diagnostic",current_sequences)
        self.assertEqual(current_sequences["enhanced-b0-diagnostic"],["validate-only-handoff","one-process-init","Z","T","S","zero-target","no-retry","read-only-post-physics"])
        self.assertEqual(current_product["path"], EXPECTED_PRODUCT_REPOSITORY)
        self.assertEqual(current_product["base_commit"], BASE_COMMIT)
        self.assertEqual(current_product["candidate_commit"], "SELF")
        self.assertEqual(current_product["candidate_tree"], "SELF")
        self.assertEqual(current_product["branch"], "r5-stage1-review-evidence-process-successor-20260929")
        changed_product_sources = {"isaaclab-training/instinctlab/evaluation/raycaster_observability.py", "isaaclab-training/instinctlab/envs/unified_h1_direct_env.py"}
        self.assertEqual(set(current_product["source_sha256"]), set(baseline_product["source_sha256"]))
        product_root=Path(current_product["path"])
        readable_product_root=ROOT/"product_snapshot"
        for relative, expected in current_product["source_sha256"].items():
            self.assertEqual(digest(readable_product_root / relative), expected, relative)
            if relative not in changed_product_sources:
                self.assertEqual(expected, baseline_product["source_sha256"][relative], relative)
        local_asset_fields = {
            "isaacsim_asset_root_env", "isaacsim_asset_root", "isaacsim_asset_manifest_sha256",
        }
        self.assertEqual({key: value for key, value in current_execution.items() if key not in local_asset_fields},
                         baseline_execution)
        self.assertEqual(current_execution["isaacsim_asset_root_env"], "ISAACSIM_ASSET_ROOT")
        self.assertTrue(Path(current_execution["isaacsim_asset_root"]).is_absolute())
        self.assertEqual(len(current_execution["isaacsim_asset_manifest_sha256"]), 64)
        self.assertEqual(set(current_workload), set(baseline_workload))
        self.assertEqual(current_workload["path"], EXPECTED_REPOSITORY)
        self.assertEqual(current_workload["base_commit"], BASE_COMMIT)
        self.assertEqual(current_workload["candidate_commit"], "SELF")
        self.assertEqual(current_workload["candidate_tree"], "SELF")
        self.assertEqual(current_workload["branch"], "r5-stage1-product-pin-fixture-successor-20260929")
        self.assertNotEqual(current_workload["path"], current_product["path"])
        for relative, expected in current_workload["source_sha256"].items():
            self.assertEqual(digest(ROOT / relative), expected, relative)
        self.assertIn("training_workloads/unified_h1/asset_closure.py", current_workload["source_sha256"])

    def test_d0005_sealed_accounting_is_preserved(self):
        evidence = json.loads((ROOT / "review_bundle/evidence/d0005/EVIDENCE.json").read_text())
        accounting = evidence["accounting"]["d0005"]
        self.assertEqual(accounting, {
            "actor_optimizer_steps": 0, "applauncher_constructions": 0,
            "checkpoint_writes": 0, "critic_optimizer_steps": 0,
            "initialization_physics_ticks": 0, "live_starts": 1,
            "ppo_updates": 0, "retries": 0,
            "simulator_initialization_slots_reserved": 1, "transitions": 0,
        })
        self.assertEqual(evidence["status"], "SEALED_FAILURE")
        self.assertEqual(evidence["outcome"]["remaining_owned_processes"], [])

    def test_readiness_inputs_do_not_authorize_a_real_run(self):
        inputs = json.loads((ROOT / "READINESS_INPUTS.json").read_text())
        self.assertFalse(inputs["real_run_authorized"])
        self.assertFalse(inputs["training_authorized"])
        self.assertFalse(inputs["d0006_authorized"])
        self.assertFalse(inputs["budgets_inherited"])


if __name__ == "__main__":
    unittest.main()
