from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_at_commit(commit: str, relative: str) -> str:
    value = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                           check=True, stdout=subprocess.PIPE).stdout
    return hashlib.sha256(value).hexdigest()


class CurrentHandoffBindingTests(unittest.TestCase):
    def test_current_config_manifest_and_b0_hash_graph(self):
        config_path = ROOT / "training_workloads/unified_h1/candidate_config.json"
        manifest_path = ROOT / "r5_delivery/CANDIDATE_SOURCE_MANIFEST.json"
        spec_path = ROOT / "r5_delivery/B0_SPEC_PREIMAGE.json"
        auth_path = ROOT / "r5_delivery/B0_AUTHORIZATION_PREIMAGE.json"
        executed_commit = "0c1a7c05ee8a721590ddf2427c557dc4fecd6434"
        config_sha = digest_at_commit(executed_commit, "training_workloads/unified_h1/candidate_config.json")
        manifest_sha = digest(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        spec = json.loads(spec_path.read_text())
        wrapper = json.loads(auth_path.read_text())
        payload = wrapper["authorization_payload"]
        self.assertEqual(manifest["files_sha256"]["training_workloads/unified_h1/candidate_config.json"], config_sha)
        for relative, expected in manifest["files_sha256"].items():
            self.assertEqual(digest_at_commit(executed_commit, relative), expected)
        self.assertEqual(spec["candidate_binding"]["config_sha256"], config_sha)
        self.assertEqual(spec["candidate_binding"]["source_manifest_sha256"], manifest_sha)
        self.assertEqual(spec["workload"]["source_manifest_sha256"], manifest_sha)
        self.assertEqual(wrapper["candidate_source_manifest_sha256"], manifest_sha)
        self.assertEqual(wrapper["spec_preimage_sha256"], digest(spec_path))
        self.assertEqual(payload["config_sha256"], config_sha)

    def test_postrun_analyzer_cleanup_repair_manifest_binds_current_bytes(self):
        manifest = json.loads((ROOT / "r5_delivery/POSTRUN_ANALYZER_CLEANUP_REPAIR_MANIFEST.json").read_text())
        self.assertEqual(manifest["base_commit"], "0c1a7c05ee8a721590ddf2427c557dc4fecd6434")
        self.assertEqual(manifest["executed_b0_job"], "J-ea67539f3baab11d")
        self.assertEqual(manifest["live_rerun_authorized"], False)
        successor_changes={
            "training_workloads/unified_h1/live_adapter.py",
            "training_workloads/unified_h1/candidate_config.json",
            "tests/training_workloads/test_d7_cpu_unittest.py",
            "tests/training_workloads/test_d7_handoff_bindings_unittest.py",
        }
        for relative, expected in manifest["files_sha256"].items():
            if relative in successor_changes:
                self.assertNotEqual(digest(ROOT / relative), expected)
            else:
                self.assertEqual(digest(ROOT / relative), expected)

    def test_installed_close_source_evidence_is_exactly_bound(self):
        evidence_path = ROOT / "r5_delivery/INSTALLED_SIMULATION_APP_CLOSE_EVIDENCE.json"
        manifest = json.loads((ROOT / "r5_delivery/CANDIDATE_SOURCE_MANIFEST.json").read_text())
        evidence = json.loads(evidence_path.read_text())
        self.assertEqual(
            manifest["files_sha256"]["r5_delivery/INSTALLED_SIMULATION_APP_CLOSE_EVIDENCE.json"],
            digest(evidence_path),
        )
        self.assertEqual(evidence["status"], "READ_ONLY_SOURCE_INSPECTION_NO_IMPORT_NO_RUNTIME_INITIALIZATION")
        self.assertEqual(
            evidence["source"]["path"],
            "/home/mint/miniconda3/envs/navrl_blackwell_test/lib/python3.11/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py",
        )
        self.assertEqual(evidence["source"]["sha256"], "7cbaa6f00e935a6f14bf1c28ec0db089fd924e931f3b0deee07a822f9b7d0090")
        self.assertEqual(evidence["inspection"]["signature"], "def close(self, wait_for_replicator=True, skip_cleanup=False) -> None:")
        self.assertIn("skip_cleanup=True sets _exiting, disables logging, and calls self._app.shutdown_and_release_framework().", evidence["inspection"]["facts"])
        self.assertIn("wait_for_replicator gates rep.orchestrator.wait_until_complete() and the following one-second sleep.", evidence["inspection"]["facts"])
        self.assertEqual(evidence["runtime_effects"]["python_imports"], 0)
        self.assertEqual(evidence["runtime_effects"]["app_launcher_starts"], 0)
        self.assertEqual(evidence["runtime_effects"]["simulator_initializations"], 0)

    def test_b0_only_scope_and_frozen_d7_r_reuse(self):
        spec = json.loads((ROOT / "r5_delivery/B0_SPEC_PREIMAGE.json").read_text())
        wrapper = json.loads((ROOT / "r5_delivery/B0_AUTHORIZATION_PREIMAGE.json").read_text())
        self.assertEqual(spec["execution_scope"], "B0_ONLY_ONE_SHOT")
        self.assertEqual(spec["status"], "PREIMAGE_NOT_REGISTERED_NOT_AUTHORIZED_NOT_STARTED")
        self.assertEqual(spec["per_condition"]["nontraining_transitions"], 20160)
        self.assertEqual(spec["per_condition"]["live_process_starts"], 1)
        self.assertEqual(spec["per_condition"]["simulator_initializations"], 1)
        self.assertEqual(spec["per_condition"]["automatic_retries"], 0)
        self.assertEqual(spec["per_condition"]["training_transitions"], 0)
        self.assertEqual(spec["per_condition"]["optimizer_updates"], 0)
        self.assertEqual(spec["per_condition"]["checkpoint_writes"], 0)
        self.assertEqual(spec["offline_comparison"]["reuse_job_id"], "J-44f25963c38ad089")
        self.assertFalse(spec["offline_comparison"]["request_d7_r_rerun"])
        self.assertEqual(wrapper["status"], "NOT_AUTHORIZED_NOT_MATERIALIZED")
        self.assertEqual(wrapper["automatic_retries"], 0)
        self.assertEqual(wrapper["outer_contract"]["action_gpu_seconds"], 900)
        self.assertEqual(wrapper["outer_contract"]["outer_wall_seconds_including_cleanup"], 910)
        self.assertEqual(wrapper["outer_contract"]["maximum_incremental_disk_bytes"], 8 * 1024**3)
        self.assertEqual(digest(ROOT / "r5_delivery/D7_R_SPEC_PREIMAGE.json"), "af9899a594456f64e9d3af2c9f8419be6946388f1fb861f1430d5da5a46ae3f8")
        self.assertEqual(digest(ROOT / "r5_delivery/D7_R_AUTHORIZATION_PREIMAGE.json"), "1ce854eccfea011682d191a2ee1ea58f3fb7d48d7bc474c67fb4aec6d568c179")


if __name__ == "__main__":
    unittest.main()
