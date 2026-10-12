import unittest
from pathlib import Path

from training_workloads.unified_h1.host_cpu_evidence_v2 import (
    HOST_ARGV,
    load_manifest,
    parse_collected,
    parse_pass_count,
    require_exact_collection,
)


class HostCpuEvidenceV2Tests(unittest.TestCase):
    def test_current_manifest_has_nonzero_unique_exact_inventory_per_module(self):
        root = Path(__file__).resolve().parents[2]
        manifest = load_manifest(root)
        self.assertEqual(manifest["host_argv"], HOST_ARGV)
        all_nodes = []
        for module in manifest["modules"]:
            self.assertGreater(len(module["expected_node_ids"]), 0)
            self.assertTrue(all(node.startswith(module["path"] + "::") for node in module["expected_node_ids"]))
            all_nodes.extend(module["expected_node_ids"])
        self.assertEqual(len(all_nodes), len(set(all_nodes)))
        self.assertEqual(len(all_nodes), 173)

    def test_every_previously_omitted_pytest_function_has_explicit_nodes(self):
        root = Path(__file__).resolve().parents[2]
        manifest = load_manifest(root)
        counts = {row["path"]: len(row["expected_node_ids"]) for row in manifest["modules"]}
        self.assertEqual(counts["control_snapshot/tests/test_process_recovery.py"], 8)
        self.assertEqual(counts["control_snapshot/tests/test_experiments_v2.py"], 9)
        self.assertEqual(counts["control_snapshot/tests/test_h1_adapter.py"], 15)
        self.assertEqual(counts["tests/training_workloads/test_unified_h1_r3_process_identity.py"], 3)

    def test_zero_discovery_fails_closed(self):
        with self.assertRaisesRegex(PermissionError, "zero pytest discovery"):
            parse_collected("0 tests collected in 0.01s\n", "example/test_target.py")

    def test_inventory_drift_fails_closed(self):
        path = "example/test_target.py"
        expected = [path + "::test_one", path + "::test_two"]
        with self.assertRaisesRegex(PermissionError, "inventory drift"):
            require_exact_collection(path + "::test_one\n", path, expected)

    def test_duplicate_node_fails_closed(self):
        path = "example/test_target.py"
        text = path + "::test_one\n" + path + "::test_one\n"
        with self.assertRaisesRegex(PermissionError, "duplicate"):
            parse_collected(text, path)

    def test_skip_or_ambiguous_summary_fails_closed(self):
        with self.assertRaisesRegex(PermissionError, "non-pass"):
            parse_pass_count("1 passed, 1 skipped in 0.01s\n", "")
        with self.assertRaisesRegex(PermissionError, "absent or ambiguous"):
            parse_pass_count("no summary\n", "")


if __name__ == "__main__":
    unittest.main()
