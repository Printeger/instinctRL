import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from training_workloads.unified_h1.failure_evidence import (
    FailureEvidenceError, OfflineFailureEvidenceArchive, RAW_FILES,
    verify_d0005_missing_window,
)

REPOSITORY = Path(__file__).resolve().parents[2]
D0005 = REPOSITORY / "docs/history/consolidation-20261012/r5/review_bundle/evidence/d0005/EVIDENCE.json"
D0005_SHA256 = "c87d7725c5047dbd5080fabf410e8852869c6ed7445aeac5e6e7ee8856500a26"
TEST_SENTINEL_DIFF_SHA256 = "0" * 64
CANDIDATE = {
    "repository": str(REPOSITORY),
    "base_commit": "524e738a65dc6954fec797899b5e9e21d27d0324",
    "commit": "524e738a65dc6954fec797899b5e9e21d27d0324",
    "diff_sha256": TEST_SENTINEL_DIFF_SHA256,
}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def raw_case(kind, *, starts, identity_state, window_state, stdout=b"", stderr=b""):
    return {
        "status.json": encoded({"status": "FAIL", "injection_type": kind, "raw_exit_code": 17}),
        "process_identity.json": encoded({"state": identity_state, "pid": 4100 if starts else None,
                                            "process_group_id": 4100 if starts else None}),
        "observed_window.json": encoded({"state": window_state, "samples": []}),
        "counters.json": encoded({"live_starts": starts, "retries": 0, "transitions": 0,
                                   "ppo_updates": 0, "checkpoint_writes": 0}),
        "stdout.log": stdout,
        "stderr.log": stderr,
    }


class OfflineFailureEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "archive"
        self.archive = OfflineFailureEvidenceArchive(self.root, CANDIDATE)

    def test_all_five_failure_paths_preserve_raw_artifacts_and_hash_manifest(self):
        cases = (
            ("pre-construction", 0, "NOT_CONSTRUCTED", "NOT_OBSERVED"),
            ("initialization", 1, "OBSERVED", "CAPTURED"),
            ("controlled-failure", 1, "OBSERVED", "CAPTURED"),
            ("supervisor-lost", 1, "LAST_OBSERVED", "LAST_OBSERVED"),
        )
        receipts = {}
        for index, (kind, starts, identity, window) in enumerate(cases):
            raw = raw_case(kind, starts=starts, identity_state=identity, window_state=window,
                           stdout=f"raw stdout {kind}\n".encode(), stderr=f"raw stderr {kind}\n".encode())
            receipt = self.archive.capture(f"case-{index}", kind, raw)
            receipts[kind] = receipt
            verified = self.archive.verify(f"case-{index}")
            self.assertEqual(verified["manifest_sha256"], receipt["manifest_sha256"])
            for name in RAW_FILES:
                self.assertEqual((self.root / f"case-{index}" / name).read_bytes(), raw[name])
                self.assertEqual(verified["files"][name]["sha256"], hashlib.sha256(raw[name]).hexdigest())
        parent = receipts["supervisor-lost"]["manifest_sha256"]
        late = raw_case("late-evidence", starts=1, identity_state="LATE_OBSERVED",
                        window_state="LATE_OBSERVED", stdout=b"late stdout\n", stderr=b"late stderr\n")
        late_receipt = self.archive.capture("case-4", "late-evidence", late,
                                            parent_manifest_sha256=parent)
        self.assertEqual(late_receipt["parent_manifest_sha256"], parent)
        self.assertEqual(self.archive.verify("case-3")["manifest_sha256"], parent)
        self.assertEqual(set(receipts) | {"late-evidence"}, {
            "pre-construction", "initialization", "controlled-failure", "supervisor-lost", "late-evidence"})

    def test_existing_evidence_cannot_be_overwritten_or_silently_tampered(self):
        raw = raw_case("controlled-failure", starts=1, identity_state="OBSERVED", window_state="CAPTURED")
        self.archive.capture("once", "controlled-failure", raw)
        before = {name: (self.root / "once" / name).read_bytes() for name in (*RAW_FILES, "manifest.json")}
        with self.assertRaisesRegex(FailureEvidenceError, "overwrite"):
            self.archive.capture("once", "controlled-failure", raw)
        self.assertEqual(before, {name: (self.root / "once" / name).read_bytes()
                                  for name in (*RAW_FILES, "manifest.json")})
        (self.root / "once" / "unmanifested.log").write_bytes(b"changed")
        with self.assertRaisesRegex(FailureEvidenceError, "missing or unmanifested"):
            self.archive.verify("once")

    def test_missing_artifacts_and_any_retry_claim_fail_closed(self):
        raw = raw_case("initialization", starts=1, identity_state="OBSERVED", window_state="CAPTURED")
        del raw["stderr.log"]
        with self.assertRaisesRegex(FailureEvidenceError, "all raw"):
            self.archive.capture("missing", "initialization", raw)
        retried = raw_case("initialization", starts=1, identity_state="OBSERVED", window_state="CAPTURED")
        retried["counters.json"] = encoded({"live_starts": 2, "retries": 1, "transitions": 0, "ppo_updates": 0, "checkpoint_writes": 0})
        with self.assertRaisesRegex(FailureEvidenceError, "zero retries"):
            self.archive.capture("retried", "initialization", retried)
        self.assertFalse((self.root / "retried").exists())

    def test_late_evidence_is_append_only_and_requires_parent_binding(self):
        late = raw_case("late-evidence", starts=1, identity_state="LATE_OBSERVED",
                        window_state="LATE_OBSERVED")
        with self.assertRaisesRegex(FailureEvidenceError, "parent manifest"):
            self.archive.capture("late", "late-evidence", late)
        with self.assertRaisesRegex(FailureEvidenceError, "parent manifest"):
            self.archive.capture("late", "late-evidence", late, parent_manifest_sha256="0" * 64)
        self.assertFalse((self.root / "late").exists())

    def test_d0005_gap_is_verified_without_reconstruction_or_mutation(self):
        before = D0005.read_bytes()
        failure_path = D0005.parent / "attempt/results/prelaunch/prelaunch_resource_gate_failure.json"
        failure_before = failure_path.read_bytes()
        receipt = verify_d0005_missing_window(D0005, D0005_SHA256)
        self.assertFalse(receipt["observed_window_present"])
        self.assertFalse(receipt["reconstructed"])
        self.assertEqual(receipt["accounting"]["live_starts"], 1)
        self.assertEqual(receipt["accounting"]["retries"], 0)
        self.assertNotIn("observed_window", json.loads(failure_before))
        self.assertEqual(D0005.read_bytes(), before)
        self.assertEqual(failure_path.read_bytes(), failure_before)


if __name__ == "__main__":
    unittest.main()
