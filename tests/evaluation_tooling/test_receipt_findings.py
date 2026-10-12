from __future__ import annotations

import unittest

from evaluation_tooling.findings import FindingCollector, RuntimeFailureClass, classify_runtime_fail_fast
from evaluation_tooling.receipt import build_launcher_receipt


class ReceiptAndFindingTests(unittest.TestCase):
    def receipt(self, **kwargs):
        defaults = dict(raw_exit=0, semantic_exit=0, semantic_evidence_present=True, cleanup_exit=None, timed_out=False)
        defaults.update(kwargs)
        return build_launcher_receipt(**defaults)

    def test_positive_receipt(self) -> None:
        self.assertEqual(self.receipt().receipt_status, "PASS")

    def test_raw_exit_failure(self) -> None:
        self.assertIn("RAW_EXIT", self.receipt(raw_exit=3).primary_failure)

    def test_semantic_exit_failure(self) -> None:
        self.assertIn("SEMANTIC_EXIT", self.receipt(semantic_exit=4).primary_failure)

    def test_missing_pass_failure(self) -> None:
        self.assertEqual(self.receipt(semantic_evidence_present=False).primary_failure, "MISSING_PASS_EVIDENCE")

    def test_timeout_primary_preserved_over_cleanup(self) -> None:
        receipt = self.receipt(raw_exit=124, cleanup_exit=9, timed_out=True)
        self.assertEqual(receipt.primary_failure, "TIMEOUT")
        self.assertEqual(receipt.cleanup_status, "FAIL")

    def test_raw_primary_preserved_over_cleanup(self) -> None:
        self.assertTrue(self.receipt(raw_exit=5, cleanup_exit=6).primary_failure.startswith("RAW_EXIT"))

    def test_cleanup_only_failure(self) -> None:
        self.assertEqual(self.receipt(cleanup_exit=7).primary_failure, "CLEANUP_EXIT:7")

    def test_collect_all_is_stable_and_deduplicated(self) -> None:
        collector = FindingCollector()
        collector.add("Z", "b", "last")
        collector.add("A", "a", "first")
        collector.add("A", "a", "first")
        self.assertEqual([item.code for item in collector.findings], ["A", "Z"])

    def test_every_runtime_failure_class(self) -> None:
        for failure_class in RuntimeFailureClass:
            with self.subTest(failure_class=failure_class):
                self.assertEqual(classify_runtime_fail_fast({failure_class: True}), failure_class)

    def test_fail_fast_priority_stable(self) -> None:
        flags = {failure_class: True for failure_class in RuntimeFailureClass}
        self.assertEqual(classify_runtime_fail_fast(flags), RuntimeFailureClass.FROZEN_IDENTITY_LOSS)
