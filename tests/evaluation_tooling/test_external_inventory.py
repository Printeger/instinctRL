from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from evaluation_tooling.external_inventory import ExternalInventorySpec, compare_external, scan_external


class ExternalInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.spec = ExternalInventorySpec("logs", self.root, ("*.log", "links/*"))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_create(self) -> None:
        spec = ExternalInventorySpec("logs", self.root, ("*.log",), expected_created=("new.log",))
        before = scan_external(spec)
        (self.root / "new.log").write_text("new")
        self.assertEqual(compare_external(spec, before, scan_external(spec))["status"], "PASS")

    def test_delete(self) -> None:
        (self.root / "old.log").write_text("old")
        before = scan_external(self.spec)
        (self.root / "old.log").unlink()
        spec = ExternalInventorySpec("logs", self.root, ("*.log",), expected_deleted=("old.log",))
        self.assertEqual(compare_external(spec, before, scan_external(spec))["deleted"], ["old.log"])

    def test_modify(self) -> None:
        (self.root / "one.log").write_text("one")
        spec = ExternalInventorySpec("logs", self.root, ("*.log",), expected_modified=("one.log",))
        before = scan_external(spec)
        (self.root / "one.log").write_text("two")
        self.assertEqual(compare_external(spec, before, scan_external(spec))["status"], "PASS")

    def test_directory_mtime_ignored(self) -> None:
        before = scan_external(self.spec)
        os.utime(self.root, None)
        self.assertEqual(compare_external(self.spec, before, scan_external(self.spec))["status"], "PASS")

    def test_undeclared_path_fails(self) -> None:
        before = scan_external(self.spec)
        (self.root / "surprise.log").write_text("x")
        result = compare_external(self.spec, before, scan_external(self.spec))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["findings"][0]["code"], "UNDECLARED_EXTERNAL_CHANGE")

    def test_unmatched_path_not_scanned(self) -> None:
        (self.root / "ignored.txt").write_text("x")
        self.assertEqual(scan_external(self.spec)["entries"], [])

    def test_symlink_target_bound(self) -> None:
        (self.root / "target").write_text("x")
        (self.root / "links").mkdir()
        (self.root / "links" / "current").symlink_to("../target")
        entry = scan_external(self.spec)["entries"][0]
        self.assertEqual((entry["type"], entry["target"]), ("symlink", "../target"))

    def test_kit_rotation_pass(self) -> None:
        (self.root / "kit_1.log").write_text("old")
        spec = ExternalInventorySpec("kit", self.root, ("kit_*.log",), expected_created=("kit_2.log",), expected_deleted=("kit_1.log",), kit_rotation={"pattern": "kit_*.log", "maximum_files": 1, "create_count": 1, "delete_count": 1})
        before = scan_external(spec)
        (self.root / "kit_1.log").unlink()
        (self.root / "kit_2.log").write_text("new")
        self.assertEqual(compare_external(spec, before, scan_external(spec))["status"], "PASS")

    def test_kit_retention_fails(self) -> None:
        (self.root / "kit_1.log").write_text("one")
        spec = ExternalInventorySpec("kit", self.root, ("kit_*.log",), kit_rotation={"pattern": "kit_*.log", "maximum_files": 0, "create_count": 0, "delete_count": 0})
        result = compare_external(spec, scan_external(spec), scan_external(spec))
        self.assertIn("KIT_ROTATION_RETENTION", {item["code"] for item in result["findings"]})
