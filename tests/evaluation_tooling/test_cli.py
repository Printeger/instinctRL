from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from tests.evaluation_tooling.helpers import make_attempt


class CLITests(unittest.TestCase):
    def test_module_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "evaluation_tooling", "--help"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn(b"preflight", completed.stdout)
        self.assertIn(b"inspect", completed.stdout)

    def test_inspect_missing_packet_is_nonzero_and_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, "-m", "evaluation_tooling", "inspect", str(Path(temporary) / "missing")],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertTrue(completed.stdout.endswith(b"\n"))
        self.assertIn(b'"status":"FAIL"', completed.stdout)
