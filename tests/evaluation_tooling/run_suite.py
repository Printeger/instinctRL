#!/usr/bin/env python3
"""Pinned standard-library entry point for the evaluation-tooling test suite."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    suite = unittest.defaultTestLoader.discover(
        str(repository / "tests" / "evaluation_tooling"),
        top_level_dir=str(repository),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
