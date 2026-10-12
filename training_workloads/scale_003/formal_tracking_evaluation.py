#!/usr/bin/env python3
"""Independent direct entry for the fixed tracking diagnostic."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training_workloads.scale_003.formal_evaluation import main


if __name__ == "__main__":
    raise SystemExit(main(fixed_case="tracking"))
