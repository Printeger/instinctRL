#!/usr/bin/env python3
"""Direct SCALE-003 training entry; all runtime imports follow authorization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training_workloads.scale_003.authorization import load_execution_authorization
from training_workloads.scale_003.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authorized SCALE-003 exact 872k-transition training workload")
    parser.add_argument("--validate-only", action="store_true", help="validate source configuration without runtime imports")
    parser.add_argument("--attempt-dir", type=Path)
    parser.add_argument("--execution-authorization", type=Path)
    args = parser.parse_args(argv)
    config = load_config()
    if args.validate_only:
        print(json.dumps({"status": "VALID", "order": config["order"], "live_execution": False}, sort_keys=True))
        return 0
    attempt_dir = args.attempt_dir or (Path(os.environ["SCALE003_ATTEMPT_DIR"]) if "SCALE003_ATTEMPT_DIR" in os.environ else None)
    execution_authorization = args.execution_authorization or (Path(os.environ["SCALE003_EXECUTION_AUTHORIZATION"]) if "SCALE003_EXECUTION_AUTHORIZATION" in os.environ else None)
    if attempt_dir is None or execution_authorization is None:
        parser.error("live execution requires --attempt-dir and --execution-authorization")
    authorization = load_execution_authorization(
        execution_authorization, expected_mode="training", expected_attempt_dir=attempt_dir,
    )
    from training_workloads.scale_003.live_training import run_live
    return run_live(authorization, attempt_dir)


if __name__ == "__main__":
    raise SystemExit(main())
