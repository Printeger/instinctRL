#!/usr/bin/env python3
"""Direct SCALE-003 evaluation entry; all runtime imports follow authorization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training_workloads.scale_003.authorization import load_execution_authorization
from training_workloads.scale_003.config import load_config, load_evaluation_case_config


def main(argv: list[str] | None = None, *, fixed_case: str | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authorized SCALE-003 deterministic final-checkpoint evaluation")
    parser.add_argument("--validate-only", action="store_true", help="validate source configuration without runtime imports")
    parser.add_argument("--attempt-dir", type=Path)
    parser.add_argument("--execution-authorization", type=Path)
    parser.add_argument("--case", choices=("station", "tracking"), default=fixed_case)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    config = load_config()
    if fixed_case is not None and args.case != fixed_case:
        parser.error(f"this entry is fixed to the {fixed_case} case")
    if args.case is not None:
        load_evaluation_case_config(args.case)
    if args.validate_only:
        print(json.dumps({"status": "VALID", "cases": config["evaluation"]["cases"], "live_execution": False}, sort_keys=True))
        return 0
    attempt_dir = args.attempt_dir or (Path(os.environ["SCALE003_ATTEMPT_DIR"]) if "SCALE003_ATTEMPT_DIR" in os.environ else None)
    execution_authorization = args.execution_authorization or (Path(os.environ["SCALE003_EXECUTION_AUTHORIZATION"]) if "SCALE003_EXECUTION_AUTHORIZATION" in os.environ else None)
    checkpoint = args.checkpoint or (Path(os.environ["SCALE003_EVALUATION_CHECKPOINT"]) if "SCALE003_EVALUATION_CHECKPOINT" in os.environ else None)
    if None in (attempt_dir, execution_authorization, args.case, checkpoint):
        parser.error("live execution requires --attempt-dir, --execution-authorization, --case and --checkpoint")
    authorization = load_execution_authorization(
        execution_authorization, expected_mode="evaluation", expected_attempt_dir=attempt_dir,
    )
    from training_workloads.scale_003.live_evaluation import run_live
    return run_live(authorization, attempt_dir, args.case, checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
