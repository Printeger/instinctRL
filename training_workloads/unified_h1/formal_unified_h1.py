"""Future R3 entry; R2 supports validation and otherwise fails closed."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from .budget import budget_manifest
from .config import load_config


def validate_only() -> dict[str, object]:
    from .evaluation import evaluation_manifest
    config = load_config()
    budget = budget_manifest(); evaluation = evaluation_manifest()
    if budget["evaluation_transitions_per_checkpoint"] != 32960 or evaluation["transitions_per_checkpoint"] != 32960:
        raise ValueError("evaluation budget mismatch")
    return {"status": "VALID", "method": config["method"]["version"], "live_authorized": False,
            "product_commit": config["resolved_identity"]["product_commit"], "future_modes": config["future_entry"]["modes"]}


def main(argv=None) -> int:
    process_started_monotonic = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-enhanced-b0-handoff", action="store_true")
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--mode", choices=("research-r6-training", "research-r6-evaluation", "main", "controlled-failure", "r4-training", "evaluation", "r5-resume", "control-diagnostic", "pilot-training", "pilot-evaluation", "enhanced-b0-diagnostic", "research-training", "research-evaluation", "research-transition", "research-formal-training", "research-formal-evaluation"))
    args = parser.parse_args(argv)
    if args.validate_only:
        print(validate_only()); return 0
    if args.validate_enhanced_b0_handoff:
        if args.handoff is None: parser.error("--validate-enhanced-b0-handoff requires --handoff")
        from .b0_handoff import load_and_validate
        print(load_and_validate(args.handoff.resolve())); return 0
    if args.authorization is None or args.attempt is None or args.mode is None:
        parser.error("live entry requires --authorization, --attempt and --mode")
    # Source/method/config validation is the mandatory predecessor of both
    # validate-only and every future live mode.
    load_config()
    from .authorization import load_r3_authorization
    authorization = load_r3_authorization(args.authorization.resolve(), args.attempt.resolve(), args.mode)
    from .live_adapter import run_main_sequence
    return run_main_sequence(authorization=authorization, attempt=args.attempt.resolve(),
                             process_started_monotonic=process_started_monotonic)


def finalize_process_exit(exit_code: int, argv: list[str], exit_process=os._exit) -> int:
    """Skip unsafe native finalizers only after the live lifecycle has returned.

    The live lifecycle has already closed the environment and SimulationApp and
    durably written PASS/FAIL. Isaac Sim 5.1 can segfault later while Python
    unloads native libraries, so terminate with the lifecycle's exact code.
    Validation-only remains a normal Python exit for CLI output and CPU tests.
    """
    if "--validate-only" not in argv and "--validate-enhanced-b0-handoff" not in argv:
        exit_process(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(finalize_process_exit(main(), sys.argv[1:]))
