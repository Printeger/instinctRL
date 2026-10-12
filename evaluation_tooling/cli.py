"""Command-line interface for shared evaluation tooling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

from .canonical import canonical_dumps
from .runner import EvaluationRunner, preflight
from .sealer import seal_attempt
from .validation import validate_sealed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation_tooling")
    subcommands = parser.add_subparsers(dest="command", required=True)

    def identity(command: argparse.ArgumentParser) -> None:
        command.add_argument("attempt_dir", type=Path)
        command.add_argument("--tooling-commit", required=True)
        command.add_argument("--product-commit", required=True)

    identity(subcommands.add_parser("preflight", help="collect all cheap DRAFT submission defects"))
    identity(subcommands.add_parser("run", help="preflight then run declared commands fail-fast"))
    run_one = subcommands.add_parser("run-command", help="run one zero-based manifest command")
    identity(run_one)
    run_one.add_argument("command_index", type=int)
    run_one.add_argument("--repair-of", type=int)
    identity(subcommands.add_parser("seal", help="perform the terminal DRAFT-to-SEALED transition"))
    inspect = subcommands.add_parser("inspect", help="validate a SEALED packet offline")
    inspect.add_argument("attempt_dir", type=Path)
    return parser


def _emit(value: Any) -> None:
    sys.stdout.buffer.write(canonical_dumps(value))
    sys.stdout.buffer.flush()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "preflight":
            result = preflight(arguments.attempt_dir, tooling_commit=arguments.tooling_commit, product_commit=arguments.product_commit)
            _emit(result)
            return 0 if result["status"] == "PASS" else 2
        if arguments.command == "run":
            runner = EvaluationRunner(arguments.attempt_dir, tooling_commit=arguments.tooling_commit, product_commit=arguments.product_commit)
            exit_code = runner.run_all()
            _emit({"schema_version": "evaluation-run-result.v1", "status": "PASS" if exit_code == 0 else "FAIL", "exit": exit_code})
            return exit_code
        if arguments.command == "run-command":
            runner = EvaluationRunner(arguments.attempt_dir, tooling_commit=arguments.tooling_commit, product_commit=arguments.product_commit)
            # A repair invocation reuses existing prepared DRAFT state. A new
            # invocation prepares only if no state exists.
            if not (arguments.attempt_dir / "TOOL_STATE.json").is_file():
                prepared = runner.prepare()
                if prepared["status"] != "PASS":
                    _emit(prepared)
                    return 2
            receipt, exit_code = runner.run_command(arguments.command_index, repair_of=arguments.repair_of)
            _emit({"schema_version": "evaluation-run-result.v1", "status": receipt.receipt_status, "exit": exit_code, "receipt": receipt.as_dict()})
            return exit_code
        if arguments.command == "seal":
            result = seal_attempt(arguments.attempt_dir, tooling_commit=arguments.tooling_commit, product_commit=arguments.product_commit)
            _emit(result)
            return 0 if result["status"] == "PASS" else 2
        result = validate_sealed(arguments.attempt_dir)
        _emit(result)
        return 0 if result["status"] == "PASS" else 2
    except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
        _emit({"schema_version": "evaluation-cli-error.v1", "status": "FAIL", "error": f"{type(error).__name__}: {error}"})
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
