"""Create the exact certified-results runtime support directories."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path

from attempt_binding import binding_from_environment, bound_path, path_from_environment
from io_utils import atomic_canonical_replace


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    binding = binding_from_environment()
    expected_root = binding.directory / "results/runtime_support"
    expected_output = expected_root / "status.json"
    output = bound_path(
        binding, path_from_environment("EVAL_STATUS_OUTPUT"), "results/runtime_support/status.json",
        must_exist=False,
    )
    if expected_root.is_symlink() or expected_output.exists():
        raise ValueError("runtime support path is symlinked or pre-existing")
    temporary = expected_root / "tmp"
    cache = expected_root / "cache"
    temporary.mkdir(parents=True, exist_ok=False)
    cache.mkdir(exist_ok=False)
    result = OrderedDict((
        ("schema_version", "eval-001-runtime-support.v1"),
        ("status", "PASS"),
        ("tmp", str(temporary)),
        ("cache", str(cache)),
    ))
    atomic_canonical_replace(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
