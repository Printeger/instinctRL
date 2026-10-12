"""Read-only real-artifact replay for the immutable D0019 formal A result."""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import sys

WORKLOAD = Path("/home/mint/rl_dev/instinctRL_coordination/evaluation_workloads/eval_001")
if str(WORKLOAD) not in sys.path:
    sys.path.insert(0, str(WORKLOAD))

from attempt_binding import bind_attempt, bound_path
from io_utils import atomic_canonical_replace
from semantic_validator import load_semantic, validate_files, validate_semantic_mirror


ATTEMPTS_ROOT = Path(
    "/home/mint/rl_dev/instinctRL_coordination/attempts/development/EVAL-001"
)
ATTEMPT = ATTEMPTS_ROOT / "EVAL-001-D0019"
RUN = ATTEMPT / "results/mean_cert_a"


def main() -> int:
    binding = bind_attempt("EVAL-001-D0019", ATTEMPT, attempts_root=ATTEMPTS_ROOT)
    result = validate_files(
        binding,
        RUN / "semantic.pt",
        RUN / "semantic.json",
        RUN / "receipt.json",
        RUN / "status.json",
        RUN / "kit.log",
        run_id="mean_cert_a",
    )
    proof = validate_semantic_mirror(load_semantic(RUN / "semantic.pt"), RUN / "semantic.json")
    output = OrderedDict((
        ("schema_version", "eval-001-d0019-read-only-replay.v1"),
        ("status", "PASS"),
        ("validation", result),
        ("pt_root_key_order", proof["pt_root_key_order"]),
        ("mirror_root_key_order", proof["mirror_root_key_order"]),
        ("root_insertion_order_differs", proof["root_insertion_order_differs"]),
        ("canonical_bytes_equal", proof["canonical_bytes_equal"]),
        ("semantic_value_difference_count", proof["semantic_value_difference_count"]),
    ))
    output_path_value = os.environ.get("EVAL_D0019_REPLAY_OUTPUT")
    if output_path_value is not None:
        current = ATTEMPTS_ROOT / "EVAL-001-D0020"
        current_binding = bind_attempt(
            "EVAL-001-D0020", current, attempts_root=ATTEMPTS_ROOT
        )
        output_path = bound_path(
            current_binding,
            Path(output_path_value),
            "results/d0019_read_only_replay.json",
            must_exist=False,
        )
        atomic_canonical_replace(output_path, output)
    print(json.dumps(output, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
