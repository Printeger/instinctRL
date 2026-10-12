"""CPU-only real-artifact test for immutable D0020 V2 closure."""

from __future__ import annotations

import json
from pathlib import Path
import sys

WORKLOAD = Path("/home/mint/rl_dev/instinctRL_coordination/evaluation_workloads/eval_001")
if str(WORKLOAD) not in sys.path:
    sys.path.insert(0, str(WORKLOAD))

from offline_closure import EXPECTED_AGGREGATE, close_immutable_source
from pair_comparator import V2_LIVE_METADATA_PATHS


def main() -> int:
    source, pair, summary = close_immutable_source()
    if source["aggregate_before"] != EXPECTED_AGGREGATE:
        raise AssertionError("unexpected D0020 aggregate")
    if source["aggregate_after"] != source["aggregate_before"]:
        raise AssertionError("D0020 changed during read-only closure test")
    if pair["status"] != "PASS":
        raise AssertionError("real D0020 V2 comparison failed")
    if pair["all_difference_paths"] != list(V2_LIVE_METADATA_PATHS):
        raise AssertionError("real D0020 difference path set is not exact")
    if pair["deterministic_domain_maximum_error"] != 0.0:
        raise AssertionError("real D0020 deterministic domain is not exact")
    if summary["new_formal_processes"] != 0 or summary["new_environment_transitions"] != 0:
        raise AssertionError("offline closure claimed new runtime work")
    result = {
        "schema_version": "eval-001-offline-closure-test.v1",
        "status": "PASS",
        "aggregate": source["aggregate_after"],
        "difference_paths": pair["all_difference_paths"],
        "deterministic_domain_maximum_error": pair["deterministic_domain_maximum_error"],
        "formal_processes": summary["formal_processes"],
        "environment_transitions": summary["environment_transitions"],
        "new_formal_processes": 0,
        "new_environment_transitions": 0,
    }
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
