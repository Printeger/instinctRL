"""CPU-only immutable D0020 evidence closure under the V2 pair domain."""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import Any

from attempt_binding import bind_attempt, bound_path
from io_utils import atomic_canonical_replace, canonical_bytes, sha256_file
from pair_comparator import compare_v2
from semantic_validator import load_semantic, validate_files


COORDINATION_ROOT = Path("/home/mint/rl_dev/instinctRL_coordination")
ATTEMPTS_ROOT = COORDINATION_ROOT / "attempts/development/EVAL-001"
SOURCE_ATTEMPT = ATTEMPTS_ROOT / "EVAL-001-D0020"
OUTPUT_ATTEMPT = ATTEMPTS_ROOT / "EVAL-001-D0021"
EXPECTED_AGGREGATE = "eee6d41ef5652891ecd8f297003d1981445634c6e49238f9e1b74dd553fd9a48"
EXPECTED_HASHES = OrderedDict((
    ("ATTEMPT_MANIFEST.json", "093feee8d9af9184206d82036c0acd214188b28b46c038fc0a79de90000f3ee5"),
    ("INPUT_HASHES.json", "fed729ecc44daaee996d6e4e35bd186e6a28e4281befa01b7d35c63bf2176bd8"),
    ("COMMAND_INDEX.json", "ae3025756a2dda58fa3a12de2a449b96275dc256c26bfedd2e5996433d4004bd"),
    ("COMMAND_LEDGER.json", "db7af3a93dd1d88419f76953254347c224d6fc4dbe0253b553b33f5a1ab52f65"),
    ("results/mean_cert_a/semantic.pt", "4e9618779e12bb8f84c767d59c1b4f64505f399e5a9184882157d43437c3bf0b"),
    ("results/mean_cert_a/semantic.json", "753838752d30c9ccff3ecfd7b48dc88af7a4d5bdfab7ba6dc5b3de39b34eda32"),
    ("results/mean_cert_a/receipt.json", "5bd048097ae764d8791176ce765d0f3bc7bbe1222576f6dc493f6a50b242de53"),
    ("results/mean_cert_a/status.json", "94d6c9d1e623dcfb98c1e5b116aa06dab32abeaa1418a707b6c7583d81a07784"),
    ("results/mean_cert_a/validation.json", "b983d1600bfe4c3fcd8de210842bf2d994ced8a3afab94c5faa67fdc88fc0338"),
    ("results/mean_cert_a/kit.log", "e53bec8b2506f2eda57add3c56cf0c8c940b8adffc3d00f10c7af6f8c7b16280"),
    ("results/mean_cert_b/semantic.pt", "edb16bdf3a6a0e514a0daf118af3c2f447168e580d7148758a118fed5440a0fe"),
    ("results/mean_cert_b/semantic.json", "95229ab87fbaf1b38d789a2ac5f0eab4817f6b7ebb9c30f91b6dcce7921301d8"),
    ("results/mean_cert_b/receipt.json", "9fdc1f8d7e713ce75f876aa22ac5ab1c37208e40203b2fc37f8da4dfc3c5b1e9"),
    ("results/mean_cert_b/status.json", "ff79249580e7c2ba1c552202e52870954f63fccd351d3a34937e8dea2100d564"),
    ("results/mean_cert_b/validation.json", "9d5717b50d7886bb0d17af9cf661412ea6ba8513708efc283c9f3c9bac2c5be3"),
    ("results/mean_cert_b/kit.log", "75144264aceae801dc502c6c5fa32bec5a1fb48130af2e9cefaf8ce25af55519"),
))


def source_aggregate() -> str:
    import hashlib

    lines = bytearray()
    paths = sorted(path for path in SOURCE_ATTEMPT.rglob("*") if not path.is_dir())
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"D0020 inventory entry is not a regular file: {path}")
        relative = path.relative_to(COORDINATION_ROOT).as_posix()
        lines.extend(f"{sha256_file(path)}  {relative}\n".encode("utf-8"))
    return hashlib.sha256(lines).hexdigest()


def _canonical_object(path: Path) -> OrderedDict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"), object_pairs_hook=OrderedDict)
    if not isinstance(value, OrderedDict) or raw != canonical_bytes(value):
        raise ValueError(f"not a canonical JSON object: {path}")
    return value


def _require_pass(relative: str) -> OrderedDict[str, Any]:
    value = _canonical_object(SOURCE_ATTEMPT / relative)
    if value.get("status") != "PASS":
        raise ValueError(f"D0020 prerequisite is not PASS: {relative}")
    return value


def close_immutable_source() -> tuple[OrderedDict[str, Any], OrderedDict[str, Any], OrderedDict[str, Any]]:
    aggregate_before = source_aggregate()
    if aggregate_before != EXPECTED_AGGREGATE:
        raise ValueError(f"D0020 aggregate mismatch before validation: {aggregate_before}")
    verified = []
    for relative, expected in EXPECTED_HASHES.items():
        path = SOURCE_ATTEMPT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"D0020 pinned hash mismatch: {relative}")
        verified.append(OrderedDict((
            ("path", relative), ("bytes", path.stat().st_size), ("sha256", actual),
        )))

    source_binding = bind_attempt("EVAL-001-D0020", SOURCE_ATTEMPT, attempts_root=ATTEMPTS_ROOT)
    validations = OrderedDict()
    semantics = OrderedDict()
    for run_id in ("mean_cert_a", "mean_cert_b"):
        root = SOURCE_ATTEMPT / f"results/{run_id}"
        result = validate_files(
            source_binding, root / "semantic.pt", root / "semantic.json",
            root / "receipt.json", root / "status.json", root / "kit.log",
            run_id=run_id,
        )
        stored = _require_pass(f"results/{run_id}/validation.json")
        if canonical_bytes(result) != canonical_bytes(stored):
            raise ValueError(f"stored validation differs from replay: {run_id}")
        validations[run_id] = result
        semantics[run_id] = load_semantic(root / "semantic.pt")
    if validations["mean_cert_a"]["pid"] == validations["mean_cert_b"]["pid"]:
        raise ValueError("D0020 formal PIDs are not distinct")
    for relative in (
        "PREFLIGHT.json", "results/cpu_tests.json",
        "results/d0019_read_only_replay.json", "results/checkpoint_audit.json",
        "results/gpu_gate.json", "external/default_kit_logs.comparison.json",
        "external/tmp_isaaclab.comparison.json",
    ):
        _require_pass(relative)
    original_pair = _canonical_object(SOURCE_ATTEMPT / "results/pair_comparison.json")
    expected_failure = "ValueError: scalar mismatch at root.cuda_resources.before.free_memory_bytes"
    if original_pair.get("status") != "FAIL" or original_pair.get("message") != expected_failure:
        raise ValueError("D0020 V1 failure evidence mismatch")

    pair = compare_v2(semantics["mean_cert_a"], semantics["mean_cert_b"])
    if pair["status"] != "PASS":
        raise ValueError(f"D0020 V2 pair did not pass: {pair}")
    aggregate_after = source_aggregate()
    if aggregate_after != aggregate_before:
        raise ValueError("D0020 aggregate changed during offline closure")
    source = OrderedDict((
        ("schema_version", "eval-001-immutable-source-verification.v1"),
        ("status", "PASS"), ("source_attempt", "EVAL-001-D0020"),
        ("aggregate_before", aggregate_before), ("aggregate_after", aggregate_after),
        ("verified_files", verified), ("individual_validations", validations),
        ("original_v1_failure", original_pair),
    ))
    metrics = OrderedDict()
    for run_id, semantic in semantics.items():
        record = json.loads(semantic["runtime_evaluation"]["canonical_record"], object_pairs_hook=OrderedDict)
        metrics[run_id] = record["metrics"]
    summary = OrderedDict((
        ("schema_version", "eval-001-offline-closure.v1"),
        ("status", "PASS"), ("profile", "EVAL_DETERMINISTIC_MEAN_PAIR_V2"),
        ("source_attempt", "EVAL-001-D0020"),
        ("formal_processes", 2), ("environment_transitions", 16384),
        ("ray_samples", 347996160), ("campaign_formal_processes", 4),
        ("new_formal_processes", 0), ("new_environment_transitions", 0),
        ("formal_pids", [validations["mean_cert_a"]["pid"], validations["mean_cert_b"]["pid"]]),
        ("metrics", metrics),
    ))
    return source, pair, summary


def main() -> int:
    output_binding = bind_attempt("EVAL-001-D0021", OUTPUT_ATTEMPT, attempts_root=ATTEMPTS_ROOT)
    source, pair, summary = close_immutable_source()
    outputs = (
        ("EVAL_SOURCE_VERIFICATION_OUTPUT", "results/immutable_source_verification.json", source),
        ("EVAL_V2_PAIR_OUTPUT", "results/v2_pair_comparison.json", pair),
        ("EVAL_CLOSURE_SUMMARY_OUTPUT", "results/closure_summary.json", summary),
    )
    for environment_name, relative, value in outputs:
        path = bound_path(
            output_binding, Path(os.environ[environment_name]), relative,
            must_exist=False,
        )
        atomic_canonical_replace(path, value)
    print(json.dumps(summary, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
