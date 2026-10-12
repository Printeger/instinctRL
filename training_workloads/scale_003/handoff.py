"""Immutable sealed training-attempt to evaluation-attempt handoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import load_strict_json, sha256_file
from .semantic import require_sha256
from .training import validate_training_semantic


HANDOFF_FIELDS = {
    "schema_version", "training_attempt_id", "training_attempt_path",
    "terminal_manifest_path", "terminal_manifest_sha256",
    "training_semantic_path", "training_semantic_sha256",
    "training_status_path", "training_status_sha256",
    "final_checkpoint_path", "final_checkpoint_sha256", "final_checkpoint_bytes", "final_checkpoint_counter",
}


@dataclass(frozen=True)
class TrainingHandoff:
    training_attempt_id: str
    training_attempt_path: Path
    terminal_manifest_path: Path
    training_semantic_path: Path
    training_status_path: Path
    final_checkpoint_path: Path
    final_checkpoint_sha256: str


def _bound_file(root: Path, authored: Any, relative: str, digest: Any) -> Path:
    authored_path = Path(authored) if isinstance(authored, str) else Path("")
    expected = root / relative
    if not authored_path.is_absolute() or authored_path != expected or expected.is_symlink() or not expected.is_file():
        raise ValueError(f"training handoff file binding mismatch: {relative}")
    path = expected.resolve()
    require_sha256(digest, f"{relative} sha256")
    if sha256_file(path) != digest:
        raise ValueError(f"training handoff file hash mismatch: {relative}")
    return path


def validate_training_handoff(value: Any, *, evaluation_attempt: Path,
                              attempts_root: Path | None = None) -> TrainingHandoff:
    if not isinstance(value, Mapping) or set(value) != HANDOFF_FIELDS or value["schema_version"] != "scale-003-training-handoff.v1":
        raise ValueError("training handoff schema mismatch")
    training_attempt = Path(value["training_attempt_path"]).resolve() if isinstance(value["training_attempt_path"], str) else Path("")
    evaluation_attempt = evaluation_attempt.resolve()
    if training_attempt == evaluation_attempt or training_attempt.name != value["training_attempt_id"]:
        raise ValueError("training/evaluation attempts must be distinct and exactly named")
    if attempts_root is not None:
        root = attempts_root.resolve()
        if root not in training_attempt.parents or root not in evaluation_attempt.parents:
            raise ValueError("handoff attempts are outside the canonical development root")
    terminal = _bound_file(training_attempt, value["terminal_manifest_path"], "TERMINAL_MANIFEST.json", value["terminal_manifest_sha256"])
    semantic = _bound_file(training_attempt, value["training_semantic_path"], "results/training/semantic.json", value["training_semantic_sha256"])
    status = _bound_file(training_attempt, value["training_status_path"], "results/training/status.json", value["training_status_sha256"])
    checkpoint = _bound_file(training_attempt, value["final_checkpoint_path"], "results/training/checkpoints/update_1954_final.pt", value["final_checkpoint_sha256"])
    if type(value["final_checkpoint_bytes"]) is not int or value["final_checkpoint_bytes"] <= 0 or checkpoint.stat().st_size != value["final_checkpoint_bytes"] or value["final_checkpoint_counter"] != 1954:
        raise ValueError("training handoff final checkpoint size/counter mismatch")
    terminal_value = load_strict_json(terminal)
    if terminal_value.get("schema_version") != "evaluation-terminal-manifest.v2" or terminal_value.get("lifecycle") != "SEALED" or terminal_value.get("attempt_id") != value["training_attempt_id"]:
        raise ValueError("source training attempt is not an exact sealed lifecycle")
    terminal_rows = terminal_value.get("files")
    if not isinstance(terminal_rows, list):
        raise ValueError("terminal manifest file inventory is absent")
    indexed = {row.get("path"): row for row in terminal_rows if isinstance(row, Mapping)}
    if len(indexed) != len(terminal_rows):
        raise ValueError("terminal manifest contains malformed or duplicate file rows")
    for relative, path in (("results/training/semantic.json", semantic), ("results/training/status.json", status), ("results/training/checkpoints/update_1954_final.pt", checkpoint)):
        row = indexed.get(relative)
        if not isinstance(row, Mapping) or row.get("sha256") != sha256_file(path) or row.get("bytes") != path.stat().st_size:
            raise ValueError(f"terminal manifest does not bind {relative}")
    status_value = load_strict_json(status)
    if status_value != {"mode": "training", "schema_version": "scale-003-status.v1", "status": "PASS"}:
        raise ValueError("training handoff status is not exact PASS")
    semantic_value = load_strict_json(semantic)
    validate_training_semantic(semantic_value, result_root=training_attempt / "results/training")
    final_receipt = semantic_value["checkpoints"][-1]
    if final_receipt["path"] != str(checkpoint) or final_receipt["sha256"] != value["final_checkpoint_sha256"] or semantic_value["final_reload_audit"]["checkpoint_sha256"] != value["final_checkpoint_sha256"]:
        raise ValueError("training semantic/final checkpoint handoff mismatch")
    return TrainingHandoff(
        value["training_attempt_id"], training_attempt, terminal, semantic, status,
        checkpoint, value["final_checkpoint_sha256"],
    )
