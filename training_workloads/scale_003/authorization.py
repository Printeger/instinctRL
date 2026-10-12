"""Fail-closed validation of a future Commander/user live execution grant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

from .config import ConfigError, load_config, load_strict_json, sha256_file
from .handoff import TrainingHandoff, validate_training_handoff


class AuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class ExecutionAuthorization:
    mode: str
    attempt_id: str
    attempt_path: Path
    execution_order_id: str
    candidate_commit: str
    candidate_tree: str
    document_path: Path
    input_handoff: TrainingHandoff | None


FIELDS = {
    "schema_version", "status", "mode", "attempt_id", "attempt_path",
    "execution_order_id", "execution_order_revision", "execution_order_path", "execution_order_sha256",
    "user_authorization_path", "user_authorization_sha256", "candidate_repository", "candidate_commit",
    "candidate_tree", "proposal_sha256", "method_decisions_approved", "limits", "input_handoff",
}


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(repository), *arguments], check=True, capture_output=True, text=True).stdout.strip()


def _canonical_commander_file(path: Path, coordination: Path) -> bool:
    if path.is_symlink():
        return False
    path = path.resolve()
    commander = (coordination / "commander").resolve()
    return path.is_file() and not path.is_symlink() and commander in path.parents


def load_execution_authorization(path: str | Path, *, expected_mode: str, expected_attempt_dir: str | Path) -> ExecutionAuthorization:
    config = load_config()
    authored_document_path = Path(path)
    document_path = authored_document_path.resolve()
    coordination = Path(config["paths"]["coordination_repository"])
    if not _canonical_commander_file(authored_document_path, coordination):
        raise AuthorizationError("authorization document must be a canonical Commander file")
    try:
        value: Any = load_strict_json(document_path)
    except (OSError, ConfigError) as exc:
        raise AuthorizationError("authorization document is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise AuthorizationError("authorization fields differ from the execution schema")
    if value["schema_version"] != "scale-003-live-execution-authorization.v1" or value["status"] != "AUTHORIZED":
        raise AuthorizationError("live execution is not authorized")
    if expected_mode not in {"training", "evaluation"} or value["mode"] != expected_mode:
        raise AuthorizationError("authorization mode mismatch")
    expected_attempt = Path(expected_attempt_dir).resolve()
    attempts_root = (coordination / "attempts/development").resolve()
    if attempts_root not in expected_attempt.parents or expected_attempt != Path(value["attempt_path"]).resolve() or expected_attempt.name != value["attempt_id"]:
        raise AuthorizationError("attempt binding mismatch")
    order_path = Path(value["execution_order_path"])
    user_path = Path(value["user_authorization_path"])
    if not _canonical_commander_file(order_path, coordination) or not _canonical_commander_file(user_path, coordination):
        raise AuthorizationError("execution Order/user authorization is not canonical")
    if sha256_file(order_path) != value["execution_order_sha256"] or sha256_file(user_path) != value["user_authorization_sha256"]:
        raise AuthorizationError("execution authority identity mismatch")
    repository = Path(config["paths"]["candidate_repository"])
    identity = (_git(repository, "rev-parse", "HEAD"), _git(repository, "rev-parse", "HEAD^{tree}"), _git(repository, "status", "--porcelain"))
    if value["candidate_repository"] != str(repository) or identity != (value["candidate_commit"], value["candidate_tree"], ""):
        raise AuthorizationError("candidate commit/tree/clean identity mismatch")
    if value["proposal_sha256"] != config["proposal"]["sha256"]:
        raise AuthorizationError("proposal binding mismatch")
    decisions = value["method_decisions_approved"]
    if not isinstance(decisions, dict) or set(decisions) != {"cumulative_1m", "checkpoint_resume", "scene_32_copy", "command_curriculum"} or not all(item is True for item in decisions.values()):
        raise AuthorizationError("all method decisions require explicit approval")
    limits = value["limits"]
    expected_limits = ({
        "process_starts": 1, "environment_transitions": 872000, "optimizer_updates": 1704,
        "checkpoint_writes": 70, "maximum_wall_time_seconds": 7200, "automatic_retries": 0,
    } if expected_mode == "training" else {
        "process_starts": 2, "environment_transitions": 1024, "optimizer_updates": 0,
        "checkpoint_writes": 0, "maximum_wall_time_seconds": 1800, "automatic_retries": 0,
    })
    if limits != expected_limits:
        raise AuthorizationError("live resource limits differ from the reviewed candidate")
    if value["execution_order_id"] == "SCALE-003" and value["execution_order_revision"] == 0:
        raise AuthorizationError("the source-only SCALE-003 revision 0 Order cannot authorize runtime")
    handoff = None
    if expected_mode == "training":
        if value["input_handoff"] is not None:
            raise AuthorizationError("training authorization must not claim a predecessor handoff")
    else:
        try:
            handoff = validate_training_handoff(
                value["input_handoff"], evaluation_attempt=expected_attempt,
                attempts_root=attempts_root,
            )
        except (OSError, ValueError) as exc:
            raise AuthorizationError(f"immutable training handoff validation failed: {exc}") from exc
    return ExecutionAuthorization(expected_mode, value["attempt_id"], expected_attempt, value["execution_order_id"], value["candidate_commit"], value["candidate_tree"], document_path, handoff)
