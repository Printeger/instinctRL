"""Append-only, CPU-only failure evidence capture for offline H1 validation.

The recorder accepts observations as data, creates no process or backend, and
never retries work.  Every capture is a new immutable directory containing the
six raw artifacts required by the readiness contract and a hash manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "unified-h1-offline-failure-evidence.v1"
INJECTION_TYPES = frozenset({
    "pre-construction", "initialization", "controlled-failure",
    "supervisor-lost", "late-evidence",
})
RAW_FILES = (
    "status.json", "process_identity.json", "observed_window.json",
    "counters.json", "stdout.log", "stderr.log",
)


class FailureEvidenceError(RuntimeError):
    """Evidence was incomplete, unsafe, mutable, or inconsistent."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _safe_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or Path(value).name != value:
        raise FailureEvidenceError(f"{label} is not a safe path component")
    return value


def _regular_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise FailureEvidenceError(f"evidence artifact is not a regular file: {path.name}")
    return path.read_bytes()


def _parse_object(data: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FailureEvidenceError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise FailureEvidenceError(f"{label} must contain a JSON object")
    return value


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


class OfflineFailureEvidenceArchive:
    """Create and verify append-only evidence without executing observed work."""

    def __init__(self, root: Path, candidate: Mapping[str, str]) -> None:
        self.root = Path(root)
        if self.root.is_symlink():
            raise FailureEvidenceError("archive root cannot be a symlink")
        required = {"repository", "base_commit", "commit", "diff_sha256"}
        if set(candidate) != required or any(not isinstance(candidate[key], str) or not candidate[key] for key in required):
            raise FailureEvidenceError("exact candidate binding is required")
        self.candidate = dict(candidate)
        self.root.mkdir(parents=True, exist_ok=True)

    def capture(self, evidence_id: str, injection_type: str, raw: Mapping[str, bytes], *,
                parent_manifest_sha256: str | None = None) -> Mapping[str, Any]:
        """Persist one complete observation exactly once; never run or retry it."""
        _safe_component(evidence_id, "evidence_id")
        if injection_type not in INJECTION_TYPES:
            raise FailureEvidenceError("unknown failure injection type")
        if set(raw) != set(RAW_FILES) or any(type(raw[name]) is not bytes for name in RAW_FILES):
            raise FailureEvidenceError("all raw status, identity, window, counters, stdout and stderr bytes are required")
        status = _parse_object(raw["status.json"], "status.json")
        identity = _parse_object(raw["process_identity.json"], "process_identity.json")
        window = _parse_object(raw["observed_window.json"], "observed_window.json")
        counters = _parse_object(raw["counters.json"], "counters.json")
        if status.get("injection_type") != injection_type:
            raise FailureEvidenceError("raw status does not match the injection type")
        if "status" not in status or "state" not in identity or "state" not in window:
            raise FailureEvidenceError("raw status, process identity, or observed window is incomplete")
        count_keys = {"live_starts", "retries", "transitions", "ppo_updates", "checkpoint_writes"}
        if not count_keys.issubset(counters) or any(type(counters[key]) is not int or counters[key] < 0 for key in count_keys):
            raise FailureEvidenceError("raw counters are incomplete or invalid")
        if counters["live_starts"] not in {0, 1} or counters["retries"] != 0:
            raise FailureEvidenceError("offline evidence must show at most one start and zero retries")
        if injection_type == "pre-construction" and counters["live_starts"] != 0:
            raise FailureEvidenceError("pre-construction evidence cannot claim a process start")
        if counters["live_starts"] == 1:
            if (type(identity.get("pid")) is not int or identity["pid"] <= 0
                    or type(identity.get("process_group_id")) is not int
                    or identity["process_group_id"] <= 0):
                raise FailureEvidenceError("started evidence requires raw positive process identity")
        if not isinstance(window.get("samples"), list):
            raise FailureEvidenceError("raw observed window samples are missing")
        if injection_type == "late-evidence":
            if not isinstance(parent_manifest_sha256, str) or len(parent_manifest_sha256) != 64:
                raise FailureEvidenceError("late evidence requires its immutable parent manifest digest")
            matching_parents = []
            for entry in self.root.iterdir():
                if entry.is_dir() and entry.name != evidence_id:
                    try:
                        receipt = self.verify(entry.name)
                    except FailureEvidenceError:
                        continue
                    if receipt["manifest_sha256"] == parent_manifest_sha256:
                        matching_parents.append(receipt)
            if len(matching_parents) != 1 or matching_parents[0]["injection_type"] == "late-evidence":
                raise FailureEvidenceError("late evidence parent manifest is not a unique verified primary observation")
        elif parent_manifest_sha256 is not None:
            raise FailureEvidenceError("only late evidence may bind a parent manifest")

        destination = self.root / evidence_id
        try:
            destination.mkdir(mode=0o755)
        except FileExistsError as error:
            raise FailureEvidenceError("refusing to overwrite existing evidence") from error
        for name in RAW_FILES:
            _write_exclusive(destination / name, raw[name])
        files = {
            name: {"bytes": len(raw[name]), "sha256": _sha256(raw[name])}
            for name in RAW_FILES
        }
        manifest = {
            "schema_version": SCHEMA,
            "evidence_id": evidence_id,
            "injection_type": injection_type,
            "candidate": self.candidate,
            "parent_manifest_sha256": parent_manifest_sha256,
            "automatic_retry": False,
            "files": files,
        }
        manifest_bytes = _canonical(manifest)
        _write_exclusive(destination / "manifest.json", manifest_bytes)
        return {**manifest, "manifest_sha256": _sha256(manifest_bytes)}

    def verify(self, evidence_id: str) -> Mapping[str, Any]:
        _safe_component(evidence_id, "evidence_id")
        directory = self.root / evidence_id
        manifest_bytes = _regular_bytes(directory / "manifest.json")
        manifest = _parse_object(manifest_bytes, "manifest.json")
        if manifest.get("schema_version") != SCHEMA or manifest.get("evidence_id") != evidence_id:
            raise FailureEvidenceError("evidence manifest identity mismatch")
        if manifest.get("candidate") != self.candidate or manifest.get("automatic_retry") is not False:
            raise FailureEvidenceError("evidence candidate or retry binding mismatch")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != set(RAW_FILES):
            raise FailureEvidenceError("evidence manifest file set mismatch")
        actual = {entry.name for entry in directory.iterdir() if entry.is_file() and not entry.is_symlink()}
        if actual != {*RAW_FILES, "manifest.json"}:
            raise FailureEvidenceError("evidence directory contains missing or unmanifested files")
        for name in RAW_FILES:
            data = _regular_bytes(directory / name)
            if files[name] != {"bytes": len(data), "sha256": _sha256(data)}:
                raise FailureEvidenceError(f"evidence hash mismatch: {name}")
        return {**manifest, "manifest_sha256": _sha256(manifest_bytes)}


def verify_d0005_missing_window(evidence_path: Path, expected_sha256: str) -> Mapping[str, Any]:
    """Verify the sealed D0005 gap without creating or reconstructing a sample."""
    evidence_path = Path(evidence_path)
    evidence_bytes = _regular_bytes(evidence_path)
    if _sha256(evidence_bytes) != expected_sha256:
        raise FailureEvidenceError("sealed D0005 evidence digest mismatch")
    evidence = _parse_object(evidence_bytes, "D0005 EVIDENCE.json")
    if evidence.get("attempt_id") != "UNIFIED-H1-R3-D0005" or evidence.get("status") != "SEALED_FAILURE":
        raise FailureEvidenceError("sealed D0005 identity mismatch")
    relative = "results/prelaunch/prelaunch_resource_gate_failure.json"
    row = evidence.get("files", {}).get(relative)
    artifact = evidence_path.parent / "attempt" / relative
    artifact_bytes = _regular_bytes(artifact)
    if row != {"bytes": len(artifact_bytes), "sha256": _sha256(artifact_bytes)}:
        raise FailureEvidenceError("sealed D0005 failure artifact mismatch")
    failure = _parse_object(artifact_bytes, "D0005 prelaunch failure")
    if "observed_window" in failure:
        raise FailureEvidenceError("D0005 unexpectedly contains a reconstructed observed_window")
    accounting = evidence.get("accounting", {}).get("d0005", {})
    expected = {"live_starts": 1, "retries": 0, "applauncher_constructions": 0,
                "transitions": 0, "ppo_updates": 0, "checkpoint_writes": 0}
    if any(accounting.get(key) != value for key, value in expected.items()):
        raise FailureEvidenceError("sealed D0005 accounting changed")
    return {
        "attempt_id": "UNIFIED-H1-R3-D0005",
        "evidence_sha256": expected_sha256,
        "observed_window_present": False,
        "reconstructed": False,
        "accounting": expected,
    }


__all__ = [
    "FailureEvidenceError", "INJECTION_TYPES", "OfflineFailureEvidenceArchive",
    "RAW_FILES", "SCHEMA", "verify_d0005_missing_window",
]
