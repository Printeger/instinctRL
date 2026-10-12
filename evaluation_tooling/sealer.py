"""Terminal DRAFT-to-SEALED transition with one atomic final write."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from typing import Any

from .canonical import atomic_canonical_create, canonical_load
from .external_inventory import ExternalInventorySpec, compare_external, scan_external
from .runner import INPUT_HASHES_NAME, MANIFEST_NAME, TERMINAL_NAME, EvaluationRunner, sha256_file, utc_now, write_canonical_replace
from .schemas import TERMINAL_SCHEMA_VERSION
from .validation import collect_draft_findings


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directories(root: Path) -> None:
    directories = [root] + sorted((path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()), key=lambda path: len(path.parts), reverse=True)
    for directory in directories:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _binding_category(relative: str, required_results: set[str]) -> str:
    if relative == MANIFEST_NAME:
        return "manifest"
    if relative == INPUT_HASHES_NAME:
        return "input_hashes"
    if relative == "COMMAND_INDEX.json":
        return "command_index"
    if relative == "COMMAND_LEDGER.json":
        return "command_ledger"
    if relative.startswith("launcher_receipts/"):
        return "launcher_receipts"
    if relative.startswith("runner_receipts/"):
        return "runner_receipts"
    if relative.startswith("external/"):
        return "external_inventory"
    if relative in required_results or relative.startswith("results/"):
        return "result_files"
    if relative.endswith("/stdout.bin") or relative.endswith("/stderr.bin"):
        return "streams"
    return "metadata"


def _inventory(root: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[str]], int]:
    rows: list[dict[str, Any]] = []
    bindings = {name: [] for name in (
        "manifest", "input_hashes", "streams", "metadata", "command_ledger", "command_index",
        "launcher_receipts", "runner_receipts", "result_files", "external_inventory",
    )}
    maximum_mtime = 0
    required_results = set(manifest["required_results"])
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"attempt symlink is forbidden: {path.relative_to(root)}")
        if not path.is_file() or path.name == TERMINAL_NAME:
            continue
        relative = path.relative_to(root).as_posix()
        stat_result = path.stat()
        maximum_mtime = max(maximum_mtime, stat_result.st_mtime_ns)
        rows.append({
            "path": relative,
            "type": "file",
            "mode": stat.S_IMODE(stat_result.st_mode),
            "bytes": stat_result.st_size,
            "sha256": sha256_file(path),
        })
        bindings[_binding_category(relative, required_results)].append(relative)
    for paths in bindings.values():
        paths.sort()
    return rows, bindings, maximum_mtime


def seal_attempt(attempt_dir: Path, *, tooling_commit: str, product_commit: str) -> dict[str, Any]:
    """Seal only after a complete collect-all pass; failure leaves DRAFT."""

    attempt_dir = attempt_dir.resolve()
    terminal_path = attempt_dir / TERMINAL_NAME
    if terminal_path.exists():
        raise RuntimeError("pre-existing terminal manifest; SEALED namespace is immutable")
    runner = EvaluationRunner(attempt_dir, tooling_commit=tooling_commit, product_commit=product_commit)
    runner._ensure_loaded()

    # Persist post inventories and comparisons before terminal aggregates.
    for declaration in runner.manifest["external_inventories"]:
        spec = ExternalInventorySpec.from_manifest(declaration)
        before_path = attempt_dir / "external" / f"{spec.name}.pre.json"
        after = scan_external(spec)
        write_canonical_replace(attempt_dir / "external" / f"{spec.name}.post.json", after)
        try:
            before = canonical_load(before_path)
        except (OSError, ValueError):
            continue
        write_canonical_replace(attempt_dir / "external" / f"{spec.name}.comparison.json", compare_external(spec, before, after))

    started_at = utc_now()
    findings = collect_draft_findings(
        attempt_dir,
        tooling_commit=tooling_commit,
        product_commit=product_commit,
        require_complete=True,
    )
    receipt_id = len(runner._receipt_files()) + 1
    if findings:
        index_bytes = (attempt_dir / "COMMAND_INDEX.json").read_bytes()
        ledger_bytes = (attempt_dir / "COMMAND_LEDGER.json").read_bytes()
        failed_receipt = {
            "schema_version": "evaluation-runner-receipt.v1",
            "attempt_id": runner.manifest["attempt_id"],
            "invocation_id": receipt_id,
            "mode": "seal",
            "command_ids": [],
            "command_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "command_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
            "identity_proof": canonical_load(attempt_dir / "TOOL_STATE.json")["identity_proof"],
            "failure": "PRESEAL_VALIDATION",
            "started_at": started_at,
            "ended_at": utc_now(),
            "result": "FAIL",
        }
        atomic_canonical_create(attempt_dir / "runner_receipts" / f"{receipt_id:04d}.json", failed_receipt)
        return {
            "schema_version": "evaluation-seal-result.v1",
            "status": "FAIL",
            "lifecycle": "DRAFT",
            "findings": [finding.as_dict() for finding in findings],
        }

    index_bytes, ledger_bytes = runner._rebuild_aggregates(terminal=True)
    terminal_receipt = {
        "schema_version": "evaluation-runner-receipt.v1",
        "attempt_id": runner.manifest["attempt_id"],
        "invocation_id": receipt_id,
        "mode": "seal",
        "command_ids": [],
        "command_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "command_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "identity_proof": canonical_load(attempt_dir / "TOOL_STATE.json")["identity_proof"],
        "failure": None,
        "started_at": started_at,
        "ended_at": utc_now(),
        "result": "PASS",
    }
    atomic_canonical_create(attempt_dir / "runner_receipts" / f"{receipt_id:04d}.json", terminal_receipt)

    # The state is final metadata and is persisted before inventory construction.
    state = canonical_load(attempt_dir / "TOOL_STATE.json")
    draft_state = dict(state)
    state["lifecycle"] = "SEALED"
    state["sealed_at"] = utc_now()
    write_canonical_replace(attempt_dir / "TOOL_STATE.json", state)
    rows, bindings, maximum_mtime = _inventory(attempt_dir, runner.manifest)
    terminal = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "attempt_id": runner.manifest["attempt_id"],
        "lifecycle": "SEALED",
        "created_at": utc_now(),
        "product_commit": product_commit,
        "tooling_commit": tooling_commit,
        "identity_proof": state["identity_proof"],
        "self_exclusion": [TERMINAL_NAME],
        "max_preseal_mtime_ns": maximum_mtime,
        "files": rows,
        "bindings": bindings,
    }
    # Verify complete construction in memory before making the sole terminal write.
    if set(row["path"] for row in rows) != set().union(*(set(paths) for paths in bindings.values())):
        raise RuntimeError("terminal binding construction is incomplete")
    try:
        for row in rows:
            _fsync_file(attempt_dir / row["path"])
        _fsync_directories(attempt_dir)
        atomic_canonical_create(terminal_path, terminal)
    except BaseException:
        # Terminal creation is the lifecycle boundary. If it did not occur,
        # restore the authoritative persisted state to DRAFT for repair.
        if not terminal_path.exists():
            write_canonical_replace(attempt_dir / "TOOL_STATE.json", draft_state)
            preserved = attempt_dir / "failed_seal_receipts" / f"{receipt_id:04d}.pass.json"
            preserved.parent.mkdir(parents=True, exist_ok=True)
            receipt_path = attempt_dir / "runner_receipts" / f"{receipt_id:04d}.json"
            if receipt_path.exists():
                receipt_path.replace(preserved)
            index_bytes, ledger_bytes = runner._rebuild_aggregates(terminal=False)
            atomic_canonical_create(receipt_path, {
                "schema_version": "evaluation-runner-receipt.v1",
                "attempt_id": runner.manifest["attempt_id"],
                "invocation_id": receipt_id,
                "mode": "seal",
                "command_ids": [],
                "command_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
                "command_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
                "identity_proof": draft_state["identity_proof"],
                "failure": "TERMINAL_CREATE_INTERRUPTED",
                "started_at": started_at,
                "ended_at": utc_now(),
                "result": "FAIL",
            })
        raise
    _fsync_file(terminal_path)
    _fsync_directories(attempt_dir)
    return {
        "schema_version": "evaluation-seal-result.v1",
        "status": "PASS",
        "lifecycle": "SEALED",
        "terminal_manifest_sha256": sha256_file(terminal_path),
        "findings": [],
    }
