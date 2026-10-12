"""Collect-all DRAFT checks and execution-free SEALED packet inspection."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any

from .canonical import CanonicalJSONError, canonical_load
from .external_inventory import ExternalInventorySpec, compare_external
from .findings import Finding, FindingCollector
from .runner import (
    INPUT_HASHES_NAME,
    MANIFEST_NAME,
    TERMINAL_NAME,
    EvaluationRunner,
    load_submission,
    sha256_file,
    verify_pins_and_inputs,
)
from .schemas import (
    LAUNCH_PROFILE,
    LEGACY_TERMINAL_SCHEMA_VERSION,
    TERMINAL_SCHEMA_VERSION,
    executable_source_paths,
)


TERMINAL_FIELDS = {
    "schema_version",
    "attempt_id",
    "lifecycle",
    "created_at",
    "product_commit",
    "tooling_commit",
    "self_exclusion",
    "max_preseal_mtime_ns",
    "files",
    "bindings",
    "identity_proof",
}
LEGACY_FILE_FIELDS = {"path", "bytes", "sha256"}
FILE_FIELDS = LEGACY_FILE_FIELDS | {"type", "mode"}
LEGACY_TERMINAL_TOOLING_COMMITS = frozenset({
    "860ce2c79ffa6a829d10028117de35a95daa34d4",
    "d55addd6af4758aff83e20d600d3678eda0e752f",
    "3611ef4b4ec8e6ef62011688d2dd6e227cd250b5",
    "f810720137825515e83e37a2029de72b955cb2ce",
})
BINDING_FIELDS = {
    "manifest",
    "input_hashes",
    "streams",
    "metadata",
    "command_ledger",
    "command_index",
    "launcher_receipts",
    "runner_receipts",
    "result_files",
    "external_inventory",
}
LAUNCHER_RECEIPT_FIELDS = {
    "schema_version", "attempt_id", "command_id", "raw_exit", "semantic_exit",
    "receipt_status", "primary_failure", "cleanup_status", "cleanup_exit", "timed_out",
}
RUNNER_RECEIPT_FIELDS = {
    "schema_version", "attempt_id", "invocation_id", "mode", "command_ids",
    "command_index_sha256", "command_ledger_sha256", "identity_proof", "failure",
    "started_at", "ended_at", "result",
}
COMMAND_METADATA_FIELDS = {
    "schema_version", "attempt_id", "command_id", "manifest_command_index", "label", "argv",
    "cwd", "environment", "started_at", "ended_at", "expected_exit", "raw_exit", "semantic_exit",
    "receipt_status", "primary_failure", "cleanup_status", "repair_of", "stdout_path", "stdout_bytes",
    "stdout_sha256", "stderr_path", "stderr_bytes", "stderr_sha256", "cleanup_stdout_path",
    "cleanup_stdout_bytes", "cleanup_stdout_sha256", "cleanup_stderr_path", "cleanup_stderr_bytes",
    "cleanup_stderr_sha256", "launcher_receipt_path", "launcher_receipt_sha256", "metadata_path",
}
INDEX_FIELDS = {"schema_version", "attempt_id", "terminal", "command_count", "commands"}
LEDGER_FIELDS = {
    "schema_version", "attempt_id", "terminal", "command_count", "unexpected_commands",
    "repaired_commands", "unresolved_commands", "commands",
}
TOOL_STATE_DRAFT_FIELDS = {
    "schema_version", "attempt_id", "lifecycle", "manifest_sha256", "input_hashes_sha256",
    "product_commit", "tooling_commit", "runtime_started", "identity_proof",
}
TOOL_STATE_SEALED_FIELDS = TOOL_STATE_DRAFT_FIELDS | {"sealed_at"}


def _exact_keys(value: dict[str, Any], expected: set[str], collector: FindingCollector, location: str) -> None:
    for field in sorted(expected - set(value)):
        collector.add("MISSING_FIELD", f"{location}.{field}", "required field is missing")
    for field in sorted(set(value) - expected):
        collector.add("UNKNOWN_FIELD", f"{location}.{field}", "field is not defined")


def _load_control(path: Path, collector: FindingCollector, location: str) -> Any | None:
    if not path.is_file() or path.is_symlink():
        collector.add("MISSING_FILE", location, "required regular file is missing")
        return None
    try:
        return canonical_load(path)
    except CanonicalJSONError as error:
        collector.add("CANONICAL_JSON", location, str(error))
        return None


def _parse_time(value: Any, collector: FindingCollector, location: str) -> dt.datetime | None:
    if not isinstance(value, str):
        collector.add("TIMESTAMP", location, "expected ISO-8601 timestamp string")
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        collector.add("TIMESTAMP", location, "invalid ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None:
        collector.add("TIMESTAMP", location, "timestamp must include timezone")
        return None
    return parsed


def _confined_reference(attempt_dir: Path, value: Any, collector: FindingCollector, location: str) -> Path | None:
    if not isinstance(value, str):
        collector.add("PATH", location, "expected confined relative path")
        return None
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        collector.add("PATH", location, "path is absolute or traverses outside the attempt")
        return None
    path = (attempt_dir / Path(*relative.parts)).resolve(strict=False)
    if attempt_dir != path and attempt_dir not in path.parents:
        collector.add("PATH", location, "path resolves outside the attempt")
        return None
    return path


def _collect_command_rows(attempt_dir: Path, collector: FindingCollector) -> tuple[list[dict[str, Any]], list[Path]]:
    root = attempt_dir / "commands"
    directories = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name) if root.is_dir() else []
    rows: list[dict[str, Any]] = []
    valid_directories: list[Path] = []
    for index, directory in enumerate(directories, start=1):
        if not directory.name.startswith(f"{index:04d}_"):
            collector.add("COMMAND_DIRECTORY_ID", directory.name, f"expected contiguous prefix {index:04d}_")
        metadata = _load_control(directory / "metadata.json", collector, f"{directory.name}/metadata.json")
        if isinstance(metadata, dict):
            rows.append(metadata)
            valid_directories.append(directory)
        transaction = _load_control(directory / "transaction.json", collector, f"{directory.name}/transaction.json")
        complete = _load_control(directory / "transaction_complete.json", collector, f"{directory.name}/transaction_complete.json")
        if isinstance(transaction, dict):
            if set(transaction) != {"schema_version", "command_id", "manifest_command_index", "repair_of", "started_at"}:
                collector.add("TRANSACTION_SCHEMA", directory.name, "transaction fields are not exact")
            if transaction.get("command_id") != index:
                collector.add("TRANSACTION_ID", directory.name, "transaction command ID differs from directory")
        if isinstance(complete, dict):
            if set(complete) != {"schema_version", "command_id", "result"}:
                collector.add("TRANSACTION_SCHEMA", directory.name, "completion fields are not exact")
            if complete.get("command_id") != index or complete.get("result") not in {"COMMITTED", "RECOVERED_FAIL", "BLOCKED_FAIL"}:
                collector.add("TRANSACTION_RESULT", directory.name, "completion identity/result is invalid")
    return rows, valid_directories


def _validate_identity_proof(
    proof: Any,
    manifest: dict[str, Any],
    input_hashes: dict[str, Any],
    collector: FindingCollector,
    location: str,
) -> None:
    if not isinstance(proof, dict):
        collector.add("IDENTITY_SCHEMA", location, "identity proof top-level fields are not exact")
        return
    version = proof.get("schema_version")
    expected_fields = {"schema_version", "tooling", "product", "status"}
    if version == "evaluation-identity-proof.v2":
        expected_fields.add("launch_profile")
    if set(proof) != expected_fields or version not in {
        "evaluation-identity-proof.v1", "evaluation-identity-proof.v2",
    }:
        collector.add("IDENTITY_SCHEMA", location, "identity proof top-level fields are not exact")
        return
    if proof.get("status") != "PASS":
        collector.add("IDENTITY_STATUS", location, "identity proof must be versioned PASS")
    tooling = proof.get("tooling")
    tooling_fields = {"package_root", "repository", "commit", "git_toplevel", "tree", "files", "file_set_sha256"}
    if not isinstance(tooling, dict) or set(tooling) != tooling_fields:
        collector.add("IDENTITY_SCHEMA", f"{location}.tooling", "tooling proof fields are not exact")
    else:
        if tooling.get("repository") != manifest.get("tooling_repository") or tooling.get("git_toplevel") != manifest.get("tooling_repository"):
            collector.add("IDENTITY_BINDING", f"{location}.tooling.repository", "does not match manifest")
        if tooling.get("commit") != manifest.get("tooling_commit") or tooling.get("tree") != manifest.get("tooling_tree"):
            collector.add("IDENTITY_BINDING", f"{location}.tooling", "commit/tree do not match manifest")
        files = tooling.get("files")
        if not isinstance(files, list):
            collector.add("IDENTITY_SCHEMA", f"{location}.tooling.files", "expected ordered file rows")
        else:
            paths: list[str] = []
            valid_rows: list[dict[str, Any]] = []
            for index, row in enumerate(files):
                if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
                    collector.add("IDENTITY_SCHEMA", f"{location}.tooling.files[{index}]", "file row fields are not exact")
                    continue
                paths.append(str(row.get("path")))
                valid_rows.append(row)
            if paths != sorted(set(paths)):
                collector.add("IDENTITY_FILE_SET", f"{location}.tooling.files", "paths must be unique and sorted")
            expected_set_hash = hashlib.sha256(
                "\n".join(f"{row['path']}\0{row['bytes']}\0{row['sha256']}" for row in valid_rows).encode("utf-8")
            ).hexdigest()
            if tooling.get("file_set_sha256") != expected_set_hash:
                collector.add("IDENTITY_FILE_SET", f"{location}.tooling.file_set_sha256", "does not bind file rows")
    product = proof.get("product")
    product_fields = {"repository", "commit", "clean_required", "head", "tree", "clean", "status_sha256"}
    if not isinstance(product, dict) or set(product) != product_fields:
        collector.add("IDENTITY_SCHEMA", f"{location}.product", "product proof fields are not exact")
    else:
        if product.get("repository") != manifest.get("product_repository"):
            collector.add("IDENTITY_BINDING", f"{location}.product.repository", "does not match manifest")
        if product.get("commit") != manifest.get("product_commit") or product.get("head") != manifest.get("product_commit"):
            collector.add("IDENTITY_BINDING", f"{location}.product.commit", "does not match manifest/HEAD")
        if product.get("tree") != manifest.get("product_tree") or product.get("clean_required") != manifest.get("product_clean_required"):
            collector.add("IDENTITY_BINDING", f"{location}.product", "tree/clean policy does not match manifest")
        if product.get("clean_required") is True and product.get("clean") is not True:
            collector.add("IDENTITY_STATUS", f"{location}.product.clean", "required clean product was not proven")
    if version == "evaluation-identity-proof.v2":
        _validate_launch_profile_proof(
            proof.get("launch_profile"), manifest, input_hashes, collector, f"{location}.launch_profile"
        )


def _validate_launch_profile_proof(
    proof: Any,
    manifest: dict[str, Any],
    input_hashes: dict[str, Any],
    collector: FindingCollector,
    location: str,
) -> None:
    fields = {"schema_version", "profile", "executing_interpreter", "commands", "status"}
    if not isinstance(proof, dict) or set(proof) != fields:
        collector.add("LAUNCH_PROOF_SCHEMA", location, "launch-profile proof fields are not exact")
        return
    if (
        proof.get("schema_version") != "evaluation-launch-profile-proof.v1"
        or proof.get("profile") != LAUNCH_PROFILE
        or proof.get("status") != "PASS"
    ):
        collector.add("LAUNCH_PROOF_STATUS", location, "launch-profile proof must be versioned DIRECT_PINNED PASS")
    interpreter = proof.get("executing_interpreter")
    if not isinstance(interpreter, str) or not Path(interpreter).is_absolute():
        collector.add("LAUNCH_PROOF_SCHEMA", f"{location}.executing_interpreter", "expected absolute path")
    pinned = {
        str(Path(os.path.abspath(row["path"]))): {"bytes": row.get("bytes"), "sha256": row.get("sha256")}
        for row in input_hashes.get("files", [])
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    declarations = manifest.get("commands")
    rows = proof.get("commands")
    if not isinstance(declarations, list) or not isinstance(rows, list) or len(rows) != len(declarations):
        collector.add("LAUNCH_PROOF_SCHEMA", f"{location}.commands", "must match manifest command count")
        return
    command_fields = {"command_index", "argv", "cleanup_argv"}
    launch_fields = {"executable", "executable_pin", "source", "source_index", "source_pin"}
    for index, (declaration, row) in enumerate(zip(declarations, rows)):
        row_location = f"{location}.commands[{index}]"
        if not isinstance(row, dict) or set(row) != command_fields or row.get("command_index") != index:
            collector.add("LAUNCH_PROOF_SCHEMA", row_location, "command proof fields/index are not exact")
            continue
        for field in ("argv", "cleanup_argv"):
            declared_argv = declaration.get(field) if isinstance(declaration, dict) else None
            observed = row.get(field)
            if declared_argv is None:
                if observed is not None:
                    collector.add("LAUNCH_PROOF_BINDING", f"{row_location}.{field}", "must be null with manifest argv")
                continue
            sources = executable_source_paths(declared_argv, declaration.get("cwd"))
            if not isinstance(observed, dict) or set(observed) != launch_fields or len(sources) != 1:
                collector.add("LAUNCH_PROOF_SCHEMA", f"{row_location}.{field}", "launch row is not exact")
                continue
            source_index, source, was_relative = sources[0]
            executable_path = str(Path(os.path.abspath(declared_argv[0])))
            source_path = str(Path(os.path.abspath(source)))
            expected = {
                "executable": executable_path,
                "executable_pin": pinned.get(executable_path),
                "source": source_path,
                "source_index": source_index,
                "source_pin": pinned.get(source_path),
            }
            if was_relative or observed != expected or executable_path != interpreter:
                collector.add("LAUNCH_PROOF_BINDING", f"{row_location}.{field}", "does not bind manifest, pins and executing interpreter")


def collect_draft_findings(
    attempt_dir: Path,
    *,
    tooling_commit: str,
    product_commit: str,
    require_complete: bool,
    verify_live_inputs: bool = True,
) -> tuple[Finding, ...]:
    """Validate persisted evidence without launching or importing attempt code."""

    attempt_dir = attempt_dir.resolve()
    collector = FindingCollector()
    manifest, input_hashes, submission_findings = load_submission(attempt_dir, allow_tool_outputs=True)
    collector.extend(submission_findings)
    if manifest is None or input_hashes is None:
        return collector.findings
    if verify_live_inputs:
        collector.extend(
            verify_pins_and_inputs(
                manifest,
                input_hashes,
                tooling_commit=tooling_commit,
                product_commit=product_commit,
            )
        )
    else:
        if manifest.get("tooling_commit") != tooling_commit or input_hashes.get("tooling_commit") != tooling_commit:
            collector.add("TOOLING_PIN", "offline pins", "tooling commit documents disagree")
        if manifest.get("product_commit") != product_commit or input_hashes.get("product_commit") != product_commit:
            collector.add("PRODUCT_PIN", "offline pins", "product commit documents disagree")
    state = _load_control(attempt_dir / "TOOL_STATE.json", collector, "TOOL_STATE.json")
    if isinstance(state, dict):
        expected_lifecycle = "SEALED" if (attempt_dir / TERMINAL_NAME).is_file() else "DRAFT"
        expected_state_fields = TOOL_STATE_SEALED_FIELDS if expected_lifecycle == "SEALED" else TOOL_STATE_DRAFT_FIELDS
        _exact_keys(state, expected_state_fields, collector, "TOOL_STATE")
        if state.get("schema_version") != "evaluation-tool-state.v1":
            collector.add("SCHEMA_VERSION", "TOOL_STATE.schema_version", "expected evaluation-tool-state.v1")
        if state.get("lifecycle") != expected_lifecycle:
            collector.add("LIFECYCLE", "TOOL_STATE.lifecycle", f"expected {expected_lifecycle}")
        if state.get("attempt_id") != manifest.get("attempt_id"):
            collector.add("ATTEMPT_ID", "TOOL_STATE.attempt_id", "does not match manifest")
        if not isinstance(state.get("runtime_started"), bool):
            collector.add("TYPE", "TOOL_STATE.runtime_started", "expected boolean")
        if expected_lifecycle == "SEALED":
            _parse_time(state.get("sealed_at"), collector, "TOOL_STATE.sealed_at")
        if state.get("manifest_sha256") != sha256_file(attempt_dir / MANIFEST_NAME):
            collector.add("FROZEN_IDENTITY", "TOOL_STATE.manifest_sha256", "manifest changed after preparation")
        if state.get("input_hashes_sha256") != sha256_file(attempt_dir / INPUT_HASHES_NAME):
            collector.add("FROZEN_IDENTITY", "TOOL_STATE.input_hashes_sha256", "input hashes changed after preparation")
        if state.get("product_commit") != product_commit or state.get("tooling_commit") != tooling_commit:
            collector.add("FROZEN_IDENTITY", "TOOL_STATE", "state commit pins differ")
        _validate_identity_proof(
            state.get("identity_proof"), manifest, input_hashes, collector, "TOOL_STATE.identity_proof"
        )

    runner = EvaluationRunner(attempt_dir, tooling_commit=tooling_commit, product_commit=product_commit)
    runner.manifest = manifest
    runner.input_hashes = input_hashes
    command_rows, command_directories = _collect_command_rows(attempt_dir, collector)
    try:
        receipt_files = runner._receipt_files()
    except RuntimeError as error:
        collector.add("RUNNER_RECEIPT", "runner_receipts", str(error))
        receipt_files = []

    declarations = manifest.get("commands") if isinstance(manifest.get("commands"), list) else []
    required_results = manifest.get("required_results") if isinstance(manifest.get("required_results"), list) else []
    external_declarations = manifest.get("external_inventories") if isinstance(manifest.get("external_inventories"), list) else []

    if require_complete:
        observed_indexes = {row.get("manifest_command_index") for row in command_rows if isinstance(row, dict)}
        for command_index in range(len(declarations)):
            if command_index not in observed_indexes:
                collector.add("MISSING_COMMAND", f"commands[{command_index}]", "declared command has no execution record")

    previous_end: dt.datetime | None = None
    for row_index, row in enumerate(command_rows):
        location = f"commands[{row_index}]"
        directory = command_directories[row_index]
        _exact_keys(row, COMMAND_METADATA_FIELDS, collector, location)
        command_id = row.get("command_id")
        expected_command_id = int(directory.name.split("_", 1)[0]) if directory.name.split("_", 1)[0].isdigit() else row_index + 1
        if command_id != expected_command_id:
            collector.add("COMMAND_ID", f"{location}.command_id", "metadata ID differs from directory ID")
        if row.get("attempt_id") != manifest.get("attempt_id"):
            collector.add("ATTEMPT_ID", f"{location}.attempt_id", "does not match manifest")
        expected_metadata = f"commands/{directory.name}/metadata.json"
        if row.get("metadata_path") != expected_metadata:
            collector.add("PATH", f"{location}.metadata_path", f"expected {expected_metadata}")
        manifest_index = row.get("manifest_command_index")
        if not isinstance(manifest_index, int) or not 0 <= manifest_index < len(declarations):
            collector.add("COMMAND_INDEX", location, "manifest command index is invalid")
            continue
        declaration = declarations[manifest_index]
        if not isinstance(declaration, dict):
            collector.add("COMMAND_DECLARATION", location, "manifest command is not an object")
            continue
        for field in ("label", "argv", "cwd", "environment", "expected_exit"):
            if row.get(field) != declaration[field]:
                collector.add("COMMAND_DECLARATION", f"{location}.{field}", "record differs from manifest")
        start = _parse_time(row.get("started_at"), collector, f"{location}.started_at")
        end = _parse_time(row.get("ended_at"), collector, f"{location}.ended_at")
        if start is not None and end is not None and start > end:
            collector.add("CHRONOLOGY", location, "command ends before it starts")
        if previous_end is not None and start is not None and start < previous_end:
            collector.add("CHRONOLOGY", location, "command overlaps prior persisted command")
        if end is not None:
            previous_end = end
        for stream_name in ("stdout", "stderr", "cleanup_stdout", "cleanup_stderr"):
            relative = row.get(f"{stream_name}_path")
            expected_relative = f"commands/{directory.name}/{stream_name}.bin"
            if relative is None:
                if stream_name in {"stdout", "stderr"}:
                    collector.add("STREAM_PATH", f"{location}.{stream_name}_path", "mandatory stream path is missing")
                continue
            stream = _confined_reference(attempt_dir, relative, collector, f"{location}.{stream_name}_path")
            if relative != expected_relative:
                collector.add("STREAM_PATH", f"{location}.{stream_name}_path", f"expected {expected_relative}")
            if stream is not None and stream.is_file() and not stream.is_symlink():
                if row.get(f"{stream_name}_bytes") != stream.stat().st_size:
                    collector.add("STREAM_SIZE", relative, "metadata byte count differs")
                if row.get(f"{stream_name}_sha256") != sha256_file(stream):
                    collector.add("STREAM_HASH", relative, "metadata hash differs")
            else:
                collector.add("MISSING_FILE", str(relative), "stream is not a regular file")
        launcher_path = row.get("launcher_receipt_path")
        if not isinstance(launcher_path, str):
            collector.add("LAUNCHER_RECEIPT", location, "launcher receipt path missing")
            continue
        launcher = _confined_reference(attempt_dir, launcher_path, collector, f"{location}.launcher_receipt_path")
        expected_launcher = f"launcher_receipts/{expected_command_id:04d}.json"
        if launcher_path != expected_launcher:
            collector.add("PATH", f"{location}.launcher_receipt_path", f"expected {expected_launcher}")
        receipt = _load_control(launcher, collector, launcher_path) if launcher is not None else None
        if isinstance(receipt, dict):
            _exact_keys(receipt, LAUNCHER_RECEIPT_FIELDS, collector, launcher_path)
            if receipt.get("schema_version") != "evaluation-launcher-receipt.v1":
                collector.add("SCHEMA_VERSION", f"{launcher_path}.schema_version", "invalid launcher receipt schema")
            if receipt.get("attempt_id") != manifest.get("attempt_id") or receipt.get("command_id") != expected_command_id:
                collector.add("LAUNCHER_RECEIPT_ID", launcher_path, "attempt/command identity differs")
            if launcher is not None and row.get("launcher_receipt_sha256") != sha256_file(launcher):
                collector.add("LAUNCHER_RECEIPT_HASH", launcher_path, "metadata hash differs")
            for field in ("raw_exit", "semantic_exit", "receipt_status", "primary_failure", "cleanup_status"):
                if receipt.get(field) != row.get(field):
                    collector.add("LAUNCHER_RECEIPT_BINDING", f"{launcher_path}.{field}", "receipt differs from metadata")
            should_pass = (
                receipt.get("raw_exit") == 0
                and receipt.get("semantic_exit") == 0
                and receipt.get("receipt_status") == "PASS"
                and receipt.get("cleanup_status") in {"PASS", "NOT_REQUESTED"}
            )
            if receipt.get("receipt_status") == "PASS" and not should_pass:
                collector.add("FALSE_PASS", launcher_path, "PASS receipt violates exit/evidence rules")
            if receipt.get("receipt_status") not in {"PASS", "FAIL"}:
                collector.add("ENUM", f"{launcher_path}.receipt_status", "expected PASS or FAIL")
            if receipt.get("cleanup_status") not in {"PASS", "FAIL", "NOT_REQUESTED"}:
                collector.add("ENUM", f"{launcher_path}.cleanup_status", "invalid cleanup status")

    rows_by_id = {
        row.get("command_id"): row for row in command_rows
        if isinstance(row.get("command_id"), int) and not isinstance(row.get("command_id"), bool)
    }
    if len(rows_by_id) != len(command_rows):
        collector.add("DUPLICATE", "commands.command_id", "command IDs are duplicated or invalid")
    successful_manifest_indexes: list[int] = []
    for row in command_rows:
        repair_of = row.get("repair_of")
        if row.get("receipt_status") == "PASS" and isinstance(row.get("manifest_command_index"), int):
            successful_manifest_indexes.append(row["manifest_command_index"])
        if repair_of is not None:
            target = rows_by_id.get(repair_of)
            if target is None or not isinstance(row.get("command_id"), int) or repair_of >= row["command_id"]:
                collector.add("REPAIR_GRAPH", f"commands[{row.get('command_id')}].repair_of", "repair target must be one earlier command")
            elif target.get("receipt_status") != "FAIL" or target.get("manifest_command_index") != row.get("manifest_command_index"):
                collector.add("REPAIR_GRAPH", f"commands[{row.get('command_id')}].repair_of", "repair target must be a failed execution of the same declaration")
    if require_complete:
        for command_index in range(len(declarations)):
            count = successful_manifest_indexes.count(command_index)
            if count == 0:
                collector.add("MISSING_COMMAND", f"commands[{command_index}]", "declared command has no successful execution record")
            elif count > 1:
                collector.add("DUPLICATE_COMMAND_COVERAGE", f"commands[{command_index}]", "declared command has multiple successful executions")

    index = _load_control(attempt_dir / "COMMAND_INDEX.json", collector, "COMMAND_INDEX.json")
    ledger = _load_control(attempt_dir / "COMMAND_LEDGER.json", collector, "COMMAND_LEDGER.json")
    if isinstance(index, dict) and index.get("command_count") != len(command_rows):
        collector.add("COMMAND_INDEX", "COMMAND_INDEX.command_count", "count differs from command records")
    if isinstance(index, dict):
        _exact_keys(index, INDEX_FIELDS, collector, "COMMAND_INDEX")
        if index.get("schema_version") != "evaluation-command-index.v1" or index.get("attempt_id") != manifest.get("attempt_id"):
            collector.add("COMMAND_INDEX", "COMMAND_INDEX", "schema or attempt identity differs")
        expected_terminal = (attempt_dir / TERMINAL_NAME).is_file()
        if index.get("terminal") is not expected_terminal:
            collector.add("COMMAND_INDEX", "COMMAND_INDEX.terminal", f"expected {expected_terminal}")
        expected_index_rows = []
        for row, directory in zip(command_rows, command_directories):
            metadata_path = directory / "metadata.json"
            if metadata_path.is_file():
                expected_index_rows.append({**row, "metadata_sha256": sha256_file(metadata_path)})
        if index.get("commands") != expected_index_rows:
            collector.add("COMMAND_INDEX", "COMMAND_INDEX.commands", "rows differ from command metadata")
    if isinstance(ledger, dict):
        _exact_keys(ledger, LEDGER_FIELDS, collector, "COMMAND_LEDGER")
        if ledger.get("schema_version") != "evaluation-command-ledger.v1" or ledger.get("attempt_id") != manifest.get("attempt_id"):
            collector.add("COMMAND_LEDGER", "COMMAND_LEDGER", "schema or attempt identity differs")
        expected_terminal = (attempt_dir / TERMINAL_NAME).is_file()
        if ledger.get("terminal") is not expected_terminal:
            collector.add("COMMAND_LEDGER", "COMMAND_LEDGER.terminal", f"expected {expected_terminal}")
        unresolved = ledger.get("unresolved_commands")
        if require_complete and unresolved:
            collector.add("UNRESOLVED_COMMAND", "COMMAND_LEDGER.unresolved_commands", "failed commands remain unresolved")
        if ledger.get("command_count") != len(command_rows):
            collector.add("COMMAND_LEDGER", "COMMAND_LEDGER.command_count", "count differs from command records")
        expected_ledger_rows = [{
            "command_id": row.get("command_id"),
            "label": row.get("label"),
            "expected_exit": row.get("expected_exit"),
            "raw_exit": row.get("raw_exit"),
            "semantic_exit": row.get("semantic_exit"),
            "receipt_status": row.get("receipt_status"),
            "repair_of": row.get("repair_of"),
        } for row in command_rows]
        if ledger.get("commands") != expected_ledger_rows:
            collector.add("COMMAND_LEDGER", "COMMAND_LEDGER.commands", "rows differ from command metadata")
        unexpected = [row["command_id"] for row in expected_ledger_rows if isinstance(row["command_id"], int) and row["receipt_status"] != "PASS"]
        repaired = sorted({row["repair_of"] for row in expected_ledger_rows if isinstance(row["repair_of"], int) and row["receipt_status"] == "PASS"})
        expected_sets = {
            "unexpected_commands": unexpected,
            "repaired_commands": repaired,
            "unresolved_commands": sorted(set(unexpected) - set(repaired)),
        }
        for field, expected_value in expected_sets.items():
            if ledger.get(field) != expected_value:
                collector.add("COMMAND_LEDGER", f"COMMAND_LEDGER.{field}", "derived set differs from command metadata")
    index_path = attempt_dir / "COMMAND_INDEX.json"
    ledger_path = attempt_dir / "COMMAND_LEDGER.json"
    receipt_coverage: list[int] = []
    prior_receipt_end: dt.datetime | None = None
    for receipt_index, path in enumerate(receipt_files, start=1):
        receipt = _load_control(path, collector, path.relative_to(attempt_dir).as_posix())
        if isinstance(receipt, dict):
            receipt_location = path.relative_to(attempt_dir).as_posix()
            _exact_keys(receipt, RUNNER_RECEIPT_FIELDS, collector, receipt_location)
            if receipt.get("schema_version") != "evaluation-runner-receipt.v1":
                collector.add("SCHEMA_VERSION", f"{receipt_location}.schema_version", "invalid runner receipt schema")
            if receipt.get("attempt_id") != manifest.get("attempt_id") or receipt.get("invocation_id") != receipt_index:
                collector.add("RUNNER_RECEIPT_ID", receipt_location, "attempt/invocation identity differs from filename")
            mode = receipt.get("mode")
            result = receipt.get("result")
            command_ids = receipt.get("command_ids")
            if mode not in {"run", "seal"}:
                collector.add("ENUM", f"{receipt_location}.mode", "expected run or seal")
            if result not in {"PASS", "FAIL"}:
                collector.add("ENUM", f"{receipt_location}.result", "expected PASS or FAIL")
            if (result == "PASS" and receipt.get("failure") is not None) or (result == "FAIL" and not isinstance(receipt.get("failure"), str)):
                collector.add("RUNNER_RECEIPT_RESULT", f"{receipt_location}.failure", "failure must be null for PASS and a string for FAIL")
            if not isinstance(command_ids, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in command_ids):
                collector.add("RUNNER_RECEIPT_COVERAGE", f"{receipt_location}.command_ids", "expected integer command ID list")
            elif mode == "seal" and command_ids:
                collector.add("RUNNER_RECEIPT_COVERAGE", f"{receipt_location}.command_ids", "seal receipt must have no child command")
            elif mode == "run":
                if len(command_ids) != 1:
                    collector.add("RUNNER_RECEIPT_COVERAGE", f"{receipt_location}.command_ids", "run receipt must cover exactly one child command")
                receipt_coverage.extend(command_ids)
                child = rows_by_id.get(command_ids[0]) if len(command_ids) == 1 else None
                if child is None:
                    collector.add("RUNNER_RECEIPT_COVERAGE", receipt_location, "child command is absent")
                elif result != child.get("receipt_status"):
                    collector.add("RUNNER_RECEIPT_RESULT", receipt_location, "result differs from child launcher result")
            if isinstance(state, dict) and receipt.get("identity_proof") != state.get("identity_proof"):
                collector.add("IDENTITY_BINDING", f"{receipt_location}.identity_proof", "does not bind TOOL_STATE identity proof")
            start = _parse_time(receipt.get("started_at"), collector, f"{receipt_location}.started_at")
            end = _parse_time(receipt.get("ended_at"), collector, f"{receipt_location}.ended_at")
            if start is not None and end is not None and start > end:
                collector.add("CHRONOLOGY", receipt_location, "receipt ends before it starts")
            if prior_receipt_end is not None and start is not None and start < prior_receipt_end:
                collector.add("CHRONOLOGY", receipt_location, "receipt overlaps the preceding invocation")
            if end is not None:
                prior_receipt_end = end
            if index_path.is_file() and receipt.get("command_index_sha256") != sha256_file(index_path):
                # Earlier receipts intentionally bind their invocation snapshot.
                if path == receipt_files[-1]:
                    collector.add("RUNNER_RECEIPT_HASH", path.name, "latest receipt does not bind command index")
            if ledger_path.is_file() and receipt.get("command_ledger_sha256") != sha256_file(ledger_path):
                if path == receipt_files[-1]:
                    collector.add("RUNNER_RECEIPT_HASH", path.name, "latest receipt does not bind command ledger")
    expected_coverage = sorted(rows_by_id)
    if sorted(receipt_coverage) != expected_coverage or len(receipt_coverage) != len(set(receipt_coverage)):
        collector.add("RUNNER_RECEIPT_COVERAGE", "runner_receipts", "child command coverage is missing, extra, or duplicated")
    if (attempt_dir / TERMINAL_NAME).is_file():
        if not receipt_files:
            collector.add("TERMINAL_RECEIPT", "runner_receipts", "terminal PASS receipt is missing")
        else:
            latest = _load_control(receipt_files[-1], collector, receipt_files[-1].relative_to(attempt_dir).as_posix())
            if not isinstance(latest, dict) or latest.get("mode") != "seal" or latest.get("result") != "PASS" or latest.get("command_ids") != []:
                collector.add("TERMINAL_RECEIPT", receipt_files[-1].name, "latest receipt must be an exact seal PASS with no child")

    for result_path in required_results:
        if not isinstance(result_path, str):
            continue
        confined_result = _confined_reference(attempt_dir, result_path, collector, f"required_results:{result_path}")
        result = _load_control(confined_result, collector, result_path) if confined_result is not None else None
        if isinstance(result, dict) and result.get("status") != "PASS":
            collector.add("RESULT_STATUS", result_path, "required result is not PASS")

    for declaration in external_declarations:
        if not isinstance(declaration, dict):
            continue
        try:
            spec = ExternalInventorySpec.from_manifest(declaration)
        except (KeyError, TypeError, ValueError) as error:
            collector.add("EXTERNAL_SCHEMA", declaration.get("name", "?"), str(error))
            continue
        before = _load_control(attempt_dir / "external" / f"{spec.name}.pre.json", collector, f"external/{spec.name}.pre.json")
        after = _load_control(attempt_dir / "external" / f"{spec.name}.post.json", collector, f"external/{spec.name}.post.json")
        comparison = _load_control(attempt_dir / "external" / f"{spec.name}.comparison.json", collector, f"external/{spec.name}.comparison.json")
        if isinstance(before, dict) and isinstance(after, dict):
            try:
                expected_comparison = compare_external(spec, before, after)
            except (KeyError, TypeError, ValueError) as error:
                collector.add("EXTERNAL_INVENTORY", spec.name, f"malformed inventory: {error}")
                continue
            if comparison != expected_comparison:
                collector.add("EXTERNAL_COMPARISON", spec.name, "comparison is absent or differs from inventories")
            if expected_comparison["status"] != "PASS":
                collector.add("EXTERNAL_CHANGE", spec.name, "external changes do not match declaration")
    return collector.findings


def validate_sealed(attempt_dir: Path) -> dict[str, Any]:
    """Validate a SEALED packet offline; never executes files from the packet."""

    attempt_dir = attempt_dir.resolve()
    collector = FindingCollector()
    terminal_path = attempt_dir / TERMINAL_NAME
    terminal = _load_control(terminal_path, collector, TERMINAL_NAME)
    if not isinstance(terminal, dict):
        return {"schema_version": "evaluation-inspection.v1", "status": "FAIL", "attempt_dir": str(attempt_dir), "findings": collector.as_dicts()}
    missing = TERMINAL_FIELDS - set(terminal)
    unknown = set(terminal) - TERMINAL_FIELDS
    for field in sorted(missing):
        collector.add("MISSING_FIELD", f"TERMINAL.{field}", "required field is missing")
    for field in sorted(unknown):
        collector.add("UNKNOWN_FIELD", f"TERMINAL.{field}", "field is not defined")
    terminal_schema = terminal.get("schema_version")
    if terminal_schema not in {LEGACY_TERMINAL_SCHEMA_VERSION, TERMINAL_SCHEMA_VERSION}:
        collector.add(
            "SCHEMA_VERSION",
            "TERMINAL.schema_version",
            f"expected {LEGACY_TERMINAL_SCHEMA_VERSION} or {TERMINAL_SCHEMA_VERSION}",
        )
    elif (
        terminal_schema == LEGACY_TERMINAL_SCHEMA_VERSION
        and terminal.get("tooling_commit") not in LEGACY_TERMINAL_TOOLING_COMMITS
    ):
        collector.add(
            "SCHEMA_VERSION",
            "TERMINAL.schema_version",
            "v1 inventory is restricted to the retained D0002-D0005 tooling commits",
        )
    if terminal.get("lifecycle") != "SEALED":
        collector.add("LIFECYCLE", "TERMINAL.lifecycle", "must be SEALED")
    if terminal.get("self_exclusion") != [TERMINAL_NAME]:
        collector.add("SELF_EXCLUSION", "TERMINAL.self_exclusion", "terminal manifest must be the sole self-exclusion")
    _parse_time(terminal.get("created_at"), collector, "TERMINAL.created_at")
    if not isinstance(terminal.get("max_preseal_mtime_ns"), int) or isinstance(terminal.get("max_preseal_mtime_ns"), bool) or terminal.get("max_preseal_mtime_ns", -1) < 0:
        collector.add("TYPE", "TERMINAL.max_preseal_mtime_ns", "expected nonnegative integer")
    files = terminal.get("files")
    listed: dict[str, dict[str, Any]] = {}
    if not isinstance(files, list):
        collector.add("TYPE", "TERMINAL.files", "expected ordered path inventory")
    else:
        prior = ""
        for index, row in enumerate(files):
            location = f"TERMINAL.files[{index}]"
            expected_file_fields = (
                LEGACY_FILE_FIELDS
                if terminal_schema == LEGACY_TERMINAL_SCHEMA_VERSION
                else FILE_FIELDS
            )
            if not isinstance(row, dict) or set(row) != expected_file_fields:
                collector.add("FILE_SCHEMA", location, "file row fields are not exact")
                continue
            relative = row.get("path")
            if not isinstance(relative, str) or relative == TERMINAL_NAME or relative.startswith("/") or ".." in Path(relative).parts:
                collector.add("PATH", location, "file path is invalid or self-excluded")
                continue
            if relative <= prior:
                collector.add("FILE_ORDER", location, "inventory paths are not strictly sorted")
            prior = relative
            if relative in listed:
                collector.add("DUPLICATE", location, "duplicate file path")
            if terminal_schema == TERMINAL_SCHEMA_VERSION:
                if row.get("type") != "file":
                    collector.add("FILE_TYPE", location, "inventoried entry must be a regular file")
                mode = row.get("mode")
                if not isinstance(mode, int) or isinstance(mode, bool) or not 0 <= mode <= 0o7777:
                    collector.add("FILE_MODE", location, "mode must be an integer permission mask")
            listed[relative] = row
    actual: dict[str, Path] = {}
    for path in attempt_dir.rglob("*") if attempt_dir.exists() else []:
        if path.is_symlink():
            collector.add("ATTEMPT_SYMLINK", path.relative_to(attempt_dir).as_posix(), "sealed packets may contain only regular files/directories")
        elif path.is_file() and path != terminal_path:
            actual[path.relative_to(attempt_dir).as_posix()] = path
    for relative in sorted(set(actual) - set(listed)):
        collector.add("UNLISTED_FILE", relative, "file is absent from terminal inventory")
    for relative in sorted(set(listed) - set(actual)):
        collector.add("MISSING_FILE", relative, "inventoried file is absent")
    for relative in sorted(set(actual) & set(listed)):
        row = listed[relative]
        if terminal_schema == TERMINAL_SCHEMA_VERSION:
            if row.get("type") != "file":
                collector.add("FILE_TYPE", relative, "type differs from regular file")
            if row.get("mode") != (actual[relative].stat().st_mode & 0o7777):
                collector.add("FILE_MODE", relative, "mode differs from terminal inventory")
        if actual[relative].stat().st_size != row.get("bytes"):
            collector.add("FILE_SIZE", relative, "size differs from terminal inventory")
        if sha256_file(actual[relative]) != row.get("sha256"):
            collector.add("FILE_HASH", relative, "content differs from terminal inventory")
    bindings = terminal.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != BINDING_FIELDS:
        collector.add("BINDING_SCHEMA", "TERMINAL.bindings", "binding categories are not exact")
    else:
        bound: set[str] = set()
        for category in sorted(BINDING_FIELDS):
            paths = bindings[category]
            if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
                collector.add("BINDING_TYPE", f"TERMINAL.bindings.{category}", "expected path list")
                continue
            if paths != sorted(set(paths)):
                collector.add("BINDING_ORDER", f"TERMINAL.bindings.{category}", "paths must be unique and sorted")
            for relative in paths:
                if relative not in listed:
                    collector.add("BINDING_TARGET", f"{category}:{relative}", "bound path is absent from inventory")
                if relative in bound:
                    collector.add("DUPLICATE_BINDING", f"{category}:{relative}", "file appears in more than one binding category")
                bound.add(relative)
        if set(listed) != bound:
            for relative in sorted(set(listed) - bound):
                collector.add("UNBOUND_FILE", relative, "inventoried file has no binding category")
    if terminal_path.is_file():
        terminal_mtime = terminal_path.stat().st_mtime_ns
        maximum_preseal = terminal.get("max_preseal_mtime_ns")
        actual_maximum_preseal = max((path.stat().st_mtime_ns for path in actual.values()), default=0)
        if maximum_preseal != actual_maximum_preseal:
            collector.add("MAX_PRESEAL_MTIME", "TERMINAL.max_preseal_mtime_ns", "must equal the actual maximum inventoried file mtime")
        if isinstance(maximum_preseal, int) and not isinstance(maximum_preseal, bool) and terminal_mtime < maximum_preseal:
            collector.add("LAST_WRITE", TERMINAL_NAME, "terminal manifest predates a sealed file")
        for relative, path in actual.items():
            if path.stat().st_mtime_ns > terminal_mtime:
                collector.add("POST_SEAL_MUTATION", relative, "file timestamp is newer than terminal manifest")
    collector.extend(
        collect_draft_findings(
            attempt_dir,
            tooling_commit=terminal.get("tooling_commit", ""),
            product_commit=terminal.get("product_commit", ""),
            require_complete=True,
            verify_live_inputs=False,
        )
    )
    manifest = _load_control(attempt_dir / MANIFEST_NAME, collector, MANIFEST_NAME)
    if isinstance(manifest, dict):
        if terminal.get("attempt_id") != manifest.get("attempt_id"):
            collector.add("ATTEMPT_ID", "TERMINAL.attempt_id", "does not match manifest")
        if terminal.get("product_commit") != manifest.get("product_commit"):
            collector.add("PRODUCT_PIN", "TERMINAL.product_commit", "does not match manifest")
        if terminal.get("tooling_commit") != manifest.get("tooling_commit"):
            collector.add("TOOLING_PIN", "TERMINAL.tooling_commit", "does not match manifest")
    state = _load_control(attempt_dir / "TOOL_STATE.json", collector, "TOOL_STATE.json")
    if isinstance(state, dict) and terminal.get("identity_proof") != state.get("identity_proof"):
        collector.add("IDENTITY_BINDING", "TERMINAL.identity_proof", "does not match TOOL_STATE")
    return {
        "schema_version": "evaluation-inspection.v1",
        "status": "PASS" if collector.ok else "FAIL",
        "attempt_dir": str(attempt_dir),
        "terminal_manifest_sha256": sha256_file(terminal_path) if terminal_path.is_file() else None,
        "findings": collector.as_dicts(),
    }
