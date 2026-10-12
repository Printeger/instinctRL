"""Stable explicit-argv runner and DRAFT lifecycle implementation."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Callable

from .canonical import CanonicalJSONError, atomic_canonical_create, canonical_dumps, canonical_load
from .external_inventory import ExternalInventorySpec, scan_external
from .findings import Finding, FindingCollector, RuntimeFailure, RuntimeFailureClass
from .identity import establish_identity
from .receipt import LauncherReceipt, build_launcher_receipt
from .schemas import (
    LAUNCH_PROFILE,
    executable_source_paths,
    path_within,
    validate_input_hashes,
    validate_live_launch_profile,
    validate_manifest,
)


MANIFEST_NAME = "ATTEMPT_MANIFEST.json"
INPUT_HASHES_NAME = "INPUT_HASHES.json"
TERMINAL_NAME = "TERMINAL_MANIFEST.json"
TOOL_OUTPUT_ROOTS = {"commands", "runner_receipts", "launcher_receipts", "failed_seal_receipts", "results", "external"}
TOOL_OUTPUT_FILES = {"COMMAND_INDEX.json", "COMMAND_LEDGER.json", "PREFLIGHT.json", "TOOL_STATE.json"}
INERT_RUNTIME_ROOTS = (
    PurePosixPath("results/runtime_support/tmp"),
    PurePosixPath("results/runtime_support/cache"),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_replace(path: Path, value: Any) -> bytes:
    data = canonical_dumps(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary file: {temporary}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return data


def _attempt_files(attempt_dir: Path) -> list[str]:
    return sorted(path.relative_to(attempt_dir).as_posix() for path in attempt_dir.rglob("*") if path.is_file() or path.is_symlink())


def _is_generated(path: str) -> bool:
    first = path.split("/", 1)[0]
    return path in TOOL_OUTPUT_FILES or first in TOOL_OUTPUT_ROOTS or path == TERMINAL_NAME


def _looks_executable(relative: str, path: Path) -> bool:
    forbidden_names = {"runner", "validator", "launcher", "comparator", "inventory", "sealer"}
    suffixes = {".py", ".pyc", ".pyo", ".sh", ".bash", ".zsh", ".fish", ".exe"}
    return (
        path.suffix.lower() in suffixes
        or path.name.lower() in forbidden_names
        or (path.exists() and not path.is_symlink() and bool(path.stat().st_mode & 0o111))
    )


def _runtime_started_is_proven(attempt_dir: Path) -> bool:
    """Accept runtime-artifact trust only from an intact persisted tool state."""

    state_path = attempt_dir / "TOOL_STATE.json"
    try:
        state = canonical_load(state_path)
    except (OSError, CanonicalJSONError):
        return False
    if not isinstance(state, dict) or state.get("runtime_started") is not True:
        return False
    if state.get("lifecycle") not in {"DRAFT", "SEALED"}:
        return False
    try:
        return (
            state.get("manifest_sha256") == sha256_file(attempt_dir / MANIFEST_NAME)
            and state.get("input_hashes_sha256") == sha256_file(attempt_dir / INPUT_HASHES_NAME)
        )
    except OSError:
        return False


def _is_inert_runtime_artifact(
    relative: str,
    path: Path,
    *,
    allow_tool_outputs: bool,
    runtime_started: bool,
) -> bool:
    """Recognize an inert output by lifecycle, exact root and regular-file type."""

    candidate = PurePosixPath(relative)
    return (
        allow_tool_outputs
        and runtime_started
        and path.is_file()
        and not path.is_symlink()
        and any(root in candidate.parents for root in INERT_RUNTIME_ROOTS)
    )


def load_submission(
    attempt_dir: Path,
    *,
    allow_tool_outputs: bool = False,
    verify_live_launch: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, tuple[Finding, ...]]:
    """Collect every cheap submission defect without mutating the attempt."""

    attempt_dir = attempt_dir.resolve()
    collector = FindingCollector()
    manifest: dict[str, Any] | None = None
    input_hashes: dict[str, Any] | None = None
    for name in (MANIFEST_NAME, INPUT_HASHES_NAME):
        path = attempt_dir / name
        if not path.is_file():
            collector.add("MISSING_SUBMISSION", name, "required Builder-authored file is missing")
            continue
        try:
            value = canonical_load(path)
        except CanonicalJSONError as error:
            collector.add("CANONICAL_JSON", name, str(error))
            continue
        if name == MANIFEST_NAME:
            manifest = value
        else:
            input_hashes = value
    files = _attempt_files(attempt_dir) if attempt_dir.exists() else []
    runtime_started = _runtime_started_is_proven(attempt_dir) if allow_tool_outputs else False
    for relative in files:
        path = attempt_dir / relative
        if relative not in {MANIFEST_NAME, INPUT_HASHES_NAME}:
            if not allow_tool_outputs or not _is_generated(relative):
                collector.add("EXTRA_SUBMISSION", relative, "attempt contains a non-tool-generated file")
            if path.is_symlink():
                collector.add("ATTEMPT_SYMLINK", relative, "attempt-local symlinks are forbidden")
            if _looks_executable(relative, path) and not _is_inert_runtime_artifact(
                relative,
                path,
                allow_tool_outputs=allow_tool_outputs,
                runtime_started=runtime_started,
            ):
                collector.add("ATTEMPT_LOCAL_EXECUTABLE", relative, "attempt-local executable/source is forbidden")
    if manifest is not None:
        collector.extend(validate_manifest(manifest, attempt_dir))
        if not isinstance(manifest, dict):
            manifest = None
    if input_hashes is not None:
        collector.extend(validate_input_hashes(input_hashes, manifest))
        if not isinstance(input_hashes, dict):
            input_hashes = None
    if manifest is not None and input_hashes is not None:
        actual_input_hash = sha256_file(attempt_dir / INPUT_HASHES_NAME)
        if manifest.get("input_hashes_sha256") != actual_input_hash:
            collector.add("INPUT_HASH_DOCUMENT_HASH", INPUT_HASHES_NAME, "manifest hash does not match canonical input-hash bytes")
        if manifest.get("input_hashes_path") != INPUT_HASHES_NAME:
            collector.add("INPUT_HASH_DOCUMENT_PATH", "$.input_hashes_path", "must name INPUT_HASHES.json")
        if verify_live_launch:
            collector.extend(validate_live_launch_profile(manifest, input_hashes, attempt_dir, Path(sys.executable)))
    return manifest, input_hashes, collector.findings


def verify_pins_and_inputs(
    manifest: dict[str, Any],
    input_hashes: dict[str, Any],
    *,
    tooling_commit: str,
    product_commit: str,
) -> tuple[Finding, ...]:
    collector = FindingCollector()
    if manifest.get("tooling_commit") != tooling_commit:
        collector.add("TOOLING_PIN", "$.tooling_commit", "manifest does not pin the executing tooling commit")
    if manifest.get("product_commit") != product_commit:
        collector.add("PRODUCT_PIN", "$.product_commit", "manifest does not pin the ordered product commit")
    if input_hashes.get("tooling_commit") != tooling_commit:
        collector.add("TOOLING_PIN", "INPUT_HASHES.tooling_commit", "input document does not pin executing tooling")
    if input_hashes.get("product_commit") != product_commit:
        collector.add("PRODUCT_PIN", "INPUT_HASHES.product_commit", "input document does not pin ordered product")
    for index, row in enumerate(input_hashes.get("files", [])):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            continue
        path = Path(row["path"])
        location = f"INPUT_HASHES.files[{index}]"
        if not path.is_file():
            collector.add("MISSING_INPUT", location, f"input does not exist: {path}")
            continue
        if path.is_symlink():
            collector.add("INPUT_SYMLINK", location, "input must be a regular non-symlink file")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if row.get("bytes") != actual_size:
            collector.add("INPUT_SIZE", location, f"expected {row.get('bytes')}, observed {actual_size}")
        if row.get("sha256") != actual_hash:
            collector.add("INPUT_HASH", location, f"expected {row.get('sha256')}, observed {actual_hash}")
    return collector.findings


def build_launch_profile_proof(
    manifest: dict[str, Any],
    input_hashes: dict[str, Any],
    executing_interpreter: Path,
) -> dict[str, Any]:
    """Persist the live launch decision using only packet-verifiable values."""

    pinned = {
        Path(os.path.abspath(row["path"])): {
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in input_hashes.get("files", [])
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    commands: list[dict[str, Any]] = []
    for command_index, command in enumerate(manifest.get("commands", [])):
        row: dict[str, Any] = {"command_index": command_index}
        for field in ("argv", "cleanup_argv"):
            argv = command.get(field)
            if argv is None:
                row[field] = None
                continue
            source_index, source, _ = executable_source_paths(argv, command.get("cwd"))[0]
            executable_path = Path(os.path.abspath(argv[0]))
            source_path = Path(os.path.abspath(source))
            row[field] = {
                "executable": str(executable_path),
                "executable_pin": pinned[executable_path],
                "source": str(source_path),
                "source_index": source_index,
                "source_pin": pinned[source_path],
            }
        commands.append(row)
    return {
        "schema_version": "evaluation-launch-profile-proof.v1",
        "profile": LAUNCH_PROFILE,
        "executing_interpreter": str(executing_interpreter.resolve(strict=True)),
        "commands": commands,
        "status": "PASS",
    }


def _versioned_identity_proof(
    manifest: dict[str, Any],
    input_hashes: dict[str, Any],
    proof: dict[str, Any],
) -> dict[str, Any]:
    result = dict(proof)
    result["schema_version"] = "evaluation-identity-proof.v2"
    result["launch_profile"] = build_launch_profile_proof(manifest, input_hashes, Path(sys.executable))
    return result


def preflight(
    attempt_dir: Path,
    *,
    tooling_commit: str,
    product_commit: str,
) -> dict[str, Any]:
    manifest, input_hashes, findings = load_submission(
        attempt_dir,
        allow_tool_outputs=False,
        verify_live_launch=True,
    )
    collector = FindingCollector()
    collector.extend(findings)
    identity_proof: dict[str, Any] | None = None
    if manifest is not None and input_hashes is not None:
        collector.extend(
            verify_pins_and_inputs(
                manifest,
                input_hashes,
                tooling_commit=tooling_commit,
                product_commit=product_commit,
            )
        )
        identity_proof, identity_findings = establish_identity(
            manifest,
            tooling_commit=tooling_commit,
            product_commit=product_commit,
        )
        collector.extend(identity_findings)
        if not validate_live_launch_profile(manifest, input_hashes, attempt_dir, Path(sys.executable)):
            identity_proof = _versioned_identity_proof(manifest, input_hashes, identity_proof)
    if (attempt_dir / TERMINAL_NAME).exists():
        collector.add("ALREADY_SEALED", TERMINAL_NAME, "SEALED namespace cannot return to DRAFT")
    return {
        "schema_version": "evaluation-preflight.v1",
        "status": "PASS" if collector.ok else "FAIL",
        "attempt_dir": str(attempt_dir.resolve()),
        "tooling_commit": tooling_commit,
        "product_commit": product_commit,
        "identity_proof": identity_proof,
        "findings": collector.as_dicts(),
    }


class EvaluationRunner:
    def __init__(
        self,
        attempt_dir: Path,
        *,
        tooling_commit: str,
        product_commit: str,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.attempt_dir = attempt_dir.resolve()
        self.tooling_commit = tooling_commit
        self.product_commit = product_commit
        self.manifest: dict[str, Any] | None = None
        self.input_hashes: dict[str, Any] | None = None
        self.identity_proof: dict[str, Any] | None = None
        self.fault_injector = fault_injector

    def _fault(self, boundary: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector(boundary)

    def prepare(self) -> dict[str, Any]:
        if (self.attempt_dir / TERMINAL_NAME).exists():
            raise RuntimeError("SEALED namespace is immutable")
        result = preflight(
            self.attempt_dir,
            tooling_commit=self.tooling_commit,
            product_commit=self.product_commit,
        )
        if result["status"] != "PASS":
            return result
        self.manifest = canonical_load(self.attempt_dir / MANIFEST_NAME)
        self.input_hashes = canonical_load(self.attempt_dir / INPUT_HASHES_NAME)
        self.identity_proof = result["identity_proof"]
        write_canonical_replace(self.attempt_dir / "PREFLIGHT.json", result)
        for name in TOOL_OUTPUT_ROOTS:
            (self.attempt_dir / name).mkdir(parents=True, exist_ok=True)
        for external in self.manifest["external_inventories"]:
            spec = ExternalInventorySpec.from_manifest(external)
            inventory = scan_external(spec)
            write_canonical_replace(self.attempt_dir / "external" / f"{spec.name}.pre.json", inventory)
        state = {
            "schema_version": "evaluation-tool-state.v1",
            "attempt_id": self.manifest["attempt_id"],
            "lifecycle": "DRAFT",
            "manifest_sha256": sha256_file(self.attempt_dir / MANIFEST_NAME),
            "input_hashes_sha256": sha256_file(self.attempt_dir / INPUT_HASHES_NAME),
            "product_commit": self.product_commit,
            "tooling_commit": self.tooling_commit,
            "runtime_started": False,
            "identity_proof": self.identity_proof,
        }
        write_canonical_replace(self.attempt_dir / "TOOL_STATE.json", state)
        self._rebuild_aggregates(terminal=False)
        return result

    def _ensure_loaded(self) -> None:
        if self.manifest is None:
            self.manifest = canonical_load(self.attempt_dir / MANIFEST_NAME)
        if self.input_hashes is None:
            self.input_hashes = canonical_load(self.attempt_dir / INPUT_HASHES_NAME)

    def _assert_frozen_identity(self) -> None:
        state = canonical_load(self.attempt_dir / "TOOL_STATE.json")
        expected = {
            "manifest_sha256": sha256_file(self.attempt_dir / MANIFEST_NAME),
            "input_hashes_sha256": sha256_file(self.attempt_dir / INPUT_HASHES_NAME),
            "product_commit": self.product_commit,
            "tooling_commit": self.tooling_commit,
        }
        mismatches = [field for field, observed in expected.items() if state.get(field) != observed]
        live_findings = verify_pins_and_inputs(
            self.manifest or {},
            self.input_hashes or {},
            tooling_commit=self.tooling_commit,
            product_commit=self.product_commit,
        )
        identity_proof, identity_findings = establish_identity(
            self.manifest or {},
            tooling_commit=self.tooling_commit,
            product_commit=self.product_commit,
        )
        if not validate_live_launch_profile(
            self.manifest or {}, self.input_hashes or {}, self.attempt_dir, Path(sys.executable)
        ):
            identity_proof = _versioned_identity_proof(
                self.manifest or {}, self.input_hashes or {}, identity_proof
            )
        if state.get("identity_proof") != identity_proof:
            mismatches.append("identity_proof")
        if mismatches or live_findings or identity_findings:
            details = sorted(set(mismatches + [finding.code for finding in (*live_findings, *identity_findings)]))
            raise RuntimeFailure(
                RuntimeFailureClass.FROZEN_IDENTITY_LOSS,
                f"persisted/live identity changed: {','.join(details)}",
            )

    def _assert_execution_confinement(self, spec: dict[str, Any], field: str) -> None:
        _, _, findings = load_submission(self.attempt_dir, allow_tool_outputs=True)
        dangerous = [finding for finding in findings if finding.code in {
            "ATTEMPT_LOCAL_EXECUTABLE", "EXTRA_SUBMISSION", "ATTEMPT_SYMLINK",
        }]
        pinned = {}
        for row in (self.input_hashes or {}).get("files", []):
            if isinstance(row, dict) and isinstance(row.get("path"), str):
                pinned[Path(row["path"]).resolve(strict=False)] = row
        resolved_attempt = self.attempt_dir.resolve()
        argv = spec.get(field)
        if not isinstance(argv, list) or not argv:
            return
        authored_executable = Path(argv[0])
        executable = authored_executable.resolve(strict=False)
        if not authored_executable.exists():
            raise RuntimeFailure(RuntimeFailureClass.PROCESS_CRASH, f"missing executable blocked before launch: {authored_executable}")
        if not authored_executable.is_absolute():
            dangerous.append(Finding("EXECUTABLE_PATH", f"{field}[0]", "executable must be absolute"))
        if authored_executable.is_symlink():
            dangerous.append(Finding("EXECUTABLE_SYMLINK", f"{field}[0]", "executable path must not be a symlink"))
        current_interpreter = Path(sys.executable).resolve(strict=False)
        if executable != current_interpreter:
            dangerous.append(Finding(
                "LAUNCH_PROFILE",
                f"{field}[0]",
                f"executable is not the current certified Python interpreter: {executable}",
            ))
        if executable == resolved_attempt or resolved_attempt in executable.parents:
            dangerous.append(Finding("ATTEMPT_LOCAL_EXECUTABLE", field, "attempt-local executable is forbidden"))
        executable_row = pinned.get(executable)
        if executable_row is None:
            dangerous.append(Finding("UNPINNED_EXECUTABLE", field, f"executable is not pinned: {executable}"))
        elif not executable.is_file() or executable.is_symlink():
            dangerous.append(Finding("EXECUTABLE_FILE", field, f"executable is not a regular non-symlink file: {executable}"))
        else:
            if executable.stat().st_size != executable_row.get("bytes"):
                dangerous.append(Finding("EXECUTABLE_SIZE", field, f"executable size changed: {executable}"))
            if sha256_file(executable) != executable_row.get("sha256"):
                dangerous.append(Finding("EXECUTABLE_HASH", field, f"executable hash changed: {executable}"))
        sources = executable_source_paths(argv, spec.get("cwd"))
        if not sources:
            dangerous.append(Finding("LAUNCH_PROFILE", field, "DIRECT_PINNED_PYTHON_SOURCE_V1 requires one identifiable source"))
        for index, source, was_relative in sources:
            authored_source = Path(argv[index])
            resolved_source = source.resolve(strict=False)
            if was_relative:
                dangerous.append(Finding("RELATIVE_SOURCE", f"{field}[{index}]", "interpreter source must be absolute"))
            if authored_source.is_symlink():
                dangerous.append(Finding("SOURCE_SYMLINK", f"{field}[{index}]", "source path must not be a symlink"))
            source_row = pinned.get(resolved_source)
            if source_row is None:
                dangerous.append(Finding("UNPINNED_SOURCE", f"{field}[{index}]", f"source is not pinned: {resolved_source}"))
            elif not resolved_source.is_file() or resolved_source.is_symlink():
                dangerous.append(Finding("SOURCE_FILE", f"{field}[{index}]", f"source is not a regular non-symlink file: {resolved_source}"))
            else:
                if resolved_source.stat().st_size != source_row.get("bytes"):
                    dangerous.append(Finding("SOURCE_SIZE", f"{field}[{index}]", f"source size changed: {resolved_source}"))
                if sha256_file(resolved_source) != source_row.get("sha256"):
                    dangerous.append(Finding("SOURCE_HASH", f"{field}[{index}]", f"source hash changed: {resolved_source}"))
        if dangerous:
            message = "; ".join(f"{item.code}@{item.location}" for item in sorted(set(dangerous)))
            raise RuntimeFailure(RuntimeFailureClass.FORBIDDEN_WRITE_OR_ACTION, message)

    def _command_directories(self) -> list[Path]:
        root = self.attempt_dir / "commands"
        directories = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name) if root.exists() else []
        expected = [f"{index:04d}_" for index in range(1, len(directories) + 1)]
        if any(not path.name.startswith(prefix) for path, prefix in zip(directories, expected)):
            raise RuntimeError("non-contiguous command directories")
        return directories

    def _receipt_files(self) -> list[Path]:
        root = self.attempt_dir / "runner_receipts"
        files = sorted(root.glob("*.json")) if root.exists() else []
        if [path.stem for path in files] != [f"{index:04d}" for index in range(1, len(files) + 1)]:
            raise RuntimeError("non-contiguous runner receipts")
        return files

    def _command_rows(self) -> list[dict[str, Any]]:
        rows = []
        for directory in self._command_directories():
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file():
                raise RuntimeError(f"incomplete command directory: {directory}")
            metadata = canonical_load(metadata_path)
            for stream_name in ("stdout", "stderr", "cleanup_stdout", "cleanup_stderr"):
                relative = metadata.get(f"{stream_name}_path")
                expected_hash = metadata.get(f"{stream_name}_sha256")
                expected_size = metadata.get(f"{stream_name}_bytes")
                if relative is None:
                    continue
                stream = self.attempt_dir / relative
                if not stream.is_file() or sha256_file(stream) != expected_hash or stream.stat().st_size != expected_size:
                    raise RuntimeError(f"stream integrity mismatch: {relative}")
            rows.append(metadata)
        return rows

    def _incomplete_directories(self) -> list[Path]:
        incomplete = []
        for directory in self._command_directories():
            pending_transaction = (directory / "transaction.json").is_file()
            complete_transaction = (directory / "transaction_complete.json").is_file()
            if not (directory / "metadata.json").is_file() or (pending_transaction and not complete_transaction):
                incomplete.append(directory)
        return incomplete

    def _recover_interrupted(self) -> int | None:
        """Convert one interrupted persistence transaction into an immutable FAIL row."""

        incomplete = self._incomplete_directories()
        if not incomplete:
            return None
        if len(incomplete) != 1 or incomplete[0] != self._command_directories()[-1]:
            raise RuntimeError("multiple or nonterminal incomplete command transactions")
        directory = incomplete[0]
        command_id = self._command_directories().index(directory) + 1
        transaction_path = directory / "transaction.json"
        transaction: dict[str, Any] = {}
        if transaction_path.is_file():
            try:
                loaded = canonical_load(transaction_path)
                if isinstance(loaded, dict):
                    transaction = loaded
            except CanonicalJSONError:
                pass
        command_index = transaction.get("manifest_command_index", min(command_id - 1, len(self.manifest["commands"]) - 1))
        if not isinstance(command_index, int) or not 0 <= command_index < len(self.manifest["commands"]):
            command_index = 0
        spec = self.manifest["commands"][command_index]
        for name in ("stdout.bin", "stderr.bin"):
            path = directory / name
            if not path.exists():
                path.write_bytes(b"")
        if spec["cleanup_argv"] is not None:
            for name in ("cleanup_stdout.bin", "cleanup_stderr.bin"):
                path = directory / name
                if not path.exists():
                    path.write_bytes(b"")

        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            metadata_path.replace(directory / "interrupted_metadata.json")
        launcher_path = self.attempt_dir / "launcher_receipts" / f"{command_id:04d}.json"
        if launcher_path.exists():
            launcher_path.replace(directory / "interrupted_launcher_receipt.json")
        receipt_files = self._receipt_files()
        if receipt_files:
            try:
                last_receipt = canonical_load(receipt_files[-1])
            except CanonicalJSONError:
                last_receipt = None
            if isinstance(last_receipt, dict) and last_receipt.get("command_ids") == [command_id]:
                receipt_files[-1].replace(directory / "interrupted_runner_receipt.json")
        for aggregate in ("COMMAND_INDEX.json", "COMMAND_LEDGER.json"):
            path = self.attempt_dir / aggregate
            if path.exists():
                path.replace(directory / f"interrupted_{aggregate}")

        recovery_receipt = LauncherReceipt(
            raw_exit=126,
            semantic_exit=1,
            receipt_status="FAIL",
            primary_failure="PROCESS_CRASH:INTERRUPTED_PERSISTENCE",
            cleanup_status="FAIL" if spec["cleanup_argv"] is not None else "NOT_REQUESTED",
            cleanup_exit=None,
            timed_out=False,
        )
        launcher_object = {
            "schema_version": "evaluation-launcher-receipt.v1",
            "attempt_id": self.manifest["attempt_id"],
            "command_id": command_id,
            **recovery_receipt.as_dict(),
        }
        atomic_canonical_create(launcher_path, launcher_object)
        started_at = transaction.get("started_at") if isinstance(transaction.get("started_at"), str) else utc_now()
        stdout = (directory / "stdout.bin").read_bytes()
        stderr = (directory / "stderr.bin").read_bytes()
        cleanup_stdout = (directory / "cleanup_stdout.bin").read_bytes() if spec["cleanup_argv"] is not None else b""
        cleanup_stderr = (directory / "cleanup_stderr.bin").read_bytes() if spec["cleanup_argv"] is not None else b""
        metadata = self._metadata_object(
            spec=spec,
            command_id=command_id,
            command_index=command_index,
            repair_of=None,
            started_at=started_at,
            ended_at=utc_now(),
            raw_exit=126,
            semantic_exit=1,
            receipt=recovery_receipt,
            stdout=stdout,
            stderr=stderr,
            cleanup_stdout=cleanup_stdout,
            cleanup_stderr=cleanup_stderr,
            directory=directory,
            launcher_path=launcher_path,
        )
        atomic_canonical_create(metadata_path, metadata)
        index_bytes, ledger_bytes = self._rebuild_aggregates(terminal=False)
        receipt_id = len(self._receipt_files()) + 1
        state = canonical_load(self.attempt_dir / "TOOL_STATE.json")
        runner_receipt = {
            "schema_version": "evaluation-runner-receipt.v1",
            "attempt_id": self.manifest["attempt_id"],
            "invocation_id": receipt_id,
            "mode": "run",
            "command_ids": [command_id],
            "command_index_sha256": sha256_bytes(index_bytes),
            "command_ledger_sha256": sha256_bytes(ledger_bytes),
            "identity_proof": state["identity_proof"],
            "failure": recovery_receipt.primary_failure,
            "started_at": started_at,
            "ended_at": utc_now(),
            "result": "FAIL",
        }
        atomic_canonical_create(self.attempt_dir / "runner_receipts" / f"{receipt_id:04d}.json", runner_receipt)
        atomic_canonical_create(directory / "transaction_complete.json", {
            "schema_version": "evaluation-command-transaction-complete.v1",
            "command_id": command_id,
            "result": "RECOVERED_FAIL",
        })
        return command_id

    def _metadata_object(
        self,
        *,
        spec: dict[str, Any],
        command_id: int,
        command_index: int,
        repair_of: int | None,
        started_at: str,
        ended_at: str,
        raw_exit: int,
        semantic_exit: int,
        receipt: LauncherReceipt,
        stdout: bytes,
        stderr: bytes,
        cleanup_stdout: bytes,
        cleanup_stderr: bytes,
        directory: Path,
        launcher_path: Path,
    ) -> dict[str, Any]:
        return {
            "schema_version": "evaluation-command-metadata.v1",
            "attempt_id": self.manifest["attempt_id"],
            "command_id": command_id,
            "manifest_command_index": command_index,
            "label": spec["label"],
            "argv": spec["argv"],
            "cwd": spec["cwd"],
            "environment": spec["environment"],
            "started_at": started_at,
            "ended_at": ended_at,
            "expected_exit": spec["expected_exit"],
            "raw_exit": raw_exit,
            "semantic_exit": semantic_exit,
            "receipt_status": receipt.receipt_status,
            "primary_failure": receipt.primary_failure,
            "cleanup_status": receipt.cleanup_status,
            "repair_of": repair_of,
            "stdout_path": (directory / "stdout.bin").relative_to(self.attempt_dir).as_posix(),
            "stdout_bytes": len(stdout),
            "stdout_sha256": sha256_bytes(stdout),
            "stderr_path": (directory / "stderr.bin").relative_to(self.attempt_dir).as_posix(),
            "stderr_bytes": len(stderr),
            "stderr_sha256": sha256_bytes(stderr),
            "cleanup_stdout_path": (directory / "cleanup_stdout.bin").relative_to(self.attempt_dir).as_posix() if spec["cleanup_argv"] is not None else None,
            "cleanup_stdout_bytes": len(cleanup_stdout) if spec["cleanup_argv"] is not None else None,
            "cleanup_stdout_sha256": sha256_bytes(cleanup_stdout) if spec["cleanup_argv"] is not None else None,
            "cleanup_stderr_path": (directory / "cleanup_stderr.bin").relative_to(self.attempt_dir).as_posix() if spec["cleanup_argv"] is not None else None,
            "cleanup_stderr_bytes": len(cleanup_stderr) if spec["cleanup_argv"] is not None else None,
            "cleanup_stderr_sha256": sha256_bytes(cleanup_stderr) if spec["cleanup_argv"] is not None else None,
            "launcher_receipt_path": launcher_path.relative_to(self.attempt_dir).as_posix(),
            "launcher_receipt_sha256": sha256_file(launcher_path),
            "metadata_path": (directory / "metadata.json").relative_to(self.attempt_dir).as_posix(),
        }

    def _record_blocked_command(
        self,
        command_index: int,
        error: RuntimeFailure,
        *,
        repair_of: int | None,
    ) -> tuple[LauncherReceipt, int]:
        spec = self.manifest["commands"][command_index]
        command_id = len(self._command_directories()) + 1
        safe_label = re.sub(r"[^a-z0-9_-]", "_", spec["label"])
        directory = self.attempt_dir / "commands" / f"{command_id:04d}_{safe_label}"
        directory.mkdir(parents=False, exist_ok=False)
        started_at = utc_now()
        atomic_canonical_create(directory / "transaction.json", {
            "schema_version": "evaluation-command-transaction.v1",
            "command_id": command_id,
            "manifest_command_index": command_index,
            "repair_of": repair_of,
            "started_at": started_at,
        })
        stdout = b""
        stderr = (str(error) + "\n").encode("utf-8")
        (directory / "stdout.bin").write_bytes(stdout)
        (directory / "stderr.bin").write_bytes(stderr)
        cleanup_stdout = b""
        cleanup_stderr = b""
        if spec["cleanup_argv"] is not None:
            (directory / "cleanup_stdout.bin").write_bytes(cleanup_stdout)
            (directory / "cleanup_stderr.bin").write_bytes(cleanup_stderr)
        receipt = LauncherReceipt(
            raw_exit=126,
            semantic_exit=1,
            receipt_status="FAIL",
            primary_failure=str(error),
            cleanup_status="NOT_REQUESTED",
            cleanup_exit=None,
            timed_out=False,
        )
        launcher_path = self.attempt_dir / "launcher_receipts" / f"{command_id:04d}.json"
        atomic_canonical_create(launcher_path, {
            "schema_version": "evaluation-launcher-receipt.v1",
            "attempt_id": self.manifest["attempt_id"],
            "command_id": command_id,
            **receipt.as_dict(),
        })
        metadata = self._metadata_object(
            spec=spec,
            command_id=command_id,
            command_index=command_index,
            repair_of=repair_of,
            started_at=started_at,
            ended_at=utc_now(),
            raw_exit=126,
            semantic_exit=1,
            receipt=receipt,
            stdout=stdout,
            stderr=stderr,
            cleanup_stdout=cleanup_stdout,
            cleanup_stderr=cleanup_stderr,
            directory=directory,
            launcher_path=launcher_path,
        )
        atomic_canonical_create(directory / "metadata.json", metadata)
        index_bytes, ledger_bytes = self._rebuild_aggregates(terminal=False)
        state = canonical_load(self.attempt_dir / "TOOL_STATE.json")
        receipt_id = len(self._receipt_files()) + 1
        atomic_canonical_create(self.attempt_dir / "runner_receipts" / f"{receipt_id:04d}.json", {
            "schema_version": "evaluation-runner-receipt.v1",
            "attempt_id": self.manifest["attempt_id"],
            "invocation_id": receipt_id,
            "mode": "run",
            "command_ids": [command_id],
            "command_index_sha256": sha256_bytes(index_bytes),
            "command_ledger_sha256": sha256_bytes(ledger_bytes),
            "identity_proof": state["identity_proof"],
            "failure": str(error),
            "started_at": started_at,
            "ended_at": utc_now(),
            "result": "FAIL",
        })
        atomic_canonical_create(directory / "transaction_complete.json", {
            "schema_version": "evaluation-command-transaction-complete.v1",
            "command_id": command_id,
            "result": "BLOCKED_FAIL",
        })
        return receipt, 1

    def _rebuild_aggregates(self, *, terminal: bool) -> tuple[bytes, bytes]:
        rows = self._command_rows()
        index_rows = []
        ledger_rows = []
        for row in rows:
            metadata_path = self.attempt_dir / row["metadata_path"]
            index_rows.append({**row, "metadata_sha256": sha256_file(metadata_path)})
            ledger_rows.append({
                "command_id": row["command_id"],
                "label": row["label"],
                "expected_exit": row["expected_exit"],
                "raw_exit": row["raw_exit"],
                "semantic_exit": row["semantic_exit"],
                "receipt_status": row["receipt_status"],
                "repair_of": row["repair_of"],
            })
        index = {
            "schema_version": "evaluation-command-index.v1",
            "attempt_id": self.manifest["attempt_id"] if self.manifest else None,
            "terminal": terminal,
            "command_count": len(index_rows),
            "commands": index_rows,
        }
        unexpected = [row["command_id"] for row in ledger_rows if row["receipt_status"] != "PASS"]
        repaired = sorted({row["repair_of"] for row in ledger_rows if row["repair_of"] is not None and row["receipt_status"] == "PASS"})
        ledger = {
            "schema_version": "evaluation-command-ledger.v1",
            "attempt_id": self.manifest["attempt_id"] if self.manifest else None,
            "terminal": terminal,
            "command_count": len(ledger_rows),
            "unexpected_commands": unexpected,
            "repaired_commands": repaired,
            "unresolved_commands": sorted(set(unexpected) - set(repaired)),
            "commands": ledger_rows,
        }
        return (
            write_canonical_replace(self.attempt_dir / "COMMAND_INDEX.json", index),
            write_canonical_replace(self.attempt_dir / "COMMAND_LEDGER.json", ledger),
        )

    def run_command(self, command_index: int, *, repair_of: int | None = None) -> tuple[LauncherReceipt, int]:
        self._ensure_loaded()
        if (self.attempt_dir / TERMINAL_NAME).exists():
            raise RuntimeError("post-seal mutation is forbidden")
        recovered = self._recover_interrupted()
        if recovered is not None:
            raise RuntimeFailure(
                RuntimeFailureClass.PROCESS_CRASH,
                f"recovered interrupted command {recovered}; retry with repair_of={recovered}",
            )
        commands = self.manifest["commands"]
        if not 0 <= command_index < len(commands):
            raise IndexError(command_index)
        spec = commands[command_index]
        try:
            self._assert_frozen_identity()
            self._assert_execution_confinement(spec, "argv")
        except RuntimeFailure as error:
            return self._record_blocked_command(command_index, error, repair_of=repair_of)
        if repair_of is not None:
            prior_rows = self._command_rows()
            targets = [row for row in prior_rows if row.get("command_id") == repair_of]
            if len(targets) != 1:
                raise ValueError("repair_of must identify exactly one prior command")
            target = targets[0]
            if target.get("receipt_status") != "FAIL":
                raise ValueError("repair_of must identify a failed command")
            if target.get("manifest_command_index") != command_index:
                raise ValueError("repair must rerun the same manifest command")
        state_path = self.attempt_dir / "TOOL_STATE.json"
        state = canonical_load(state_path)
        if not state["runtime_started"]:
            state["runtime_started"] = True
            write_canonical_replace(state_path, state)
        command_id = len(self._command_directories()) + 1
        safe_label = re.sub(r"[^a-z0-9_-]", "_", spec["label"])
        directory = self.attempt_dir / "commands" / f"{command_id:04d}_{safe_label}"
        self._fault("before_directory")
        directory.mkdir(parents=False, exist_ok=False)
        started_at = utc_now()
        atomic_canonical_create(directory / "transaction.json", {
            "schema_version": "evaluation-command-transaction.v1",
            "command_id": command_id,
            "manifest_command_index": command_index,
            "repair_of": repair_of,
            "started_at": started_at,
        })
        self._fault("after_directory")
        timed_out = False
        process_error: str | None = None
        try:
            completed = subprocess.run(
                spec["argv"],
                cwd=spec["cwd"],
                env=dict(spec["environment"]),
                capture_output=True,
                timeout=spec["timeout_seconds"],
                check=False,
            )
            raw_exit = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            raw_exit = 124
            stdout = error.stdout or b""
            stderr = error.stderr or b""
        except OSError as error:
            raw_exit = 127
            stdout = b""
            stderr = f"{type(error).__name__}: {error}\n".encode("utf-8", errors="backslashreplace")
            process_error = f"PROCESS_CRASH:{type(error).__name__}"
        (directory / "stdout.bin").write_bytes(stdout)
        self._fault("after_stdout")
        (directory / "stderr.bin").write_bytes(stderr)
        self._fault("after_stderr")
        cleanup_exit: int | None = None
        cleanup_stdout = b""
        cleanup_stderr = b""
        cleanup_confinement_error: RuntimeFailure | None = None
        post_identity_error: RuntimeFailure | None = None
        try:
            self._assert_frozen_identity()
            self._assert_execution_confinement(spec, "argv")
        except RuntimeFailure as error:
            post_identity_error = error
        if spec["cleanup_argv"] is not None:
            if post_identity_error is not None:
                cleanup_confinement_error = post_identity_error
                cleanup_exit = 126
                cleanup_stderr = (str(post_identity_error) + "\n").encode("utf-8")
            if post_identity_error is None:
                try:
                    self._assert_frozen_identity()
                    self._assert_execution_confinement(spec, "cleanup_argv")
                except RuntimeFailure as error:
                    cleanup_confinement_error = error
                    cleanup_exit = 126
                    cleanup_stderr = (str(error) + "\n").encode("utf-8")
            if cleanup_confinement_error is None and post_identity_error is None:
                try:
                    cleanup = subprocess.run(
                        spec["cleanup_argv"],
                        cwd=spec["cwd"],
                        env=dict(spec["environment"]),
                        capture_output=True,
                        timeout=spec["timeout_seconds"],
                        check=False,
                    )
                    cleanup_exit = cleanup.returncode
                    cleanup_stdout = cleanup.stdout
                    cleanup_stderr = cleanup.stderr
                except subprocess.TimeoutExpired as error:
                    cleanup_exit = 124
                    cleanup_stdout = error.stdout or b""
                    cleanup_stderr = error.stderr or b""
                try:
                    self._assert_frozen_identity()
                    self._assert_execution_confinement(spec, "cleanup_argv")
                except RuntimeFailure as error:
                    cleanup_confinement_error = error
            (directory / "cleanup_stdout.bin").write_bytes(cleanup_stdout)
            self._fault("after_cleanup_stdout")
            (directory / "cleanup_stderr.bin").write_bytes(cleanup_stderr)
            self._fault("after_cleanup_stderr")
        semantic_present = spec["semantic_result_path"] is None
        semantic_exit = 0 if semantic_present else 1
        semantic_failure_class = None
        if spec["semantic_result_path"] is not None:
            semantic_path = self.attempt_dir / spec["semantic_result_path"]
            if semantic_path.is_file():
                try:
                    semantic = canonical_load(semantic_path)
                    if not isinstance(semantic, dict) or semantic.get("status") not in {"PASS", "FAIL"}:
                        raise ValueError("invalid semantic receipt")
                    semantic_present = True
                    semantic_exit = int(semantic.get("semantic_exit", 0 if semantic["status"] == "PASS" else 1))
                    if semantic["status"] == "FAIL" and semantic_exit == 0:
                        semantic_exit = 1
                    semantic_failure_class = semantic.get("failure_class")
                    allowed_failure_classes = {item.value for item in RuntimeFailureClass}
                    if semantic_exit != 0 and semantic_failure_class not in allowed_failure_classes:
                        semantic_failure_class = RuntimeFailureClass.PRODUCT_SEMANTIC_FAILURE.value
                except (CanonicalJSONError, AttributeError, TypeError, ValueError):
                    semantic_present = False
                    semantic_exit = 1
        receipt = build_launcher_receipt(
            raw_exit=raw_exit,
            semantic_exit=semantic_exit,
            semantic_evidence_present=semantic_present,
            cleanup_exit=cleanup_exit,
            timed_out=timed_out,
        )
        if process_error is None and raw_exit < 0:
            process_error = f"PROCESS_CRASH:SIGNAL:{-raw_exit}"
        if process_error is not None and not timed_out:
            receipt = LauncherReceipt(
                raw_exit=receipt.raw_exit,
                semantic_exit=receipt.semantic_exit,
                receipt_status="FAIL",
                primary_failure=process_error,
                cleanup_status=receipt.cleanup_status,
                cleanup_exit=receipt.cleanup_exit,
                timed_out=receipt.timed_out,
            )
        if semantic_failure_class is not None and receipt.receipt_status == "FAIL" and receipt.primary_failure and receipt.primary_failure.startswith("SEMANTIC_EXIT"):
            receipt = LauncherReceipt(
                raw_exit=receipt.raw_exit,
                semantic_exit=receipt.semantic_exit,
                receipt_status=receipt.receipt_status,
                primary_failure=f"{semantic_failure_class}:{receipt.primary_failure}",
                cleanup_status=receipt.cleanup_status,
                cleanup_exit=receipt.cleanup_exit,
                timed_out=receipt.timed_out,
            )
        if cleanup_confinement_error is not None and receipt.primary_failure and receipt.primary_failure.startswith("CLEANUP_EXIT"):
            receipt = LauncherReceipt(
                raw_exit=receipt.raw_exit,
                semantic_exit=receipt.semantic_exit,
                receipt_status="FAIL",
                primary_failure=str(cleanup_confinement_error),
                cleanup_status="FAIL",
                cleanup_exit=cleanup_exit,
                timed_out=receipt.timed_out,
            )
        if post_identity_error is not None:
            receipt = LauncherReceipt(
                raw_exit=raw_exit,
                semantic_exit=1,
                receipt_status="FAIL",
                primary_failure=str(post_identity_error),
                cleanup_status=receipt.cleanup_status,
                cleanup_exit=receipt.cleanup_exit,
                timed_out=receipt.timed_out,
            )
        launcher_path = self.attempt_dir / "launcher_receipts" / f"{command_id:04d}.json"
        launcher_object = {
            "schema_version": "evaluation-launcher-receipt.v1",
            "attempt_id": self.manifest["attempt_id"],
            "command_id": command_id,
            **receipt.as_dict(),
        }
        atomic_canonical_create(launcher_path, launcher_object)
        self._fault("after_launcher_receipt")
        ended_at = utc_now()
        metadata = self._metadata_object(
            spec=spec,
            command_id=command_id,
            command_index=command_index,
            repair_of=repair_of,
            started_at=started_at,
            ended_at=ended_at,
            raw_exit=raw_exit,
            semantic_exit=receipt.semantic_exit,
            receipt=receipt,
            stdout=stdout,
            stderr=stderr,
            cleanup_stdout=cleanup_stdout,
            cleanup_stderr=cleanup_stderr,
            directory=directory,
            launcher_path=launcher_path,
        )
        atomic_canonical_create(directory / "metadata.json", metadata)
        self._fault("after_metadata")
        index_bytes, ledger_bytes = self._rebuild_aggregates(terminal=False)
        self._fault("after_aggregates")
        receipt_id = len(self._receipt_files()) + 1
        runner_receipt = {
            "schema_version": "evaluation-runner-receipt.v1",
            "attempt_id": self.manifest["attempt_id"],
            "invocation_id": receipt_id,
            "mode": "run",
            "command_ids": [command_id],
            "command_index_sha256": sha256_bytes(index_bytes),
            "command_ledger_sha256": sha256_bytes(ledger_bytes),
            "identity_proof": state["identity_proof"],
            "failure": receipt.primary_failure,
            "started_at": started_at,
            "ended_at": utc_now(),
            "result": receipt.receipt_status,
        }
        atomic_canonical_create(self.attempt_dir / "runner_receipts" / f"{receipt_id:04d}.json", runner_receipt)
        self._fault("after_runner_receipt")
        atomic_canonical_create(directory / "transaction_complete.json", {
            "schema_version": "evaluation-command-transaction-complete.v1",
            "command_id": command_id,
            "result": "COMMITTED",
        })
        exit_code = 0 if receipt.receipt_status == "PASS" else 1
        return receipt, exit_code

    def run_all(self) -> int:
        result = self.prepare()
        if result["status"] != "PASS":
            return 2
        for index in range(len(self.manifest["commands"])):
            _, exit_code = self.run_command(index)
            if exit_code != 0:
                return exit_code
        return 0
