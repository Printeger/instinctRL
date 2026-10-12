"""Dependency-complete, discovery-exact successor host CPU evidence runner."""
from __future__ import annotations

import hashlib
import json
import importlib
import os
import platform
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "instinctrl-host-cpu-evidence/3"
PACKAGE_ID = "INSTINCTRL-R5-STAGE1-HOST-EXECUTOR-SUCCESSOR-20260929"
TASK_ID = "T-93e0afe401775fb0"
BASE_COMMIT = "a0816149ced88bd46c1e0104413b73050626a7e9"
REVIEWED_PREDECESSOR_DIFF = "44bb356a9496995553481fb4b23e457ff091f86eba9f7d5453fc26917d0f96a8"
MANIFEST_PATH = "review_evidence/host_cpu_test_manifest_successor.json"
HOST_ARGV = ["/home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python", "-m", "training_workloads.unified_h1.host_cpu_evidence_successor"]
OUTPUT_PATHS = [
    "review_evidence/host_cpu_successor.collection.stdout",
    "review_evidence/host_cpu_successor.collection.stderr",
    "review_evidence/host_cpu_successor.test.stdout",
    "review_evidence/host_cpu_successor.test.stderr",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def regular(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise PermissionError("unsafe host evidence path")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise PermissionError("host evidence byte is absent or not a regular file")
    return path


def immutable_regular(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError("host evidence byte is absent, unsafe, or mutable")
    return path


def load_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise PermissionError("host evidence JSON is absent or not a regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PermissionError("host evidence JSON is unreadable") from error


def load_manifest(root: Path) -> dict[str, Any]:
    value = load_json(root / MANIFEST_PATH)
    expected = {"schema_version", "package_id", "task_id", "base_commit", "reviewed_predecessor_diff", "host_argv", "pytest_version", "pytest_prefix", "interpreter", "modules", "source_paths"}
    if not isinstance(value, dict) or set(value) != expected or value["schema_version"] != SCHEMA:
        raise PermissionError("host CPU successor manifest schema mismatch")
    identity = (value["package_id"], value["task_id"], value["base_commit"], value["reviewed_predecessor_diff"], value["host_argv"], value["pytest_version"])
    required = (PACKAGE_ID, TASK_ID, BASE_COMMIT, REVIEWED_PREDECESSOR_DIFF, HOST_ARGV, "8.3.5")
    if identity != required:
        raise PermissionError("host CPU successor manifest identity mismatch")
    prefix = value["pytest_prefix"]
    if prefix != [HOST_ARGV[0], "-m", "pytest", "-s", "-p", "no:cacheprovider", "-c", "/dev/null", "--rootdir=."]:
        raise PermissionError("host CPU successor pytest argv mismatch")
    modules = value["modules"]
    if not isinstance(modules, list) or not modules:
        raise PermissionError("host CPU successor module inventory is empty")
    seen_paths: set[str] = set()
    seen_nodes: set[str] = set()
    for row in modules:
        if not isinstance(row, dict) or set(row) != {"path", "expected_node_ids"}:
            raise PermissionError("host CPU successor module row is malformed")
        path, nodes = row["path"], row["expected_node_ids"]
        if path in seen_paths or not isinstance(nodes, list) or not nodes or len(nodes) != len(set(nodes)):
            raise PermissionError("host CPU successor module inventory is duplicated or empty")
        if any(not isinstance(node, str) or not node.startswith(path + "::") or node in seen_nodes for node in nodes):
            raise PermissionError("host CPU successor node inventory is malformed or duplicated")
        seen_paths.add(path)
        seen_nodes.update(nodes)
    paths = value["source_paths"]
    if not isinstance(paths, list) or not paths or len(paths) != len(set(paths)) or any(not isinstance(path, str) for path in paths):
        raise PermissionError("host CPU successor source inventory is malformed")
    for row in modules:
        if row["path"] not in paths:
            raise PermissionError("invoked test source is not hash-bound")
    if MANIFEST_PATH not in paths or "training_workloads/unified_h1/host_cpu_evidence_successor.py" not in paths:
        raise PermissionError("runner or manifest is not hash-bound")
    return value


def validate_executor_identity(manifest: dict[str, Any], *, executable: str | None = None, pytest_version: str | None = None, importer: Any = importlib.import_module) -> None:
    identity = manifest.get("interpreter")
    expected = {"executable": HOST_ARGV[0], "pytest_version": "8.3.5", "required_imports": ["pytest", "referencing"]}
    if identity != expected:
        raise PermissionError("host CPU successor interpreter manifest drift")
    actual_executable = sys.executable if executable is None else executable
    if actual_executable != identity["executable"]:
        raise PermissionError("host CPU successor interpreter identity drift")
    pytest_module = importer("pytest")
    actual_pytest = getattr(pytest_module, "__version__", None) if pytest_version is None else pytest_version
    if actual_pytest != identity["pytest_version"] or actual_pytest != manifest["pytest_version"]:
        raise PermissionError("host CPU successor pytest identity drift")
    for module_name in identity["required_imports"]:
        try:
            importer(module_name)
        except (ImportError, ModuleNotFoundError) as error:
            raise PermissionError("host CPU successor dependency drift: " + module_name) from error


def load_failed_attempt(root: Path) -> dict[str, Any]:
    relative = "review_evidence/host_cpu_failed_predecessor_attempt.json"
    path = immutable_regular(root / relative)
    value = load_json(path)
    expected = {
        "schema_version": "instinctrl-host-cpu-failed-attempt/1",
        "package_id": "INSTINCTRL-R5-STAGE1-PYTEST-DISCOVERY-SUCCESSOR-20260929",
        "host_argv": ["/usr/bin/python3", "-m", "training_workloads.unified_h1.host_cpu_evidence_v2"],
        "exit_code": 1,
        "receipt_created": False,
        "first_module": "control_snapshot/tests/test_process_recovery.py",
        "setup_errors": 8,
        "failure": "ModuleNotFoundError: No module named 'referencing'",
        "classification": "executor-environment-implementation-finding",
        "source": "frozen-orchestration-input",
    }
    if value != expected:
        raise PermissionError("failed predecessor attempt provenance drift")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": digest(path), "record": value}


def parse_collected(stdout: str, module_path: str) -> list[str]:
    nodes = [line.strip() for line in stdout.splitlines() if line.strip().startswith(module_path + "::")]
    if not nodes:
        raise PermissionError("zero pytest discovery for " + module_path)
    if len(nodes) != len(set(nodes)):
        raise PermissionError("duplicate pytest node IDs for " + module_path)
    return nodes


def require_exact_collection(stdout: str, module_path: str, expected: list[str]) -> list[str]:
    actual = parse_collected(stdout, module_path)
    if actual != expected:
        raise PermissionError("pytest node inventory drift for " + module_path)
    return actual


def parse_pass_count(stdout: str, stderr: str) -> int:
    combined = stdout + "\n" + stderr
    if re.search(r"\b(?:failed|error|errors|skipped|xfailed|xpassed)\b", combined, re.IGNORECASE):
        raise PermissionError("pytest reported a non-pass outcome")
    matches = re.findall(r"(?:^|\s)(\d+) passed in ", combined)
    if len(matches) != 1:
        raise PermissionError("pytest pass summary is absent or ambiguous")
    return int(matches[0])


def source_inventory(root: Path, paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in paths:
        path = regular(root, relative)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    return rows


def _validate_rows(root: Path, rows: Any) -> None:
    if not isinstance(rows, list) or not rows:
        raise PermissionError("host CPU successor byte inventory is empty")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"} or row["path"] in seen:
            raise PermissionError("host CPU successor byte row is malformed or duplicated")
        seen.add(row["path"])
        path = regular(root, row["path"])
        if type(row["bytes"]) is not int or row["bytes"] < 0 or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise PermissionError("host CPU successor byte drift: " + row["path"])


def _write_sealed(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)


def verify_host_cpu_evidence(root: Path) -> dict[str, Any]:
    root = Path(root)
    receipt_path = immutable_regular(root / "review_evidence/HOST_CPU_EVIDENCE_SUCCESSOR.json")
    digest_path = immutable_regular(root / "review_evidence/HOST_CPU_EVIDENCE_SUCCESSOR.sha256")
    receipt = load_json(receipt_path)
    digest_text = digest_path.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", digest_text) or digest(receipt_path) != digest_text:
        raise PermissionError("HOST_CPU_EVIDENCE_SUCCESSOR receipt hash mismatch")
    expected_keys = {"schema_version", "package_id", "task_id", "base_commit", "reviewed_predecessor_diff", "host_argv", "executor", "pytest_version", "modules", "outputs", "sources", "predecessor_evidence", "failed_predecessor_attempt", "cleanup_audit"}
    if not isinstance(receipt, dict) or set(receipt) != expected_keys or receipt["schema_version"] != SCHEMA:
        raise PermissionError("HOST_CPU_EVIDENCE_SUCCESSOR schema mismatch")
    manifest = load_manifest(root)
    for key in ("package_id", "task_id", "base_commit", "reviewed_predecessor_diff", "host_argv", "pytest_version"):
        if receipt[key] != manifest[key]:
            raise PermissionError("HOST_CPU_EVIDENCE_SUCCESSOR identity or argv drift")
    executor = receipt["executor"]
    if not isinstance(executor, dict) or set(executor) != {"kind", "executable", "python", "platform", "fork", "signals"} or executor["kind"] != "external-host-cpu" or executor["executable"] != HOST_ARGV[0] or executor["fork"] is not True or executor["signals"] is not True:
        raise PermissionError("HOST_CPU_EVIDENCE_SUCCESSOR executor mismatch")
    expected_modules = manifest["modules"]
    rows = receipt["modules"]
    if not isinstance(rows, list) or len(rows) != len(expected_modules):
        raise PermissionError("HOST_CPU_EVIDENCE_SUCCESSOR module results mismatch")
    for expected, row in zip(expected_modules, rows):
        if not isinstance(row, dict) or set(row) != {"path", "collection_argv", "test_argv", "expected_node_ids", "collected", "passed", "exit_code"}:
            raise PermissionError("HOST_CPU_EVIDENCE_SUCCESSOR module result malformed")
        if row["path"] != expected["path"] or row["expected_node_ids"] != expected["expected_node_ids"]:
            raise PermissionError("HOST_CPU_EVIDENCE_SUCCESSOR expected inventory drift")
        expected_collection_argv = manifest["pytest_prefix"] + ["--collect-only", "-q", expected["path"]]
        expected_test_argv = manifest["pytest_prefix"] + ["-q", *expected["expected_node_ids"]]
        if row["collection_argv"] != expected_collection_argv or row["test_argv"] != expected_test_argv:
            raise PermissionError("HOST_CPU_EVIDENCE_SUCCESSOR per-module argv drift")
        count = len(expected["expected_node_ids"])
        if row["collected"] != count or row["passed"] != count or row["exit_code"] != 0:
            raise PermissionError("HOST_CPU_EVIDENCE_SUCCESSOR does not record complete passing execution")
    failed = load_failed_attempt(root)
    if receipt["failed_predecessor_attempt"] != failed:
        raise PermissionError("failed predecessor attempt receipt binding drift")
    cleanup_nodes = [node for module in manifest["modules"] for node in module["expected_node_ids"] if "detached" in node or "double_fork" in node]
    if receipt["cleanup_audit"] != {"identity_bound_detached_cleanup_nodes": cleanup_nodes, "passed": len(cleanup_nodes)}:
        raise PermissionError("detached cleanup audit mismatch")
    if receipt["predecessor_evidence"] != {"path": "review_evidence/HOST_CPU_EVIDENCE.json", "role": "historical-provenance-only", "tests_run": 132}:
        raise PermissionError("HOST_CPU_EVIDENCE_SUCCESSOR predecessor provenance mismatch")
    if [row.get("path") for row in receipt["outputs"] if isinstance(row, dict)] != OUTPUT_PATHS:
        raise PermissionError("HOST_CPU_EVIDENCE_SUCCESSOR output inventory mismatch")
    _validate_rows(root, receipt["outputs"])
    for row in receipt["outputs"]:
        immutable_regular(root / row["path"])
    if [row.get("path") for row in receipt["sources"] if isinstance(row, dict)] != manifest["source_paths"]:
        raise PermissionError("HOST_CPU_EVIDENCE_SUCCESSOR source inventory mismatch")
    _validate_rows(root, receipt["sources"])
    return receipt


def run_host_cpu(root: Path) -> int:
    root = Path(root).resolve()
    manifest = load_manifest(root)
    if not hasattr(os, "fork") or not hasattr(__import__("signal"), "SIGTERM"):
        raise PermissionError("host executor lacks fork/signal support")
    validate_executor_identity(manifest)
    import pytest
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = "control_snapshot:control_snapshot/tests"
    module_results = []
    collect_out: list[str] = []
    collect_err: list[str] = []
    test_out: list[str] = []
    test_err: list[str] = []
    prefix = manifest["pytest_prefix"]
    for module in manifest["modules"]:
        path = module["path"]
        expected = module["expected_node_ids"]
        collection_argv = prefix + ["--collect-only", "-q", path]
        collected = subprocess.run(collection_argv, cwd=root, env=env, capture_output=True, text=True, check=False)
        collect_out.append("$ " + " ".join(collection_argv) + "\n" + collected.stdout)
        collect_err.append("$ " + " ".join(collection_argv) + "\n" + collected.stderr)
        if collected.returncode != 0:
            raise PermissionError("pytest collection failed for " + path)
        actual = require_exact_collection(collected.stdout, path, expected)
        test_argv = prefix + ["-q", *expected]
        tested = subprocess.run(test_argv, cwd=root, env=env, capture_output=True, text=True, check=False)
        test_out.append("$ " + " ".join(test_argv) + "\n" + tested.stdout)
        test_err.append("$ " + " ".join(test_argv) + "\n" + tested.stderr)
        if tested.returncode != 0:
            raise PermissionError("pytest execution failed for " + path)
        passed = parse_pass_count(tested.stdout, tested.stderr)
        if passed != len(expected):
            raise PermissionError("pytest pass count mismatch for " + path)
        module_results.append({"path": path, "collection_argv": collection_argv, "test_argv": test_argv, "expected_node_ids": expected, "collected": len(actual), "passed": passed, "exit_code": tested.returncode})
    output_data = ["\n".join(collect_out), "\n".join(collect_err), "\n".join(test_out), "\n".join(test_err)]
    for relative, text in zip(OUTPUT_PATHS, output_data):
        _write_sealed(root / relative, text.encode("utf-8"))
    receipt = {
        "schema_version": SCHEMA, "package_id": PACKAGE_ID, "task_id": TASK_ID, "base_commit": BASE_COMMIT,
        "reviewed_predecessor_diff": REVIEWED_PREDECESSOR_DIFF, "host_argv": HOST_ARGV,
        "executor": {"kind": "external-host-cpu", "executable": sys.executable, "python": sys.version.split()[0], "platform": platform.platform(), "fork": True, "signals": True},
        "pytest_version": manifest["pytest_version"], "modules": module_results,
        "outputs": source_inventory(root, OUTPUT_PATHS), "sources": source_inventory(root, manifest["source_paths"]),
        "predecessor_evidence": {"path": "review_evidence/HOST_CPU_EVIDENCE.json", "role": "historical-provenance-only", "tests_run": 132},
        "failed_predecessor_attempt": load_failed_attempt(root),
        "cleanup_audit": {"identity_bound_detached_cleanup_nodes": [node for module in manifest["modules"] for node in module["expected_node_ids"] if "detached" in node or "double_fork" in node], "passed": sum(1 for module in manifest["modules"] for node in module["expected_node_ids"] if "detached" in node or "double_fork" in node)},
    }
    receipt_path = root / "review_evidence/HOST_CPU_EVIDENCE_SUCCESSOR.json"
    _write_sealed(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    _write_sealed(root / "review_evidence/HOST_CPU_EVIDENCE_SUCCESSOR.sha256", (digest(receipt_path) + "\n").encode("ascii"))
    verify_host_cpu_evidence(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_host_cpu(Path(__file__).resolve().parents[2]))
