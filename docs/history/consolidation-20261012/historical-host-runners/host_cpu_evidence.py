"""Create and verify the immutable host-CPU regression receipt."""
from __future__ import annotations
import hashlib, json, os, platform, re, signal, stat, subprocess, sys
from pathlib import Path
from typing import Any

SCHEMA = "instinctrl-host-cpu-evidence/1"
PACKAGE_ID = "INSTINCTRL-R5-STAGE1-REVIEW-EVIDENCE-PROCESS-SUCCESSOR-20260929"
TASK_ID = "T-562c13f18d3323d3"
BASE_COMMIT = "a0816149ced88bd46c1e0104413b73050626a7e9"
PREDECESSOR = {
    "task_id": "T-d219cc10961c75be",
    "base_head": BASE_COMMIT,
    "diff_sha256": "ea2873300779e325af07bda2a0be716675eceba5d74ba837fd13962cb822a044",
    "tree_files_sha256": "bf3916f06e1a239d63cf82e72b73f07e6440b00e86a33f3020b48209aaed9a17",
    "role": "provenance-only",
}


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
        raise PermissionError("HOST_CPU_EVIDENCE byte is absent, unsafe, or mutable")
    return path


def load_regular(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise PermissionError("HOST_CPU_EVIDENCE is absent or not a regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PermissionError("HOST_CPU_EVIDENCE is unreadable") from error


def load_manifest(root: Path) -> dict[str, Any]:
    value = load_regular(root / "review_evidence/host_cpu_test_manifest.json")
    expected = {"schema_version", "package_id", "task_id", "base_commit", "test_argv", "source_paths"}
    if not isinstance(value, dict) or set(value) != expected or value["schema_version"] != SCHEMA:
        raise PermissionError("host CPU manifest schema mismatch")
    if value["package_id"] != PACKAGE_ID or value["task_id"] != TASK_ID or value["base_commit"] != BASE_COMMIT:
        raise PermissionError("host CPU manifest identity mismatch")
    if not isinstance(value["test_argv"], list) or not value["test_argv"] or not all(isinstance(v, str) and v for v in value["test_argv"]):
        raise PermissionError("host CPU test argv is malformed")
    if not isinstance(value["source_paths"], list) or not value["source_paths"] or len(set(value["source_paths"])) != len(value["source_paths"]):
        raise PermissionError("host CPU source inventory is malformed")
    return value


def source_inventory(root: Path, paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in paths:
        path = regular(root, relative)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    return rows


def _validate_rows(root: Path, rows: Any) -> None:
    if not isinstance(rows, list) or not rows:
        raise PermissionError("host CPU source inventory is empty")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"} or row["path"] in seen:
            raise PermissionError("host CPU source row is malformed or duplicated")
        seen.add(row["path"])
        path = regular(root, row["path"])
        if type(row["bytes"]) is not int or row["bytes"] < 0 or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise PermissionError("host CPU source drift: " + row["path"])


def verify_host_cpu_evidence(root: Path, receipt_path: Path | None = None, digest_path: Path | None = None) -> dict[str, Any]:
    root = Path(root)
    receipt_path = receipt_path or root / "review_evidence/HOST_CPU_EVIDENCE.json"
    digest_path = digest_path or root / "review_evidence/HOST_CPU_EVIDENCE.sha256"
    immutable_regular(receipt_path)
    receipt = load_regular(receipt_path)
    digest_file = immutable_regular(digest_path).read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", digest_file) or digest(receipt_path) != digest_file:
        raise PermissionError("HOST_CPU_EVIDENCE receipt hash mismatch")
    expected = {"schema_version", "package_id", "task_id", "base_commit", "executor", "predecessor", "test_argv", "result", "outputs", "sources"}
    if not isinstance(receipt, dict) or set(receipt) != expected or receipt["schema_version"] != SCHEMA:
        raise PermissionError("HOST_CPU_EVIDENCE schema mismatch")
    if receipt["package_id"] != PACKAGE_ID or receipt["task_id"] != TASK_ID or receipt["base_commit"] != BASE_COMMIT:
        raise PermissionError("HOST_CPU_EVIDENCE identity mismatch")
    if receipt["predecessor"] != PREDECESSOR:
        raise PermissionError("HOST_CPU_EVIDENCE predecessor provenance mismatch")
    manifest = load_manifest(root)
    if receipt["test_argv"] != manifest["test_argv"]:
        raise PermissionError("HOST_CPU_EVIDENCE argv drift")
    executor = receipt["executor"]
    if not isinstance(executor, dict) or set(executor) != {"kind", "python", "platform", "fork", "signals"} or executor["kind"] != "external-host-cpu" or executor["fork"] is not True or executor["signals"] is not True:
        raise PermissionError("HOST_CPU_EVIDENCE executor provenance mismatch")
    result = receipt["result"]
    if not isinstance(result, dict) or set(result) != {"exit_code", "tests_run", "failures", "errors", "skipped"} or any(type(result[k]) is not int or result[k] < 0 for k in result):
        raise PermissionError("HOST_CPU_EVIDENCE result mismatch")
    if result["exit_code"] != 0 or result["tests_run"] <= 0 or result["failures"] or result["errors"]:
        raise PermissionError("HOST_CPU_EVIDENCE does not record a passing run")
    outputs = receipt["outputs"]
    if not isinstance(outputs, list) or {row.get("path") for row in outputs if isinstance(row, dict)} != {"review_evidence/host_cpu.stdout", "review_evidence/host_cpu.stderr"}:
        raise PermissionError("HOST_CPU_EVIDENCE output inventory mismatch")
    _validate_rows(root, outputs)
    for row in outputs:
        immutable_regular(root / row["path"])
    if [row.get("path") for row in receipt["sources"] if isinstance(row, dict)] != manifest["source_paths"]:
        raise PermissionError("HOST_CPU_EVIDENCE source inventory drift")
    _validate_rows(root, receipt["sources"])
    return receipt


def _write_sealed(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)


def run_host_cpu(root: Path) -> int:
    root = Path(root).resolve()
    manifest = load_manifest(root)
    if not hasattr(os, "fork") or not hasattr(signal, "SIGTERM"):
        raise PermissionError("host executor lacks fork/signal support")
    completed = subprocess.run(manifest["test_argv"], cwd=root, capture_output=True, text=True, check=False)
    output_root = root / "review_evidence"
    stdout_path = output_root / "host_cpu.stdout"
    stderr_path = output_root / "host_cpu.stderr"
    _write_sealed(stdout_path, completed.stdout.encode("utf-8"))
    _write_sealed(stderr_path, completed.stderr.encode("utf-8"))
    combined = completed.stdout + "\n" + completed.stderr
    match = re.search(r"Ran (\d+) tests?", combined)
    failures = len(re.findall(r"^FAIL:", combined, re.MULTILINE))
    errors = len(re.findall(r"^ERROR:", combined, re.MULTILINE))
    skipped = sum(int(v) for v in re.findall(r"skipped=(\d+)", combined))
    result = {"exit_code": completed.returncode, "tests_run": int(match.group(1)) if match else 0, "failures": failures, "errors": errors, "skipped": skipped}
    receipt = {
        "schema_version": SCHEMA, "package_id": PACKAGE_ID, "task_id": TASK_ID, "base_commit": BASE_COMMIT,
        "executor": {"kind": "external-host-cpu", "python": sys.version.split()[0], "platform": platform.platform(), "fork": True, "signals": True},
        "predecessor": PREDECESSOR, "test_argv": manifest["test_argv"], "result": result,
        "outputs": source_inventory(root, ["review_evidence/host_cpu.stdout", "review_evidence/host_cpu.stderr"]),
        "sources": source_inventory(root, manifest["source_paths"]),
    }
    receipt_path = output_root / "HOST_CPU_EVIDENCE.json"
    _write_sealed(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    digest_path = output_root / "HOST_CPU_EVIDENCE.sha256"
    _write_sealed(digest_path, (digest(receipt_path) + "\n").encode("ascii"))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(run_host_cpu(Path(__file__).resolve().parents[2]))
