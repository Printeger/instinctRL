from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import stat
import sys
from typing import Any

from evaluation_tooling.canonical import canonical_dumps
from evaluation_tooling.schemas import executable_source_paths


TOOLING_REPOSITORY = Path(__file__).resolve().parents[2]


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=TOOLING_REPOSITORY, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


TOOLING_COMMIT = _git("rev-parse", "HEAD")
TOOLING_TREE = _git("rev-parse", "HEAD^{tree}")
PRODUCT_COMMIT = TOOLING_COMMIT
PRODUCT_TREE = TOOLING_TREE


def write_canonical(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_dumps(value))


def terminal_file_row(packet: Path, path: Path, schema_version: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path.relative_to(packet).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if schema_version == "evaluation-terminal-manifest.v2":
        row.update(type="file", mode=stat.S_IMODE(path.stat().st_mode))
    return row


def python_source_argv(root: Path, name: str, source: str) -> list[str]:
    path = root / f"fixture_{name}.py"
    path.write_text(source, encoding="utf-8")
    return [str(Path(sys.executable).resolve()), "-B", str(path.resolve())]


def _adapt_native_argv(root: Path, name: str, argv: list[str]) -> list[str]:
    executable = Path(argv[0]).name
    if executable == "printf":
        source = f"import sys\nsys.stdout.write({''.join(argv[1:])!r})\n"
    elif executable == "true":
        source = "raise SystemExit(0)\n"
    elif executable == "false":
        source = "raise SystemExit(1)\n"
    elif executable == "touch":
        source = f"from pathlib import Path\nPath({argv[1]!r}).touch()\n"
    elif executable == "truncate" and argv[1:3] == ["-s", "0"]:
        source = f"from pathlib import Path\nPath({argv[3]!r}).write_bytes(b'')\n"
    elif executable == "cp":
        source = f"import shutil\nshutil.copyfile({argv[1]!r}, {argv[2]!r})\n"
    elif executable == "sleep":
        source = f"import time\ntime.sleep({float(argv[1])!r})\n"
    elif executable == "evaluation-tooling-does-not-exist":
        source = "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n"
    else:
        return argv
    return python_source_argv(root, name, source)


def make_attempt(
    root: Path,
    *,
    commands: list[dict[str, Any]] | None = None,
    required_results: list[str] | None = None,
    external: list[dict[str, Any]] | None = None,
    float_policy: str = "FLOAT32_GEOMETRY_ULP2_ABS1E5_V1",
    pin_command_files: bool = True,
    adapt_native_commands: bool = True,
) -> Path:
    attempt = root / "attempt"
    attempt.mkdir()
    source = root / "input.dat"
    source.write_bytes(b"frozen-input\n")
    if commands is None:
        commands = [{
            "label": "positive",
            "argv": ["/usr/bin/printf", "fixture-output"],
            "cwd": str(root),
            "environment": {},
            "expected_exit": 0,
            "timeout_seconds": 5,
            "semantic_result_path": None,
            "cleanup_argv": None,
        }]
    if adapt_native_commands:
        for index, spec in enumerate(commands):
            for field in ("argv", "cleanup_argv"):
                argv = spec.get(field)
                if isinstance(argv, list) and argv:
                    spec[field] = _adapt_native_argv(root, f"{index:04d}_{field}", argv)
    pinned_files: dict[Path, str] = {source.resolve(): "fixture"}
    if pin_command_files:
        for spec in commands:
            for field in ("argv", "cleanup_argv"):
                argv = spec.get(field)
                if not isinstance(argv, list) or not argv:
                    continue
                executable = Path(argv[0])
                if executable.exists():
                    resolved = executable.resolve()
                    argv[0] = str(resolved)
                    if resolved.is_file() and not resolved.is_symlink():
                        pinned_files[resolved] = "executable"
                for _, candidate, was_relative in executable_source_paths(argv, spec.get("cwd")):
                    resolved_source = candidate.resolve(strict=False)
                    if not was_relative and resolved_source.is_file() and not resolved_source.is_symlink():
                        pinned_files[resolved_source] = "source"
    input_hashes = {
        "schema_version": "evaluation-input-hashes.v1",
        "product_commit": PRODUCT_COMMIT,
        "tooling_commit": TOOLING_COMMIT,
        "files": [{
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "role": pinned_files[path],
        } for path in sorted(pinned_files)],
    }
    input_bytes = canonical_dumps(input_hashes)
    (attempt / "INPUT_HASHES.json").write_bytes(input_bytes)
    input_roots = sorted({str(root.resolve()), *(str(path.parent) for path in pinned_files)})
    manifest = {
        "schema_version": "evaluation-attempt-manifest.v1",
        "attempt_id": "FIXTURE-D0001",
        "product_commit": PRODUCT_COMMIT,
        "tooling_commit": TOOLING_COMMIT,
        "tooling_repository": str(TOOLING_REPOSITORY),
        "tooling_tree": TOOLING_TREE,
        "product_repository": str(TOOLING_REPOSITORY),
        "product_tree": PRODUCT_TREE,
        "product_clean_required": False,
        "input_hashes_path": "INPUT_HASHES.json",
        "input_hashes_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "allowed_input_roots": input_roots,
        "allowed_working_roots": [str(root)],
        "float_policy": float_policy,
        "commands": commands,
        "external_inventories": external or [],
        "required_results": required_results or [],
    }
    write_canonical(attempt / "ATTEMPT_MANIFEST.json", manifest)
    return attempt
