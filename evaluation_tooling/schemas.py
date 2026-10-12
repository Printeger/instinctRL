"""Versioned declarative schemas and strict validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable

from .findings import Finding, FindingCollector


MANIFEST_SCHEMA_VERSION = "evaluation-attempt-manifest.v1"
INPUT_HASHES_SCHEMA_VERSION = "evaluation-input-hashes.v1"
TERMINAL_SCHEMA_VERSION = "evaluation-terminal-manifest.v2"
LEGACY_TERMINAL_SCHEMA_VERSION = "evaluation-terminal-manifest.v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
LAUNCH_PROFILE = "DIRECT_PINNED_PYTHON_SOURCE_V1"

MANIFEST_FIELDS = {
    "schema_version",
    "attempt_id",
    "product_commit",
    "tooling_commit",
    "tooling_repository",
    "tooling_tree",
    "product_repository",
    "product_tree",
    "product_clean_required",
    "input_hashes_path",
    "input_hashes_sha256",
    "allowed_input_roots",
    "allowed_working_roots",
    "float_policy",
    "commands",
    "external_inventories",
    "required_results",
}
INPUT_HASH_FIELDS = {"schema_version", "product_commit", "tooling_commit", "files"}
COMMAND_FIELDS = {
    "label",
    "argv",
    "cwd",
    "environment",
    "expected_exit",
    "timeout_seconds",
    "semantic_result_path",
    "cleanup_argv",
}
INPUT_FILE_FIELDS = {"path", "bytes", "sha256", "role"}
EXTERNAL_FIELDS = {"name", "root", "patterns", "expected", "kit_rotation"}


def confined_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def path_within(path: Path, roots: list[Path]) -> bool:
    absolute = Path(os.path.abspath(path))
    return any(absolute == Path(os.path.abspath(root)) or Path(os.path.abspath(root)) in absolute.parents for root in roots)


_INTERPRETERS = {
    "python", "python3", "bash", "dash", "sh", "zsh", "fish", "node", "perl", "ruby",
}

def _interpreter_name(executable: str) -> str:
    name = Path(executable).name.lower()
    if name.startswith("python"):
        return "python"
    return name


def _dynamic_code_argument(argv: list[str]) -> bool:
    if not argv:
        return False
    interpreter = _interpreter_name(argv[0])
    if interpreter not in _INTERPRETERS:
        return False
    for argument in argv[1:]:
        lowered = argument.lower()
        if argument in {"-", "/dev/stdin", "/proc/self/fd/0"}:
            return True
        if lowered in {"-c", "--command", "--eval", "-e"}:
            return True
        if lowered.startswith(("--command=", "--eval=")):
            return True
        if interpreter in {"python", "bash", "dash", "sh", "zsh", "fish"} and lowered.startswith("-") and not lowered.startswith("--") and "c" in lowered[1:]:
            return True
        if interpreter in {"node", "perl", "ruby"} and lowered.startswith("-e") and len(argument) > 2:
            return True
    return False


def _module_execution(argv: list[str]) -> bool:
    return bool(argv and _interpreter_name(argv[0]) == "python" and "-m" in argv[1:])


def executable_source_paths(argv: list[str], cwd: str | None) -> list[tuple[int, Path, bool]]:
    """Return the one source in the DIRECT_PINNED_PYTHON_SOURCE_V1 shape."""

    if not argv or not isinstance(cwd, str):
        return []
    index = 1
    while index < len(argv) and argv[index] == "-B":
        index += 1
    if index >= len(argv) or argv[index].startswith("-"):
        return []
    candidate = Path(argv[index])
    return [(index, candidate if candidate.is_absolute() else Path(cwd) / candidate, not candidate.is_absolute())]


def _source_confinement(
    argv: list[str],
    *,
    cwd: Any,
    attempt_dir: Path | None,
    location: str,
) -> tuple[Finding, ...]:
    collector = FindingCollector()
    for index, path, was_relative in executable_source_paths(argv, cwd if isinstance(cwd, str) else None):
        if was_relative:
            collector.add("ATTEMPT_LOCAL_EXECUTABLE", f"{location}[{index}]", "relative interpreter source is forbidden")
        if attempt_dir is not None:
            absolute_attempt = Path(os.path.abspath(attempt_dir))
            absolute = Path(os.path.abspath(path))
            if absolute == absolute_attempt or absolute_attempt in absolute.parents:
                collector.add("ATTEMPT_LOCAL_EXECUTABLE", f"{location}[{index}]", "attempt-local executable/source is forbidden")
    return collector.findings


def _type_name(value: Any) -> str:
    return type(value).__name__


def _exact_fields(value: Any, fields: set[str], location: str, collector: FindingCollector) -> bool:
    if not isinstance(value, dict):
        collector.add("TYPE", location, f"expected object, got {_type_name(value)}")
        return False
    for key in sorted(fields - set(value)):
        collector.add("MISSING_FIELD", f"{location}.{key}", "required field is missing")
    for key in sorted(set(value) - fields):
        collector.add("UNKNOWN_FIELD", f"{location}.{key}", "field is not defined by this schema")
    return set(value) == fields


def _string(value: Any, location: str, collector: FindingCollector, pattern: re.Pattern[str] | None = None) -> bool:
    if not isinstance(value, str):
        collector.add("TYPE", location, f"expected string, got {_type_name(value)}")
        return False
    if pattern is not None and pattern.fullmatch(value) is None:
        collector.add("FORMAT", location, "string does not match required format")
        return False
    return True


def validate_manifest(value: Any, attempt_dir: Path | None = None) -> tuple[Finding, ...]:
    collector = FindingCollector()
    if not _exact_fields(value, MANIFEST_FIELDS, "$", collector):
        if not isinstance(value, dict):
            return collector.findings
    if not isinstance(value, dict):
        return collector.findings
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        collector.add("SCHEMA_VERSION", "$.schema_version", f"expected {MANIFEST_SCHEMA_VERSION}")
    _string(value.get("attempt_id"), "$.attempt_id", collector, re.compile(r"[A-Z0-9][A-Z0-9._-]*\Z"))
    _string(value.get("product_commit"), "$.product_commit", collector, COMMIT_RE)
    _string(value.get("tooling_commit"), "$.tooling_commit", collector, COMMIT_RE)
    for field in ("tooling_repository", "product_repository"):
        item = value.get(field)
        if not isinstance(item, str) or not Path(item).is_absolute():
            collector.add("PATH", f"$.{field}", "repository must be an absolute path")
    _string(value.get("tooling_tree"), "$.tooling_tree", collector, COMMIT_RE)
    _string(value.get("product_tree"), "$.product_tree", collector, COMMIT_RE)
    if not isinstance(value.get("product_clean_required"), bool):
        collector.add("TYPE", "$.product_clean_required", "expected boolean")
    if value.get("input_hashes_path") != "INPUT_HASHES.json":
        collector.add("PATH", "$.input_hashes_path", "must be exactly INPUT_HASHES.json")
    _string(value.get("input_hashes_sha256"), "$.input_hashes_sha256", collector, SHA256_RE)
    if value.get("float_policy") not in {"EXACT_FLOAT32_V1", "EXACT_FLOAT64_V1", "FLOAT32_GEOMETRY_ULP2_ABS1E5_V1"}:
        collector.add("FLOAT_POLICY", "$.float_policy", "unknown policy")
    for field in ("allowed_input_roots", "allowed_working_roots", "required_results"):
        items = value.get(field)
        if not isinstance(items, list):
            collector.add("TYPE", f"$.{field}", "expected list")
            continue
        if len(items) != len(set(map(str, items))):
            collector.add("DUPLICATE", f"$.{field}", "set-labelled list contains duplicate values")
        for index, item in enumerate(items):
            if not isinstance(item, str):
                collector.add("TYPE", f"$.{field}[{index}]", "expected string")
            elif field == "required_results" and not confined_relative_path(item):
                collector.add("PATH", f"$.{field}[{index}]", "result path is not confined")
            elif field != "required_results" and not Path(item).is_absolute():
                collector.add("PATH", f"$.{field}[{index}]", "root must be absolute")
    commands = value.get("commands")
    if not isinstance(commands, list):
        collector.add("TYPE", "$.commands", "expected ordered list")
    else:
        labels: list[str] = []
        working_roots = [Path(item) for item in value.get("allowed_working_roots", []) if isinstance(item, str) and Path(item).is_absolute()]
        for index, command in enumerate(commands):
            location = f"$.commands[{index}]"
            _exact_fields(command, COMMAND_FIELDS, location, collector)
            if not isinstance(command, dict):
                continue
            label = command.get("label")
            if _string(label, f"{location}.label", collector, re.compile(r"[a-z0-9][a-z0-9_-]*\Z")):
                labels.append(label)
            argv = command.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
                collector.add("ARGV", f"{location}.argv", "must be a nonempty list of nonempty strings")
            elif not Path(argv[0]).is_absolute():
                collector.add("ARGV", f"{location}.argv[0]", "executable must be absolute")
            elif _dynamic_code_argument(argv):
                collector.add("DYNAMIC_CODE", f"{location}.argv", "dynamic/stdin execution is forbidden")
            elif _module_execution(argv):
                collector.add("UNPINNED_SOURCE", f"{location}.argv", "module execution cannot prove exact source bytes")
            elif _interpreter_name(argv[0]) != "python":
                collector.add("LAUNCH_PROFILE", f"{location}.argv", f"{LAUNCH_PROFILE} requires direct Python execution")
            elif not executable_source_paths(argv, command.get("cwd")):
                collector.add("LAUNCH_PROFILE", f"{location}.argv", f"{LAUNCH_PROFILE} requires one identifiable source")
            else:
                collector.extend(_source_confinement(argv, cwd=command.get("cwd"), attempt_dir=attempt_dir, location=f"{location}.argv"))
            cwd = command.get("cwd")
            if not isinstance(cwd, str) or not Path(cwd).is_absolute() or not path_within(Path(cwd), working_roots):
                collector.add("PATH", f"{location}.cwd", "cwd is outside declared working roots")
            environment = command.get("environment")
            if not isinstance(environment, dict) or not all(isinstance(key, str) and isinstance(item, str) for key, item in environment.items()):
                collector.add("ENVIRONMENT", f"{location}.environment", "must map strings to strings")
            expected_exit = command.get("expected_exit")
            if not isinstance(expected_exit, int) or isinstance(expected_exit, bool):
                collector.add("TYPE", f"{location}.expected_exit", "expected integer")
            timeout = command.get("timeout_seconds")
            if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
                collector.add("RANGE", f"{location}.timeout_seconds", "must be positive")
            semantic_path = command.get("semantic_result_path")
            if semantic_path is not None and (not isinstance(semantic_path, str) or not confined_relative_path(semantic_path)):
                collector.add("PATH", f"{location}.semantic_result_path", "must be null or a confined relative path")
            cleanup = command.get("cleanup_argv")
            if cleanup is not None:
                if not isinstance(cleanup, list) or not cleanup or not all(isinstance(item, str) and item for item in cleanup) or not Path(cleanup[0]).is_absolute():
                    collector.add("ARGV", f"{location}.cleanup_argv", "must be null or an absolute explicit argv")
                elif _dynamic_code_argument(cleanup):
                    collector.add("DYNAMIC_CODE", f"{location}.cleanup_argv", "dynamic/stdin execution is forbidden")
                elif _module_execution(cleanup):
                    collector.add("UNPINNED_SOURCE", f"{location}.cleanup_argv", "module execution cannot prove exact source bytes")
                elif _interpreter_name(cleanup[0]) != "python":
                    collector.add("LAUNCH_PROFILE", f"{location}.cleanup_argv", f"{LAUNCH_PROFILE} requires direct Python execution")
                elif not executable_source_paths(cleanup, command.get("cwd")):
                    collector.add("LAUNCH_PROFILE", f"{location}.cleanup_argv", f"{LAUNCH_PROFILE} requires one identifiable source")
                else:
                    collector.extend(_source_confinement(cleanup, cwd=command.get("cwd"), attempt_dir=attempt_dir, location=f"{location}.cleanup_argv"))
        if len(labels) != len(set(labels)):
            collector.add("DUPLICATE", "$.commands", "command labels must be unique")
        required_results = value.get("required_results", [])
        if isinstance(required_results, list):
            for index, command in enumerate(commands):
                if isinstance(command, dict) and command.get("semantic_result_path") is not None and command.get("semantic_result_path") not in required_results:
                    collector.add(
                        "SEMANTIC_BINDING",
                        f"$.commands[{index}].semantic_result_path",
                        "semantic result must also be declared in required_results",
                    )
    external = value.get("external_inventories")
    if not isinstance(external, list):
        collector.add("TYPE", "$.external_inventories", "expected list")
    else:
        names: list[str] = []
        for index, entry in enumerate(external):
            location = f"$.external_inventories[{index}]"
            _exact_fields(entry, EXTERNAL_FIELDS, location, collector)
            if not isinstance(entry, dict):
                continue
            if _string(entry.get("name"), f"{location}.name", collector, re.compile(r"[a-z0-9][a-z0-9_-]*\Z")):
                names.append(entry["name"])
            root = entry.get("root")
            if not isinstance(root, str) or not Path(root).is_absolute():
                collector.add("PATH", f"{location}.root", "external root must be absolute")
            patterns = entry.get("patterns")
            if not isinstance(patterns, list) or not patterns or not all(isinstance(item, str) and item for item in patterns):
                collector.add("PATTERN", f"{location}.patterns", "must be a nonempty string list")
            else:
                if len(patterns) != len(set(patterns)):
                    collector.add("DUPLICATE", f"{location}.patterns", "patterns must be unique")
                for pattern_index, pattern in enumerate(patterns):
                    pattern_path = PurePosixPath(pattern)
                    if pattern_path.is_absolute() or ".." in pattern_path.parts:
                        collector.add("PATTERN", f"{location}.patterns[{pattern_index}]", "pattern must be root-relative")
            expected = entry.get("expected")
            if not isinstance(expected, dict):
                collector.add("TYPE", f"{location}.expected", "expected object")
            else:
                expected_fields = {"created", "deleted", "modified"}
                _exact_fields(expected, expected_fields, f"{location}.expected", collector)
                for change_type in sorted(expected_fields):
                    paths = expected.get(change_type)
                    if not isinstance(paths, list) or not all(isinstance(item, str) and confined_relative_path(item) for item in paths):
                        collector.add("PATH", f"{location}.expected.{change_type}", "expected confined path list")
                    elif len(paths) != len(set(paths)):
                        collector.add("DUPLICATE", f"{location}.expected.{change_type}", "expected paths must be unique")
            rotation = entry.get("kit_rotation")
            if rotation is not None:
                if not isinstance(rotation, dict):
                    collector.add("TYPE", f"{location}.kit_rotation", "expected null or object")
                else:
                    rotation_fields = {"pattern", "maximum_files", "create_count", "delete_count"}
                    _exact_fields(rotation, rotation_fields, f"{location}.kit_rotation", collector)
                    if not isinstance(rotation.get("pattern"), str) or not rotation.get("pattern"):
                        collector.add("PATTERN", f"{location}.kit_rotation.pattern", "expected nonempty pattern")
                    for field in ("maximum_files", "create_count", "delete_count"):
                        count = rotation.get(field)
                        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                            collector.add("RANGE", f"{location}.kit_rotation.{field}", "expected nonnegative integer")
        if len(names) != len(set(names)):
            collector.add("DUPLICATE", "$.external_inventories", "external inventory names must be unique")
    if attempt_dir is not None and isinstance(value.get("commands"), list):
        absolute_attempt = Path(os.path.abspath(attempt_dir))
        for index, command in enumerate(value["commands"]):
            if not isinstance(command, dict):
                continue
            for field in ("argv", "cleanup_argv"):
                argv = command.get(field)
                if not isinstance(argv, list):
                    continue
                for argument_index, argument in enumerate(argv):
                    if not isinstance(argument, str):
                        continue
                    candidate = Path(argument)
                    absolute_candidate = Path(os.path.abspath(candidate))
                    if candidate.is_absolute() and (absolute_candidate == absolute_attempt or absolute_attempt in absolute_candidate.parents):
                        collector.add("ATTEMPT_LOCAL_EXECUTABLE", f"$.commands[{index}].{field}[{argument_index}]", "attempt-local executable/source is forbidden")
    return collector.findings


def validate_live_launch_profile(
    manifest: dict[str, Any],
    input_hashes: dict[str, Any],
    attempt_dir: Path,
    executing_interpreter: Path,
) -> tuple[Finding, ...]:
    """Validate live-only interpreter, source and exact-pin requirements."""

    collector = FindingCollector()
    pinned = {
        Path(row["path"]).resolve(strict=False): row
        for row in input_hashes.get("files", [])
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    expected_interpreter = executing_interpreter.resolve(strict=False)
    resolved_attempt = attempt_dir.resolve()
    for command_index, command in enumerate(manifest.get("commands", [])):
        if not isinstance(command, dict):
            continue
        for field in ("argv", "cleanup_argv"):
            argv = command.get(field)
            if argv is None or not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
                continue
            location = f"$.commands[{command_index}].{field}"
            authored_executable = Path(argv[0])
            executable = authored_executable.resolve(strict=False)
            if authored_executable.is_symlink() or not executable.is_file() or executable != expected_interpreter:
                collector.add("LAUNCH_PROFILE", f"{location}[0]", "executable must be the current regular non-symlink Python interpreter")
            _validate_live_pin(executable, pinned, collector, location, "EXECUTABLE")
            sources = executable_source_paths(argv, command.get("cwd"))
            if len(sources) != 1:
                collector.add("LAUNCH_PROFILE", location, f"{LAUNCH_PROFILE} requires exactly one identifiable source")
                continue
            index, source, was_relative = sources[0]
            authored_source = Path(argv[index])
            resolved_source = source.resolve(strict=False)
            if was_relative or not authored_source.is_absolute():
                collector.add("RELATIVE_SOURCE", f"{location}[{index}]", "source must be absolute")
            if authored_source.is_symlink():
                collector.add("SOURCE_SYMLINK", f"{location}[{index}]", "source path must not be a symlink")
            elif not resolved_source.is_file():
                collector.add("SOURCE_FILE", f"{location}[{index}]", "source must be a regular non-symlink file")
            if resolved_source == resolved_attempt or resolved_attempt in resolved_source.parents:
                collector.add("ATTEMPT_LOCAL_EXECUTABLE", f"{location}[{index}]", "attempt-local source is forbidden")
            _validate_live_pin(resolved_source, pinned, collector, f"{location}[{index}]", "SOURCE")
    return collector.findings


def _validate_live_pin(
    path: Path,
    pinned: dict[Path, dict[str, Any]],
    collector: FindingCollector,
    location: str,
    prefix: str,
) -> None:
    row = pinned.get(path)
    if row is None:
        collector.add(f"UNPINNED_{prefix}", location, f"file is not an exact input pin: {path}")
        return
    if not path.is_file() or path.is_symlink():
        collector.add(f"{prefix}_FILE", location, f"pinned path is not a regular non-symlink file: {path}")
        return
    data = path.read_bytes()
    if len(data) != row.get("bytes"):
        collector.add(f"{prefix}_SIZE", location, f"pinned size changed: {path}")
    if hashlib.sha256(data).hexdigest() != row.get("sha256"):
        collector.add(f"{prefix}_HASH", location, f"pinned hash changed: {path}")


def validate_input_hashes(value: Any, manifest: dict[str, Any] | None = None) -> tuple[Finding, ...]:
    collector = FindingCollector()
    if not _exact_fields(value, INPUT_HASH_FIELDS, "$", collector):
        if not isinstance(value, dict):
            return collector.findings
    if not isinstance(value, dict):
        return collector.findings
    if value.get("schema_version") != INPUT_HASHES_SCHEMA_VERSION:
        collector.add("SCHEMA_VERSION", "$.schema_version", f"expected {INPUT_HASHES_SCHEMA_VERSION}")
    _string(value.get("product_commit"), "$.product_commit", collector, COMMIT_RE)
    _string(value.get("tooling_commit"), "$.tooling_commit", collector, COMMIT_RE)
    files = value.get("files")
    paths: list[str] = []
    roots = [Path(item) for item in (manifest or {}).get("allowed_input_roots", []) if isinstance(item, str)]
    if not isinstance(files, list):
        collector.add("TYPE", "$.files", "expected set-labelled list")
    else:
        for index, item in enumerate(files):
            location = f"$.files[{index}]"
            _exact_fields(item, INPUT_FILE_FIELDS, location, collector)
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if isinstance(path, str):
                paths.append(path)
                if not Path(path).is_absolute() or (roots and not path_within(Path(path), roots)):
                    collector.add("PATH", f"{location}.path", "input is outside declared roots")
            else:
                collector.add("TYPE", f"{location}.path", "expected absolute path string")
            size = item.get("bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                collector.add("RANGE", f"{location}.bytes", "must be a nonnegative integer")
            _string(item.get("sha256"), f"{location}.sha256", collector, SHA256_RE)
            _string(item.get("role"), f"{location}.role", collector)
    if len(paths) != len(set(paths)):
        collector.add("DUPLICATE", "$.files", "input paths must be unique")
    if manifest is not None:
        if value.get("product_commit") != manifest.get("product_commit"):
            collector.add("PIN_MISMATCH", "$.product_commit", "does not match manifest")
        if value.get("tooling_commit") != manifest.get("tooling_commit"):
            collector.add("PIN_MISMATCH", "$.tooling_commit", "does not match manifest")
    return collector.findings
