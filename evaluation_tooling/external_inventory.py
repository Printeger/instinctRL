"""Content-focused, manifest-confined external inventory and attribution."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Iterable

from .findings import Finding, FindingCollector


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


@dataclass(frozen=True)
class ExternalInventorySpec:
    name: str
    root: Path
    patterns: tuple[str, ...]
    expected_created: tuple[str, ...] = ()
    expected_deleted: tuple[str, ...] = ()
    expected_modified: tuple[str, ...] = ()
    kit_rotation: dict[str, Any] | None = None

    @classmethod
    def from_manifest(cls, value: dict[str, Any]) -> "ExternalInventorySpec":
        expected = value["expected"]
        allowed = {"created", "deleted", "modified"}
        unknown = set(expected) - allowed
        missing = allowed - set(expected)
        if unknown or missing:
            raise ValueError(f"external expected fields mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")
        return cls(
            name=value["name"],
            root=Path(value["root"]),
            patterns=tuple(value["patterns"]),
            expected_created=tuple(sorted(expected["created"])),
            expected_deleted=tuple(sorted(expected["deleted"])),
            expected_modified=tuple(sorted(expected["modified"])),
            kit_rotation=value["kit_rotation"],
        )


def scan_external(spec: ExternalInventorySpec) -> dict[str, Any]:
    """Scan only one declared root and its declared patterns, without following links."""

    root = spec.root.resolve(strict=False)
    entries: list[dict[str, Any]] = []
    if root.exists() and not root.is_dir():
        raise ValueError(f"external root is not a directory: {root}")
    if root.exists():
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            symlink_directories = []
            retained_directories = []
            for name in sorted(directories):
                path = current_path / name
                if path.is_symlink():
                    symlink_directories.append(path)
                else:
                    retained_directories.append(name)
            directories[:] = retained_directories
            candidates = symlink_directories + [current_path / name for name in sorted(files)]
            for path in candidates:
                relative = path.relative_to(root).as_posix()
                if not _matches(relative, spec.patterns):
                    continue
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    entries.append({"path": relative, "type": "symlink", "target": os.readlink(path)})
                elif stat.S_ISREG(mode):
                    entries.append({"path": relative, "type": "file", "bytes": path.stat().st_size, "sha256": _sha256(path)})
                else:
                    entries.append({"path": relative, "type": "other"})
    return {
        "name": spec.name,
        "root": str(root),
        "patterns": list(spec.patterns),
        "entries": sorted(entries, key=lambda row: row["path"]),
    }


def compare_external(
    spec: ExternalInventorySpec,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    collector = FindingCollector()
    for label, inventory in (("before", before), ("after", after)):
        if inventory.get("root") != str(spec.root.resolve(strict=False)):
            collector.add("ROOT_MISMATCH", label, "inventory root differs from declaration")
        if tuple(inventory.get("patterns", [])) != spec.patterns:
            collector.add("PATTERN_MISMATCH", label, "inventory patterns differ from declaration")
    before_rows = {row["path"]: row for row in before.get("entries", [])}
    after_rows = {row["path"]: row for row in after.get("entries", [])}
    created = tuple(sorted(set(after_rows) - set(before_rows)))
    deleted = tuple(sorted(set(before_rows) - set(after_rows)))
    modified = tuple(sorted(path for path in set(before_rows) & set(after_rows) if before_rows[path] != after_rows[path]))
    actual = {"created": created, "deleted": deleted, "modified": modified}
    expected = {
        "created": spec.expected_created,
        "deleted": spec.expected_deleted,
        "modified": spec.expected_modified,
    }
    for change_type in ("created", "deleted", "modified"):
        for path in sorted(set(actual[change_type]) - set(expected[change_type])):
            collector.add("UNDECLARED_EXTERNAL_CHANGE", f"{change_type}:{path}", "change was not declared")
        for path in sorted(set(expected[change_type]) - set(actual[change_type])):
            collector.add("MISSING_EXTERNAL_CHANGE", f"{change_type}:{path}", "declared change did not occur")
        for path in expected[change_type]:
            if not _matches(path, spec.patterns):
                collector.add("UNDECLARED_PATH", f"{change_type}:{path}", "declared path is outside scan patterns")
    rotation = spec.kit_rotation
    if rotation is not None:
        allowed = {"pattern", "maximum_files", "create_count", "delete_count"}
        if set(rotation) != allowed:
            collector.add("KIT_ROTATION_SCHEMA", spec.name, "rotation fields are not exact")
        else:
            pattern = rotation["pattern"]
            created_rotation = [path for path in created if fnmatch.fnmatchcase(path, pattern)]
            deleted_rotation = [path for path in deleted if fnmatch.fnmatchcase(path, pattern)]
            retained = [path for path in after_rows if fnmatch.fnmatchcase(path, pattern)]
            if len(created_rotation) != rotation["create_count"]:
                collector.add("KIT_ROTATION_CREATE", spec.name, "created log count differs")
            if len(deleted_rotation) != rotation["delete_count"]:
                collector.add("KIT_ROTATION_DELETE", spec.name, "deleted log count differs")
            if len(retained) > rotation["maximum_files"]:
                collector.add("KIT_ROTATION_RETENTION", spec.name, "retained log count exceeds maximum")
    return {
        "status": "PASS" if collector.ok else "FAIL",
        "name": spec.name,
        "created": list(created),
        "deleted": list(deleted),
        "modified": list(modified),
        "findings": collector.as_dicts(),
    }
