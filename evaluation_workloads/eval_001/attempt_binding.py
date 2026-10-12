"""Strict explicit binding for one fresh EVAL-001 attempt namespace."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


COORDINATION_ROOT = Path("/home/mint/rl_dev/instinctRL_coordination")
EVAL_ATTEMPTS_ROOT = COORDINATION_ROOT / "attempts/development/EVAL-001"
ATTEMPT_PATTERN = re.compile(r"EVAL-001-D[0-9]{4}")


@dataclass(frozen=True)
class AttemptBinding:
    attempt_id: str
    directory: Path
    attempts_root: Path


def _authored_absolute(path: Path, name: str) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be an absolute lexical path without traversal")
    return path


def _reject_symlink_components(path: Path, stop: Path, name: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{name} has a symlink component")
        if current == stop:
            return
        if stop not in current.parents:
            raise ValueError(f"{name} escapes the bound attempt")
        current = current.parent


def bind_attempt(
    attempt_id: str,
    attempt_dir: Path,
    *,
    attempts_root: Path = EVAL_ATTEMPTS_ROOT,
) -> AttemptBinding:
    """Validate an explicit current-attempt ID/directory pair."""

    if ATTEMPT_PATTERN.fullmatch(attempt_id) is None:
        raise ValueError("attempt ID must be exactly EVAL-001-D plus four decimal digits")
    root = _authored_absolute(attempts_root, "attempts root")
    directory = _authored_absolute(attempt_dir, "attempt directory")
    expected = root / attempt_id
    if directory != expected or directory.name != attempt_id or directory.parent != root:
        raise ValueError("attempt directory is not the direct matching EVAL-001 child")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("attempts root must be a regular non-symlink directory")
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("attempt directory must be a regular non-symlink directory")
    _reject_symlink_components(root, Path(root.anchor), "attempts root")
    _reject_symlink_components(directory, root, "attempt directory")
    if root.resolve(strict=True) != directory.parent.resolve(strict=True):
        raise ValueError("attempt directory resolved parent mismatch")
    if directory.resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError("attempt directory resolved identity mismatch")
    return AttemptBinding(attempt_id=attempt_id, directory=directory, attempts_root=root)


def binding_from_environment() -> AttemptBinding:
    try:
        attempt_id = os.environ["EVAL_ATTEMPT_ID"]
        attempt_dir = Path(os.environ["EVAL_ATTEMPT_DIR"])
    except KeyError as error:
        raise ValueError(f"missing exact attempt environment: {error.args[0]}") from error
    return bind_attempt(attempt_id, attempt_dir)


def path_from_environment(name: str) -> Path:
    try:
        return Path(os.environ[name])
    except KeyError as error:
        raise ValueError(f"missing exact path environment: {name}") from error


def bound_path(
    binding: AttemptBinding,
    candidate: Path,
    relative: str,
    *,
    kind: str | None = None,
    must_exist: bool,
) -> Path:
    """Require one exact authored path below the validated current attempt."""

    path = _authored_absolute(candidate, "attempt path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("invalid bound relative path")
    expected = binding.directory / relative_path
    if path != expected:
        raise ValueError("path is outside or aliased from the bound current attempt")
    _reject_symlink_components(path, binding.directory, "attempt path")
    if path.exists():
        if path.resolve(strict=True) != expected.resolve(strict=True):
            raise ValueError("attempt path resolved identity mismatch")
        if kind == "file" and not path.is_file():
            raise ValueError("attempt path must be a regular file")
        if kind == "directory" and not path.is_dir():
            raise ValueError("attempt path must be a directory")
    elif must_exist:
        raise ValueError("required attempt path does not exist")
    else:
        existing = path.parent
        while not existing.exists():
            if existing == binding.directory:
                break
            existing = existing.parent
        _reject_symlink_components(existing, binding.directory, "attempt path parent")
        resolved_parent = Path(os.path.realpath(existing))
        if binding.directory.resolve(strict=True) not in (resolved_parent, *resolved_parent.parents):
            raise ValueError("attempt path parent resolves outside current attempt")
    return path
