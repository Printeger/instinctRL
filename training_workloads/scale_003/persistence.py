"""Atomic workload-owned progress, semantic, and failure preservation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .config import canonical_bytes


class PersistenceError(RuntimeError):
    pass


def _confined(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise PersistenceError("output path must be confined and relative")
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise PersistenceError("output escapes the result root")
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root and parent.exists()):
        raise PersistenceError("symlinked output path is forbidden")
    return path


def atomic_json_replace(root: Path, relative: str, value: Any) -> Path:
    path = _confined(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def atomic_torch_replace(root: Path, relative: str, value: Any) -> Path:
    """Atomically persist tensor evidence without importing torch at module import time."""
    import torch

    path = _confined(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


class ProgressStore:
    def __init__(self, result_root: Path, mode: str) -> None:
        if mode not in {"training", "station", "tracking"}:
            raise ValueError("invalid progress mode")
        self.result_root = result_root
        self.mode = mode
        self.latest: dict[str, Any] = {}

    def write(self, **fields: Any) -> None:
        self.latest = {"schema_version": "scale-003-progress.v1", "mode": self.mode, **fields}
        atomic_json_replace(self.result_root, "progress.json", self.latest)

    def fail(self, error: BaseException, **fields: Any) -> None:
        classification = "RAW_RUNTIME_FAILURE"
        if isinstance(error, (ValueError, TypeError, AssertionError)):
            classification = "SEMANTIC_CONTRACT_FAILURE"
        if isinstance(error, MemoryError) or "memory" in str(error).lower() or "oom" in str(error).lower():
            classification = "RESOURCE_SAFETY_FAILURE"
        payload = {
            "schema_version": "scale-003-failure.v1", "mode": self.mode,
            "failure_class": classification, "error_type": type(error).__name__,
            "message": str(error), **fields,
        }
        atomic_json_replace(self.result_root, "failure.json", payload)
        self.write(status="FAIL", failure_class=classification, **fields)
