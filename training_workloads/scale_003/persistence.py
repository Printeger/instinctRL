"""Atomic workload-owned progress, semantic, and failure preservation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

from .config import canonical_bytes


class PersistenceError(RuntimeError):
    pass


class OutputWriteQuota:
    """Conservative cumulative bytes, charged before every controlled write.

    Rewrites and failed serialization are never refunded. This bounds disk
    growth without estimating tensor archive sizes or buffering huge inputs.
    Installation is explicit; historical workloads retain their old behavior.
    """
    def __init__(self, root: Path, maximum: int):
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("output write quota must be a positive integer")
        self.root = root.resolve(); self.maximum = maximum
        self.used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        self.lock = threading.Lock()
        if self.used > maximum: raise PersistenceError("existing output exceeds write quota")

    def reserve(self, amount: int) -> None:
        if type(amount) is not int or amount < 0: raise ValueError("invalid write size")
        with self.lock:
            if self.used + amount > self.maximum:
                raise PersistenceError("authorized output write quota exhausted before write")
            self.used += amount


_output_quota: OutputWriteQuota | None = None


def install_output_write_quota(root: Path, maximum: int) -> OutputWriteQuota:
    global _output_quota
    if _output_quota is not None: raise PersistenceError("output quota is already installed")
    _output_quota = OutputWriteQuota(root, maximum)
    return _output_quota


def reserve_output_bytes(path: Path, amount: int) -> None:
    quota = _output_quota
    if quota is not None and quota.root in path.resolve().parents:
        quota.reserve(amount)


class _QuotaStream:
    def __init__(self, stream, path: Path): self.stream = stream; self.path = path
    def write(self, data):
        reserve_output_bytes(self.path, len(data))
        return self.stream.write(data)
    def __getattr__(self, name): return getattr(self.stream, name)


def guard_stream(stream, path: Path | None = None):
    if _output_quota is None: return stream
    target = path if path is not None else getattr(stream, "name", None)
    if not isinstance(target, (str, Path)):
        raise PersistenceError("quota-enabled fd stream requires an explicit output path")
    return _QuotaStream(stream, Path(target))


def save_tensor_evidence(value, path: Path):
    import torch
    if _output_quota is None:
        return torch.save(value, path)
    with path.open("wb") as stream:
        torch.save(value, guard_stream(stream))
        stream.flush(); os.fsync(stream.fileno())


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
            guard_stream(stream, Path(temporary)).write(data)
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
            torch.save(value, guard_stream(stream, Path(temporary)))
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
