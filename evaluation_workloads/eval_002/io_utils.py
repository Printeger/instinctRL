"""Small fail-closed serialization and hashing helpers."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_canonical_replace(path: Path, value: Any) -> None:
    data = canonical_bytes(value)
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
        fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_torch_replace(path: Path, value: Any) -> None:
    """Durably replace one torch payload without exposing a partial result."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary file: {temporary}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def finite_tree(value: Any, path: str = "root") -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover - every ordered interpreter has torch
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"nonfinite tensor at {path}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite scalar at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, f"{path}[{index}]")


def canonical_value(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu().contiguous()
        return OrderedDict((
            ("tensor_dtype", str(cpu.dtype)),
            ("tensor_shape", list(cpu.shape)),
            ("tensor_data", cpu.tolist()),
        ))
    if isinstance(value, dict):
        return OrderedDict((str(key), canonical_value(child)) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return [canonical_value(child) for child in value]
    return value
