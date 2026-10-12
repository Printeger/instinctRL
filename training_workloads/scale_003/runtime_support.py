"""Attempt-confined runtime directories and IsaacLab logging configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .persistence import atomic_json_replace


def runtime_support_paths(attempt_dir: Path) -> dict[str, Path]:
    attempt = attempt_dir.resolve()
    results = (attempt / "results").resolve()
    if results.parent != attempt:
        raise ValueError("attempt results path is not confined")
    root = results / "runtime_support"
    return {"root": root, "tmp": root / "tmp", "cache": root / "cache", "logs": root / "logs"}


def runtime_environment(attempt_dir: Path) -> dict[str, str]:
    paths = runtime_support_paths(attempt_dir)
    return {
        "TMPDIR": str(paths["tmp"]),
        "TMP": str(paths["tmp"]),
        "TEMP": str(paths["tmp"]),
        "XDG_CACHE_HOME": str(paths["cache"]),
        "SCALE003_ISAACLAB_LOG_DIR": str(paths["logs"]),
    }


def prepare_runtime_support(attempt_dir: Path, *, mode: str) -> Mapping[str, str]:
    if mode not in {"training", "evaluation"}:
        raise ValueError("runtime support mode must be training or evaluation")
    paths = runtime_support_paths(attempt_dir)
    results = attempt_dir.resolve() / "results"
    if not results.is_dir() or results.is_symlink():
        raise ValueError("attempt results root must be a regular directory")
    for path in paths.values():
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise ValueError(f"runtime support path is unsafe: {path}")
        path.mkdir(parents=True, exist_ok=True)
    expected = runtime_environment(attempt_dir)
    mismatches = {name: (os.environ.get(name), value) for name, value in expected.items() if os.environ.get(name) != value}
    if mismatches:
        raise ValueError(f"runtime support environment differs from the manifest: {mismatches}")
    payload = {
        "schema_version": "scale-003-runtime-support.v1",
        "mode": mode,
        "paths": {name: str(path) for name, path in paths.items()},
        "environment": expected,
        "external_inventories_required": False,
    }
    atomic_json_replace(results, "runtime_support/configuration.json", payload)
    return payload


def configure_simulation_logging(cfg: Any, attempt_dir: Path) -> str:
    paths = runtime_support_paths(attempt_dir)
    if not hasattr(cfg, "sim") or not hasattr(cfg.sim, "log_dir"):
        raise TypeError("environment config does not expose SimulationCfg.log_dir")
    log_dir = paths["logs"].resolve()
    if paths["root"].resolve() not in log_dir.parents:
        raise ValueError("IsaacLab log directory escapes runtime support")
    cfg.sim.log_dir = str(log_dir)
    return str(log_dir)
