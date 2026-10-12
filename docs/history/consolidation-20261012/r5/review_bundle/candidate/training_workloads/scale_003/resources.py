"""Workload-local adapter for the frozen EVAL-002 coexistence thresholds."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from .config import load_config, sha256_file
from .persistence import atomic_json_replace


GPU_IDENTITY = {"index": 0, "name": "NVIDIA GeForce RTX 5070", "driver": "580.173.02", "memory_total_mib": 12227}
PRELAUNCH_MAX_UTILIZATION_PERCENT = 5
PRELAUNCH_MAX_USED_MEMORY_MIB = 1024
PRELAUNCH_MIN_FREE_MEMORY_MIB = 10240
RUNTIME_MIN_FREE_MEMORY_BYTES = 1024 ** 3
ALLOWED_COMPUTE_APP = "gnome-remote-desktop-daemon"
ALLOWED_COMPUTE_APP_MAX_MIB = 256


def validate_shutdown_runtime_sources() -> dict[str, dict[str, Any]]:
    """Bind the public graceful-shutdown configuration to the frozen runtime sources."""
    pins = load_config()["pins"]
    result: dict[str, dict[str, Any]] = {}
    for name in ("app_launcher", "simulation_app"):
        path = Path(pins[f"{name}_path"])
        expected = pins[f"{name}_sha256"]
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen {name} source identity changed")
        result[name] = {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}
    return result


def _integer(value: str) -> int:
    return int(value.strip().replace(" MiB", "").replace(" %", ""))


def collect_resource_sample() -> dict[str, Any]:
    config = load_config(); executable = config["pins"]["nvidia_smi_path"]
    gpu = subprocess.run(
        [executable, "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=15,
    )
    if gpu.returncode != 0:
        raise RuntimeError(f"nvidia-smi GPU query failed ({gpu.returncode}): {gpu.stderr.strip()}")
    rows = [line for line in gpu.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("coexistence gate requires exactly one GPU")
    fields = [item.strip() for item in rows[0].split(",")]
    if len(fields) != 6:
        raise ValueError("GPU sample field count mismatch")
    apps_result = subprocess.run(
        [executable, "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=15,
    )
    if apps_result.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute-app query failed ({apps_result.returncode}): {apps_result.stderr.strip()}")
    apps = []
    text = apps_result.stdout.strip()
    if text and "No running processes found" not in text:
        for line in text.splitlines():
            values = [item.strip() for item in line.split(",")]
            if len(values) != 3:
                raise ValueError("compute application field count mismatch")
            apps.append({"pid": int(values[0]), "name": values[1], "used_memory_mib": _integer(values[2])})
    total = _integer(fields[3]); used = _integer(fields[4])
    return {
        "monotonic_ns": time.monotonic_ns(),
        "gpu": {"index": int(fields[0]), "name": fields[1], "driver": fields[2],
                "memory_total_mib": total, "memory_used_mib": used,
                "free_memory_mib": total - used, "utilization_percent": _integer(fields[5])},
        "compute_apps": apps,
    }


def validate_coexistence_window(window: Mapping[str, Any]) -> dict[str, Any]:
    if set(window) != {"sample_count", "span_ns", "samples"} or window["sample_count"] != 3:
        raise ValueError("coexistence window must contain exactly three samples")
    samples = window["samples"]
    if not isinstance(samples, list) or len(samples) != 3 or type(window["span_ns"]) is not int or window["span_ns"] < 10_000_000_000:
        raise ValueError("coexistence window must span at least ten seconds")
    prior = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or set(sample) != {"monotonic_ns", "gpu", "compute_apps"}:
            raise ValueError(f"resource sample {index + 1} schema mismatch")
        stamp = sample["monotonic_ns"]
        if type(stamp) is not int or (prior is not None and stamp <= prior):
            raise ValueError("resource sample timestamps must be strictly increasing")
        prior = stamp
        gpu = sample["gpu"]
        required_gpu = {*GPU_IDENTITY, "memory_used_mib", "free_memory_mib", "utilization_percent"}
        if not isinstance(gpu, Mapping) or set(gpu) != required_gpu or any(gpu[key] != value for key, value in GPU_IDENTITY.items()):
            raise ValueError("GPU identity mismatch")
        for key in ("memory_used_mib", "free_memory_mib", "utilization_percent"):
            if type(gpu[key]) is not int or gpu[key] < 0:
                raise ValueError("GPU resource value is not a nonnegative integer")
        if gpu["memory_used_mib"] + gpu["free_memory_mib"] != gpu["memory_total_mib"]:
            raise ValueError("GPU used/free accounting mismatch")
        if gpu["utilization_percent"] > PRELAUNCH_MAX_UTILIZATION_PERCENT:
            raise RuntimeError("GPU utilization exceeds 5 percent")
        if gpu["memory_used_mib"] > PRELAUNCH_MAX_USED_MEMORY_MIB or gpu["free_memory_mib"] < PRELAUNCH_MIN_FREE_MEMORY_MIB:
            raise MemoryError("GPU memory coexistence threshold failed")
        apps = sample["compute_apps"]
        if not isinstance(apps, list) or len(apps) > 1:
            raise RuntimeError("unexpected compute application count")
        if apps:
            app = apps[0]
            if (not isinstance(app, Mapping) or set(app) != {"pid", "name", "used_memory_mib"}
                    or type(app["pid"]) is not int or Path(str(app["name"])).name != ALLOWED_COMPUTE_APP
                    or type(app["used_memory_mib"]) is not int or app["used_memory_mib"] > ALLOWED_COMPUTE_APP_MAX_MIB):
                raise RuntimeError("unexpected compute application")
    if samples[-1]["monotonic_ns"] - samples[0]["monotonic_ns"] != window["span_ns"]:
        raise ValueError("coexistence span does not bind sample timestamps")
    return {"schema_version": "scale-003-coexistence-window.v1", "status": "PASS", **dict(window)}


def collect_coexistence_window(
    sample_provider: Callable[[], dict[str, Any]] = collect_resource_sample,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    # Anchor the ten-second window after the first sample is complete.  The
    # sample provider performs two nvidia-smi calls before recording its
    # timestamp, so anchoring at function entry can make the first-to-last
    # timestamp span shorter than ten seconds even when the final sample is
    # requested at entry + 10s.
    samples = [sample_provider()]; start = monotonic()
    for offset in (5.0, 10.0):
        remaining = start + offset - monotonic()
        if remaining > 0:
            sleeper(remaining)
        samples.append(sample_provider())
    # Validation remains in run_prelaunch_gate so a rejected real window can
    # be persisted verbatim before the error escapes.
    return {
        "sample_count": 3, "span_ns": samples[-1]["monotonic_ns"] - samples[0]["monotonic_ns"], "samples": samples,
    }


def run_prelaunch_gate(output_root: Path, *, mode: str, attempt_id: str, authorization_sha256: str,
                       collector: Callable[[], dict[str, Any]] = collect_coexistence_window) -> dict[str, Any]:
    config = load_config(); pins = config["pins"]
    window: Mapping[str, Any] | None = None
    try:
        if sha256_file(Path(pins["eval_002_gpu_gate_path"])) != pins["eval_002_gpu_gate_sha256"]:
            raise ValueError("frozen EVAL-002 coexistence gate source changed")
        executable = Path(pins["nvidia_smi_path"])
        if executable.is_symlink() or not executable.is_file() or sha256_file(executable) != pins["nvidia_smi_sha256"]:
            raise ValueError("nvidia-smi identity changed")
        window = collector()
        validated = validate_coexistence_window({key: window[key] for key in ("sample_count", "span_ns", "samples")})
        result = {**validated, "mode": mode, "attempt_id": attempt_id, "pid": os.getpid(),
                  "authorization_sha256": authorization_sha256,
                  "frozen_gate_source_sha256": pins["eval_002_gpu_gate_sha256"],
                  "nvidia_smi_sha256": pins["nvidia_smi_sha256"]}
        atomic_json_replace(output_root, "prelaunch_resource_gate.json", result)
        return result
    except BaseException as error:
        failure = {
            "schema_version": "scale-003-resource-failure.v1", "status": "FAIL",
            "mode": mode, "attempt_id": attempt_id, "pid": os.getpid(),
            "authorization_sha256": authorization_sha256, "error_type": type(error).__name__, "message": str(error),
        }
        if isinstance(window, Mapping):
            failure["observed_window"] = dict(window)
        atomic_json_replace(output_root, "prelaunch_resource_gate_failure.json", failure)
        raise


class RuntimeResourceGuard:
    """Persist every post-launch free-memory decision before new work begins."""

    def __init__(self, output_root: Path, observer: Callable[[], Mapping[str, int]] | None = None) -> None:
        self.output_root = output_root
        self.observer = observer or self._torch_observer
        self.rows: list[dict[str, Any]] = []

    @staticmethod
    def _torch_observer() -> Mapping[str, int]:
        import torch
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {"free_memory_bytes": int(free), "total_memory_bytes": int(total),
                "allocated_memory_bytes": int(torch.cuda.memory_allocated())}

    def check(self, *, stage: str, ordinal: int) -> dict[str, Any]:
        raw = self.observer()
        if set(raw) != {"free_memory_bytes", "total_memory_bytes", "allocated_memory_bytes"}:
            raise ValueError("runtime resource observer schema mismatch")
        if any(type(raw[key]) is not int or raw[key] < 0 for key in raw):
            raise ValueError("runtime resource observation must contain nonnegative integers")
        row = {"stage": stage, "ordinal": ordinal, **dict(raw), "minimum_free_memory_bytes": RUNTIME_MIN_FREE_MEMORY_BYTES}
        self.rows.append(row)
        atomic_json_replace(self.output_root, "runtime_resources.json", {
            "schema_version": "scale-003-runtime-resources.v1", "observations": self.rows,
        })
        if raw["free_memory_bytes"] < RUNTIME_MIN_FREE_MEMORY_BYTES:
            atomic_json_replace(self.output_root, "runtime_resource_failure.json", {
                "schema_version": "scale-003-resource-failure.v1", "status": "FAIL", **row,
            })
            raise MemoryError("runtime free CUDA memory is below 1 GiB")
        return row


def start_after_coexistence_gate(
    output_root: Path, *, mode: str, attempt_id: str, authorization_sha256: str,
    gate_runner: Callable[..., dict[str, Any]] = run_prelaunch_gate,
    launcher_factory: Callable[[Path], Any] | None = None,
) -> tuple[dict[str, Any], Any, Any]:
    """Make the resource gate a mandatory predecessor of AppLauncher creation."""
    gate = gate_runner(
        output_root, mode=mode, attempt_id=attempt_id,
        authorization_sha256=authorization_sha256,
    )
    validate_shutdown_runtime_sources()
    if launcher_factory is None:
        from isaaclab.app import AppLauncher
        launcher_factory = lambda log: AppLauncher(
            headless=True, enable_cameras=False, fast_shutdown=False,
            kit_args=f"--/log/file={log}",
        )
    launcher = launcher_factory(output_root / "kit.log")
    return gate, launcher, launcher.app


def validate_graceful_shutdown_configuration(simulation_app: Any) -> dict[str, Any]:
    """Fail closed unless SimulationApp is configured for a returning cleanup path."""
    config = getattr(simulation_app, "config", None)
    if not isinstance(config, Mapping) or config.get("fast_shutdown") is not False:
        raise RuntimeError("SimulationApp fast_shutdown must be explicitly disabled")
    return {
        "fast_shutdown": False,
        "shutdown_mode": "GRACEFUL_RETURN_REQUIRED",
        "runtime_sources": validate_shutdown_runtime_sources(),
    }
