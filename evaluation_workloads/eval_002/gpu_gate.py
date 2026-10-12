"""Collect and validate passive RTX 5070 coexistence windows."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from attempt_binding import binding_from_environment, bound_path, path_from_environment
from io_utils import atomic_canonical_replace, sha256_file


NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
NVIDIA_SMI_SHA256 = "22964713c1701fb62b4dd10b26b0dd25d174e100af5bda20c65e0b0fcc32b3be"


def _run_query(arguments: list[str]) -> str:
    completed = subprocess.run(
        [str(NVIDIA_SMI), *arguments], check=False, capture_output=True,
        text=True, timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed ({completed.returncode}): {completed.stderr.strip()}")
    return completed.stdout


def _integer(value: str) -> int:
    return int(value.strip().replace(" MiB", "").replace(" %", ""))


def collect_sample() -> OrderedDict[str, Any]:
    gpu_text = _run_query([
        "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]).strip()
    rows = [line for line in gpu_text.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("expected exactly one GPU")
    fields = [item.strip() for item in rows[0].split(",")]
    if len(fields) != 6:
        raise ValueError("GPU sample field count mismatch")
    apps_text = _run_query([
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]).strip()
    apps = []
    if apps_text and "No running processes found" not in apps_text:
        for line in apps_text.splitlines():
            app_fields = [item.strip() for item in line.split(",")]
            if len(app_fields) != 3:
                raise ValueError("compute application field count mismatch")
            apps.append(OrderedDict((
                ("pid", int(app_fields[0])),
                ("name", app_fields[1]),
                ("used_memory_mib", _integer(app_fields[2])),
            )))
    total = _integer(fields[3])
    used = _integer(fields[4])
    return OrderedDict((
        ("monotonic_ns", time.monotonic_ns()),
        ("gpu", OrderedDict((
            ("index", int(fields[0])), ("name", fields[1]),
            ("driver", fields[2]), ("memory_total_mib", total),
            ("memory_used_mib", used), ("free_memory_mib", total - used),
            ("utilization_percent", _integer(fields[5])),
        ))),
        ("compute_apps", apps),
    ))


def validate_sample(sample: OrderedDict[str, Any]) -> list[str]:
    gpu = sample["gpu"]
    failures = []
    if (gpu["index"], gpu["name"], gpu["driver"], gpu["memory_total_mib"]) != (0, "NVIDIA GeForce RTX 5070", "580.173.02", 12227):
        failures.append("GPU identity mismatch")
    if gpu["utilization_percent"] > 5:
        failures.append("GPU utilization exceeds 5 percent")
    if gpu["memory_used_mib"] > 1024 or gpu["free_memory_mib"] < 10240:
        failures.append("GPU memory coexistence threshold failed")
    apps = sample["compute_apps"]
    if len(apps) > 1:
        failures.append("unexpected compute application count")
    if apps:
        name = Path(apps[0]["name"]).name
        if name != "gnome-remote-desktop-daemon" or apps[0]["used_memory_mib"] > 256:
            failures.append("unexpected compute application")
    return failures


def collect_window() -> OrderedDict[str, Any]:
    samples = []
    start = time.monotonic()
    for offset in (0.0, 5.0, 10.0):
        remaining = start + offset - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        samples.append(collect_sample())
    span_ns = samples[-1]["monotonic_ns"] - samples[0]["monotonic_ns"]
    failures = []
    if span_ns < 10_000_000_000:
        failures.append("sample window spans less than ten seconds")
    for index, sample in enumerate(samples, 1):
        failures.extend(f"sample {index}: {message}" for message in validate_sample(sample))
    return OrderedDict((
        ("sample_count", 3), ("span_ns", span_ns), ("samples", samples),
        ("failures", failures), ("status", "PASS" if not failures else "FAIL"),
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, default=12)
    args = parser.parse_args()
    binding = binding_from_environment()
    output = bound_path(
        binding, path_from_environment("EVAL_GPU_GATE_OUTPUT"),
        "results/gpu_gate.json", must_exist=False,
    )
    if not 1 <= args.max_windows <= 12:
        raise ValueError("max-windows must be in [1,12]")
    windows = []
    try:
        if NVIDIA_SMI.is_symlink() or not NVIDIA_SMI.is_file() or sha256_file(NVIDIA_SMI) != NVIDIA_SMI_SHA256:
            raise ValueError("nvidia-smi identity mismatch")
        for _ in range(args.max_windows):
            window = collect_window()
            windows.append(window)
            if window["status"] == "PASS":
                result = OrderedDict((
                    ("schema_version", "eval-002-gpu-gate.v1"),
                    ("status", "PASS"), ("window_count", len(windows)),
                    ("selected_window", len(windows) - 1), ("windows", windows),
                ))
                atomic_canonical_replace(output, result)
                print(json.dumps(result, sort_keys=True))
                return 0
        raise RuntimeError("no passive GPU window passed")
    except BaseException as error:
        atomic_canonical_replace(output, {
            "failure_class": "UNSAFE_EXTERNAL_STATE",
            "message": f"{type(error).__name__}: {error}",
            "semantic_exit": 1,
            "status": "FAIL",
            "windows": windows,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
