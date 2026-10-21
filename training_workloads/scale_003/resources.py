"""Workload-local adapter for the frozen EVAL-002 coexistence thresholds."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
import json
import stat
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


def validate_shutdown_runtime_sources(*, resource_pins: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Bind the public graceful-shutdown configuration to the frozen runtime sources."""
    pins = load_config()["pins"] if resource_pins is None else resource_pins
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


def collect_resource_sample(*, resource_pins: Mapping[str, Any] | None = None,
                            deadline_monotonic_ns: int | None = None,
                            before_query: Callable[[], None] | None = None) -> dict[str, Any]:
    pins = load_config()["pins"] if resource_pins is None else resource_pins
    executable = pins["nvidia_smi_path"]
    if sha256_file(Path(executable)) != pins["nvidia_smi_sha256"]:
        raise ValueError("nvidia-smi identity changed")
    def timeout():
        if before_query is not None: before_query()
        if deadline_monotonic_ns is None: return 15
        remaining = (deadline_monotonic_ns - time.monotonic_ns()) / 1e9
        if remaining <= 0: raise TimeoutError("resource collection deadline before query")
        return min(15, remaining)
    gpu = subprocess.run(
        [executable, "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=timeout(),
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
        check=False, capture_output=True, text=True, timeout=timeout(),
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
    if before_query is not None: before_query()
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
                       collector: Callable[[], dict[str, Any]] | None = None,
                       resource_pins: Mapping[str, Any] | None = None) -> dict[str, Any]:
    pins = load_config()["pins"] if resource_pins is None else resource_pins
    if collector is None:
        collector = (collect_coexistence_window if resource_pins is None else
            lambda: collect_coexistence_window(sample_provider=lambda: collect_resource_sample(resource_pins=resource_pins)))
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


def _r6_capture_binding() -> dict[str, Any]:
    """Require the current dispatcher-owned bounded stdout/stderr pipe."""
    try:
        binding=json.loads(os.environ['INSTINCTRL_R6_NATIVE_CAPTURE'])
        parent=os.getppid()
        parent_start=Path(f'/proc/{parent}/stat').read_text().rsplit(')',1)[1].split()[19]
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if (binding.get('schema_version')!='r6-native-capture/1'
                or type(binding.get('owner_pid')) is not int or binding['owner_pid']!=parent
                or binding.get('owner_starttime')!=parent_start or binding.get('boot_id')!=boot
                or type(binding.get('cap_bytes')) is not int or not 0<binding['cap_bytes']<=256*1024**2
                or type(binding.get('pipe_device')) is not int or type(binding.get('pipe_inode')) is not int
                or not isinstance(binding.get('path'),str) or not Path(binding['path']).is_absolute()):
            raise ValueError('capture owner or limit mismatch')
        for fd in (1,2):
            info=os.fstat(fd)
            if not stat.S_ISFIFO(info.st_mode) or (info.st_dev,info.st_ino)!=(binding['pipe_device'],binding['pipe_inode']):
                raise ValueError('capture pipe mismatch')
        return binding
    except (KeyError,ValueError,TypeError,OSError,AttributeError,IndexError) as error:
        raise RuntimeError('R6 bounded native capture required before AppLauncher') from error


def _r6_native_settings(output_root: Path, native_volume_pin=None) -> dict[str, Any]:
    """Pinned Kit setting names; no runtime import or disabled crash evidence."""
    if (not output_root.is_absolute() or output_root.resolve()!=output_root
            or output_root.is_symlink() or any(c.isspace() for c in str(output_root))):
        raise ValueError("R6 native output requires an exact whitespace-free path")
    result={"/log/file":"/proc/self/fd/1",
            "/crashreporter/dumpDir":str(output_root/"native_crash"),
            "/crashreporter/preserveDump":True,"/crashreporter/skipOldDumpUpload":True}
    if native_volume_pin is not None:
        from training_workloads.unified_h1.native_volume import validate, settings
        result.update(settings(validate(native_volume_pin,minimum_free_bytes=0)))
    return result


def _read_r6_native_settings(keys):
    # Only called after authorized AppLauncher construction, never CPU preflight.
    import carb.settings
    settings=carb.settings.get_settings()
    return {key:settings.get(key) for key in keys}


def verify_r6_native_paths(output_root: Path, native_volume_pin=None) -> None:
    capture=_r6_capture_binding()
    requested=_r6_native_settings(output_root,native_volume_pin)
    if native_volume_pin is not None:
        from training_workloads.unified_h1.native_volume import validate, require_capture
        require_capture(validate(native_volume_pin,minimum_free_bytes=0),capture)
    observed=_read_r6_native_settings(requested)
    matched=all(type(observed.get(k))is type(v) and observed.get(k)==v for k,v in requested.items())
    atomic_json_replace(output_root,'native_path_runtime.json',{
        'schema_version':'r6-native-paths/1','status':'SETTINGS_VERIFIED' if matched else 'SETTINGS_MISMATCH',
        'requested':requested,'observed':observed,'native_capture':capture,'total_native_capacity_verified':False})
    if not matched:raise RuntimeError('R6 native output settings differ after AppLauncher')


def start_after_coexistence_gate(
    output_root: Path, *, mode: str, attempt_id: str, authorization_sha256: str,
    gate_runner: Callable[..., dict[str, Any]] = run_prelaunch_gate,
    launcher_factory: Callable[[Path], Any] | None = None,
    asset_root: Path | None = None,
    resource_pins: Mapping[str, Any] | None = None,
    native_volume_pin: Mapping[str, str] | None = None,
    extension_inputs_pin: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], Any, Any]:
    """Make the resource gate a mandatory predecessor of AppLauncher creation."""
    extension_args=''
    if native_volume_pin is not None:
        if mode not in ('research-r6-evaluation','research-r6-training','research-r6-candidate-evaluation'):raise ValueError('Native volume requires exact R6 mode')
        if extension_inputs_pin is None:raise ValueError('R6 native volume requires frozen extension inputs')
        from training_workloads.unified_h1.extension_inputs import validate as validate_extensions
        extension_args=' --ext-folder '+validate_extensions(extension_inputs_pin)
        from training_workloads.unified_h1.extension_inputs import user_config_bytes
        persistent_config=user_config_bytes(extension_inputs_pin)
        from training_workloads.unified_h1.native_volume import validate, require_capture
        require_capture(validate(dict(native_volume_pin)),_r6_capture_binding())
    gate = gate_runner(
        output_root, mode=mode, attempt_id=attempt_id,
        authorization_sha256=authorization_sha256,
    )
    if resource_pins is None: validate_shutdown_runtime_sources()
    else: validate_shutdown_runtime_sources(resource_pins=resource_pins)
    kit_asset_settings: tuple[str, ...] = ()
    if asset_root is not None:
        observed = os.environ.get("ISAACSIM_ASSET_ROOT")
        if observed != str(asset_root) or not asset_root.is_absolute() or asset_root.is_symlink() or not asset_root.is_dir():
            raise RuntimeError("ISAACSIM_ASSET_ROOT must name the verified local Isaac Sim asset root")
        # IsaacLab 2.3/Isaac Sim 5.1 builds ISAAC_NUCLEUS_DIR from the
        # ``cloud`` setting. Keep ``default`` aligned so Kit and IsaacLab
        # resolve the same immutable, offline root.
        kit_asset_settings = (
            "/persistent/isaac/asset_root/default",
            "/persistent/isaac/asset_root/cloud",
        )
    kit_asset_args = "".join(f" --{setting}={asset_root}" for setting in kit_asset_settings)
    native_settings = _r6_native_settings(output_root,native_volume_pin) if mode in ('research-r6-evaluation','research-r6-training','research-r6-candidate-evaluation') else {}
    if native_settings:
        capture=_r6_capture_binding()
        paths=[Path(native_settings['/crashreporter/dumpDir'])]
        if native_volume_pin is not None:
            paths.extend(Path(native_settings[key]) for key in (
                '/rtx/shaderDb/shaderCachePath','/rtx/shaderDb/driverShaderCachePath',
                '/rtx-transient/resourcemanager/localTextureCachePath',
                '/app/extensions/registryCacheFull','/exts/omni.kit.registry.nucleus/cachePath',
                '/app/extensions/registryCache','/exts/omni.kit.pipapi/envPath',
                '/structuredLog/logDirectory'))
        for path in paths:path.mkdir(mode=0o700,exist_ok=False)
        if native_volume_pin is not None:
            destination=Path(native_settings['/app/userConfigPath'])
            fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as stream:
                stream.write(persistent_config);stream.flush();os.fsync(stream.fileno())
        atomic_json_replace(output_root,'native_path_request.json',{
            'schema_version':'r6-native-paths/1','status':'REQUESTED_NOT_VERIFIED',
            'settings':native_settings,'attempt_id':attempt_id,'authorization_sha256':authorization_sha256,
            'native_capture':capture,
            'extension_inputs':extension_inputs_pin,
            'total_native_capacity_verified':False})
    native_args=''.join(f" --{key}={str(value).lower() if type(value)is bool else value}" for key,value in native_settings.items() if key!='/log/file')
    if launcher_factory is None:
        from isaaclab.app import AppLauncher
        launcher_factory = lambda log: AppLauncher(
            headless=True, enable_cameras=False, fast_shutdown=False,
            kit_args=f"--/log/file={log}{kit_asset_args}{native_args}{extension_args}",
        )
    launcher = launcher_factory(Path('/proc/self/fd/1') if native_settings else output_root / "kit.log")
    if asset_root is not None:
        atomic_json_replace(output_root, "asset_root_runtime.json", {
            "schema_version":"unified-h1-local-asset-root.v2", "environment_variable":"ISAACSIM_ASSET_ROOT",
            "asset_root":str(asset_root), "kit_settings":list(kit_asset_settings),
            "isaac_nucleus_dir":str(asset_root / "Isaac"),
            "remote_fallback":False,
        })
    return gate, launcher, launcher.app


def validate_graceful_shutdown_configuration(simulation_app: Any, *, resource_pins: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fail closed unless SimulationApp is configured for a returning cleanup path."""
    config = getattr(simulation_app, "config", None)
    if not isinstance(config, Mapping) or config.get("fast_shutdown") is not False:
        raise RuntimeError("SimulationApp fast_shutdown must be explicitly disabled")
    return {
        "fast_shutdown": False,
        "shutdown_mode": "GRACEFUL_RETURN_REQUIRED",
        "runtime_sources": (validate_shutdown_runtime_sources() if resource_pins is None else
                            validate_shutdown_runtime_sources(resource_pins=resource_pins)),
    }
