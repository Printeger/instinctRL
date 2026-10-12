"""CPU-testable held-child dispatcher for a future authorized local launch."""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import signal
import time
from typing import Any, Callable, Mapping, Sequence

# exec()-loaded CPU tests may prebind an inert syscall substitute. A normal
# production import always uses the real os module.
_PROCESS_API = globals().get("_INERT_PROCESS_API", os)

_INHERITED_ENV_ALLOWLIST = frozenset({
    "HOME", "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM",
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "PYTHONHASHSEED", "PYTHONPATH",
    "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "OMNI_KIT_ACCEPT_EULA",
    "ISAACSIM_ASSET_ROOT",
    "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME",
})

class DispatchError(RuntimeError):
    pass

def canonical_stop_paths(coordination_root: Path) -> tuple[Path, ...]:
    root = coordination_root.resolve()
    return (root / "commander" / "STOP", root / "executor" / "STOP", root / "STOP", root.parent / "STOP")

def filtered_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if source is None else source
    return {str(key): str(value) for key, value in source.items() if str(key) in _INHERITED_ENV_ALLOWLIST}

def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except BaseException:
        try: temporary.unlink()
        except FileNotFoundError: pass
        raise

def _stop_present(paths: Sequence[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)

def _reclaim(pid: int, pgid: int, grace: float = 0.25, *, process_api: Any = os) -> None:
    if pgid != pid:
        raise DispatchError("refusing to reclaim a process group not created by this dispatcher")
    try:
        if process_api.getpgid(pid) != pgid:
            process_api.waitpid(pid, 0); return
    except ProcessLookupError:
        try: process_api.waitpid(pid, 0)
        except ChildProcessError: pass
        return
    try: process_api.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError: pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        waited, _ = process_api.waitpid(pid, os.WNOHANG)
        if waited == pid: return
        time.sleep(0.01)
    try: process_api.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError: pass
    try: process_api.waitpid(pid, 0)
    except ChildProcessError: pass

def dispatch_held(argv: Sequence[str], *, cwd: Path, identity_path: Path,
                  coordination_root: Path, handshake_timeout_s: float = 1.0,
                  completion_timeout_s: float = 5.0, environment: Mapping[str, str] | None = None,
                  child_before_identity: Callable[[], None] | None = None,
                  before_release: Callable[[], None] | None = None) -> int:
    if not argv or handshake_timeout_s <= 0 or completion_timeout_s <= 0:
        raise ValueError("argv and positive bounded timeouts are required")
    stop_paths = canonical_stop_paths(coordination_root)
    stop = _stop_present(stop_paths)
    if stop is not None: raise DispatchError(f"STOP present before spawn: {stop}")
    env = filtered_environment(environment); process_api = _PROCESS_API
    identity_r, identity_w = process_api.pipe(); release_r, release_w = process_api.pipe()
    pid = process_api.fork()
    if pid == 0:
        try:
            process_api.close(identity_r); process_api.close(release_w); process_api.setsid()
            if child_before_identity is not None: child_before_identity()
            identity = {"pid": process_api.getpid(), "process_group_id": process_api.getpgid(0),
                        "session_id": process_api.getsid(0), "parent_pid": process_api.getppid()}
            process_api.write(identity_w, (json.dumps(identity) + "\n").encode()); process_api.close(identity_w)
            if process_api.read(release_r, 1) != b"1": process_api._exit(126)
            process_api.close(release_r); process_api.chdir(cwd); process_api.execve(argv[0], list(argv), env)
        except BaseException: process_api._exit(126)
    process_api.close(identity_w); process_api.close(release_r)
    pgid = pid
    selector = process_api.selector() if hasattr(process_api, "selector") else selectors.DefaultSelector()
    selector.register(identity_r, selectors.EVENT_READ)
    raw = b""; deadline = time.monotonic() + handshake_timeout_s
    try:
        while b"\n" not in raw:
            remaining = deadline - time.monotonic()
            if remaining <= 0: raise DispatchError("child identity handshake timeout")
            waited, _ = process_api.waitpid(pid, os.WNOHANG)
            if waited == pid: raise DispatchError("child exited before identity handshake")
            if not selector.select(remaining): raise DispatchError("child identity handshake timeout")
            chunk = process_api.read(identity_r, 4096)
            if not chunk: raise DispatchError("child identity handshake absent")
            raw += chunk
        identity = json.loads(raw.splitlines()[0])
        if identity.get("pid") != pid or identity.get("process_group_id") != pid or identity.get("session_id") != pid:
            raise DispatchError("held child identity is not an owned session/process-group leader")
        identity_record = {"schema_version": "unified-h1-local-dispatch.v1", "held_child": identity,
            "argv": list(argv), "cwd": str(cwd), "environment_keys": sorted(env),
            "released_after_durable_record": False}
        _atomic_json(identity_path, identity_record)
        if before_release is not None: before_release()
        stop = _stop_present(stop_paths)
        if stop is not None: raise DispatchError(f"STOP present immediately before release: {stop}")
        process_api.write(release_w, b"1"); process_api.close(release_w); release_w = -1
        identity_record["released_after_durable_record"] = True; _atomic_json(identity_path, identity_record)
        deadline = time.monotonic() + completion_timeout_s
        while True:
            waited, status = process_api.waitpid(pid, os.WNOHANG)
            if waited == pid: return os.waitstatus_to_exitcode(status)
            stop = _stop_present(stop_paths)
            if stop is not None: raise DispatchError(f"STOP present after release: {stop}")
            if time.monotonic() >= deadline: raise DispatchError("child completion timeout")
            time.sleep(0.01)
    except BaseException as error:
        if release_w >= 0:
            try: process_api.close(release_w)
            except OSError: pass
        _reclaim(pid, pgid, process_api=process_api)
        if isinstance(error, DispatchError):
            error.owned_pid = pid; error.owned_process_group_id = pgid
        raise
    finally:
        selector.close(); process_api.close(identity_r)

__all__ = ["DispatchError", "canonical_stop_paths", "dispatch_held", "filtered_environment"]
