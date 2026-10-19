"""CPU-testable held-child dispatcher for a future authorized local launch."""
from __future__ import annotations

import json
import hashlib
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
                  before_release: Callable[[], None] | None = None,
                  native_output_cap_bytes: int | None = None) -> int:
    if not argv or handshake_timeout_s <= 0 or completion_timeout_s <= 0:
        raise ValueError("argv and positive bounded timeouts are required")
    stop_paths = canonical_stop_paths(coordination_root)
    stop = _stop_present(stop_paths)
    if stop is not None: raise DispatchError(f"STOP present before spawn: {stop}")
    native_r = native_w = -1; native_stream = None
    native_digest = hashlib.sha256()
    native_bytes = 0; native_eof = False; native_exit = None; native_error = None
    if native_output_cap_bytes is not None:
        if type(native_output_cap_bytes) is not int or not 0 < native_output_cap_bytes <= 256*1024**2:
            raise ValueError("native output cap must be within256MiB")
        native_path = identity_path.parent / "native-output.log"
        if (identity_path.parent / "native-output.json").exists():
            raise DispatchError("native output receipt already exists")
        native_stream = native_path.open("xb", buffering=0)
        native_r, native_w = os.pipe()
        os.set_blocking(native_r, False)
    env = filtered_environment(environment); process_api = _PROCESS_API
    identity_r, identity_w = process_api.pipe(); release_r, release_w = process_api.pipe()
    pid = process_api.fork()
    if pid == 0:
        try:
            process_api.close(identity_r); process_api.close(release_w); process_api.setsid()
            if native_stream is not None:
                process_api.close(native_r); native_stream.close()
                process_api.dup2(native_w, 1); process_api.dup2(native_w, 2); process_api.close(native_w)
            if child_before_identity is not None: child_before_identity()
            identity = {"pid": process_api.getpid(), "process_group_id": process_api.getpgid(0),
                        "session_id": process_api.getsid(0), "parent_pid": process_api.getppid()}
            process_api.write(identity_w, (json.dumps(identity) + "\n").encode()); process_api.close(identity_w)
            if process_api.read(release_r, 1) != b"1": process_api._exit(126)
            process_api.close(release_r); process_api.chdir(cwd); process_api.execve(argv[0], list(argv), env)
        except BaseException: process_api._exit(126)
    process_api.close(identity_w); process_api.close(release_r)
    if native_w >= 0: os.close(native_w); native_w = -1
    pgid = pid
    selector = process_api.selector() if hasattr(process_api, "selector") else selectors.DefaultSelector()
    selector.register(identity_r, selectors.EVENT_READ)
    def drain_native():
        nonlocal native_bytes, native_eof
        if native_stream is None or native_eof: return
        try: data = os.read(native_r, 65536)
        except BlockingIOError: return
        if not data: native_eof = True; return
        available = native_output_cap_bytes - native_bytes
        accepted = data[:available]
        view = memoryview(accepted)
        while view:
            written = native_stream.write(view)
            if not written: raise DispatchError("native output short write")
            native_digest.update(view[:written]); native_bytes += written; view = view[written:]
        if len(data) > available: raise DispatchError("native output cap exceeded before further write")
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
        exited_at = None
        while True:
            drain_native()
            if native_exit is None:
                waited, status = process_api.waitpid(pid, os.WNOHANG)
                if waited == pid:
                    native_exit = os.waitstatus_to_exitcode(status); exited_at = time.monotonic()
            if native_exit is not None:
                if native_stream is None or native_eof: return native_exit
                if time.monotonic() - exited_at > 5:
                    raise DispatchError("native output pipe not closed after child exit")
            stop = _stop_present(stop_paths)
            if stop is not None: raise DispatchError(f"STOP present after release: {stop}")
            if time.monotonic() >= deadline: raise DispatchError("child completion timeout")
            time.sleep(0.01)
    except BaseException as error:
        native_error = f"{type(error).__name__}: {error}"
        if release_w >= 0:
            try: process_api.close(release_w)
            except OSError: pass
        if native_exit is None: _reclaim(pid, pgid, process_api=process_api)
        if isinstance(error, DispatchError):
            error.owned_pid = pid; error.owned_process_group_id = pgid
        raise
    finally:
        selector.close(); process_api.close(identity_r)
        if native_r >= 0: os.close(native_r)
        if native_stream is not None:
            try: native_stream.flush(); os.fsync(native_stream.fileno())
            finally: native_stream.close()
            _atomic_json(identity_path.parent / "native-output.json", {
                "schema_version":"r6-native-output/1", "path":str(native_path),
                "cap_bytes":native_output_cap_bytes, "written_bytes":native_bytes, "sha256":native_digest.hexdigest(),
                "owned_descendant_cleanup":"required from outer rlflow lifeguard receipt",
                "pipe_eof":native_eof, "native_exit_code":native_exit, "error":native_error,
                "status":"COMPLETE" if native_error is None and native_eof else "FAILED_PREFIX_PRESERVED",
                "streams":"stdout+stderr merged at pipe", "automatic_retry":False})

__all__ = ["DispatchError", "canonical_stop_paths", "dispatch_held", "filtered_environment"]
