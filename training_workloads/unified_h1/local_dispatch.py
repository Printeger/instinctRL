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

def _reclaim(pid: int, pgid: int, grace: float = 0.25, *, process_api: Any = os) -> int | None:
    """Only an unreaped direct child can authorize signaling its own group."""
    if pgid != pid:
        raise DispatchError("refusing to reclaim a process group not created by this dispatcher")
    try:
        waited, status = process_api.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return None  # Reaped elsewhere: numerical PID/PGID is no ownership proof.
    if waited == pid: return os.waitstatus_to_exitcode(status)
    def signal_child(signum):
        try:
            if process_api.getpgid(pid) == pgid: process_api.killpg(pgid,signum)
            else:
                # Setup can fail before the held child has called setsid().
                # It is still our unreaped child; never signal the parent group.
                process_api.kill(pid,signum)
        except ProcessLookupError: pass
    signal_child(signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        waited, status = process_api.waitpid(pid, os.WNOHANG)
        if waited == pid: return os.waitstatus_to_exitcode(status)
        time.sleep(0.01)
    signal_child(signal.SIGKILL)
    waited, status = process_api.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status) if waited == pid else None


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
    if native_output_cap_bytes is not None and (type(native_output_cap_bytes) is not int or not 0 < native_output_cap_bytes <= 256*1024**2):
        raise ValueError("native output cap must be within256MiB")
    env = filtered_environment(environment); process_api = _PROCESS_API
    descriptors = []; selector = None; pid = None; reaped = False
    native_r = native_w = release_w = -1; native_stream = None
    native_bytes = 0; native_eof = False; native_exit = None; native_error = None
    cleanup_errors = []; native_digest = hashlib.sha256()
    native_capture = None
    native_path = identity_path.parent / "native-output.log"

    def pipe(api):
        pair = api.pipe(); descriptors.extend((fd,api) for fd in pair); return pair
    def close(fd,api):
        if (fd,api) in descriptors:
            try: api.close(fd)
            finally: descriptors.remove((fd,api))
    def drain_native(*, fail_overflow=True):
        nonlocal native_bytes, native_eof
        if native_stream is None or native_eof: return False
        try: data = os.read(native_r,65536)
        except BlockingIOError: return False
        if not data: native_eof = True; return False
        available = native_output_cap_bytes-native_bytes
        view = memoryview(data[:available])
        while view:
            written = native_stream.write(view)
            if not written: raise DispatchError("native output short write")
            native_digest.update(view[:written]); native_bytes += written; view = view[written:]
        if len(data)>available and fail_overflow:
            raise DispatchError("native output cap exceeded before further write")
        return True

    try:
        if native_output_cap_bytes is not None:
            if (identity_path.parent/'native-output.json').exists():
                raise DispatchError("native output receipt already exists")
            native_stream = native_path.open('xb',buffering=0)
            native_r,native_w = pipe(os); os.set_blocking(native_r,False)
            pipe_info = os.fstat(native_w)
            native_capture = dict(schema_version='r6-native-capture/1',
                owner_pid=os.getpid(),
                owner_starttime=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pipe_device=pipe_info.st_dev,pipe_inode=pipe_info.st_ino,
                cap_bytes=native_output_cap_bytes,path=str(native_path.resolve()))
            # This key is deliberately not inherited by filtered_environment.
            # Only this dispatcher may describe the pipe it actually bounds.
            env['INSTINCTRL_R6_NATIVE_CAPTURE']=json.dumps(native_capture,sort_keys=True)
        identity_r,identity_w = pipe(process_api)
        release_r,release_w = pipe(process_api)
        pid = process_api.fork()
        if pid == 0:
            try:
                close(identity_r,process_api); close(release_w,process_api); process_api.setsid()
                if native_stream is not None:
                    close(native_r,os); native_stream.close()
                    process_api.dup2(native_w,1); process_api.dup2(native_w,2); close(native_w,os)
                if child_before_identity is not None: child_before_identity()
                child = {"pid":process_api.getpid(),"process_group_id":process_api.getpgid(0),
                         "session_id":process_api.getsid(0),"parent_pid":process_api.getppid()}
                process_api.write(identity_w,(json.dumps(child)+"\n").encode()); close(identity_w,process_api)
                if process_api.read(release_r,1)!=b'1': process_api._exit(126)
                close(release_r,process_api); process_api.chdir(cwd); process_api.execve(argv[0],list(argv),env)
            except BaseException: process_api._exit(126)
        close(identity_w,process_api); close(release_r,process_api)
        if native_w>=0: close(native_w,os)
        selector = process_api.selector() if hasattr(process_api,'selector') else selectors.DefaultSelector()
        selector.register(identity_r,selectors.EVENT_READ)
        raw=b''; deadline=time.monotonic()+handshake_timeout_s
        while b'\n' not in raw:
            drain_native()
            remaining=deadline-time.monotonic()
            if remaining<=0: raise DispatchError('child identity handshake timeout')
            waited,status=process_api.waitpid(pid,os.WNOHANG)
            if waited==pid:
                native_exit=os.waitstatus_to_exitcode(status); reaped=True
                raise DispatchError('child exited before identity handshake')
            if not selector.select(min(remaining,.05)): continue
            chunk=process_api.read(identity_r,4096)
            if not chunk: raise DispatchError('child identity handshake absent')
            raw+=chunk
            if len(raw)>65536: raise DispatchError('child identity handshake oversized')
        child=json.loads(raw.splitlines()[0])
        if child.get('pid')!=pid or child.get('process_group_id')!=pid or child.get('session_id')!=pid:
            raise DispatchError('held child identity is not an owned session/process-group leader')
        record={"schema_version":"unified-h1-local-dispatch.v1","held_child":child,"argv":list(argv),
                "cwd":str(cwd),"environment_keys":sorted(env),"released_after_durable_record":False}
        if native_capture is not None: record['native_capture']=native_capture
        _atomic_json(identity_path,record)
        if before_release is not None: before_release()
        stop=_stop_present(stop_paths)
        if stop is not None: raise DispatchError(f'STOP present immediately before release: {stop}')
        process_api.write(release_w,b'1'); close(release_w,process_api); release_w=-1
        record['released_after_durable_record']=True; _atomic_json(identity_path,record)
        deadline=time.monotonic()+completion_timeout_s; exited_at=None
        while True:
            drain_native()
            if not reaped:
                waited,status=process_api.waitpid(pid,os.WNOHANG)
                if waited==pid:
                    native_exit=os.waitstatus_to_exitcode(status); reaped=True; exited_at=time.monotonic()
            if reaped:
                if native_stream is None or native_eof: return native_exit
                if time.monotonic()-exited_at>5: raise DispatchError('native output pipe not closed after child exit')
            stop=_stop_present(stop_paths)
            if stop is not None: raise DispatchError(f'STOP present after release: {stop}')
            if time.monotonic()>=deadline: raise DispatchError('child completion timeout')
            time.sleep(.01)
    except BaseException as error:
        native_error=f'{type(error).__name__}: {error}'
        if pid is not None and pid>0 and not reaped:
            try:
                native_exit=_reclaim(pid,pid,process_api=process_api); reaped=True
            except BaseException as cleanup_error: cleanup_errors.append(str(cleanup_error))
        if isinstance(error,DispatchError):
            error.owned_pid=pid; error.owned_process_group_id=pid; error.native_exit_code=native_exit
        raise
    finally:
        # Closing our writer endpoints is essential on fork/selector failure.
        for fd,api in list(descriptors):
            if fd!=native_r or api is not os:
                try: close(fd,api)
                except OSError as e: cleanup_errors.append(str(e))
        if native_stream is not None and native_r>=0:
            deadline=time.monotonic()+1
            try:
                while not native_eof and native_bytes<native_output_cap_bytes and time.monotonic()<deadline:
                    if not drain_native(fail_overflow=False): time.sleep(.005)
            except BaseException as e: cleanup_errors.append(str(e))
        for fd,api in list(descriptors):
            try: close(fd,api)
            except OSError as e: cleanup_errors.append(str(e))
        if selector is not None:
            try: selector.close()
            except BaseException as e: cleanup_errors.append(str(e))
        if native_stream is not None:
            try: native_stream.flush(); os.fsync(native_stream.fileno())
            except BaseException as e: cleanup_errors.append(str(e))
            finally: native_stream.close()
            _atomic_json(identity_path.parent/'native-output.json',{
                'schema_version':'r6-native-output/1','path':str(native_path),'cap_bytes':native_output_cap_bytes,
                'written_bytes':native_bytes,'sha256':native_digest.hexdigest(),'pipe_eof':native_eof,
                'native_exit_code':native_exit,'error':native_error,'cleanup_errors':cleanup_errors,
                'status':'COMPLETE' if native_error is None and native_eof and not cleanup_errors else 'FAILED_PREFIX_PRESERVED',
                'owned_descendant_cleanup':'required from outer rlflow lifeguard receipt',
                'streams':'stdout+stderr merged at pipe','automatic_retry':False})
            if cleanup_errors and native_error is None:
                raise DispatchError('native output cleanup/persistence failed: '+str(cleanup_errors))

__all__ = ["DispatchError", "canonical_stop_paths", "dispatch_held", "filtered_environment"]
