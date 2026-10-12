"""Fail-closed CPU-only boundary for offline H1 supervision validation.

This module does not import rlflow, a GPU runtime, AppLauncher, the simulator,
or a training backend.  It verifies the protected control snapshot bytes before
allowing an explicitly labelled CPU substitute to be constructed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

PROTECTED_CONTROL_RELEASE = "43d44b3f98820f792e516c41e1f7d7f3bae6640db05ab56f39cd5d2102bc497f"
PROTECTED_RELEASE_MANIFEST_SHA256 = "be55e12a81f8eab9790c5a7021d9f37ca537467313319d5c45d5f4e30ab38c16"
CPU_SUBSTITUTE = "cpu-substitute"


class OfflineBoundaryError(RuntimeError):
    """The offline supervision boundary is absent, drifted, or bypassed."""


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _regular_file_beneath(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise OfflineBoundaryError("control snapshot manifest contains an unsafe path")
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise OfflineBoundaryError(f"control snapshot file is unavailable: {relative}") from error
    if path.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(root):
        raise OfflineBoundaryError(f"control snapshot file escaped its protected root: {relative}")
    return resolved


def validate_control_snapshot(snapshot_root: Path) -> Mapping[str, Any]:
    """Verify the exact protected rlflow release and every manifested byte."""
    root = Path(snapshot_root)
    if not root.is_absolute() or root.is_symlink():
        raise OfflineBoundaryError("control snapshot root must be an exact absolute directory")
    try:
        root = root.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise OfflineBoundaryError("control snapshot root is unavailable") from error
    if not root.is_dir():
        raise OfflineBoundaryError("control snapshot root is not a directory")
    release_path = _regular_file_beneath(root, "release.json")
    if _digest(release_path) != PROTECTED_RELEASE_MANIFEST_SHA256:
        raise OfflineBoundaryError("control snapshot protected manifest mismatch")
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OfflineBoundaryError("control snapshot release metadata is invalid") from error
    if not isinstance(release, dict) or set(release) != {"files", "id"}:
        raise OfflineBoundaryError("control snapshot release schema mismatch")
    if release["id"] != PROTECTED_CONTROL_RELEASE:
        raise OfflineBoundaryError("control snapshot release identity mismatch")
    files = release["files"]
    if not isinstance(files, dict) or not files:
        raise OfflineBoundaryError("control snapshot release manifest is empty")

    # The signed manifest fixes both the bytes and the complete source-file
    # namespace.  Enumerate from the snapshot root so root-level additions and
    # entirely new top-level directories cannot escape coverage.  release.json
    # authenticates the manifest and is therefore the sole non-manifested file.
    # Interpreter caches are runtime products rather than snapshot source.
    manifest_paths = set(files)
    actual_paths: set[str] = set()
    for candidate in root.rglob("*"):
        relative_path = candidate.relative_to(root)
        # Ignore generated cache *artifacts*, never a directory wholesale:
        # source planted beneath a cache-named directory is still drift.
        parts = relative_path.parts
        is_interpreter_cache = (
            "__pycache__" in parts and candidate.suffix in {".pyc", ".pyo"}
        )
        pytest_relative = parts[parts.index(".pytest_cache") + 1:] if ".pytest_cache" in parts else ()
        is_pytest_cache = (
            pytest_relative in {(".gitignore",), ("CACHEDIR.TAG",), ("README.md",)}
            or (len(pytest_relative) >= 3 and pytest_relative[:2] == ("v", "cache")
                and candidate.suffix not in {".py", ".pyi", ".pyx"})
        )
        if is_interpreter_cache:
            raise OfflineBoundaryError(
                f"control snapshot contains unmanifested executable bytecode: {relative_path.as_posix()}"
            )
        if is_pytest_cache:
            continue
        if candidate.is_symlink():
            raise OfflineBoundaryError(f"control snapshot source entry is a symlink: {relative_path.as_posix()}")
        if candidate.is_file() and relative_path.as_posix() != "release.json":
            actual_paths.add(relative_path.as_posix())
    if actual_paths != manifest_paths:
        added = sorted(actual_paths - manifest_paths)
        missing = sorted(manifest_paths - actual_paths)
        raise OfflineBoundaryError(
            f"control snapshot source set drift: added={added!r}, missing={missing!r}"
        )

    for relative, expected in files.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise OfflineBoundaryError(f"control snapshot digest is invalid: {relative}")
        if _digest(_regular_file_beneath(root, relative)) != expected:
            raise OfflineBoundaryError(f"control snapshot byte drift: {relative}")
    return release


def validate_readiness_binding(readiness_inputs: Path, snapshot_root: Path) -> Mapping[str, Any]:
    """Require both immutable readiness references to name the protected release."""
    path = Path(readiness_inputs)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise OfflineBoundaryError("readiness inputs must be an exact absolute regular file")
    try:
        inputs = json.loads(path.read_text(encoding="utf-8"))
        snapshot_release = inputs["tools/rlflow"]["release"]
        accepted_release = inputs["evidence"]["latest_real_long_task_acceptance"]["control_release"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise OfflineBoundaryError("readiness control binding is incomplete") from error
    if snapshot_release != PROTECTED_CONTROL_RELEASE or accepted_release != PROTECTED_CONTROL_RELEASE:
        raise OfflineBoundaryError("readiness control binding does not match the protected release")
    validate_control_snapshot(Path(snapshot_root))
    return inputs


def _validate_inert_fixture(value: Any, path: str = "fixture", active: set[int] | None = None) -> None:
    """Accept only recursively inspectable data and reject capability objects."""
    if callable(value):
        raise OfflineBoundaryError(f"offline fixture contains a callable capability: {path}")
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return
    if type(value) not in {dict, list, tuple, set, frozenset}:
        raise OfflineBoundaryError(f"offline fixture contains a non-inert capability: {path}")
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise OfflineBoundaryError(f"offline fixture contains a recursive value: {path}")
    active.add(identity)
    try:
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise OfflineBoundaryError(f"offline fixture mapping key is not text: {path}")
                _validate_inert_fixture(item, f"{path}.{key}", active)
        else:
            for index, item in enumerate(value):
                _validate_inert_fixture(item, f"{path}[{index}]", active)
    finally:
        active.remove(identity)


class CpuSubstituteBackend:
    """Inert value object used by offline tests; it has no runtime capabilities."""

    def __init__(self, **fixture: Any) -> None:
        _validate_inert_fixture(fixture)
        self.fixture = dict(fixture)


def construct_offline_backend(kind: str, /, **kwargs: Any) -> CpuSubstituteBackend:
    """Construct the built-in inert substitute; caller factories are never accepted."""
    if kind != CPU_SUBSTITUTE:
        raise OfflineBoundaryError(
            "offline supervision permits only a CPU substitute; GPU, AppLauncher, simulator, training, and real backends are forbidden"
        )
    _validate_inert_fixture(kwargs)
    return CpuSubstituteBackend(**kwargs)


class OfflineSupervisionError(OfflineBoundaryError):
    """A fail-closed transition rejected an offline held-child launch."""

    def __init__(self, message: str, *, owned_child_id: int | None = None) -> None:
        super().__init__(message)
        self.owned_child_id = owned_child_id


class OfflineHeldChild:
    """Deterministic in-memory child; it never creates or executes a process."""

    def __init__(self, child_id: int, argv: tuple[str, ...], *, identity_after_polls: int | None = 0,
                 completion_after_polls: int | None = 0, exit_before_identity: bool = False) -> None:
        self.child_id = child_id
        self.argv = argv
        self.identity_after_polls = identity_after_polls
        self.completion_after_polls = completion_after_polls
        self.exit_before_identity = exit_before_identity
        self.identity_polls = 0
        self.completion_polls = 0
        self.identity_persisted = False
        self.released = False
        self.reclaimed = False

    def poll_identity(self) -> Mapping[str, int] | None:
        if self.exit_before_identity:
            raise OfflineSupervisionError("held child exited before identity handshake", owned_child_id=self.child_id)
        poll = self.identity_polls
        self.identity_polls += 1
        if self.identity_after_polls is None or poll < self.identity_after_polls:
            return None
        return {"pid": self.child_id, "process_group_id": self.child_id, "session_id": self.child_id}

    def release(self) -> None:
        if not self.identity_persisted:
            raise OfflineSupervisionError("held child cannot release before durable identity", owned_child_id=self.child_id)
        self.released = True

    def poll_completion(self) -> int | None:
        if not self.released:
            raise OfflineSupervisionError("held child is still paused", owned_child_id=self.child_id)
        poll = self.completion_polls
        self.completion_polls += 1
        if self.completion_after_polls is None or poll < self.completion_after_polls:
            return None
        return 0


class CpuHeldChildSubstitute(CpuSubstituteBackend):
    """In-memory backend with explicit ownership and an auditable event trace."""

    def __init__(self, **fixture: Any) -> None:
        super().__init__(**fixture)
        self.events: list[str] = []
        self.identities: dict[int, Mapping[str, int]] = {}
        self.owned_children: dict[int, OfflineHeldChild] = {}
        self.unrelated_children: set[int] = set(fixture.get("unrelated_children", ()))
        self._next_child_id = int(fixture.get("first_child_id", 1000))

    def spawn_held(self, argv: tuple[str, ...]) -> OfflineHeldChild:
        child = OfflineHeldChild(
            self._next_child_id,
            argv,
            identity_after_polls=self.fixture.get("identity_after_polls", 0),
            completion_after_polls=self.fixture.get("completion_after_polls", 0),
            exit_before_identity=bool(self.fixture.get("exit_before_identity", False)),
        )
        self._next_child_id += 1
        self.owned_children[child.child_id] = child
        self.events.append(f"spawn-held:{child.child_id}")
        return child

    def persist_identity(self, child: OfflineHeldChild, identity: Mapping[str, int]) -> None:
        self.identities[child.child_id] = dict(identity)
        child.identity_persisted = True
        self.events.append(f"identity-durable:{child.child_id}")

    def reclaim(self, child: OfflineHeldChild) -> None:
        if self.owned_children.get(child.child_id) is not child:
            raise OfflineSupervisionError("refusing to reclaim a child not owned by this supervisor")
        child.reclaimed = True
        self.events.append(f"reclaim-owned:{child.child_id}")


def construct_cpu_held_child_substitute(**fixture: Any) -> CpuHeldChildSubstitute:
    """Build the only backend accepted by :class:`OfflineHeldChildSupervisor`."""
    return CpuHeldChildSubstitute(**fixture)


class OfflineHeldChildSupervisor:
    """Pure-CPU model of the protected held-child launch protocol."""

    def __init__(self, backend: CpuHeldChildSubstitute, exact_authorization: Mapping[str, Any]) -> None:
        if type(backend) is not CpuHeldChildSubstitute:
            raise OfflineSupervisionError("offline supervisor requires the built-in CPU held-child substitute")
        if not exact_authorization:
            raise OfflineSupervisionError("an exact non-empty authorization binding is required")
        self.backend = backend
        _validate_inert_fixture(exact_authorization, "exact_authorization")
        # The expected value must not share nested containers with mutable
        # control state supplied by the caller.
        self.exact_authorization = copy.deepcopy(dict(exact_authorization))
        self._launch_ids: set[str] = set()

    def supervise(self, launch_id: str, argv: tuple[str, ...], *,
                  authorization_reader: Any, stop_reader: Any = lambda stage: False,
                  before_release: Any = None, handshake_limit: int = 3,
                  completion_limit: int = 3) -> int:
        if not launch_id or not argv or handshake_limit <= 0 or completion_limit <= 0:
            raise ValueError("launch id, argv, and positive bounded limits are required")
        if stop_reader("before-spawn"):
            raise OfflineSupervisionError("STOP present before spawn")
        if dict(authorization_reader()) != self.exact_authorization:
            raise OfflineSupervisionError("exact authorization mismatch before spawn")
        if launch_id in self._launch_ids:
            raise OfflineSupervisionError("duplicate launch rejected")
        self._launch_ids.add(launch_id)
        child = self.backend.spawn_held(tuple(argv))
        try:
            identity = None
            for _ in range(handshake_limit):
                identity = child.poll_identity()
                if identity is not None:
                    break
            if identity is None:
                raise OfflineSupervisionError("held child identity handshake timeout", owned_child_id=child.child_id)
            expected_identity = {"pid": child.child_id, "process_group_id": child.child_id, "session_id": child.child_id}
            if dict(identity) != expected_identity:
                raise OfflineSupervisionError("held child identity is not the owned session", owned_child_id=child.child_id)
            self.backend.persist_identity(child, identity)
            if before_release is not None:
                before_release(child)
            if stop_reader("before-release"):
                raise OfflineSupervisionError("STOP present immediately before release", owned_child_id=child.child_id)
            if dict(authorization_reader()) != self.exact_authorization:
                raise OfflineSupervisionError("exact authorization changed before release", owned_child_id=child.child_id)
            child.release()
            self.backend.events.append(f"release:{child.child_id}")
            for _ in range(completion_limit):
                if stop_reader("after-release"):
                    raise OfflineSupervisionError("STOP present after release", owned_child_id=child.child_id)
                result = child.poll_completion()
                if result is not None:
                    self.backend.events.append(f"complete:{child.child_id}")
                    return result
            raise OfflineSupervisionError("held child completion timeout", owned_child_id=child.child_id)
        except BaseException:
            self.backend.reclaim(child)
            raise


__all__ = [
    "CPU_SUBSTITUTE", "CpuSubstituteBackend", "OfflineBoundaryError", "PROTECTED_CONTROL_RELEASE",
    "PROTECTED_RELEASE_MANIFEST_SHA256",
    "construct_offline_backend", "validate_control_snapshot", "validate_readiness_binding",
    "CpuHeldChildSubstitute", "OfflineHeldChild", "OfflineHeldChildSupervisor",
    "OfflineSupervisionError", "construct_cpu_held_child_substitute",
]
