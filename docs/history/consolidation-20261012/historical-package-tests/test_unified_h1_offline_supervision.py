import hashlib
import os
import json
import py_compile
import struct
from pathlib import Path
import tempfile
import unittest

from training_workloads.unified_h1.offline_supervision import (
    CPU_SUBSTITUTE,
    CpuSubstituteBackend,
    OfflineBoundaryError,
    PROTECTED_CONTROL_RELEASE,
    construct_offline_backend,
    construct_cpu_held_child_substitute,
    OfflineHeldChildSupervisor,
    OfflineSupervisionError,
    validate_control_snapshot,
    validate_readiness_binding,
)


REPOSITORY = Path(__file__).resolve().parents[2]
SNAPSHOT = REPOSITORY / "control_snapshot"
READINESS_INPUTS = REPOSITORY / "READINESS_INPUTS.json"


def copy_manifested_snapshot(destination):
    release = json.loads((SNAPSHOT / "release.json").read_text(encoding="utf-8"))
    for relative in ("release.json", *release["files"]):
        source = SNAPSHOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return destination


class ProtectedSnapshotBindingTests(unittest.TestCase):
    def test_exact_protected_release_and_manifested_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            release = validate_control_snapshot(copy_manifested_snapshot(Path(directory) / "snapshot"))
            self.assertEqual(release["id"], PROTECTED_CONTROL_RELEASE)
            self.assertIn("rlflow/h1.py", release["files"])
            self.assertIn("tests/test_h1_adapter.py", release["files"])

    def test_both_readiness_references_bind_the_exact_release(self):
        with tempfile.TemporaryDirectory() as directory:
            clean = copy_manifested_snapshot(Path(directory) / "snapshot")
            inputs = validate_readiness_binding(READINESS_INPUTS, clean)
            self.assertEqual(inputs["control_snapshot"]["release"], PROTECTED_CONTROL_RELEASE)
            self.assertEqual(inputs["evidence"]["latest_real_long_task_acceptance"]["control_release"], PROTECTED_CONTROL_RELEASE)

    def test_same_size_same_timestamp_malicious_bytecode_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = copy_manifested_snapshot(Path(directory) / "snapshot")
            source = copied / "rlflow" / "h1.py"
            original = source.stat()
            malicious = source.with_name("malicious.py")
            malicious.write_bytes(b"raise RuntimeError('BYTECODE EXECUTED')\n".ljust(original.st_size, b"#"))
            cache = copied / "rlflow" / "__pycache__" / "h1.cpython-312.pyc"
            cache.parent.mkdir()
            py_compile.compile(str(malicious), cfile=str(cache), dfile=str(source),
                               invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
            malicious.unlink()
            source.write_bytes((SNAPSHOT / "rlflow" / "h1.py").read_bytes())
            pyc_timestamp, pyc_size = struct.unpack("<II", cache.read_bytes()[8:16])
            self.assertEqual(pyc_size, source.stat().st_size)
            self.assertEqual(pyc_timestamp, int(source.stat().st_mtime))
            with self.assertRaisesRegex(OfflineBoundaryError, "unmanifested executable bytecode"):
                validate_control_snapshot(copied)

    def test_self_asserted_manifest_with_protected_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            payload = path / "payload.py"
            payload.write_text("arbitrary = True\n", encoding="utf-8")
            release = {
                "id": PROTECTED_CONTROL_RELEASE,
                "files": {"payload.py": hashlib.sha256(payload.read_bytes()).hexdigest()},
            }
            (path / "release.json").write_text(json.dumps(release, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(OfflineBoundaryError, "protected manifest mismatch"):
                validate_control_snapshot(path)

    def test_unmanifested_source_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "control_snapshot"
            release = json.loads((SNAPSHOT / "release.json").read_text(encoding="utf-8"))
            for relative in ("release.json", *release["files"]):
                source = SNAPSHOT / relative
                destination = copied / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
            (copied / "rlflow" / "unmanifested_backend.py").write_text(
                "raise RuntimeError('must never be imported')\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(OfflineBoundaryError, "source set drift"):
                validate_control_snapshot(copied)

    def test_unmanifested_root_source_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "control_snapshot"
            release = json.loads((SNAPSHOT / "release.json").read_text(encoding="utf-8"))
            for relative in ("release.json", *release["files"]):
                source = SNAPSHOT / relative
                destination = copied / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
            (copied / "unmanifested_backend.py").write_text(
                "raise RuntimeError('must never be imported')\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(OfflineBoundaryError, "source set drift"):
                validate_control_snapshot(copied)

    def test_unmanifested_top_level_package_source_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "control_snapshot"
            release = json.loads((SNAPSHOT / "release.json").read_text(encoding="utf-8"))
            for relative in ("release.json", *release["files"]):
                source = SNAPSHOT / relative
                destination = copied / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
            destination = copied / "unmanifested_package" / "backend.py"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("raise RuntimeError('must never be imported')\n", encoding="utf-8")
            with self.assertRaisesRegex(OfflineBoundaryError, "source set drift"):
                validate_control_snapshot(copied)

    def test_unmanifested_source_under_interpreter_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "control_snapshot"
            release = json.loads((SNAPSHOT / "release.json").read_text(encoding="utf-8"))
            for relative in ("release.json", *release["files"]):
                source = SNAPSHOT / relative
                destination = copied / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
            destination = copied / "__pycache__" / "unmanifested_backend.py"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("raise RuntimeError('must never be imported')\n", encoding="utf-8")
            with self.assertRaisesRegex(OfflineBoundaryError, "source set drift"):
                validate_control_snapshot(copied)

    def test_unmanifested_source_under_pytest_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "control_snapshot"
            release = json.loads((SNAPSHOT / "release.json").read_text(encoding="utf-8"))
            for relative in ("release.json", *release["files"]):
                source = SNAPSHOT / relative
                destination = copied / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
            destination = copied / ".pytest_cache" / "unmanifested_backend.py"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("raise RuntimeError('must never be imported')\n", encoding="utf-8")
            with self.assertRaisesRegex(OfflineBoundaryError, "source set drift"):
                validate_control_snapshot(copied)

    def test_release_identity_drift_fails_closed(self):
        release = json.loads((SNAPSHOT / "release.json").read_text(encoding="utf-8"))
        release["id"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "release.json").write_text(json.dumps(release), encoding="utf-8")
            with self.assertRaisesRegex(OfflineBoundaryError, "protected manifest mismatch"):
                validate_control_snapshot(path)

    def test_readiness_reference_drift_fails_before_snapshot_acceptance(self):
        inputs = json.loads(READINESS_INPUTS.read_text(encoding="utf-8"))
        inputs["control_snapshot"]["release"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.json"
            path.write_text(json.dumps(inputs), encoding="utf-8")
            with self.assertRaisesRegex(OfflineBoundaryError, "does not match"):
                validate_readiness_binding(path, SNAPSHOT)


class CpuOnlyConstructionBoundaryTests(unittest.TestCase):
    def test_cpu_substitute_is_the_only_constructible_backend(self):
        value = construct_offline_backend(CPU_SUBSTITUTE, fixture=True)
        self.assertIsInstance(value, CpuSubstituteBackend)
        self.assertEqual(value.fixture, {"fixture": True})

    def test_forbidden_capabilities_hard_fail(self):
        for kind in ("gpu", "AppLauncher", "simulator", "simulation", "training", "real-backend", "live"):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(OfflineBoundaryError, "only a CPU substitute"):
                    construct_offline_backend(kind)

    def test_caller_factory_cannot_spoof_cpu_substitute_label(self):
        calls = []

        def real_backend_factory():
            calls.append("constructed")

        with self.assertRaises(TypeError):
            construct_offline_backend(CPU_SUBSTITUTE, real_backend_factory)
        self.assertEqual(calls, [])

    def test_keyword_factory_capability_is_rejected_without_invocation(self):
        calls = []

        def real_backend_factory():
            calls.append("constructed")

        with self.assertRaisesRegex(OfflineBoundaryError, "callable capability"):
            construct_offline_backend(CPU_SUBSTITUTE, backend_factory=real_backend_factory)
        self.assertEqual(calls, [])

    def test_nested_non_inert_capability_is_rejected(self):
        capability = object()
        with self.assertRaisesRegex(OfflineBoundaryError, "non-inert capability"):
            construct_offline_backend(CPU_SUBSTITUTE, nested={"value": [capability]})


class HeldChildSupervisionIntegrationTests(unittest.TestCase):
    AUTHORIZATION = {
        "approval_id": "A-offline-fixture",
        "proposal_sha256": "1" * 64,
        "proposal_version": 1,
        "scope": "PACKAGE",
    }

    def supervisor(self, **fixture):
        backend = construct_cpu_held_child_substitute(**fixture)
        return backend, OfflineHeldChildSupervisor(backend, self.AUTHORIZATION)

    def test_held_child_releases_only_after_durable_identity_and_exact_recheck(self):
        backend, supervisor = self.supervisor()
        reads = []
        def authorization_reader():
            reads.append("read")
            return dict(self.AUTHORIZATION)
        def before_release(child):
            self.assertTrue(child.identity_persisted)
            self.assertFalse(child.released)
            self.assertIn(f"identity-durable:{child.child_id}", backend.events)
        result = supervisor.supervise(
            "launch-1", ("inert-cpu-fixture",), authorization_reader=authorization_reader,
            before_release=before_release,
        )
        self.assertEqual(result, 0)
        self.assertEqual(len(reads), 2)
        self.assertLess(backend.events.index("identity-durable:1000"), backend.events.index("release:1000"))

    def test_stop_fail_closed_at_every_transition_and_reclaims_only_spawned_owned_child(self):
        for stage in ("before-spawn", "before-release", "after-release"):
            with self.subTest(stage=stage):
                backend, supervisor = self.supervisor(unrelated_children={777})
                with self.assertRaisesRegex(OfflineSupervisionError, "STOP"):
                    supervisor.supervise(
                        f"launch-{stage}", ("inert",),
                        authorization_reader=lambda: dict(self.AUTHORIZATION),
                        stop_reader=lambda current, target=stage: current == target,
                    )
                self.assertEqual(backend.unrelated_children, {777})
                if stage == "before-spawn":
                    self.assertEqual(backend.owned_children, {})
                else:
                    self.assertTrue(backend.owned_children[1000].reclaimed)

    def test_nested_authorization_alias_drift_before_release_fails_closed(self):
        authorization = {**self.AUTHORIZATION, "budgets": {"wall_seconds": 10},
                         "parameters": {"curriculum": ["held"]}}
        backend = construct_cpu_held_child_substitute()
        supervisor = OfflineHeldChildSupervisor(backend, authorization)
        def mutate_original(_child):
            authorization["budgets"]["wall_seconds"] = 11
            authorization["parameters"]["curriculum"].append("released")
        with self.assertRaisesRegex(OfflineSupervisionError, "authorization changed"):
            supervisor.supervise("launch-nested", ("inert",), authorization_reader=lambda: authorization,
                                 before_release=mutate_original)
        self.assertNotIn("release:1000", backend.events)
        self.assertTrue(backend.owned_children[1000].reclaimed)

    def test_authorization_drift_before_release_fails_closed_and_reclaims(self):
        backend, supervisor = self.supervisor()
        current = dict(self.AUTHORIZATION)
        def drift(_child):
            current["proposal_version"] = 2
        with self.assertRaisesRegex(OfflineSupervisionError, "authorization changed"):
            supervisor.supervise(
                "launch-auth", ("inert",), authorization_reader=lambda: dict(current), before_release=drift,
            )
        self.assertTrue(backend.owned_children[1000].reclaimed)
        self.assertNotIn("release:1000", backend.events)

    def test_duplicate_launch_is_rejected_while_active_and_after_completion(self):
        backend, supervisor = self.supervisor()
        nested_errors = []
        def attempt_duplicate(_child):
            try:
                supervisor.supervise(
                    "launch-duplicate", ("inert",), authorization_reader=lambda: dict(self.AUTHORIZATION),
                )
            except OfflineSupervisionError as error:
                nested_errors.append(str(error))
        self.assertEqual(supervisor.supervise(
            "launch-duplicate", ("inert",), authorization_reader=lambda: dict(self.AUTHORIZATION),
            before_release=attempt_duplicate,
        ), 0)
        self.assertEqual(nested_errors, ["duplicate launch rejected"])
        with self.assertRaisesRegex(OfflineSupervisionError, "duplicate launch rejected"):
            supervisor.supervise(
                "launch-duplicate", ("inert",), authorization_reader=lambda: dict(self.AUTHORIZATION),
            )
        self.assertEqual(list(backend.owned_children), [1000])

    def test_handshake_early_exit_and_completion_timeouts_are_bounded_and_owned(self):
        cases = (
            ({"identity_after_polls": None}, "handshake timeout"),
            ({"exit_before_identity": True}, "exited before identity"),
            ({"completion_after_polls": None}, "completion timeout"),
        )
        for fixture, message in cases:
            with self.subTest(message=message):
                backend, supervisor = self.supervisor(unrelated_children={777}, **fixture)
                with self.assertRaisesRegex(OfflineSupervisionError, message) as caught:
                    supervisor.supervise(
                        f"launch-{message}", ("inert",),
                        authorization_reader=lambda: dict(self.AUTHORIZATION),
                        handshake_limit=2, completion_limit=2,
                    )
                self.assertEqual(caught.exception.owned_child_id, 1000)
                self.assertTrue(backend.owned_children[1000].reclaimed)
                self.assertEqual(backend.unrelated_children, {777})

    def test_real_backend_objects_are_rejected_before_supervision(self):
        class SpoofedBackend:
            pass
        with self.assertRaisesRegex(OfflineSupervisionError, "built-in CPU"):
            OfflineHeldChildSupervisor(SpoofedBackend(), self.AUTHORIZATION)


if __name__ == "__main__":
    unittest.main()
