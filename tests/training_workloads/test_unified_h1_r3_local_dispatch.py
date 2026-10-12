import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import hashlib
import sys
import subprocess

from training_workloads.unified_h1 import budget as budget_module
from training_workloads.unified_h1 import local_dispatch
from training_workloads.unified_h1.budget import (
    OfflineResourceSettlementLedger, RESOURCE_PRODUCER_IDENTITY, RESOURCE_PRODUCER_KEY_ID,
    RESOURCE_SOURCE_MANIFEST_SCHEMA, ResourceSettlementError,
)
from training_workloads.unified_h1.local_dispatch import DispatchError, filtered_environment


TEST_RESOURCE_RSA_N = 5692905827857634435890480251706180638328854894874122768253066260460561868460070892151807011167977405807964856570723917615489618358056329013483500298692979
TEST_RESOURCE_RSA_D = 3361087709191684837998540555400266222726984488874443936585682787066855675058664240229333431597782486408404782390272523988203132532361716865252265814778153


def trusted_resource_signature(manifest):
    fields = ("schema_version", "producer_identity", "producer_key_id", "candidate",
              "observation_id", "launch_identity", "measurement_context", "dimensions")
    payload = {key: manifest[key] for key in fields}
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest_info = bytes.fromhex("3031300d060960864801650304020105000420") + hashlib.sha256(data).digest()
    size = (TEST_RESOURCE_RSA_N.bit_length() + 7) // 8
    encoded = b"\x00\x01" + b"\xff" * (size - len(digest_info) - 3) + b"\x00" + digest_info
    return pow(int.from_bytes(encoded, "big"), TEST_RESOURCE_RSA_D,
               TEST_RESOURCE_RSA_N).to_bytes(size, "big").hex()


class InertSelector:
    def __init__(self, api): self.api = api
    def register(self, *_): pass
    def select(self, _timeout): return [(object(), 1)] if self.api.identity_mode == "ready" else []
    def close(self): pass


class InertProcessApi:
    """Deterministic syscall substitute: no process or executable is created."""
    def __init__(self, *, identity_mode="ready", complete=True, on_release=None):
        self.pid = 4242; self.identity_mode = identity_mode; self.complete = complete
        self.on_release = on_release; self.released = False; self.killed = False
        self.signalled_groups = []; self.pipe_count = 0; self.identity_read = False
    def pipe(self):
        base = 10 + self.pipe_count * 2; self.pipe_count += 1; return base, base + 1
    def fork(self): return self.pid
    def close(self, _fd): pass
    def selector(self): return InertSelector(self)
    def waitpid(self, pid, _flags):
        if self.identity_mode == "early-exit" and not self.identity_read: return pid, 23 << 8
        if self.killed or (self.released and self.complete): return pid, 0
        return 0, 0
    def read(self, fd, _size):
        if fd == 10 and self.identity_mode == "ready" and not self.identity_read:
            self.identity_read = True
            return (json.dumps({"pid": self.pid, "process_group_id": self.pid,
                                "session_id": self.pid, "parent_pid": 1}) + "\n").encode()
        return b""
    def write(self, fd, data):
        if fd == 13 and data == b"1":
            self.released = True
            if self.on_release: self.on_release()
        return len(data)
    def getpgid(self, pid): return self.pid
    def killpg(self, pgid, _signal):
        self.signalled_groups.append(pgid); self.killed = True


class EnvironmentFilterTests(unittest.TestCase):
    def test_explicit_allowlist_preserves_home_and_rejects_credentials(self):
        source = {"HOME": "/inherited", "PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "",
                  "ISAACSIM_ASSET_ROOT": "/frozen/Assets/Isaac/5.1",
                  "AWS_ACCESS_KEY_ID": "x", "GITHUB_TOKEN": "x", "MY_TOKEN": "x"}
        self.assertEqual(filtered_environment(source),
                         {"HOME": "/inherited", "PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "",
                          "ISAACSIM_ASSET_ROOT": "/frozen/Assets/Isaac/5.1"})


class InertProductionDispatcherTests(unittest.TestCase):
    def dispatch(self, directory, api, **kwargs):
        root = Path(directory); coordination = root / "coordination"; coordination.mkdir(exist_ok=True)
        previous = local_dispatch._PROCESS_API; local_dispatch._PROCESS_API = api
        try:
            return local_dispatch.dispatch_held(
                ("inert-cpu-fixture",), cwd=root, identity_path=root / "identity.json",
                coordination_root=coordination, environment={"HOME": "/cpu", "MY_TOKEN": "secret"},
                **kwargs)
        finally:
            local_dispatch._PROCESS_API = previous

    def test_identity_is_durable_before_release_and_environment_is_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            observed = []
            def gate():
                record = json.loads((Path(directory) / "identity.json").read_text())
                observed.append(record["released_after_durable_record"])
            api = InertProcessApi()
            self.assertEqual(self.dispatch(directory, api, before_release=gate), 0)
            self.assertEqual(observed, [False]); self.assertTrue(api.released)
            record = json.loads((Path(directory) / "identity.json").read_text())
            self.assertTrue(record["released_after_durable_record"])
            self.assertEqual(record["environment_keys"], ["HOME"])

    def test_stop_before_spawn_and_at_both_release_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); stop = root / "coordination" / "STOP"
            (root / "coordination").mkdir(); stop.touch()
            with self.assertRaisesRegex(DispatchError, "before spawn"):
                self.dispatch(directory, InertProcessApi())
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "coordination" / "STOP"; api = InertProcessApi()
            with self.assertRaisesRegex(DispatchError, "immediately before release"):
                self.dispatch(directory, api, before_release=lambda: stop.touch())
            self.assertEqual(api.signalled_groups, [api.pid]); self.assertFalse(api.released)
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "coordination" / "STOP"
            api = InertProcessApi(complete=False, on_release=lambda: stop.touch())
            with self.assertRaisesRegex(DispatchError, "after release"):
                self.dispatch(directory, api)
            self.assertEqual(api.signalled_groups, [api.pid])

    def test_authorization_drift_in_protected_style_gate_prevents_release(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = {"budgets": {"wall_seconds": 10}, "parameters": {"mode": "held"}}
            current = json.loads(json.dumps(expected))
            def protected_gate():
                current["budgets"]["wall_seconds"] = 11
                if current != expected: raise RuntimeError("authorization changed before release")
            api = InertProcessApi()
            with self.assertRaisesRegex(RuntimeError, "authorization changed"):
                self.dispatch(directory, api, before_release=protected_gate)
            self.assertFalse(api.released); self.assertEqual(api.signalled_groups, [api.pid])

    def test_handshake_early_exit_and_completion_timeout_are_bounded_and_owned(self):
        for api, message in ((InertProcessApi(identity_mode="absent"), "handshake timeout"),
                             (InertProcessApi(identity_mode="early-exit"), "exited before identity"),
                             (InertProcessApi(complete=False), "completion timeout")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(DispatchError, message) as caught:
                    self.dispatch(directory, api, handshake_timeout_s=.01, completion_timeout_s=.01)
                self.assertEqual(caught.exception.owned_pid, api.pid)
                self.assertEqual(api.signalled_groups, [api.pid])

    def test_reclaim_refuses_nonowned_group(self):
        api = InertProcessApi()
        with self.assertRaisesRegex(DispatchError, "not created"):
            local_dispatch._reclaim(api.pid, api.pid + 1, process_api=api)
        self.assertEqual(api.signalled_groups, [])


class ProtectedExperimentWorkerGateTests(unittest.TestCase):
    def run_worker(self, directory, *, drift=False, cleanup_failure=False):
        control = Path(__file__).resolve().parents[2] / "tools/rlflow"
        sys.path.insert(0, str(control))
        try:
            from rlflow import experiments, isolation, lifeguard, releases
        finally:
            sys.path.remove(str(control))
        root = Path(directory) / "state"; root.mkdir()
        job = root / "jobs" / "J-offline"; job.mkdir(parents=True)
        dispatcher = Path(local_dispatch.__file__).resolve()
        spec = {"backend": "runner_cpu", "argv": ["inert-cpu-fixture"], "cwd": directory,
                "resources": {"wall_seconds": 1},
                "pins": {"dispatcher": {"path": str(dispatcher),
                          "sha256": hashlib.sha256(dispatcher.read_bytes()).hexdigest()},
                         "product": {"path": directory}, "workload": {"path": directory}}}
        request = {"root": str(root), "job_id": "J-offline", "spec": spec}
        request_path = job / "request.json"; request_path.write_text(json.dumps(request))
        fake_identity = lambda pid: {"pid": pid, "pgid": pid, "starttime": "inert", "boot_id": "inert"}
        (job / "supervisor-release.json").write_text(json.dumps({
            "job_id": "J-offline", "identity": fake_identity(os.getpid()), "control_release": "43d44b3f"}))
        calls = []; events = []
        def authorized(_state, _spec, _root):
            calls.append("authorized")
            if drift and len(calls) == 2: raise experiments.Conflict("nested authorization drift")
            return {"approval_id": "A-offline"}
        class Store:
            def __init__(self, _root): self.root = root
            def replay(self): return {"authorization": {"budgets": {"wall_seconds": 1}}}
        class Guard:
            def wait(self, timeout): self.timeout = timeout
        def await_cleanup(directory, guard, owner):
            self.assertTrue((Path(directory) / "lifeguard-done").exists())
            self.assertFalse((Path(directory) / "receipt.json").exists())
            events.append("guardian-ack")
            if cleanup_failure: raise RuntimeError("forced cleanup failure")
            return {"no_survivors": True, "owner": owner}
        api = InertProcessApi()
        real_exec = __builtins__["exec"] if isinstance(__builtins__, dict) else __builtins__.exec
        def injecting_exec(code, globals_dict, locals_dict=None):
            if getattr(code, "co_filename", "") == str(dispatcher): globals_dict["_INERT_PROCESS_API"] = api
            return real_exec(code, globals_dict, locals_dict)
        with mock.patch.object(experiments, "Store", Store), \
             mock.patch.object(experiments, "authorized", authorized), \
             mock.patch.object(experiments, "identity", fake_identity), \
             mock.patch.object(experiments, "stopped", lambda _root: False), \
             mock.patch.object(releases, "verify", lambda _root: {"id": "43d44b3f"}), \
             mock.patch.object(lifeguard, "start", lambda *_args, **_kwargs: Guard()), \
             mock.patch.object(lifeguard, "await_normal_cleanup", side_effect=await_cleanup), \
             mock.patch("builtins.exec", side_effect=injecting_exec):
            experiments.worker(request_path)
        events.append("receipt")
        return job, api, calls, events

    def test_protected_submit_rejects_duplicate_start_by_reconciling_existing_job(self):
        control = Path(__file__).resolve().parents[2] / "tools/rlflow"
        sys.path.insert(0, str(control))
        try:
            from rlflow import experiments
        finally:
            sys.path.remove(str(control))
        spec = {"id": "same-exact-spec", "nested": {"budgets": {"starts": 1}}}
        jid = "J-" + experiments.sha({"id": spec["id"], "spec": spec})[:16]
        class Store:
            root = Path("/inert")
            def replay(self): return {"jobs": {jid: {"id": jid, "spec": spec}}}
        calls = []
        with mock.patch.object(experiments, "reconcile", lambda _store, observed: calls.append(observed) or "EXISTING"), \
             mock.patch.object(experiments.subprocess, "Popen", side_effect=AssertionError("duplicate spawned")):
            self.assertEqual(experiments.submit(Store(), spec), "EXISTING")
        self.assertEqual(calls, [jid])

    def test_protected_worker_gate_persists_identity_then_releases_once(self):
        with tempfile.TemporaryDirectory() as directory:
            job, api, calls, events = self.run_worker(directory)
            receipt = json.loads((job / "receipt.json").read_text())
            self.assertEqual(calls, ["authorized", "authorized"])
            self.assertTrue(api.released); self.assertTrue((job / "process-identity.json").exists())
            self.assertTrue((job / "release-intent.json").exists())
            self.assertEqual(receipt["status"], "SUCCEEDED")
            self.assertEqual(events, ["guardian-ack", "receipt"])

    def test_protected_worker_gate_nested_auth_drift_reclaims_without_release(self):
        with tempfile.TemporaryDirectory() as directory:
            job, api, calls, events = self.run_worker(directory, drift=True)
            receipt = json.loads((job / "receipt.json").read_text())
            self.assertEqual(calls, ["authorized", "authorized"])
            self.assertFalse(api.released); self.assertEqual(api.signalled_groups, [api.pid])
            self.assertFalse((job / "release-intent.json").exists())
            self.assertTrue(receipt["definitely_not_released"])


    def test_production_worker_cleanup_failure_writes_truthful_failed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            job, api, calls, events = self.run_worker(directory, cleanup_failure=True)
            receipt = json.loads((job / "receipt.json").read_text())
            self.assertTrue(api.released)
            self.assertEqual(events, ["guardian-ack", "receipt"])
            self.assertEqual(receipt["status"], "FAILED")
            self.assertIn("forced cleanup failure", receipt["error"])



class OfflineResourceSettlementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        repository = Path(__file__).resolve().parents[2]
        diff_sha256 = os.environ.get("RL_CANDIDATE_DIFF_SHA256")
        if diff_sha256 is None:
            raw_diff = subprocess.run(
                ["git", "diff", "903afc5abd0fbf82297527e5790c2cd40827b0b2", "--"],
                cwd=repository, check=True, stdout=subprocess.PIPE).stdout
            diff_sha256 = hashlib.sha256(raw_diff).hexdigest()
        self.candidate = {
            "repository": str(repository),
            "base_commit": "903afc5abd0fbf82297527e5790c2cd40827b0b2",
            "commit": "903afc5abd0fbf82297527e5790c2cd40827b0b2",
            "diff_sha256": diff_sha256,
        }
        self.candidate_diff_sha256 = diff_sha256
        base = Path(self.temporary.name)
        self.root = base / "settlement"
        self.manifests = base / "raw-manifests"
        self.manifests.mkdir()
        self.key_patch = mock.patch.object(
            budget_module, "RESOURCE_PRODUCER_RSA_N", TEST_RESOURCE_RSA_N)
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)
        self.ledger = OfflineResourceSettlementLedger(self.root, self.candidate)

    def observation(self, identifier, *, phase="primary", parent=None, wall=None, gpu=None,
                    starts=1, transitions=0, context="cpu_offline_fixture", launch="launch-001"):
        dimensions = {
            "live_starts": {"state": "OBSERVED", "value": starts},
            "wall_seconds": ({"state": "OBSERVED", "value": wall}
                             if wall is not None else {"state": "UNKNOWN"}),
            "allocated_gpu_seconds_by_device": (
                {"state": "OBSERVED", "value": gpu} if gpu is not None else {"state": "RESERVED"}),
            "transitions": {"state": "OBSERVED", "value": transitions},
        }
        manifest = {
            "schema_version": RESOURCE_SOURCE_MANIFEST_SCHEMA,
            "producer_identity": RESOURCE_PRODUCER_IDENTITY,
            "producer_key_id": RESOURCE_PRODUCER_KEY_ID,
            "candidate": self.candidate,
            "observation_id": identifier,
            "launch_identity": launch,
            "measurement_context": context,
            "dimensions": dimensions,
        }
        manifest["producer_signature"] = trusted_resource_signature(manifest)
        data = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        manifest_path = self.manifests / f"{identifier}-{hashlib.sha256(data).hexdigest()}.json"
        manifest_path.write_bytes(data)
        return {
            "observation_id": identifier,
            "phase": phase,
            "authority": "trusted_raw_observation",
            "measurement_context": context,
            "source_manifest_path": str(manifest_path),
            "source_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "launch_identity": launch,
            "parent_event_sha256": parent,
            "dimensions": dimensions,
        }

    def assertObserved(self, row, value, context="cpu_offline_fixture"):
        self.assertEqual(row["state"], "OBSERVED")
        self.assertEqual(row["value"], value)
        self.assertTrue(row["sources"])
        self.assertTrue(all(source["measurement_context"] == context for source in row["sources"]))
        self.assertTrue(all(source["producer_identity"] == RESOURCE_PRODUCER_IDENTITY
                            for source in row["sources"]))

    def test_trusted_start_wall_per_device_gpu_seconds_and_transitions_settle(self):
        self.ledger.append(self.observation(
            "complete", wall=12.25, gpu={"device-0": 12.0}, starts=1, transitions=64))
        result = self.ledger.settle()
        self.assertEqual(result["candidate"]["diff_sha256"], self.candidate_diff_sha256)
        self.assertObserved(result["dimensions"]["live_starts"], 1)
        self.assertObserved(result["dimensions"]["wall_seconds"], 12.25)
        self.assertObserved(result["dimensions"]["allocated_gpu_seconds_by_device"], {"device-0": 12.0})
        self.assertObserved(result["dimensions"]["transitions"], 64)
        self.assertEqual(result["usage_interpretation"],
                         "CPU_OFFLINE_FIXTURE_NOT_REAL_GPU_OR_TRAINING_USAGE")
        self.assertEqual(result["verified_launch_identities"], ["launch-001"])
        self.assertFalse(result["automatic_retry"])

    def test_missing_dimensions_remain_unknown_or_reserved(self):
        self.ledger.append(self.observation("partial"))
        result = self.ledger.settle()
        self.assertEqual(result["dimensions"]["wall_seconds"], {"state": "UNKNOWN"})
        self.assertEqual(result["dimensions"]["allocated_gpu_seconds_by_device"], {"state": "RESERVED"})

    def test_late_evidence_is_append_only_idempotent_and_keeps_one_start(self):
        primary = self.ledger.append(self.observation("primary"))
        original = {path.name: path.read_bytes() for path in self.root.iterdir()}
        late = self.observation("late", phase="late", parent=primary["event_sha256"],
                                wall=9.5, gpu={"device-0": 9.25})
        first = self.ledger.append(late)
        once = {path.name: path.read_bytes() for path in self.root.iterdir()}
        second = self.ledger.append(late)
        self.assertTrue(first["appended"]); self.assertFalse(second["appended"])
        self.assertEqual(once, {path.name: path.read_bytes() for path in self.root.iterdir()})
        self.assertEqual(original, {name: once[name] for name in original})
        result = self.ledger.settle()
        self.assertObserved(result["dimensions"]["live_starts"], 1)
        self.assertObserved(result["dimensions"]["wall_seconds"], 9.5)
        self.assertEqual(len(result["event_sha256"]), 2)
        self.assertFalse(result["automatic_retry"])

    def test_nonfinite_wall_and_per_device_gpu_seconds_fail_closed_on_append(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(dimension="wall_seconds", value=value):
                with self.assertRaisesRegex(ResourceSettlementError, "finite non-negative"):
                    self.ledger.append(self.observation(f"wall-{repr(value)}", wall=value))
            with self.subTest(dimension="allocated_gpu_seconds_by_device", value=value):
                with self.assertRaisesRegex(ResourceSettlementError, "finite non-negative"):
                    self.ledger.append(self.observation(
                        f"gpu-{repr(value)}", gpu={"device-0": value}))

    def test_nonfinite_source_manifest_fails_closed_during_source_review(self):
        for name, value in (("wall_seconds", float("inf")),
                            ("allocated_gpu_seconds_by_device", {"device-0": float("-inf")})):
            observation = self.observation(f"source-{name}", **(
                {"wall": value} if name == "wall_seconds" else {"gpu": value}))
            expected = {
                "candidate": self.candidate,
                "observation_id": observation["observation_id"],
                "launch_identity": observation["launch_identity"],
                "measurement_context": observation["measurement_context"],
                "dimensions": observation["dimensions"],
            }
            with self.subTest(dimension=name), self.assertRaisesRegex(
                    ResourceSettlementError, "finite non-negative"):
                self.ledger._validate_source_manifest(
                    observation["source_manifest_path"],
                    observation["source_manifest_sha256"], expected)

    def test_nonfinite_event_fails_closed_during_settlement(self):
        valid = self.ledger.append(self.observation("settle-source", wall=1.0))
        event = {key: value for key, value in valid.items()
                 if key not in {"event_sha256", "appended"}}
        event["dimensions"] = dict(event["dimensions"])
        event["dimensions"]["wall_seconds"] = {"state": "OBSERVED", "value": float("inf")}
        with mock.patch.object(self.ledger, "_read", return_value=event):
            with self.assertRaisesRegex(ResourceSettlementError, "finite non-negative"):
                self.ledger.settle()

    def test_missing_or_forged_source_manifest_fails_closed(self):
        missing = self.observation("missing", wall=4.5)
        missing["source_manifest_path"] = str(self.manifests / "does-not-exist.json")
        with self.assertRaisesRegex(ResourceSettlementError, "readable regular file"):
            self.ledger.append(missing)
        forged = self.observation("forged", wall=4.5)
        manifest_path = Path(forged["source_manifest_path"])
        manifest = json.loads(manifest_path.read_text())
        manifest["producer_identity"] = "self_asserted_untrusted_producer"
        data = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        manifest_path.write_bytes(data)
        forged["source_manifest_sha256"] = hashlib.sha256(data).hexdigest()
        with self.assertRaisesRegex(ResourceSettlementError, "producer identity is not trusted"):
            self.ledger.append(forged)
        copied = self.observation("forged", wall=4.5)
        copied_path = Path(copied["source_manifest_path"])
        copied_manifest = json.loads(copied_path.read_text())
        copied_manifest["measurement_context"] = "real_run"
        copied_manifest["dimensions"]["allocated_gpu_seconds_by_device"] = {
            "state": "OBSERVED", "value": {"device-0": 999999.0}}
        copied_data = (json.dumps(copied_manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        copied_path.write_bytes(copied_data)
        copied["measurement_context"] = "real_run"
        copied["dimensions"] = copied_manifest["dimensions"]
        copied["source_manifest_sha256"] = hashlib.sha256(copied_data).hexdigest()
        with self.assertRaisesRegex(ResourceSettlementError, "producer signature is invalid"):
            self.ledger.append(copied)

    def test_signature_rejects_candidate_and_observation_identity_relabeling(self):
        candidate_attack = self.observation("candidate-source", wall=4.5)
        candidate_path = Path(candidate_attack["source_manifest_path"])
        candidate_manifest = json.loads(candidate_path.read_text())
        relabeled_candidate = {**self.candidate, "diff_sha256": "f" * 64}
        candidate_manifest["candidate"] = relabeled_candidate
        candidate_data = (json.dumps(candidate_manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        candidate_path.write_bytes(candidate_data)
        candidate_attack["source_manifest_sha256"] = hashlib.sha256(candidate_data).hexdigest()
        relabeled_ledger = OfflineResourceSettlementLedger(
            Path(self.temporary.name) / "candidate-attack", relabeled_candidate)
        with self.assertRaisesRegex(ResourceSettlementError, "producer signature is invalid"):
            relabeled_ledger.append(candidate_attack)

        identity_attack = self.observation("identity-source", wall=4.5)
        identity_path = Path(identity_attack["source_manifest_path"])
        identity_manifest = json.loads(identity_path.read_text())
        identity_manifest["observation_id"] = "identity-relabeled"
        identity_data = (json.dumps(identity_manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        identity_path.write_bytes(identity_data)
        identity_attack["observation_id"] = "identity-relabeled"
        identity_attack["source_manifest_sha256"] = hashlib.sha256(identity_data).hexdigest()
        with self.assertRaisesRegex(ResourceSettlementError, "producer signature is invalid"):
            self.ledger.append(identity_attack)

    def test_source_manifest_bytes_remain_required_at_settlement(self):
        observation = self.observation("mutable-source", wall=4.5)
        self.ledger.append(observation)
        Path(observation["source_manifest_path"]).write_text("{}")
        with self.assertRaisesRegex(ResourceSettlementError, "source manifest digest mismatch"):
            self.ledger.settle()

    def test_two_independent_primary_starts_fail_closed_and_retry_is_derived(self):
        self.ledger.append(self.observation("one", launch="launch-001"))
        with self.assertRaisesRegex(ResourceSettlementError, "second independent live start"):
            self.ledger.append(self.observation("two", launch="launch-002"))
        result = self.ledger.settle()
        self.assertEqual(result["verified_launch_identities"], ["launch-001"])
        self.assertFalse(result["automatic_retry"])

    def test_cpu_fixture_cannot_fill_real_run_dimensions(self):
        self.ledger.append(self.observation("real", context="real_run", gpu=None))
        cpu = self.observation("cpu", phase="late", context="cpu_offline_fixture",
                               gpu={"device-0": 7.0})
        cpu["parent_event_sha256"] = next(self.root.iterdir()).stem
        with self.assertRaisesRegex(ResourceSettlementError, "cannot mix measurement contexts"):
            self.ledger.append(cpu)
        self.assertEqual(self.ledger.settle()["dimensions"]["allocated_gpu_seconds_by_device"],
                         {"state": "RESERVED"})

    def test_estimates_second_start_and_conflicting_observations_fail_closed(self):
        estimate = self.observation("estimate", wall=1, gpu={"device-0": 1})
        estimate["authority"] = "cpu_estimate"
        with self.assertRaisesRegex(ResourceSettlementError, "trusted raw"):
            self.ledger.append(estimate)
        with self.assertRaisesRegex(ResourceSettlementError, "second live start"):
            self.ledger.append(self.observation("second", starts=2))
        primary = self.ledger.append(self.observation("one", wall=1, gpu={"device-0": 1}))
        self.ledger.append(self.observation("conflict", phase="late", parent=primary["event_sha256"],
                                            wall=2, gpu={"device-0": 1}))
        with self.assertRaisesRegex(ResourceSettlementError, "conflicting raw observations for wall_seconds"):
            self.ledger.settle()


if __name__ == "__main__": unittest.main()
