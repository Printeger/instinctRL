"""Frozen R3-R6 arithmetic and action-before-use budget accounting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

EVALUATION_CASE_STEPS = {"Z": 200, "T": 100, "S": 330, "X": 200, "W": 200}


def budget_manifest() -> dict[str, int]:
    per_checkpoint = 32 * sum(EVALUATION_CASE_STEPS.values())
    values = {
        "evaluation_transitions_per_checkpoint": per_checkpoint,
        "r3_training_transitions": 2048, "r3_nontraining_transitions": per_checkpoint + 512,
        "r4_training_transitions": 131072, "r4_nontraining_transitions": 5 * per_checkpoint,
        "r5_training_transitions": 917504, "r5_nontraining_transitions": 6 * per_checkpoint,
        "live_starts_r3_through_r5": 10, "checkpoint_writes_r3_through_r5": 35,
        "initialization_ticks_r3_through_r5_maximum": 100,
    }
    values["training_transitions_r3_through_r5"] = values["r3_training_transitions"] + values["r4_training_transitions"] + values["r5_training_transitions"]
    values["nontraining_transitions_r3_through_r5"] = values["r3_nontraining_transitions"] + values["r4_nontraining_transitions"] + values["r5_nontraining_transitions"]
    values["all_transitions_r3_through_r5"] = values["training_transitions_r3_through_r5"] + values["nontraining_transitions_r3_through_r5"]
    return values


@dataclass
class ExecutionCounters:
    live_process_starts: int = 0
    simulator_initializations: int = 0
    attempted_policy_steps: int = 0
    completed_policy_steps: int = 0
    real_environment_transitions: int = 0
    training_transitions: int = 0
    nontraining_transitions: int = 0
    real_optimizer_updates: int = 0
    real_checkpoint_writes: int = 0
    initialization_physics_ticks: int = 0
    explicit_resets: int = 0
    automatic_resets: int = 0
    synthetic_samples: int = 0
    synthetic_optimizer_updates: int = 0
    synthetic_optimizer_steps: int = 0
    temporary_checkpoint_writes: int = 0
    attempted_updates: int = 0
    completed_updates: int = 0
    actor_optimizer_steps: int = 0
    critic_optimizer_steps: int = 0

    def _require(self, value: int, maximum: int | None, label: str) -> None:
        if maximum is not None and value > maximum:
            raise RuntimeError(f"authorized {label} budget exhausted")

    def reserve_live_start(self, maximum: int | None = None) -> None:
        self._require(self.live_process_starts + 1, maximum, "live-process-start")
        self.live_process_starts += 1

    def reserve_simulator_initialization(self, maximum: int | None = None) -> None:
        self._require(self.simulator_initializations + 1, maximum, "simulator-initialization")
        self.simulator_initializations += 1

    def record_initialization_ticks(self, ticks: int, maximum: int | None = None) -> None:
        if type(ticks) is not int or ticks < 0:
            raise ValueError("initialization ticks must be a nonnegative integer")
        self._require(self.initialization_physics_ticks + ticks, maximum, "initialization-tick")
        self.initialization_physics_ticks += ticks

    def reserve_reset(self, *, automatic: bool, lanes: int = 1, maximum: int | None = None) -> None:
        if type(lanes) is not int or lanes < 0:
            raise ValueError("reset lanes must be a nonnegative integer")
        field = "automatic_resets" if automatic else "explicit_resets"
        value = getattr(self, field) + lanes
        self._require(value, maximum, field.replace("_", "-"))
        setattr(self, field, value)

    def require_reset_capacity(self, *, automatic: bool, lanes: int = 1,
                               maximum: int | None = None) -> None:
        """Gate the maximum reset effect of the next environment action."""
        if type(lanes) is not int or lanes < 0:
            raise ValueError("reset lanes must be a nonnegative integer")
        field = "automatic_resets" if automatic else "explicit_resets"
        self._require(getattr(self, field) + lanes, maximum, field.replace("_", "-"))

    def require_step_capacity(self, *, kind: str, lanes: int = 32,
                              maximum: int | None = None) -> None:
        if kind not in {"training", "nontraining"}:
            raise ValueError("policy-step kind must be training or nontraining")
        current = self.training_transitions if kind == "training" else self.nontraining_transitions
        self._require(current + lanes, maximum, f"{kind}-transition")

    def attempt_step(self) -> None:
        self.attempted_policy_steps += 1

    def complete_step(self, lanes: int = 32, *, kind: str = "training") -> None:
        if self.completed_policy_steps >= self.attempted_policy_steps:
            raise RuntimeError("complete cannot precede attempt")
        if kind not in {"training", "nontraining"}:
            raise ValueError("policy-step kind must be training or nontraining")
        self.completed_policy_steps += 1; self.real_environment_transitions += lanes
        field = "training_transitions" if kind == "training" else "nontraining_transitions"
        setattr(self, field, getattr(self, field) + lanes)

    def attempt_update(self, maximum: int | None = None) -> None:
        self._require(self.real_optimizer_updates + 1, maximum, "optimizer-update")
        self.attempted_updates += 1

    def complete_update(self, actor_steps: int, critic_steps: int) -> None:
        if self.completed_updates >= self.attempted_updates:
            raise RuntimeError("update completion cannot precede its durable attempt")
        self.completed_updates += 1
        self.real_optimizer_updates += 1
        self.actor_optimizer_steps = actor_steps
        self.critic_optimizer_steps = critic_steps

    def reserve_checkpoint(self, maximum: int | None = None) -> None:
        self._require(self.real_checkpoint_writes + 1, maximum, "checkpoint-write")

    def snapshot(self) -> dict[str, int]:
        return dict(vars(self))


def required_mode_budget(mode: str, parameters: Mapping[str, object]) -> dict[str, int]:
    """Return the exact ceiling for one future process; no slack is implicit."""
    common = {"live_process_starts": 1, "simulator_initializations": 1,
              "initialization_ticks": 10, "automatic_resets": 0}
    if mode == "main":
        return {**common, "training_transitions": 2048, "nontraining_transitions": 33472,
                "optimizer_updates": 4, "checkpoint_writes": 2,
                "explicit_resets": 8, "automatic_resets": 35520, "wall_seconds": 3600}
    if mode == "controlled-failure":
        return {**common, "training_transitions": 0, "nontraining_transitions": 0,
                "optimizer_updates": 0, "checkpoint_writes": 0,
                "explicit_resets": 1, "automatic_resets": 0, "wall_seconds": 600}
    if mode == "r4-training":
        return {**common, "training_transitions": 131072, "nontraining_transitions": 0,
                "optimizer_updates": 256, "checkpoint_writes": 5,
                "explicit_resets": 1, "automatic_resets": 131072, "wall_seconds": 7200}
    if mode == "evaluation":
        stage = parameters.get("stage")
        if stage == "R4":
            return {**common, "training_transitions": 0, "nontraining_transitions": 164800,
                    "optimizer_updates": 0, "checkpoint_writes": 0,
                    "explicit_resets": 25, "automatic_resets": 164800, "wall_seconds": 7200}
        if stage in {"R5-512", "R5-1024", "R5-2048"}:
            return {**common, "training_transitions": 0, "nontraining_transitions": 65920,
                    "optimizer_updates": 0, "checkpoint_writes": 0,
                    "explicit_resets": 10, "automatic_resets": 65920, "wall_seconds": 3600}
    if mode == "control-diagnostic":
        from .control_diagnostic import CASES, TRANSITIONS
        if parameters.get("condition") in {"D7-R", "B0"} and parameters.get("cases") == list(CASES):
            return {**common, "training_transitions": 0, "nontraining_transitions": TRANSITIONS,
                    "optimizer_updates": 0, "checkpoint_writes": 0,
                    "explicit_resets": len(CASES), "automatic_resets": TRANSITIONS,
                    "wall_seconds": 900}
    if mode == "enhanced-b0-diagnostic" and set(parameters) == {"handoff"}:
        return {**common, "training_transitions":0, "nontraining_transitions":20160,
                "optimizer_updates":0, "checkpoint_writes":0, "explicit_resets":3,
                "automatic_resets":20160, "wall_seconds":900}
    if mode == "pilot-training" and parameters.get("group") in {"P0", "PLR"}:
        return {**common, "training_transitions": 32768, "nontraining_transitions": 0,
                "optimizer_updates": 64, "checkpoint_writes": 7,
                "explicit_resets": 1, "automatic_resets": 32768, "wall_seconds": 3600}
    if mode == "pilot-evaluation" and parameters.get("group") in {"P0", "PLR"}:
        return {**common, "training_transitions": 0, "nontraining_transitions": 12800,
                "optimizer_updates": 0, "checkpoint_writes": 0,
                "explicit_resets": 2, "automatic_resets": 12800, "wall_seconds": 1800}
    if mode == "r5-resume":
        target = parameters.get("target_update")
        table = {512: (131072, 256, 4), 1024: (262144, 512, 8), 2048: (524288, 1024, 16)}
        if target in table:
            transitions, updates, checkpoints = table[int(target)]
            return {**common, "training_transitions": transitions, "nontraining_transitions": 0,
                    "optimizer_updates": updates, "checkpoint_writes": checkpoints,
                    "explicit_resets": 1, "automatic_resets": transitions, "wall_seconds": 7200}
    raise PermissionError("mode parameters do not identify a frozen R3-R5 budget")


RESOURCE_SETTLEMENT_SCHEMA = "unified-h1-offline-resource-settlement.v2"
RESOURCE_SOURCE_MANIFEST_SCHEMA = "unified-h1-raw-resource-manifest.v2"
RESOURCE_PRODUCER_IDENTITY = "unified_h1.failure_evidence.raw_resource_producer.v1"
RESOURCE_PRODUCER_KEY_ID = "unified-h1-resource-producer-rsa-2026-09"
# Public verification anchor only. The corresponding private key is intentionally
# absent from this candidate and is held by the trusted evidence producer.
RESOURCE_PRODUCER_RSA_E = 65537
RESOURCE_PRODUCER_RSA_N = 163255565205114413981967112939406703840787435513956732507646752734412647889421658039397328515821843518525027219628742493077659188628804786053858089925256839635772346933778067281539120612886365911468740993564333034418849059616285853313713249876701992804743843274595596982179524788316200109892118430636200677817
RESOURCE_DIMENSIONS = (
    "live_starts", "wall_seconds", "allocated_gpu_seconds_by_device", "transitions",
)


class ResourceSettlementError(RuntimeError):
    """A resource observation was untrusted, inconsistent, or mutable."""


def _resource_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _resource_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resource_signature_payload(manifest: Mapping[str, Any]) -> bytes:
    """Return every producer-authenticated identity, ownership, and value fact."""
    fields = ("schema_version", "producer_identity", "producer_key_id", "candidate",
              "observation_id", "launch_identity", "measurement_context", "dimensions")
    return _resource_bytes({key: manifest[key] for key in fields})


def _verify_resource_signature(manifest: Mapping[str, Any]) -> None:
    if manifest.get("producer_key_id") != RESOURCE_PRODUCER_KEY_ID:
        raise ResourceSettlementError("source manifest producer key is not trusted")
    signature = manifest.get("producer_signature")
    size = (RESOURCE_PRODUCER_RSA_N.bit_length() + 7) // 8
    if (not isinstance(signature, str) or len(signature) != size * 2
            or any(character not in "0123456789abcdef" for character in signature)):
        raise ResourceSettlementError("source manifest producer signature is invalid")
    digest_info = bytes.fromhex("3031300d060960864801650304020105000420") + hashlib.sha256(
        _resource_signature_payload(manifest)).digest()
    padding_length = size - len(digest_info) - 3
    expected = b"\x00\x01" + b"\xff" * padding_length + b"\x00" + digest_info
    actual = pow(int(signature, 16), RESOURCE_PRODUCER_RSA_E,
                 RESOURCE_PRODUCER_RSA_N).to_bytes(size, "big")
    if padding_length < 8 or actual != expected:
        raise ResourceSettlementError("source manifest producer signature is invalid")


def _resource_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ResourceSettlementError("resource evidence is not a readable regular file")
    return path.read_bytes()


def _validate_dimension(name: str, row: Any) -> Mapping[str, Any]:
    if not isinstance(row, dict) or row.get("state") not in {"OBSERVED", "UNKNOWN", "RESERVED"}:
        raise ResourceSettlementError(f"invalid settlement state for {name}")
    state = row["state"]
    if state != "OBSERVED":
        if set(row) != {"state"}:
            raise ResourceSettlementError(f"{name} {state} cannot carry an inferred value")
        return {"state": state}
    if set(row) != {"state", "value"}:
        raise ResourceSettlementError(f"observed {name} requires exactly one raw value")
    value = row["value"]
    if name == "allocated_gpu_seconds_by_device":
        if (not isinstance(value, dict) or not value
                or any(not isinstance(device, str) or not device for device in value)
                or any(type(seconds) not in {int, float} or (type(seconds) is float and not math.isfinite(seconds))
                       or seconds < 0
                       for seconds in value.values())):
            raise ResourceSettlementError(
                "allocated GPU seconds require finite non-negative per-device raw values")
        value = dict(sorted(value.items()))
    elif name in {"live_starts", "transitions"}:
        if type(value) is not int or value < 0:
            raise ResourceSettlementError(f"{name} requires a non-negative integer raw value")
    elif (type(value) not in {int, float}
          or (type(value) is float and not math.isfinite(value)) or value < 0):
        raise ResourceSettlementError("wall seconds require a finite non-negative raw value")
    return {"state": "OBSERVED", "value": value}


class OfflineResourceSettlementLedger:
    """Content-addressed CPU verifier for append-only raw resource evidence."""

    def __init__(self, root: Path, candidate: Mapping[str, str]) -> None:
        self.root = Path(root)
        required = {"repository", "base_commit", "commit", "diff_sha256"}
        if (self.root.is_symlink() or set(candidate) != required
                or any(not isinstance(candidate[key], str) or not candidate[key] for key in required)):
            raise ResourceSettlementError("safe ledger root and exact candidate binding are required")
        self.candidate = dict(candidate)
        self.root.mkdir(parents=True, exist_ok=True)

    def _validate_source_manifest(self, path_value: Any, expected_digest: str,
                                  expected: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(path_value, str) or not path_value:
            raise ResourceSettlementError("readable source manifest path is required")
        path = Path(path_value)
        data = _resource_file(path)
        if _resource_digest(data) != expected_digest:
            raise ResourceSettlementError("source manifest digest mismatch")
        try:
            manifest = json.loads(data)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ResourceSettlementError("source manifest is not valid JSON") from error
        required = {"schema_version", "producer_identity", "producer_key_id", "producer_signature",
                    "candidate", "observation_id", "launch_identity", "measurement_context", "dimensions"}
        if not isinstance(manifest, dict) or set(manifest) != required:
            raise ResourceSettlementError("source manifest fields are incomplete")
        if (manifest["schema_version"] != RESOURCE_SOURCE_MANIFEST_SCHEMA
                or manifest["producer_identity"] != RESOURCE_PRODUCER_IDENTITY):
            raise ResourceSettlementError("source manifest producer identity is not trusted")
        _verify_resource_signature(manifest)
        dimensions = manifest["dimensions"]
        if not isinstance(dimensions, dict) or set(dimensions) != set(RESOURCE_DIMENSIONS):
            raise ResourceSettlementError("source manifest must represent all resource dimensions")
        for name in RESOURCE_DIMENSIONS:
            _validate_dimension(name, dimensions[name])
        for key in ("candidate", "observation_id", "launch_identity", "measurement_context", "dimensions"):
            if manifest[key] != expected[key]:
                raise ResourceSettlementError(f"source manifest {key} binding mismatch")
        return manifest

    def _read(self, path: Path) -> Mapping[str, Any]:
        data = _resource_file(path)
        if path.suffix != ".json" or path.stem != _resource_digest(data):
            raise ResourceSettlementError("resource event digest mismatch")
        try:
            event = json.loads(data)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ResourceSettlementError("resource event is not valid JSON") from error
        if (not isinstance(event, dict) or event.get("schema_version") != RESOURCE_SETTLEMENT_SCHEMA
                or event.get("candidate") != self.candidate):
            raise ResourceSettlementError("resource event candidate binding mismatch")
        dimensions = event.get("dimensions")
        if not isinstance(dimensions, dict) or set(dimensions) != set(RESOURCE_DIMENSIONS):
            raise ResourceSettlementError("resource event must represent all resource dimensions")
        for name in RESOURCE_DIMENSIONS:
            _validate_dimension(name, dimensions[name])
        expected = {key: event[key] for key in (
            "candidate", "observation_id", "launch_identity", "measurement_context", "dimensions")}
        self._validate_source_manifest(event.get("source_manifest_path"),
                                       event.get("source_manifest_sha256"), expected)
        return event

    def append(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        required = {"observation_id", "phase", "authority", "measurement_context",
                    "source_manifest_path", "source_manifest_sha256", "launch_identity",
                    "parent_event_sha256", "dimensions"}
        if not isinstance(observation, dict) or set(observation) != required:
            raise ResourceSettlementError("resource observation fields are incomplete")
        identifier = observation["observation_id"]
        if (not isinstance(identifier, str) or not identifier or identifier in {".", ".."}
                or Path(identifier).name != identifier):
            raise ResourceSettlementError("unsafe resource observation identity")
        if observation["phase"] not in {"primary", "late"}:
            raise ResourceSettlementError("resource observation phase is invalid")
        if observation["authority"] != "trusted_raw_observation":
            raise ResourceSettlementError("resource values require trusted raw observation authority")
        context = observation["measurement_context"]
        if context not in {"cpu_offline_fixture", "real_run"}:
            raise ResourceSettlementError("measurement context is invalid")
        launch_identity = observation["launch_identity"]
        if not isinstance(launch_identity, str) or not launch_identity:
            raise ResourceSettlementError("immutable launch identity is required")
        source = observation["source_manifest_sha256"]
        if (not isinstance(source, str) or len(source) != 64
                or any(character not in "0123456789abcdef" for character in source)):
            raise ResourceSettlementError("source manifest digest is required")
        dimensions = observation["dimensions"]
        if not isinstance(dimensions, dict) or set(dimensions) != set(RESOURCE_DIMENSIONS):
            raise ResourceSettlementError("all resource dimensions must be explicitly represented")
        normalized = {name: _validate_dimension(name, dimensions[name]) for name in RESOURCE_DIMENSIONS}
        starts = normalized["live_starts"]
        if starts["state"] == "OBSERVED" and starts["value"] not in {0, 1}:
            raise ResourceSettlementError("resource evidence cannot claim a second live start")
        manifest_expected = {
            "candidate": self.candidate, "observation_id": identifier,
            "launch_identity": launch_identity, "measurement_context": context,
            "dimensions": normalized,
        }
        self._validate_source_manifest(observation["source_manifest_path"], source, manifest_expected)
        existing = {path.stem: self._read(path) for path in self.root.glob("*.json")}
        if existing and {event["measurement_context"] for event in existing.values()} != {context}:
            raise ResourceSettlementError("resource observations cannot mix measurement contexts")
        parent = observation["parent_event_sha256"]
        if observation["phase"] == "primary":
            if parent is not None:
                raise ResourceSettlementError("primary observation cannot bind a parent event")
        elif not isinstance(parent, str) or parent not in existing or existing[parent]["phase"] != "primary":
            raise ResourceSettlementError("late observation requires an immutable primary parent event")
        if observation["phase"] == "late" and existing[parent]["launch_identity"] != launch_identity:
            raise ResourceSettlementError("late observation launch identity differs from its primary")
        if starts["state"] == "OBSERVED" and starts["value"] == 1:
            prior_start_identities = {
                event["launch_identity"] for event in existing.values()
                if event["dimensions"]["live_starts"] == {"state": "OBSERVED", "value": 1}
            }
            if prior_start_identities and prior_start_identities != {launch_identity}:
                raise ResourceSettlementError("second independent live start observation")
        event = {
            "schema_version": RESOURCE_SETTLEMENT_SCHEMA,
            "candidate": self.candidate,
            "observation_id": identifier,
            "phase": observation["phase"],
            "authority": observation["authority"],
            "measurement_context": context,
            "launch_identity": launch_identity,
            "source_manifest_path": str(Path(observation["source_manifest_path"]).resolve()),
            "source_manifest_sha256": source,
            "source_producer_identity": RESOURCE_PRODUCER_IDENTITY,
            "parent_event_sha256": parent,
            "dimensions": normalized,
            "usage_interpretation": ("CPU_OFFLINE_FIXTURE_NOT_REAL_GPU_OR_TRAINING_USAGE"
                                     if context == "cpu_offline_fixture"
                                     else "RAW_REAL_RUN_OBSERVATION"),
        }
        data = _resource_bytes(event)
        digest = _resource_digest(data)
        for old_digest, old in existing.items():
            if old["observation_id"] == identifier and old_digest != digest:
                raise ResourceSettlementError("resource observation identity is immutable")
        destination = self.root / f"{digest}.json"
        if destination.exists():
            if _resource_file(destination) != data:
                raise ResourceSettlementError("content-addressed resource event was mutated")
            return {**event, "event_sha256": digest, "appended": False}
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return {**event, "event_sha256": digest, "appended": True}

    def settle(self) -> Mapping[str, Any]:
        events = [self._read(path) for path in sorted(self.root.glob("*.json"))]
        if not events:
            raise ResourceSettlementError("at least one trusted resource observation is required")
        by_digest = {_resource_digest(_resource_bytes(event)): event for event in events}
        contexts = {event["measurement_context"] for event in events}
        if len(contexts) != 1:
            raise ResourceSettlementError("resource observations cannot mix measurement contexts")
        for event in events:
            parent = event["parent_event_sha256"]
            if event["phase"] == "late" and parent not in by_digest:
                raise ResourceSettlementError("late observation parent is missing")
            if event["phase"] == "late" and by_digest[parent]["launch_identity"] != event["launch_identity"]:
                raise ResourceSettlementError("late observation launch identity differs from its primary")
        start_identities = {
            event["launch_identity"] for event in events
            if event["dimensions"]["live_starts"] == {"state": "OBSERVED", "value": 1}
        }
        automatic_retry = len(start_identities) > 1
        if automatic_retry:
            raise ResourceSettlementError("multiple launch identities reveal a second start or retry")
        settled = {}
        for name in RESOURCE_DIMENSIONS:
            for event in events:
                _validate_dimension(name, event["dimensions"][name])
            observed = [(digest, event, event["dimensions"][name]["value"])
                        for digest, event in by_digest.items()
                        if event["dimensions"][name]["state"] == "OBSERVED"]
            if observed and any(row[2] != observed[0][2] for row in observed[1:]):
                raise ResourceSettlementError(f"conflicting raw observations for {name}")
            if observed:
                sources = [{
                    "event_sha256": digest,
                    "source_manifest_sha256": event["source_manifest_sha256"],
                    "producer_identity": event["source_producer_identity"],
                    "measurement_context": event["measurement_context"],
                    "launch_identity": event["launch_identity"],
                } for digest, event, _value in sorted(observed)]
                settled[name] = {"state": "OBSERVED", "value": observed[0][2], "sources": sources}
            else:
                states = {event["dimensions"][name]["state"] for event in events}
                settled[name] = {"state": "RESERVED" if "RESERVED" in states else "UNKNOWN"}
        context = next(iter(contexts))
        return {
            "schema_version": RESOURCE_SETTLEMENT_SCHEMA,
            "candidate": self.candidate,
            "event_sha256": sorted(by_digest),
            "dimensions": settled,
            "measurement_contexts": [context],
            "usage_interpretation": ("CPU_OFFLINE_FIXTURE_NOT_REAL_GPU_OR_TRAINING_USAGE"
                                     if context == "cpu_offline_fixture"
                                     else "RAW_REAL_RUN_OBSERVATION"),
            "verified_launch_identities": sorted(start_identities),
            "append_only": True,
            "automatic_retry": automatic_retry,
        }


__all__ = [
    "EVALUATION_CASE_STEPS", "ExecutionCounters", "OfflineResourceSettlementLedger",
    "RESOURCE_DIMENSIONS", "RESOURCE_PRODUCER_IDENTITY", "RESOURCE_PRODUCER_KEY_ID",
    "RESOURCE_PRODUCER_RSA_E", "RESOURCE_PRODUCER_RSA_N", "RESOURCE_SETTLEMENT_SCHEMA",
    "RESOURCE_SOURCE_MANIFEST_SCHEMA", "ResourceSettlementError",
    "budget_manifest", "required_mode_budget",
]
