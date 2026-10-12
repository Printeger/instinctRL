from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from training_workloads.scale_003.audit import _entry_import_audit, affine_rank_negative_control
from training_workloads.scale_003.config import CONFIG_PATH, ConfigError, load_config, load_strict_json
from training_workloads.scale_003.evaluation import evaluation_command_at, run_evaluation, validate_evaluation_semantic
from training_workloads.scale_003.geometry import (
    MESH_PRIM_PATH, build_combined_geometry, fixture_grid_origins,
    validate_exact_observability_gate, validate_geometry, validate_readback,
)
from training_workloads.scale_003.manifest import build_attempt_documents, build_training_handoff_document
from training_workloads.scale_003.persistence import ProgressStore, atomic_json_replace
from training_workloads.scale_003.resources import RuntimeResourceGuard, collect_coexistence_window, run_prelaunch_gate, start_after_coexistence_gate, validate_coexistence_window
from training_workloads.scale_003.schedule import CommandClock, build_segments, checkpoint_counters, command_at
from training_workloads.scale_003.semantic import EVALUATION_INPUT_KEYS, OBSERVABILITY_DETAIL_KEYS
from training_workloads.scale_003.training import run_training, validate_training_semantic


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "training_workloads/scale_003"
PYTHON = Path("/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11")


def _payload_hash(value):
    payload = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resource_window(*, free=11227, used=1000, utilization=1):
    samples = []
    for stamp in (1, 5_000_000_001, 10_000_000_001):
        samples.append({
            "monotonic_ns": stamp,
            "gpu": {"index": 0, "name": "NVIDIA GeForce RTX 5070", "driver": "580.173.02",
                    "memory_total_mib": 12227, "memory_used_mib": used,
                    "free_memory_mib": free, "utilization_percent": utilization},
            "compute_apps": [],
        })
    return {"sample_count": 3, "span_ns": 10_000_000_000, "samples": samples}


def resource_gate(mode):
    return {"schema_version": "scale-003-coexistence-window.v1", "status": "PASS", **resource_window(),
            "mode": mode, "attempt_id": "INERT-ATTEMPT", "pid": 1,
            "authorization_sha256": "a" * 64, "frozen_gate_source_sha256": "b" * 64,
            "nvidia_smi_sha256": "c" * 64}


def runtime_resource():
    return {"stage": "before_environment", "ordinal": 0, "free_memory_bytes": 2 * 1024 ** 3,
            "total_memory_bytes": 12 * 1024 ** 3, "allocated_memory_bytes": 0,
            "minimum_free_memory_bytes": 1024 ** 3}


def training_preflight():
    geometry = {"mesh_prim_path": MESH_PRIM_PATH, "points": 59424, "triangles": 113024,
                "wall_colliders": 64, "env_origins": [[float(i * 100), 0.0, 0.0] for i in range(32)],
                "inventory": [{"path": f"/World/Scale003WallColliders/wall_{i:02d}", "collision": True, "type": "Mesh"} for i in range(64)],
                "coverage_limit": "inert CPU structure; live visibility not proven"}
    geometry["sha256"] = _payload_hash(geometry)
    return {"geometry": geometry,
            "observability": {"status": "PASS", "algorithm": "exact_range_normal_jacobian", "environments": 32,
                              "surface_hits": {name: [1] * 32 for name in ("floor", "x_wall", "y_wall")},
                              "numerical_rank": [3] * 32, "sigma_min": [1.0] * 32,
                              "singular_values": [[1.0, 1.0, 1.0] for _ in range(32)],
                              "source": "actual_raw_raycaster_acquisition"},
            "prelaunch_resource_gate": resource_gate("training"), "runtime_resource_initial": runtime_resource()}


def evaluation_preflight():
    geometry = {"mesh_path": "/World/MID360StaticScene/Eval002CombinedMesh",
                "points": [[0.0, 0.0, 0.0] for _ in range(1857)], "indices": [0] * 10596,
                "topology": {"points": 1857, "triangles": 3532, "indices": 10596,
                             "floor_normal": [0.0, 0.0, 1.0], "x_wall_normal": [-1.0, 0.0, 0.0], "y_wall_normal": [0.0, -1.0, 0.0]},
                "inventory": [{"path": "/World/MID360StaticScene/Eval002CombinedMesh", "collision": False}],
                "env_origins": [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                "initial_root_state_w": [[0.0] * 13 for _ in range(2)]}
    geometry["sha256"] = _payload_hash(geometry)
    return {"geometry": geometry,
            "observability": {"surface_hits": {name: [1, 1] for name in ("floor", "x_wall", "y_wall")},
                              "numerical_rank": [3, 3], "singular_values": [[1.0] * 3 for _ in range(2)],
                              "sigma_min": [1.0, 1.0]},
            "prelaunch_resource_gate": resource_gate("evaluation"), "runtime_resource_initial": runtime_resource()}


def runtime_summary(case):
    import torch
    from instinctlab.evaluation.metrics import aggregate_metrics, compute_step_metrics
    from instinctlab.evaluation.runtime import SOURCE_MANIFEST

    inputs = OrderedDict()
    commands = [[torch.tensor(evaluation_command_at(case, step), dtype=torch.float32).tolist() for _ in range(2)] for step in range(256)]
    vectors_zero = [[[0.0, 0.0, 0.0] for _ in range(2)] for _ in range(256)]
    scalars_zero = [[0.0, 0.0] for _ in range(256)]
    bool_false = [[False, False] for _ in range(256)]
    for name in EVALUATION_INPUT_KEYS:
        if name == "commanded_velocity_b_m_s": value = commands
        elif name in {"actual_velocity_b_m_s", "issued_velocity_b_m_s"}: value = copy.deepcopy(commands)
        elif name in {"position_w_m", "station_reference_position_w_m"}: value = copy.deepcopy(vectors_zero)
        elif name in {"anchor_error_m", "ics_worst_margin_m_s"}: value = copy.deepcopy(scalars_zero)
        elif name in {"minimum_clearance_m", "ics_beta", "observability_value"}: value = [[1.0, 1.0] for _ in range(256)]
        elif name in {"anchor_valid", "clearance_valid", "observability_valid"}: value = [[True, True] for _ in range(256)]
        elif name in {"termination_reason"}: value = [[0, 0] for _ in range(256)]
        elif name == "observability_source": value = [[2, 2] for _ in range(256)]
        else: value = copy.deepcopy(bool_false)
        inputs[name] = value
    bool_names = {"anchor_valid", "clearance_valid", "collision", "ics_emergency", "terminated", "truncated", "termination_collision", "termination_below_height", "termination_above_height", "termination_timeout", "observability_valid"}
    long_names = {"termination_reason", "observability_source"}
    tensors = OrderedDict((name, torch.tensor(value, dtype=torch.bool if name in bool_names else torch.long if name in long_names else torch.float32)) for name, value in inputs.items())
    aggregate = aggregate_metrics(compute_step_metrics(tensors))
    metrics = OrderedDict()
    for metric in aggregate.metrics:
        row = asdict(metric); name = row.pop("name"); metrics[name] = row
    record = OrderedDict((
        ("schema_version", "m7.evaluation.v1"), ("platform", "TASLAB_UAV"), ("sensor", "Livox MID360"),
        ("claim", "pure-tensor contract record; no simulator evaluation or policy-performance claim"),
        ("actor_audit", {"status": "PASS", "keys": ["lidar_grid", "state_vec"],
                         "lidar_grid_shape": [2, 12, 59, 360], "state_vec_shape": [2, 52],
                         "dtype": "torch.float32", "device": "cpu",
                         "lidar_grid_sha256": "a" * 64, "state_vec_sha256": "b" * 64}),
        ("input_dimensions", {"time_steps": 256, "environments": 2, "samples": 512}),
        ("evaluation_input_order", list(EVALUATION_INPUT_KEYS)),
        ("metric_config", {"schema_version": "m7.evaluation.v1", "null_command_threshold_m_s": 0.001, "full_compliance_beta": 0.999}),
        ("observability_source_counts", {"UNAVAILABLE": 0, "PROXY": 0, "EXACT": 512}),
        ("metrics", metrics),
        ("termination_counts", {"collision": 0, "below_height": 0, "above_height": 0, "timeout": 0, "terminated": 0, "truncated": 0, "done": 0, "simultaneous_terminated_truncated": 0, "reason_none": 512, "reason_timeout": 0, "reason_above_height": 0, "reason_below_height": 0, "reason_collision": 0}),
        ("provenance", {"product_commit": "de9d6b8c03274a11760a3f06ca68d38cc0058f10", "fixture_id": f"scale-003-eval-002-{case}", "generator_id": "training_workloads/scale_003/formal_evaluation.py", "seed": 20260905}),
    ))
    canonical = json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    scalar_detail = [[1.0, 1.0] for _ in range(256)]
    vector_detail = [[[1.0, 0.0, 0.0] for _ in range(2)] for _ in range(256)]
    identity_detail = [[[index % 256] * 32 for _ in range(2)] for index in range(256)]
    details = OrderedDict()
    for name in OBSERVABILITY_DETAIL_KEYS:
        if name == "singular_values": value = [[[1.0, 1.0, 1.0] for _ in range(2)] for _ in range(256)]
        elif name == "weak_direction_w": value = copy.deepcopy(vector_detail)
        elif name in {"condition_valid", "weak_direction_valid", "source_reset_mask"}: value = [[True, True] for _ in range(256)] if name != "source_reset_mask" else copy.deepcopy(bool_false)
        elif name in {"numerical_rank", "valid_ray_count", "total_ray_count", "source_step"}: value = [[3, 3] for _ in range(256)]
        elif name in {"source_identity_sha256", "result_identity_sha256"}: value = copy.deepcopy(identity_detail)
        else: value = copy.deepcopy(scalar_detail)
        details[name] = value
    runtime = {"schema_version": "m7.runtime-evaluation.v2",
               "contract": {"num_envs": 2, "policy_steps": 256, "samples": 512, "rays_per_environment": 21240, "ray_samples": 10874880},
               "source_manifest": copy.deepcopy(SOURCE_MANIFEST),
               "policy_hashes": [f"{index:064x}" for index in range(256)], "action_hashes": [f"{index + 256:064x}" for index in range(256)],
               "station_reference_before_w_m": copy.deepcopy(vectors_zero), "station_reference_after_w_m": copy.deepcopy(vectors_zero),
               "reset_masks": copy.deepcopy(bool_false), "returned_episode_lengths": [[index + 1] * 2 for index in range(256)],
               "evaluation_inputs": inputs,
               "observability_details": details,
               "canonical_record": canonical, "gates": {"coverage_exact": True, "all_finite": True, "observability_exact": True, "reset_count": 0}}
    return {"canonical_metrics_available": True, "terminal_samples_preserved": True, "zero_denominator_is_unavailable": True,
            "runtime_evaluation": runtime, "canonical_record": record, "reset_count": 0, "checkpoint_state_unchanged": True}


class MemoryProgress:
    def __init__(self) -> None:
        self.latest = {}
        self.writes = []
        self.failures = []

    def write(self, **fields):
        self.latest = fields
        self.writes.append(fields)

    def fail(self, error, **fields):
        self.failures.append((type(error).__name__, fields))


class FakeTrainingBackend:
    def __init__(self, fail_at=None, output_root=None):
        self.segment = None
        self.optimizer = 2000
        self.saved = []
        self.preserved = False
        self.fail_at = fail_at
        self.output_root = Path(output_root) if output_root is not None else Path("/tmp/scale-003-inert-checkpoints")
        self.update_counter = 250

    def bootstrap(self, seed): assert seed == 20260907
    def create_environment(self): pass
    def reset_environment(self, seed): assert seed == 20260907
    def create_trainer(self): pass
    def load_checkpoint(self, path, sha256): assert path.is_file() and len(sha256) == 64
    def run_initial_preflight(self): return training_preflight()
    def prepare_segment(self, segment): self.segment = segment
    def collect_segment(self, segment):
        if segment.ordinal == self.fail_at: raise RuntimeError("inert interruption")
        return {"sample_count": segment.samples, "batch": segment.ordinal}
    def update(self, batch):
        assert batch == self.segment.ordinal
        self.optimizer = self.segment.target_optimizer_step
        self.update_counter = self.segment.update_counter
        return {"sample_count": self.segment.samples, "coverage_min": 1, "coverage_max": 1, "update_counter": self.segment.update_counter}
    def optimizer_steps(self): return self.optimizer, self.optimizer
    def save_checkpoint(self, counter, kind):
        path = self.output_root / "checkpoints" / f"update_{counter:04d}_{kind}.pt"
        if self.output_root != Path("/tmp/scale-003-inert-checkpoints"):
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(f"inert checkpoint {counter}\n".encode())
            digest = hashlib.sha256(path.read_bytes()).hexdigest(); size = path.stat().st_size
        else:
            digest = f"{counter:064x}"[-64:]; size = counter
        row = {"counter": counter, "kind": kind, "path": str(path), "sha256": digest, "bytes": size}
        self.saved.append(row); return row
    def final_reload_audit(self, counter, optimizer_step):
        final = self.saved[-1]
        return {"counter": counter, "optimizer_step": optimizer_step, "checkpoint_path": final["path"], "checkpoint_sha256": final["sha256"], "exact": True}
    def preserve_partial(self, error): self.preserved = True
    def failure_accounting(self):
        return {"completed_updates": self.update_counter - 250, "update_counter": self.update_counter,
                "actor_optimizer_step": self.optimizer, "critic_optimizer_step": self.optimizer,
                "optimizer_state_status": "CONFIRMED_UPDATE_BOUNDARY"}


class FakeEvaluationBackend:
    def __init__(self): self.rows = []; self.completed_transitions = 0
    def bootstrap(self, seed): assert seed == 20260905
    def create_environment(self, case): self.case = case
    def reset_environment(self, seed): pass
    def load_checkpoint(self, path, counter): assert counter == 1954
    def run_initial_preflight(self): return evaluation_preflight()
    def disable_gradients(self): pass
    def evaluate_step(self, case, index, command):
        self.completed_transitions += 2
        row = {"global_index": index, "command_b_m_s": list(command), "actor_keys": ["lidar_grid", "state_vec"],
               "terminal_environment_count": 0, "visibility": {"floor": [1, 1], "x_wall": [1, 1], "y_wall": [1, 1]}}
        self.rows.append(row); return row
    def finalize(self): return runtime_summary(self.case)
    def preserve_partial(self, error): pass
    def failure_accounting(self): return {"completed_transitions": self.completed_transitions, "completed_policy_steps": self.completed_transitions // 2}


def test_strict_config_and_source_pins(tmp_path):
    config = load_config()
    assert config["order"]["runtime_process_starts"] == 0
    assert config["training"]["additional_transitions"] == 872000
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate"):
        load_strict_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(ConfigError, match="non-standard"):
        load_strict_json(nonfinite)


def test_entries_are_import_safe_and_current_order_rejects_live(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "PYTHONPATH": f"{ROOT}:/home/mint/rl_dev/worktrees/instinctRL-scale-004-f1-product/isaaclab-training"}
    for name in ("formal_scale_training.py", "formal_evaluation.py", "formal_station_evaluation.py", "formal_tracking_evaluation.py"):
        entry = SOURCE / name
        assert _entry_import_audit(entry)["runtime_imports"] is False
        help_result = subprocess.run([PYTHON, "-B", entry, "--help"], env=env, text=True, capture_output=True)
        assert help_result.returncode == 0 and "AppLauncher" not in help_result.stderr
        valid = subprocess.run([PYTHON, "-B", entry, "--validate-only"], env=env, text=True, capture_output=True)
        assert valid.returncode == 0 and json.loads(valid.stdout)["live_execution"] is False
    attempt = tmp_path / "SCALE-003-D0001"
    order = Path("/home/mint/rl_dev/instinctRL_coordination/commander/CURRENT_ORDER.md")
    rejected = subprocess.run(
        [PYTHON, "-B", SOURCE / "formal_scale_training.py", "--attempt-dir", attempt,
         "--execution-authorization", order], env=env, text=True, capture_output=True,
    )
    assert rejected.returncode != 0 and not attempt.exists()


def test_exact_schedule_command_clock_and_checkpoint_coverage():
    segments = build_segments()
    assert len(segments) == 1704
    assert sum(row.samples for row in segments) == 872000
    assert segments[-2].policy_step_end == 27248 and segments[-1].horizon == 2
    assert segments[-1].target_optimizer_step == 15625
    assert checkpoint_counters() == tuple(range(275, 1951, 25)) + (1953, 1954)
    assert command_at(7999) == (0.0, 0.0, 0.0)
    assert command_at(8000) == (0.2, 0.0, 0.0)
    assert command_at(8032) == (-0.2, 0.0, 0.0)
    clock = CommandClock()
    for _ in range(11): clock.advance_after_action()
    clock.observe_reset()
    assert clock.index == 11


def test_geometry_exact_counts_normals_readback_and_negative_cases():
    origins = fixture_grid_origins()
    geometry = build_combined_geometry(origins)
    assert (len(geometry.points), len(geometry.triangles), len(geometry.wall_colliders)) == (59424, 113024, 64)
    rows = [{"prim_path": item.prim_path, "points": item.points, "collision": True} for item in geometry.wall_colliders]
    validate_readback(
        expected=geometry, mesh_prim_path=MESH_PRIM_PATH, points=geometry.points,
        triangle_indices=[i for triangle in geometry.triangles for i in triangle], collider_rows=rows,
        ground_collision_present=True, sensed_mesh_collision=False,
    )
    with pytest.raises(ValueError, match="inventory"):
        validate_readback(expected=geometry, mesh_prim_path=MESH_PRIM_PATH, points=geometry.points,
                          triangle_indices=[i for triangle in geometry.triangles for i in triangle], collider_rows=rows[:-1],
                          ground_collision_present=True, sensed_mesh_collision=False)
    with pytest.raises(ValueError, match="wrong mesh"):
        validate_readback(expected=geometry, mesh_prim_path="/World/wrong", points=geometry.points,
                          triangle_indices=[], collider_rows=[], ground_collision_present=True, sensed_mesh_collision=False)
    shifted = list(geometry.points); shifted[0] = (shifted[0][0] + 1.0, *shifted[0][1:])
    with pytest.raises(ValueError, match="translations"):
        validate_geometry(replace(geometry, points=tuple(shifted)), origins)
    wrong_origins = list(origins); wrong_origins[0] = (1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="origins"):
        validate_geometry(geometry, wrong_origins)


def test_exact_observability_gate_and_affine_rank_negative_control():
    accepted = validate_exact_observability_gate(
        surface_hits={name: [1] * 32 for name in ("floor", "x_wall", "y_wall")},
        numerical_rank=[3] * 32, sigma_min=[0.1] * 32,
    )
    assert accepted["status"] == "PASS"
    with pytest.raises(ValueError, match="rank"):
        validate_exact_observability_gate(
            surface_hits={name: [1] * 32 for name in ("floor", "x_wall", "y_wall")},
            numerical_rank=[2] + [3] * 31, sigma_min=[0.1] * 32,
        )
    assert affine_rank_negative_control() == {
        "point_cloud_affine_rank": 3, "normal_jacobian_rank": 2, "sigma_min": 0.0, "gate": "REJECT",
    }


def test_fake_training_executes_all_1704_segments_losslessly(tmp_path):
    output = tmp_path / "training"
    backend = FakeTrainingBackend(output_root=output); progress = MemoryProgress()
    result = run_training(backend, progress)
    assert result["additional_transitions"] == 872000
    assert result["new_updates"] == 1704
    assert result["checkpoint_count"] == 70
    assert result["target_update_counter"] == 1954
    assert progress.latest["status"] == "PASS"
    validate_training_semantic(result, result_root=output)


def test_training_semantic_rejects_missing_or_tampered_evidence(tmp_path):
    output = tmp_path / "training"
    result = run_training(FakeTrainingBackend(output_root=output), MemoryProgress())
    for mutation, match in (
        (lambda value: value.__setitem__("initial_preflight", {}), "preflight"),
        (lambda value: value["checkpoints"].__setitem__(0, {}), "receipt"),
        (lambda value: value["checkpoints"][0].__setitem__("sha256", "0" * 64), "hash"),
        (lambda value: value["final_reload_audit"].__setitem__("checkpoint_sha256", "0" * 64), "reload"),
    ):
        altered = copy.deepcopy(result); mutation(altered)
        with pytest.raises(ValueError, match=match):
            validate_training_semantic(altered, result_root=output)


def test_fake_training_interruption_preserves_partial_state():
    backend = FakeTrainingBackend(fail_at=3); progress = MemoryProgress()
    with pytest.raises(RuntimeError, match="inert interruption"):
        run_training(backend, progress)
    assert backend.preserved and progress.failures
    assert progress.failures[0][1]["completed_transitions"] == 1024
    assert progress.failures[0][1]["last_checkpoint_counter"] == 250


def test_persisted_training_failure_accounting_mid_rollout_and_partial_update(tmp_path):
    from training_workloads.scale_003.live_training import _LiveProgress

    progress = _LiveProgress(ProgressStore(tmp_path / "mid", "training"))
    class MidRollout(FakeTrainingBackend):
        def collect_segment(self, segment):
            for step in range(1, 4):
                progress.write_step_attempt(step); progress.write_step_complete(step)
            progress.write_step_attempt(4)
            raise RuntimeError("inert failure after three completed steps")
    with pytest.raises(RuntimeError): run_training(MidRollout(), progress)
    mid = json.loads((tmp_path / "mid/failure.json").read_text())
    assert (mid["attempted_transitions"], mid["completed_transitions"]) == (128, 96)
    assert mid["last_checkpoint_sha256"] == load_config()["pins"]["checkpoint_sha256"]

    progress2 = _LiveProgress(ProgressStore(tmp_path / "update", "training"))
    class PartialUpdate(FakeTrainingBackend):
        def update(self, batch):
            self.optimizer = 2001
            raise RuntimeError("inert partial optimizer update")
        def failure_accounting(self):
            return {"completed_updates": 0, "update_counter": 250, "actor_optimizer_step": 2001,
                    "critic_optimizer_step": 2000, "optimizer_state_status": "PARTIAL_UPDATE_STATE"}
    with pytest.raises(RuntimeError): run_training(PartialUpdate(), progress2)
    partial = json.loads((tmp_path / "update/failure.json").read_text())
    assert (partial["attempted_transitions"], partial["completed_transitions"]) == (512, 512)
    assert partial["completed_updates"] == 0
    assert (partial["actor_optimizer_step"], partial["critic_optimizer_step"], partial["optimizer_state_status"]) == (2001, 2000, "PARTIAL_UPDATE_STATE")


@pytest.mark.parametrize("case", ["station", "tracking"])
def test_fake_evaluation_exact_two_environment_campaign(case, tmp_path):
    backend = FakeEvaluationBackend(); progress = MemoryProgress()
    result = run_evaluation(case, tmp_path / "final.pt", backend, progress)
    assert result["environment_transitions"] == 512 and len(result["rows"]) == 256
    assert result["performance_gate"] is False and result["optimizer_updates"] == 0
    validate_evaluation_semantic(result, case=case)
    if case == "tracking":
        assert evaluation_command_at(case, 31) != evaluation_command_at(case, 32)


def test_evaluation_semantic_and_post_step_failure_are_strict(tmp_path):
    result = run_evaluation("station", tmp_path / "final.pt", FakeEvaluationBackend(), MemoryProgress())
    mutations = (
        lambda value: value.__setitem__("initial_preflight", {}),
        lambda value: value["summary"].pop("runtime_evaluation"),
        lambda value: value["summary"].__setitem__("canonical_record", {"tampered": True}),
        lambda value: value["summary"]["runtime_evaluation"].__setitem__("canonical_record", '{"metric":NaN,"metric":1}'),
        lambda value: value["summary"]["runtime_evaluation"]["source_manifest"]["collision"].__setitem__("source", "tampered"),
    )
    for mutation in mutations:
        altered = copy.deepcopy(result); mutation(altered)
        with pytest.raises((ConfigError, ValueError)):
            validate_evaluation_semantic(altered, case="station")
    for mutation in (
        lambda record: record["observability_source_counts"].__setitem__("EXACT", 511),
        lambda record: record["termination_counts"].__setitem__("reason_none", 511),
    ):
        altered = copy.deepcopy(result)
        record = altered["summary"]["canonical_record"]
        mutation(record)
        altered["summary"]["runtime_evaluation"]["canonical_record"] = json.dumps(
            record, ensure_ascii=True, allow_nan=False, separators=(",", ":"),
        )
        with pytest.raises(ValueError):
            validate_evaluation_semantic(altered, case="station")
    altered = copy.deepcopy(result)
    record = altered["summary"]["canonical_record"]
    record["metrics"]["collision_rate"].update(numerator=256.0, value=0.5)
    altered["summary"]["runtime_evaluation"]["canonical_record"] = json.dumps(
        record, ensure_ascii=True, allow_nan=False, separators=(",", ":"),
    )
    with pytest.raises(ValueError, match="source samples"):
        validate_evaluation_semantic(altered, case="station")
    progress = ProgressStore(tmp_path / "eval-failure", "station")
    class AfterStep(FakeEvaluationBackend):
        def evaluate_step(self, case, index, command):
            self.completed_transitions += 2
            raise RuntimeError("collector failed after inert env step")
    with pytest.raises(RuntimeError): run_evaluation("station", tmp_path / "final.pt", AfterStep(), progress)
    failure = json.loads((tmp_path / "eval-failure/failure.json").read_text())
    assert (failure["attempted_transitions"], failure["completed_transitions"], failure["completed_policy_steps"]) == (2, 2, 1)


def test_evaluation_float32_and_json_object_roundtrip(tmp_path):
    import torch

    result = run_evaluation("tracking", tmp_path / "final.pt", FakeEvaluationBackend(), MemoryProgress())
    command = result["summary"]["runtime_evaluation"]["evaluation_inputs"]["commanded_velocity_b_m_s"][32][0][0]
    assert command == torch.tensor(0.2, dtype=torch.float32).item()
    assert command != 0.2

    path = atomic_json_replace(tmp_path, "semantic.json", result)
    loaded = load_strict_json(path)
    validate_evaluation_semantic(loaded, case="tracking")

    def reverse_objects(value):
        if isinstance(value, dict):
            return {key: reverse_objects(value[key]) for key in reversed(tuple(value))}
        if isinstance(value, list):
            return [reverse_objects(item) for item in value]
        return value

    reordered = reverse_objects(loaded)
    validate_evaluation_semantic(reordered, case="tracking")

    wrong_schedule = copy.deepcopy(loaded)
    commands = wrong_schedule["summary"]["runtime_evaluation"]["evaluation_inputs"]["commanded_velocity_b_m_s"]
    commands[31], commands[32] = commands[32], commands[31]
    with pytest.raises(ValueError, match="schedule"):
        validate_evaluation_semantic(wrong_schedule, case="tracking")

    wrong_order = copy.deepcopy(loaded)
    record = wrong_order["summary"]["canonical_record"]
    record["evaluation_input_order"] = list(reversed(record["evaluation_input_order"]))
    wrong_order["summary"]["runtime_evaluation"]["canonical_record"] = json.dumps(
        record, ensure_ascii=True, allow_nan=False, separators=(",", ":"),
    )
    with pytest.raises(ValueError, match="dimensions/order"):
        validate_evaluation_semantic(wrong_order, case="tracking")


def test_atomic_persistence_is_confined(tmp_path):
    path = atomic_json_replace(tmp_path, "nested/value.json", {"status": "PASS", "value": 3})
    assert json.loads(path.read_text()) == {"status": "PASS", "value": 3}
    assert not list(path.parent.glob(".*.tmp"))
    with pytest.raises(Exception):
        atomic_json_replace(tmp_path, "../escape.json", {})
    progress = ProgressStore(tmp_path, "training")
    progress.fail(ValueError("semantic"), completed_transitions=64)
    assert json.loads((tmp_path / "failure.json").read_text())["failure_class"] == "SEMANTIC_CONTRACT_FAILURE"


def test_resource_gates_order_thresholds_and_failure_persistence(tmp_path):
    assert validate_coexistence_window(resource_window())["status"] == "PASS"
    with pytest.raises(MemoryError): validate_coexistence_window(resource_window(free=10239, used=1988))
    with pytest.raises(RuntimeError): validate_coexistence_window(resource_window(utilization=6))
    evidence = run_prelaunch_gate(tmp_path / "pass", mode="training", attempt_id="INERT", authorization_sha256="a" * 64,
                                  collector=lambda: resource_window())
    assert evidence["status"] == "PASS" and (tmp_path / "pass/prelaunch_resource_gate.json").is_file()
    with pytest.raises(MemoryError):
        run_prelaunch_gate(tmp_path / "fail", mode="evaluation", attempt_id="INERT", authorization_sha256="a" * 64,
                           collector=lambda: resource_window(free=10239, used=1988))
    failure = json.loads((tmp_path / "fail/prelaunch_resource_gate_failure.json").read_text())
    assert failure["observed_window"]["samples"][0]["gpu"]["free_memory_mib"] == 10239
    calls = []
    class Launcher:
        app = object()
    gate, launcher, app = start_after_coexistence_gate(
        tmp_path / "ordered", mode="training", attempt_id="INERT", authorization_sha256="a" * 64,
        gate_runner=lambda *args, **kwargs: calls.append("gate") or {"status": "PASS"},
        launcher_factory=lambda log: calls.append("launcher") or Launcher(),
    )
    assert calls == ["gate", "launcher"] and gate["status"] == "PASS" and app is launcher.app
    calls.clear()
    with pytest.raises(MemoryError):
        start_after_coexistence_gate(
            tmp_path / "ordered-fail", mode="evaluation", attempt_id="INERT", authorization_sha256="a" * 64,
            gate_runner=lambda *args, **kwargs: calls.append("gate") or (_ for _ in ()).throw(MemoryError("busy")),
            launcher_factory=lambda log: calls.append("launcher") or Launcher(),
        )
    assert calls == ["gate"]
    observations = iter((
        {"free_memory_bytes": 2 * 1024 ** 3, "total_memory_bytes": 12 * 1024 ** 3, "allocated_memory_bytes": 1},
        {"free_memory_bytes": 1024 ** 3 - 1, "total_memory_bytes": 12 * 1024 ** 3, "allocated_memory_bytes": 1},
    ))
    guard = RuntimeResourceGuard(tmp_path / "runtime", observer=lambda: next(observations))
    guard.check(stage="before_rollout", ordinal=1)
    with pytest.raises(MemoryError): guard.check(stage="before_rollout", ordinal=2)
    assert json.loads((tmp_path / "runtime/runtime_resource_failure.json").read_text())["ordinal"] == 2


def test_coexistence_collection_anchors_after_slow_first_sample():
    now = 0.0
    latencies = iter((3.0, 1.0, 1.0))

    def monotonic():
        return now

    def sleeper(seconds):
        nonlocal now
        now += seconds

    def provider():
        nonlocal now
        now += next(latencies)
        sample = resource_window()["samples"][0]
        sample["monotonic_ns"] = int(now * 1_000_000_000)
        return sample

    window = collect_coexistence_window(provider, sleeper, monotonic)
    assert window["span_ns"] == 11_000_000_000
    assert validate_coexistence_window(window)["status"] == "PASS"


def test_two_stage_manifests_preserve_training_inputs_and_bind_sealed_handoff(tmp_path):
    sys.path.insert(0, "/home/mint/rl_dev/instinctRL_coordination")
    from evaluation_tooling.schemas import validate_input_hashes, validate_live_launch_profile, validate_manifest

    training_attempt = tmp_path / "SCALE-003-T0001"; training_attempt.mkdir()
    evaluation_attempt = tmp_path / "SCALE-003-E0001"; evaluation_attempt.mkdir()
    authorization = tmp_path / "authorization.json"; authorization.write_text("{}\n")
    config = load_config()
    training_manifest, training_hashes = build_attempt_documents(
        mode="training", attempt_id=training_attempt.name, attempt_dir=training_attempt,
        execution_authorization=authorization, checkpoint=Path(config["pins"]["checkpoint_path"]),
        source_dependencies=[], candidate_commit="a" * 40, candidate_tree="b" * 40,
    )
    frozen_training = json.dumps([training_manifest, training_hashes], sort_keys=True, separators=(",", ":"))
    assert validate_manifest(training_manifest, training_attempt) == ()
    assert validate_input_hashes(training_hashes, training_manifest) == ()
    assert validate_live_launch_profile(training_manifest, training_hashes, training_attempt, PYTHON) == ()

    output = training_attempt / "results/training"
    semantic = run_training(FakeTrainingBackend(output_root=output), MemoryProgress())
    atomic_json_replace(output, "semantic.json", semantic)
    atomic_json_replace(output, "status.json", {"schema_version": "scale-003-status.v1", "status": "PASS", "mode": "training"})
    bound = []
    for relative in ("results/training/semantic.json", "results/training/status.json", "results/training/checkpoints/update_1954_final.pt"):
        path = training_attempt / relative
        bound.append({"path": relative, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mode": 420, "type": "file"})
    atomic_json_replace(training_attempt, "TERMINAL_MANIFEST.json", {
        "schema_version": "evaluation-terminal-manifest.v2", "attempt_id": training_attempt.name,
        "lifecycle": "SEALED", "files": bound,
    })
    handoff = build_training_handoff_document(training_attempt=training_attempt, evaluation_attempt=evaluation_attempt)
    evaluation_manifest, evaluation_hashes = build_attempt_documents(
        mode="evaluation", attempt_id=evaluation_attempt.name, attempt_dir=evaluation_attempt,
        execution_authorization=authorization, checkpoint=Path(handoff["final_checkpoint_path"]),
        source_dependencies=[], candidate_commit="a" * 40, candidate_tree="b" * 40,
        training_handoff=handoff,
    )
    assert json.dumps([training_manifest, training_hashes], sort_keys=True, separators=(",", ":")) == frozen_training
    assert training_manifest["commands"][0]["label"] == "scale_training"
    assert [row["label"] for row in evaluation_manifest["commands"]] == ["evaluate_station", "evaluate_tracking"]
    assert validate_manifest(evaluation_manifest, evaluation_attempt) == ()
    assert validate_input_hashes(evaluation_hashes, evaluation_manifest) == ()
    assert validate_live_launch_profile(evaluation_manifest, evaluation_hashes, evaluation_attempt, PYTHON) == ()
    pinned = {row["path"]: row["sha256"] for row in evaluation_hashes["files"]}
    assert pinned[str(training_attempt / "TERMINAL_MANIFEST.json")] == handoff["terminal_manifest_sha256"]
    assert pinned[handoff["final_checkpoint_path"]] == handoff["final_checkpoint_sha256"]
    altered = dict(handoff); altered["final_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        build_attempt_documents(
            mode="evaluation", attempt_id=evaluation_attempt.name, attempt_dir=evaluation_attempt,
            execution_authorization=authorization, checkpoint=Path(handoff["final_checkpoint_path"]),
            source_dependencies=[], candidate_commit="a" * 40, candidate_tree="b" * 40,
            training_handoff=altered,
        )
