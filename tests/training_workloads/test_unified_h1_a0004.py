from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS

import pytest
import torch

from training_workloads.unified_h1.backend import UnifiedH1LiveBackend, _state_fingerprint, create_live_backend
from training_workloads.unified_h1.budget import ExecutionCounters
from training_workloads.unified_h1.driver import UnifiedH1TrainingLoop
from training_workloads.unified_h1.evaluation import (
    case_initial_positions, continuation_decision, evaluate_episode, replay_raw_case,
    stopping_confirmation,
)
from training_workloads.unified_h1.logging import LearningLog
from training_workloads.unified_h1.schedule import command_at
from test_unified_h1_rework import _summary, _trace


class _EvaluationEnvironment:
    device = "cpu"

    def __init__(self):
        self.scene = NS(env_origins=torch.zeros(32, 3))
        self._robot = NS(data=NS(root_state_w=torch.zeros(32, 13)))
        self._evaluation_ray_snapshot = None
        self.command = torch.zeros(32, 3)

    def set_h1_evaluation_case(self, case): self.case = case
    def set_h1_initial_command(self, command, step): self.command = command.clone()

    def reset(self, seed):
        root = torch.zeros(32, 13); root[:, :3] = case_initial_positions(self.case); root[:, 3] = 1
        self._robot.data.root_state_w = root
        sensor = root[:, :3] + torch.tensor([0., 0., .05])
        self._evaluation_ray_snapshot = NS(ray_starts_w=sensor[:, None, :])
        return {}, {}

    def _refresh_task_snapshot(self): return None
    def prepare_b0_step(self, command_b, next_command_b): self.command = command_b.clone()

    def step(self, action):
        mask = torch.zeros(32, 1, dtype=torch.bool)
        sensor = self._robot.data.root_state_w[:, :3] + torch.tensor([0., 0., .05])
        extras = {
            "pre_reset": {"task": {"actual_velocity_b": self.command.clone(),
                "termination": {"masks": {"collision": mask, "below_height": mask, "above_height": mask}},
                "exact_clearance_m": torch.ones(32, 1)},
                "control": {"ics_metrics": {"beta": torch.ones(32, 1)}}},
            "evaluation_ray_source": {"ray_starts_w": sensor[:, None, :]},
            "reward_diagnostics": {"q": torch.ones(32, 1, dtype=torch.bool)},
        }
        return {}, torch.zeros(32), mask[:, 0], mask[:, 0], extras


def test_actual_backend_and_raw_replay_use_product_q_ics_scalar_lanes(tmp_path):
    backend = object.__new__(UnifiedH1LiveBackend)
    backend.environment = _EvaluationEnvironment(); backend.output = tmp_path
    backend.counters = ExecutionCounters(); backend.resource_check = lambda *unused: None
    backend.authorization = {"budgets": {"explicit_resets": 1, "nontraining_transitions": 3200,
                                            "automatic_resets": 3200}}
    backend._evaluation_rows = {}
    backend._evaluate_case("T", direct_b0=True, evaluation_label="actual_shapes")
    direct = backend._evaluation_rows["T"]
    replayed = replay_raw_case(str(tmp_path / "evaluation_raw/actual_shapes/T.jsonl"), case="T")
    assert direct[0].q.shape == direct[0].ics_beta.shape == (100,)
    assert replayed[0].q.shape == replayed[0].ics_beta.shape == (100,)
    assert evaluate_episode(direct[0]) == evaluate_episode(replayed[0])


def test_t0_post_step_and_x_deadline_boundaries_are_not_backdated():
    assert stopping_confirmation([0.] * 11, zero_effective_index=0)["confirmation_time_s"] == pytest.approx(1.1)
    s = evaluate_episode(_trace("S", 0))
    assert s["success"] and s["metrics"]["stop_time"] == pytest.approx(1.1)
    for case in ("T", "X"):
        trace = _trace(case, 0)
        positions = trace.sensor_positions_w_m.clone()
        positions[:, 2] = trace.initial_sensor_position_w_m[2] + .21
        result = evaluate_episode(replace(trace, sensor_positions_w_m=positions))
        assert not result["success"] and result["metrics"]["height"] > .20
    x = _trace("X", 0); velocity = x.velocity_b.clone(); velocity[50:70] = 0
    late = evaluate_episode(replace(x, velocity_b=velocity))
    assert not late["success"] and late["reason"] == "NONZERO_REACH_CONFIRMATION_FAILED"
    assert late["metrics"]["reach_confirmation_time_s"][0] == pytest.approx(3.1)


def test_continuation_rejects_empty_or_inconsistent_episode_evidence():
    good = dict(baseline=_summary(0, .8), current=_summary(256, .4),
                previous=_summary(192, .5), u64=_summary(64, .5))
    assert continuation_decision(**good)["extend"]
    empty = _summary(256, .4); empty["cases"] = {name: {**value, "subgroups": {}}
                                                 for name, value in empty["cases"].items()}
    assert not continuation_decision(**{**good, "current": empty})["extend"]
    inconsistent = _summary(256, .4); inconsistent["episodes"][0]["success"] = False
    assert not continuation_decision(**{**good, "current": inconsistent})["extend"]
    forged_gate = _summary(256, .4); forged_gate["engineering_gates_pass"] = False
    assert not continuation_decision(**{**good, "current": forged_gate})["extend"]


def _ray_source(valid_first=False):
    valid = torch.zeros(32, 2, dtype=torch.bool); valid[0, 0] = valid_first
    return {"accepted_ray_hits_w": torch.full((32, 2, 3), float("inf")),
            "accepted_beam_valid": valid, "mesh_path": "/World/H1RayScene/CombinedMesh",
            "maximum_distance_m": 40., "source_step": torch.ones(32, dtype=torch.long)}


def test_legal_ray_misses_are_durable_compact_and_distinct_from_bad_valid_samples(tmp_path):
    log = LearningLog(tmp_path / "legal")
    log.record_step_evidence({"evaluation_ray_source": _ray_source()}, completed_policy_steps=1)
    row = json.loads((tmp_path / "legal/partial_rollout_steps.jsonl").read_text())
    dense = row["evidence"]["evaluation_ray_source"]["dense_contract"]
    assert dense["status"] == "RECORDED" and dense["legal_positive_infinity_miss_counts"][0] == 2
    assert "accepted_ray_hits_w" not in row["evidence"]["evaluation_ray_source"]
    log.append("selected.jsonl", _ray_source())
    selected = json.loads((tmp_path / "legal/selected.jsonl").read_text())
    assert selected["accepted_ray_hits_w"][0][0][0] == {"encoding":"POSITIVE_INFINITY", "status":"RAY_MISS"}
    bad = LearningLog(tmp_path / "bad")
    bad.record_step_evidence({"evaluation_ray_source": _ray_source(valid_first=True)}, completed_policy_steps=1)
    bad_dense = json.loads((tmp_path / "bad/partial_rollout_steps.jsonl").read_text())["evidence"]["evaluation_ray_source"]["dense_contract"]
    assert bad_dense["status"] == "INVALID_VALID_BEAM" and bad_dense["valid_nonfinite_counts"][0] == 1


def test_reset_splits_command_block_and_cannot_confirm_across_episodes(tmp_path):
    log = LearningLog(tmp_path); root = torch.zeros(32, 13); root[:, 2] = 2
    for step in range(12):
        command = torch.zeros(32, 3)
        if step == 0: command[:, 0] = .2
        reset = torch.zeros(32, dtype=torch.bool)
        if step == 5: reset[0] = True
        log._update_command_blocks({"pre_reset": {"root_state_w": root,
            "task": {"actual_velocity_b": torch.zeros(32, 3)}},
            "_h1_driver": {"global_policy_step": step}}, command, reset)
    log._finish_command_block(log._block_states[0], right_censored=True)
    rows = [json.loads(line) for line in (tmp_path / "command_blocks.jsonl").read_text().splitlines()]
    lane = [row for row in rows if row["env"] == 0]
    assert all(row["stop_confirmation"]["status"] != "CONFIRMED" for row in lane)
    assert any(row["stop_confirmation"].get("reason") == "RESET_INTERRUPTED_NO_SPLICE" for row in lane)


def test_resource_decline_after_rollout_stops_before_optimizer(monkeypatch, tmp_path):
    import instinctlab.training.rollout_h1 as rollout_module
    events = []; available = {"value": True}
    def guard(ordinal):
        events.append("resource_check")
        if not available["value"]: raise RuntimeError("resource decline")
    def collect(*unused):
        events.append("collection_returns_after_resource_decline"); available["value"] = False
        return NS(segment=NS(batch=None))
    monkeypatch.setattr(rollout_module, "collect_h1_rollout", collect)
    noop = lambda *args, **kwargs: None
    journal = NS(durable_step_rows=0, record_update_attempt=noop, record_rollout=lambda *args, **kwargs: {},
                 record_update=noop, record_partial_episodes=noop, record_partial_rollout_boundary=noop,
                 record_partial_failure=noop)
    driver = NS(counters=ExecutionCounters(), observation={}, take_evidence=lambda: (), evidence_buffer=[], journal=None)
    optimizer = NS(state_dict=lambda: {"state": {}})
    trainer = NS(update_counter=0, update=lambda batch: events.append("trainer_update_entered"),
                 actor_optimizer=optimizer, critic_optimizer=optimizer)
    loop = UnifiedH1TrainingLoop(driver, trainer, journal, before_update=guard, update_limit=1)
    with pytest.raises(RuntimeError, match="resource decline"): loop.run_update()
    assert events == ["resource_check", "collection_returns_after_resource_decline", "resource_check"]
    assert driver.counters.attempted_updates == 1 and driver.counters.completed_updates == 0


def test_invalid_visibility_is_rejected_before_prelaunch_resource_start(monkeypatch, tmp_path):
    import training_workloads.unified_h1.backend as backend_module
    workload = Path(__file__).resolve().parents[2]
    product = Path("/home/mint/rl_dev/worktrees/instinctRL-unified-h1-r2-a0004-product")
    config = {"execution": {"cuda_visible_devices_required": "0"},
              "product": {"path": str(product)}, "workload": {"path": str(workload)}}
    monkeypatch.setattr(backend_module, "load_config", lambda: config)
    starts = []
    monkeypatch.setattr(backend_module, "start_after_coexistence_gate", lambda *a, **k: starts.append("started"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    attempt = tmp_path / "attempt"; (attempt / "results").mkdir(parents=True)
    with pytest.raises(PermissionError, match="visibility"):
        create_live_backend(authorization={"mode":"main", "authorization_sha256":"a" * 64}, attempt=attempt)
    assert starts == [] and not (attempt / "results/prelaunch").exists()


def test_wrong_statically_resolved_product_source_is_rejected_before_start(monkeypatch, tmp_path):
    import training_workloads.unified_h1.backend as backend_module
    workload = Path(__file__).resolve().parents[2]
    frozen = json.loads((workload / "training_workloads/unified_h1/candidate_config.json").read_text())
    config = {"execution": frozen["execution"],
              "product": {"path": "/home/mint/rl_dev/worktrees/instinctRL-unified-h1-r2-a0003-product"},
              "workload": {"path": str(workload)}}
    monkeypatch.setattr(backend_module, "load_config", lambda: config)
    starts = []
    monkeypatch.setattr(backend_module, "start_after_coexistence_gate", lambda *a, **k: starts.append("started"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("ISAACSIM_ASSET_ROOT", frozen["execution"]["isaacsim_asset_root"])
    attempt = tmp_path / "attempt"; (attempt / "results").mkdir(parents=True)
    with pytest.raises(ValueError, match="product module source"):
        create_live_backend(authorization={"mode":"main", "authorization_sha256":"a" * 64}, attempt=attempt)
    assert starts == [] and not (attempt / "results/prelaunch").exists()


def test_backend_wall_clock_includes_preconstruction_elapsed_time(tmp_path):
    (tmp_path / "results/main").mkdir(parents=True)
    backend = UnifiedH1LiveBackend(authorization={"mode":"main", "budgets":{"wall_seconds":1}},
        attempt=tmp_path, app=None, environment=NS(), resource_guard=NS(check=lambda **unused: None),
        process_started_monotonic=time.monotonic() - 2)
    with pytest.raises(TimeoutError, match="wall-time"): backend.resource_check("after_initialization", 0)


def test_reload_fingerprint_is_deterministic_and_detects_tensor_or_optimizer_changes():
    state = {"model":{"weight":torch.arange(6).reshape(2,3)}, "optimizer":{"state":{0:{"step":torch.tensor(8)}}}}
    clone = {"model":{"weight":state["model"]["weight"].clone()},
             "optimizer":{"state":{0:{"step":torch.tensor(8)}}}}
    assert _state_fingerprint(state) == _state_fingerprint(clone)
    clone["optimizer"]["state"][0]["step"] += 1
    assert _state_fingerprint(state) != _state_fingerprint(clone)
