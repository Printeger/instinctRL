from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training_workloads.scale_003.lifecycle import run_lifecycle
from training_workloads.unified_h1.authorization import load_r3_authorization
from training_workloads.unified_h1.budget import ExecutionCounters, budget_manifest
from training_workloads.unified_h1.config import load_config
from training_workloads.unified_h1.driver import UnifiedH1Driver
from training_workloads.unified_h1.evaluation import (
    case_commands, case_initial_positions, case_outcome, continuation_decision, evaluate_stop_episode, evaluation_manifest, normalized_loss, select_checkpoint,
    sensor_origin_surface_distance, stopping_confirmation, wall_motion_prerequisite, wilson_interval,
)
from training_workloads.unified_h1.formal_unified_h1 import validate_only
from training_workloads.unified_h1.logging import LearningLog, grouped_reward_summary
from training_workloads.unified_h1.schedule import (
    DIRECTIONS, NUM_ENVS, R4_STEPS, R5_MAX_STEPS, ScheduleClock, command_at, plan_step,
    reset_local_positions,
)
from training_workloads.unified_h1.r3_sequence import run_r3_controlled_failure, run_r3_main


def test_schedule_boundaries_role_swap_and_direction_counts():
    assert torch.count_nonzero(command_at(1023)) == 0
    first = command_at(1024); last = command_at(1151); swapped = command_at(1152)
    assert torch.equal(first, last)
    assert (torch.linalg.vector_norm(first[:16], dim=-1) > 0).all() and (first[16:] == 0).all()
    assert (swapped[:16] == 0).all() and (torch.linalg.vector_norm(swapped[16:], dim=-1) > 0).all()
    for block_start in range(1024, R5_MAX_STEPS, 128):
        command = command_at(block_start)
        nonzero = command[torch.linalg.vector_norm(command, dim=-1) > 0]
        assert nonzero.shape[0] == 16
        counts = [int(torch.isclose(nonzero, torch.tensor(direction)).all(dim=-1).sum()) for direction in DIRECTIONS]
        assert counts == [2] * 8


def test_every_mixed_rollout_is_256_zero_and_256_nonzero_samples():
    for start in (1024, 1136, 1152, 4080, R5_MAX_STEPS - 16):
        counts = [int((torch.linalg.vector_norm(command_at(step), dim=-1) <= 1e-3).sum()) for step in range(start, start + 16)]
        assert sum(counts) == 256


def test_clock_reset_and_resume_do_not_restart_curriculum():
    clock = ScheduleClock(1151); before = clock.plan(); clock.reset_lane(7); assert clock.plan().global_policy_step == 1151
    clock.complete_step(); assert clock.plan().global_policy_step == 1152
    restored = ScheduleClock(); restored.load_state_dict(clock.state_dict()); assert restored.plan().global_policy_step == 1152
    assert not torch.equal(before.command_b, restored.plan().command_b)


def test_reset_distribution_is_24_center_4_x_4_y_after_warmup():
    warm, warm_labels = reset_local_positions(0); mixed, labels = reset_local_positions(1024)
    assert set(warm_labels) == {"center"} and torch.equal(warm[:, :2], torch.zeros(32, 2))
    assert labels.count("center") == 24 and labels.count("near_x_wall") == 4 and labels.count("near_y_wall") == 4
    assert torch.equal(mixed[torch.tensor([6, 14, 22, 30]), 0], torch.full((4,), 4.6))


class FakeEnvironment:
    def __init__(self, initial): self.current = initial; self.prepared = None
    def prepare_h1_step(self, **value): self.prepared = value
    def step(self, action):
        state = self.current["policy"]["state_vec"].clone(); state[:, 45:48] = self.prepared["next_command_b"]
        next_obs = {"policy": {"lidar_grid": self.current["policy"]["lidar_grid"].clone(), "state_vec": state}, "critic": torch.zeros(32, 21)}
        self.current = next_obs
        return next_obs, torch.zeros(32), torch.zeros(32, dtype=torch.bool), torch.zeros(32, dtype=torch.bool), {}


def _observation(step):
    state = torch.zeros(32, 52); state[:, 45:48] = command_at(step)
    return {"policy": {"lidar_grid": torch.zeros(32, 12, 59, 360), "state_vec": state}, "critic": torch.zeros(32, 21)}


def test_driver_pairs_obs_action_reward_next_command_and_attempt_first_accounting():
    observation = _observation(1023); env = FakeEnvironment(observation); counters = ExecutionCounters()
    driver = UnifiedH1Driver(env, observation, clock=ScheduleClock(1023), counters=counters)
    returned = driver.step(torch.zeros(32, 4))
    assert env.prepared["global_policy_step"] == 1023
    assert torch.equal(env.prepared["command_b"], command_at(1023))
    assert torch.equal(returned[0]["policy"]["state_vec"][:, 45:48], command_at(1024))
    assert counters.attempted_policy_steps == counters.completed_policy_steps == 1
    assert counters.real_environment_transitions == 32


def test_driver_failure_keeps_attempt_without_completion():
    class Failing(FakeEnvironment):
        def step(self, action): raise RuntimeError("step failed")
    observation = _observation(0); counters = ExecutionCounters(); driver = UnifiedH1Driver(Failing(observation), observation, counters=counters)
    with pytest.raises(RuntimeError): driver.step(torch.zeros(32, 4))
    assert counters.attempted_policy_steps == 1 and counters.completed_policy_steps == 0 and counters.real_environment_transitions == 0


def test_stop_confirmation_uses_11_samples_and_never_splices_reset():
    assert stopping_confirmation([.08] * 10, zero_effective_index=0)["status"] == "UNAVAILABLE"
    assert stopping_confirmation([.08] * 11, zero_effective_index=0)["confirmation_time_s"] == pytest.approx(1.1)
    positions = torch.zeros(220, 3)
    passed = evaluate_stop_episode(speed_norm_m_s=[.0] * 220, positions_w_m=positions, zero_effective_index=0, episode_ids=[4] * 220)
    assert passed["success"] and passed["reference_index"] == 10
    failed = evaluate_stop_episode(speed_norm_m_s=[.0] * 220, positions_w_m=positions, zero_effective_index=0, episode_ids=[4] * 100 + [5] * 120)
    assert failed == {"success": False, "reason": "RESET_INTERRUPTED_NO_SPLICE"}


def test_wall_cannot_pass_by_never_moving_and_geometry_distance_is_independent():
    assert not wall_motion_prerequisite([0.0] * 200)
    assert wall_motion_prerequisite([0.1] * 11)
    distance, surfaces = sensor_origin_surface_distance(torch.tensor([[4.6, 0.0, 2.0], [0.0, 0.0, 0.7]]))
    assert torch.allclose(distance, torch.tensor([1.4, .7])) and surfaces == ("x_wall", "floor")
    with pytest.raises(ValueError): sensor_origin_surface_distance(torch.tensor([[float("nan"), 0.0, 1.0]]))


def test_denominators_subgroups_wilson_and_loss_penalties():
    groups = [f"d{index % 8}" for index in range(32)]
    outcome = case_outcome([True] * 32, groups, expected_subgroups=[f"d{i}" for i in range(8)])
    assert outcome["pass"] and outcome["total"] == 32 and 0 < outcome["wilson_95"][0] < 1
    failure = case_outcome([False] + [True] * 31, groups, expected_subgroups=[f"d{i}" for i in range(8)])
    assert not failure["pass"]
    metrics = {"z_speed": .08, "z_drift": .3, "z_height": .2, "t_rmse": .15, "s_stop_time": 3., "s_drift": .3, "x_rmse": .15, "w_min_distance": .8}
    assert normalized_loss(metrics, {}) == pytest.approx(7 / 8)
    assert normalized_loss(metrics, {"t_rmse": True}) > normalized_loss(metrics, {})


def test_budget_and_evaluation_arithmetic():
    budget = budget_manifest(); evaluation = evaluation_manifest()
    assert evaluation["transitions_per_checkpoint"] == 32960
    assert budget["training_transitions_r3_through_r5"] == 1050624
    assert budget["nontraining_transitions_r3_through_r5"] == 396032
    assert budget["all_transitions_r3_through_r5"] == 1446656
    assert budget["initialization_ticks_r3_through_r5_maximum"] == 100
    assert torch.count_nonzero(case_commands("Z")) == 0
    assert torch.equal(case_commands("S")[:100], case_commands("T"))
    assert torch.count_nonzero(case_commands("S")[100:]) == 0
    assert torch.equal(case_commands("X")[125:], -case_commands("T")[:75])
    assert torch.all(case_initial_positions("W")[:16, 0] == 4.6)
    assert torch.all(case_initial_positions("W")[16:, 1] == 4.6)


def test_continuation_and_checkpoint_selection_are_frozen():
    baseline = {"z_speed": .10, "z_drift": .4, "z_height": .2, "t_rmse": .20, "s_stop_time": 4., "s_drift": .4, "x_rmse": .20, "w_min_distance": .8}
    current = {"z_speed": .08, "z_drift": .3, "z_height": .2, "t_rmse": .15, "s_stop_time": 3., "s_drift": .3, "x_rmse": .15, "w_min_distance": .8}
    assert not continuation_decision(baseline=baseline, current=current, previous_collisions=0, current_collisions=0)["extend"]
    def summary(update, value, collisions=0):
        cases = {name:{"successes":32,"total":32,"pass":True,"wilson_95":[.89,1.],
            "subgroups":({f"d{i}":{"successes":4,"total":4,"pass":True} for i in range(8)} if name != "W" else
                         {axis:{"successes":16,"total":16,"pass":True} for axis in ("x_wall","y_wall")})} for name in ("Z","T","S","X","W")}
        interval=wilson_interval(32,32)
        for outcome in cases.values(): outcome["wilson_95"]=interval
        episodes=[]
        metrics={"Z":{"speed":value*.08,"drift":value*.3,"height":value*.2},
            "T":{"rmse":value*.15},"S":{"stop_time":value*3,"drift":value*.3},
            "X":{"rmse":value*.15},"W":{"min_distance":.8*(1-value)}}
        for name in ("Z","T","S","X","W"):
            for lane in range(32):
                episodes.append({"case":name,"env_index":lane,"success":True,"reason":"PASS","metrics":metrics[name],
                    "engineering":{"collision":collisions == 1 and name == "Z" and lane == 0,
                        "height_failure":False,"missing":False,"reset_interrupted":False,
                        "evidence_valid":True}})
        return {"schema_version":"metric-h1-v1.checkpoint.v1", "update":update, "episodes":episodes, "cases":cases,
                "error_components":{key:value for key in baseline}, "loss":value,
                "failures":{key:0 for key in ("Z","T","S","X","W")}, "collisions":collisions,
                "height_failures":0, "invalid_evidence_episodes":0,
                "engineering_gates_pass":collisions == 0, "all_gates_pass":collisions == 0}
    decision = continuation_decision(baseline=summary(0, .8), current=summary(256, .4), previous=summary(192, .5), u64=summary(64, .5))
    assert decision["extend"]
    blocked = continuation_decision(baseline=summary(0, .8), current=summary(256, .4), previous=summary(192, .5, 1), u64=summary(64, .5))
    assert not blocked["extend"]
    assert select_checkpoint([summary(256, .7), summary(192, .7)])["update"] == 192
    assert select_checkpoint([{"schema_version":"metric-h1-v1.checkpoint.v1", "update": 256,
                               "loss": .7, "all_gates_pass": True}]) is None
    assert select_checkpoint([{"update": 256, "loss": .1, "all_gates_pass": False}]) is None


def test_learning_log_keeps_every_minibatch_and_partial_optimizer_counters(tmp_path):
    log = LearningLog(tmp_path)
    records = [{"epoch": 0, "minibatch": index, "actor_loss": float(index)} for index in range(8)]
    log.record_update({"records": records, "update_counter": 1}, attempted_updates=1,
                      actor_optimizer_steps=8, critic_optimizer_steps=8, statistics={"zero_count": 256, "nonzero_count": 256})
    assert len((tmp_path / "minibatches.jsonl").read_text().splitlines()) == 8
    log.record_partial_failure(attempted_updates=2, completed_updates=1, actor_optimizer_steps=10,
                               critic_optimizer_steps=9, error=RuntimeError("partial"))
    failure = json.loads((tmp_path / "failures.jsonl").read_text())
    assert failure["actor_optimizer_steps"] == 10 and failure["critic_optimizer_steps"] == 9
    summary = grouped_reward_summary({"reward": torch.tensor([-1., -2., -3.])}, {"zero": torch.tensor([True, False, True]), "empty": torch.tensor([False, False, False])})
    assert summary["zero"]["reward"]["cost_p95"] > 0 and summary["empty"]["reward"]["status"] == "UNAVAILABLE"


def test_lifecycle_preserves_primary_and_cleanup_failure_order():
    seen = []
    def operation(): raise ValueError("primary")
    def cleanup(): seen.append("cleanup"); raise OSError("cleanup")
    captured = []
    assert run_lifecycle(operation, (("app.close", cleanup),), lambda value: None, captured.append) == 1
    assert seen == ["cleanup"] and str(captured[0].primary_error) == "primary" and str(captured[0].cleanup_failures[0].error) == "cleanup"


def test_config_validate_only_and_future_live_entry_fail_closed(tmp_path):
    config = load_config(); result = validate_only()
    assert result["status"] == "VALID" and not result["live_authorized"]
    assert config["future_entry"]["sequences"]["main"][1] == "B0-ZTSXW"
    missing = tmp_path / "missing.json"
    with pytest.raises(PermissionError): load_r3_authorization(missing.resolve(), tmp_path / "attempt", "main")
    old = tmp_path / "old.json"; old.write_text(json.dumps({"status": "AUTHORIZED", "order_id": "SCALE-004"}))
    with pytest.raises(PermissionError): load_r3_authorization(old.resolve(), tmp_path / "attempt", "main")


def test_live_adapter_statically_selects_unified_environment_and_b0_direct():
    root = Path(__file__).resolve().parents[2]
    source = (root / "training_workloads/unified_h1/live_adapter.py").read_text()
    assert "UnifiedH1DirectEnv" in source and "execute_b0_direct_velocity" in source
    assert "use_b0" not in source


def test_r3_main_and_controlled_failure_sequence_contracts():
    class Backend:
        def __init__(self): self.calls = []
        def resource_check(self, stage, ordinal): self.calls.append((stage, ordinal))
        def initial_preflight(self): self.calls.append("preflight")
        def evaluate_b0(self, case): self.calls.append(f"B0-{case}")
        def cold_start(self): self.calls.append("cold")
        def save_checkpoint(self, label): self.calls.append(label)
        def smoke_update(self, profile): self.calls.append(profile)
        def reload_final(self): self.calls.append("reload")
        def mixed_reload_rollout(self, steps): self.calls.append(("reload-rollout", steps))
        def preserve(self, status, outcome=None): self.calls.append(status)
        def initial_raw_rays(self): self.calls.append("raw-rays")
        def close_bounded(self): self.calls.append("close")
    main = Backend(); assert run_r3_main(main) == 0
    assert [item for item in main.calls if isinstance(item, str) and item.startswith("B0-")] == [f"B0-{case}" for case in ("Z", "T", "S", "X", "W")]
    assert main.calls.count("smoke_init") == main.calls.count("smoke_final") == 1
    assert main.calls[-3:] == ["PASS_PENDING_CLOSE", "close", "PASS"]
    failure = Backend(); assert run_r3_controlled_failure(failure) == 1
    assert "raw-rays" in failure.calls and "EXPECTED_CONTROLLED_FAILURE" in failure.calls


def test_one_complete_512_sample_ppo_update_and_updated_checkpoint_roundtrip(tmp_path):
    from instinctlab.training import HistoryActor, NetworkConfig, PPOConfig, PPOTrainer
    from instinctlab.training.checkpoint_h1 import load_checkpoint_h1, save_checkpoint_h1
    from instinctlab.training.networks_h1 import PrivilegedCriticH1
    from instinctlab.training.ppo import PPOBatch

    network = NetworkConfig(lidar_channels=(2, 2, 2), state_hidden=4, actor_hidden=8, critic_hidden=(8, 8))
    trainer = PPOTrainer(HistoryActor(network), PrivilegedCriticH1(network), PPOConfig(minibatch_size=64, epoch_count=1))
    lidar = torch.zeros(1, 12, 59, 360).expand(512, -1, -1, -1)
    state = torch.zeros(1, 52).expand(512, -1)
    batch = PPOBatch(
        policy={"lidar_grid": lidar, "state_vec": state}, privileged=torch.zeros(512, 21),
        actions=torch.full((512, 4), .5), old_log_probabilities=torch.zeros(512, 1),
        advantages=torch.linspace(-1, 1, 512).unsqueeze(1), returns=torch.linspace(-.5, .5, 512).unsqueeze(1),
    )
    result = trainer.update(batch)
    assert result["sample_count"] == 512 and len(result["records"]) == 8 and result["update_counter"] == 1
    assert {int(item["step"].item()) for item in trainer.actor_optimizer.state_dict()["state"].values()} == {8}
    generator = torch.Generator().manual_seed(3)
    method = {"method": "a" * 64}; source = {"product": "ae7cb669a31e9bdacad9c4a3ea3420146cffd4f5"}
    checkpoint = (tmp_path / "updated.pt").resolve()
    save_checkpoint_h1(checkpoint, trainer, method_hashes=method, source_identity=source,
        transition_counter=512, global_policy_step=16, curriculum={"version": "curriculum-h1-v1", "completed_policy_steps": 16},
        training_seed=20260908, reset_number=2, curriculum_generator=generator)
    restored = PPOTrainer(HistoryActor(network), PrivilegedCriticH1(network), PPOConfig(minibatch_size=64, epoch_count=1))
    metadata = load_checkpoint_h1(checkpoint, restored, expected_method_hashes=method,
        expected_source_identity=source, curriculum_generator=torch.Generator())
    assert metadata["counters"] == {"update": 1, "transition": 512, "global_policy_step": 16, "actor_optimizer": 8, "critic_optimizer": 8}
