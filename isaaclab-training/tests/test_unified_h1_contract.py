from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path

import pytest
import torch

from instinctlab.control.ics import mounted_mid360_rays_body
from instinctlab.observations.actor import APPROVED_INPUT_NAMES
from instinctlab.task.critic_h1 import CRITIC_H1_DIMENSION, build_critic_h1_state
from instinctlab.task.rewards_h1 import MeasurementCoverageUnavailable, compute_command_risk, compute_reward_h1
from instinctlab.training import HistoryActor, NetworkConfig, PPOConfig, PPOTrainer
from instinctlab.training.checkpoint import CheckpointError
from instinctlab.training.checkpoint_h1 import load_checkpoint_h1, save_checkpoint_h1
from instinctlab.training.networks_h1 import PrivilegedCriticH1
from instinctlab.training.ppo import PPOBatch, compute_gae
from instinctlab.training.rollout import RolloutConfig
from instinctlab.training.rollout_h1 import collect_h1_transition


def _lidar(batch=1, distance=40.0):
    value = torch.zeros(batch, 12, 59, 360, dtype=torch.float32)
    value[:, 0::3] = 40.0
    value[:, 1::3] = 1.0
    value[:, 2::3] = 1.0
    if distance != 40.0:
        rays = mounted_mid360_rays_body()
        index = int(rays[:, 0].argmax())
        row, column = divmod(index, 360)
        value[:, 9, row, column] = distance
    return value


def _root(batch=1, offset=0.0):
    root = torch.zeros(batch, 13, dtype=torch.float32)
    root[:, :3] = torch.tensor([offset + 1.0, offset + 2.0, 2.0])
    root[:, 3] = 1.0
    return root


def _reward_inputs(distance=40.0):
    zero3 = torch.zeros(1, 3)
    zero1 = torch.zeros(1, 1)
    return dict(
        lidar_grid=_lidar(distance=distance), command_b=zero3.clone(), v_gov_b=zero3.clone(),
        v_final_b=zero3.clone(), previous_issued_v_final_b=zero3.clone(), actual_velocity_b=zero3.clone(),
        anchor_loss=zero1.clone(), anchor_active=zero1.clone(), anchor_usable_fraction=torch.ones(1, 1),
        ics_beta=torch.ones(1, 1), exact_clearance_m=torch.ones(1, 1),
        collision=torch.zeros(1, 1, dtype=torch.bool), below_height=torch.zeros(1, 1, dtype=torch.bool),
        above_height=torch.zeros(1, 1, dtype=torch.bool),
    )


def test_command_risk_is_action_independent_and_has_frozen_boundaries():
    rays = mounted_mid360_rays_body(); index = int(rays[:, 0].argmax()); direction = rays[index:index + 1]
    command = 0.2 * direction
    near = compute_command_risk(_lidar(distance=1.005), command, rays_body=rays)
    edge = compute_command_risk(_lidar(distance=1.02), command, rays_body=rays)
    away = compute_command_risk(_lidar(distance=0.8), -command, rays_body=rays)
    zero = compute_command_risk(_lidar(distance=0.2), torch.zeros_like(command), rays_body=rays)
    assert near.dangerous_command.item() and near.beta_cmd.item() == pytest.approx(0.7071, abs=2e-3)
    assert not edge.dangerous_command.item() and edge.beta_cmd.item() == 1.0
    assert not away.dangerous_command.item() and not zero.dangerous_command.item()
    assert zero.emergency_cmd.item() and zero.zero_command.item()


def test_missing_measurement_coverage_fails_closed():
    lidar = _lidar(); lidar[:, 1::3] = 0.0; lidar[:, 2::3] = 0.0
    with pytest.raises(MeasurementCoverageUnavailable, match="MEASUREMENT_COVERAGE_UNAVAILABLE"):
        compute_command_risk(lidar, torch.zeros(1, 3))


def test_reward_formula_danger_braking_terminal_union_and_correction():
    rays = mounted_mid360_rays_body(); index = int(rays[:, 0].argmax()); command = 0.2 * rays[index:index + 1]
    values = _reward_inputs(distance=1.005)
    values.update(command_b=command, actual_velocity_b=command.clone(), v_gov_b=command.clone(),
                  v_final_b=torch.zeros_like(command), ics_beta=torch.tensor([[0.5]]),
                  collision=torch.tensor([[True]]), above_height=torch.tensor([[True]]))
    first = compute_reward_h1(**values, rays_body=rays)
    changed_action_path = dict(values); changed_action_path["v_final_b"] = command.clone(); changed_action_path["ics_beta"] = torch.ones(1, 1)
    second = compute_reward_h1(**changed_action_path, rays_body=rays)
    assert first.risk.dangerous_command.item() and second.risk.dangerous_command.item()
    assert first.components["reward_velocity"].item() == pytest.approx(-0.2, abs=2e-5)
    assert first.components["reward_failure"].item() == -300.0
    assert first.components["reward_intervention"].item() == pytest.approx(-0.035, abs=2e-5)
    assert first.components["reward_governor_proxy"].item() == 0.0


def test_reward_zero_and_safe_nonzero_formula_terms():
    zero = _reward_inputs(); zero["actual_velocity_b"][:] = torch.tensor([0.03, 0.04, 0.0])
    zero["anchor_loss"][:] = 0.25; zero["anchor_active"][:] = 1.0
    result = compute_reward_h1(**zero)
    assert result.components["reward_velocity"].item() == pytest.approx(-0.05)
    assert result.components["reward_zero_extra"].item() == 0.0
    assert result.components["reward_anchor"].item() == -1.0
    safe = _reward_inputs(); safe["command_b"][:] = torch.tensor([0.2, 0.0, 0.0])
    safe["v_gov_b"][:] = torch.tensor([0.1, 0.0, 0.0]); safe["v_final_b"][:] = torch.tensor([0.1, 0.0, 0.0])
    safe_result = compute_reward_h1(**safe)
    assert safe_result.diagnostics["g"].item() == 1.0
    assert safe_result.components["reward_governor_proxy"].item() == pytest.approx(-0.01)
    assert safe_result.components["reward_final_proxy"].item() == pytest.approx(-0.025)
    assert safe_result.components["reward_preservation"].item() == pytest.approx(-0.125)


def test_termination_bootstrap_and_gae_never_cross_reset():
    rewards = torch.tensor([[[-1.0]], [[-2.0]], [[-3.0]]])
    values = torch.zeros_like(rewards); next_values = torch.full_like(rewards, 10.0)
    terminated = torch.tensor([[[False]], [[False]], [[True]]])
    truncated = torch.tensor([[[False]], [[True]], [[True]]])
    reset = terminated | truncated
    timeout_values = torch.tensor([[[0.0]], [[7.0]], [[99.0]]])
    result = compute_gae(rewards, values, next_values, terminated, truncated, reset,
                         timeout_bootstrap_values=timeout_values, gamma=0.99, gae_lambda=0.95)
    assert result.advantages[1].item() == pytest.approx(-2 + .99 * 7)
    assert result.advantages[2].item() == -3.0
    assert result.advantages[0].item() == pytest.approx((-1 + .99 * 10) + .99 * .95 * result.advantages[1].item())


def test_critic_h1_local_position_translation_command_and_actor_boundary():
    roots = torch.cat((_root(offset=0.0), _root(offset=500.0)))
    origins = torch.tensor([[0.0, 0.0, 0.0], [500.0, 500.0, 0.0]])
    state = build_critic_h1_state(root_state_w=roots, env_origin_w=origins,
        actual_velocity_b=torch.zeros(2, 3), exact_clearance_m=torch.ones(2, 1),
        collision=torch.zeros(2, 1, dtype=torch.bool), command_b=torch.zeros(2, 3)).privileged
    assert state.shape == (2, CRITIC_H1_DIMENSION) and torch.allclose(state[0, :18], state[1, :18], atol=1e-5)
    changed = build_critic_h1_state(root_state_w=roots, env_origin_w=origins,
        actual_velocity_b=torch.zeros(2, 3), exact_clearance_m=torch.ones(2, 1),
        collision=torch.zeros(2, 1, dtype=torch.bool), command_b=torch.ones(2, 3)).privileged
    assert torch.equal(state[:, :18], changed[:, :18]) and not torch.equal(state[:, 18:], changed[:, 18:])
    assert set(APPROVED_INPUT_NAMES) == {"ranges_m", "validity", "sensor_frame_time_s", "policy_time_s", "body_angular_velocity_rad_s", "gravity_direction_body", "commanded_body_velocity_m_s", "previous_issued_body_velocity_m_s"}
    assert "root_state" not in inspect.getsource(HistoryActor.forward)


def _trainer():
    config = NetworkConfig(lidar_channels=(2, 2, 2), state_hidden=4, actor_hidden=8, critic_hidden=(8, 8))
    return PPOTrainer(HistoryActor(config), PrivilegedCriticH1(config), PPOConfig(minibatch_size=8, epoch_count=1))


def test_h1_critic_flows_through_ppo_without_mutating_actor_on_critic_step():
    trainer = _trainer(); actor_before = deepcopy(trainer.actor.state_dict())
    result = trainer.critic_only_step(torch.randn(8, 21), torch.randn(8, 1))
    assert result["value_loss"] >= 0
    assert all(torch.equal(value, actor_before[key]) for key, value in trainer.actor.state_dict().items())


METHOD = {"method": "a" * 64, "reward": "b" * 64}
SOURCE = {"product_commit": "c" * 40, "workload_commit": "d" * 40}


def _checkpoint_args(generator):
    return dict(method_hashes=METHOD, source_identity=SOURCE, transition_counter=0,
                global_policy_step=0, curriculum={"version": "curriculum-h1-v1", "completed_policy_steps": 0},
                training_seed=20260908, reset_number=0, curriculum_generator=generator)


def test_checkpoint_h1_cold_roundtrip_rejects_overwrite_and_v1(tmp_path):
    torch.manual_seed(7); trainer = _trainer(); generator = torch.Generator().manual_seed(99)
    path = (tmp_path / "init.pt").resolve(); save_checkpoint_h1(path, trainer, **_checkpoint_args(generator))
    expected = deepcopy(trainer.actor.state_dict()); restored = _trainer(); restored_generator = torch.Generator()
    metadata = load_checkpoint_h1(path, restored, expected_method_hashes=METHOD,
                                  expected_source_identity=SOURCE, curriculum_generator=restored_generator)
    assert metadata["counters"]["update"] == 0
    assert all(torch.equal(value, expected[key]) for key, value in restored.actor.state_dict().items())
    with pytest.raises(FileExistsError): save_checkpoint_h1(path, trainer, **_checkpoint_args(generator))
    with pytest.raises(CheckpointError):
        load_checkpoint_h1(path, restored, expected_method_hashes={"wrong": "x"},
                           expected_source_identity=SOURCE, curriculum_generator=restored_generator)


def test_checkpoint_h1_failure_does_not_partially_pollute_trainer(tmp_path):
    trainer = _trainer(); generator = torch.Generator().manual_seed(1)
    path = (tmp_path / "valid.pt").resolve(); save_checkpoint_h1(path, trainer, **_checkpoint_args(generator))
    payload = torch.load(path, weights_only=True); payload["critic"][next(iter(payload["critic"]))] = torch.tensor(float("nan"))
    corrupt = (tmp_path / "corrupt.pt").resolve(); torch.save(payload, corrupt)
    target = _trainer(); before_actor = deepcopy(target.actor.state_dict()); before_critic = deepcopy(target.critic.state_dict())
    with pytest.raises(CheckpointError):
        load_checkpoint_h1(corrupt, target, expected_method_hashes=METHOD,
                           expected_source_identity=SOURCE, curriculum_generator=torch.Generator())
    assert all(torch.equal(value, before_actor[key]) for key, value in target.actor.state_dict().items())
    assert all(torch.equal(value, before_critic[key]) for key, value in target.critic.state_dict().items())


def test_ppo_h1_accepts_21_and_rejects_18_dimension():
    trainer = _trainer(); batch = 8
    lidar = _lidar(batch); state = torch.zeros(batch, 52); policy = {"lidar_grid": lidar, "state_vec": state}
    with torch.no_grad():
        action, logp, _ = trainer.actor.sample(policy)
    item = PPOBatch(policy, torch.zeros(batch, 21), action, logp, torch.arange(batch).float().unsqueeze(1), torch.zeros(batch, 1))
    assert trainer.update(item)["sample_count"] == batch
    bad = PPOBatch(policy, torch.zeros(batch, 18), action, logp, torch.zeros(batch, 1), torch.zeros(batch, 1))
    with pytest.raises(ValueError): trainer.update(bad)


def test_h1_rollout_uses_pre_reset_next_command_critic_for_pure_timeout():
    trainer = _trainer()
    observation = {"policy": {"lidar_grid": torch.zeros(1, 12, 59, 360), "state_vec": torch.zeros(1, 52)},
                   "critic": torch.zeros(1, 21)}
    terminal_critic = torch.zeros(1, 21); terminal_critic[:, 18] = .2
    reset_critic = torch.zeros(1, 21); reset_critic[:, 18] = -.2
    class Environment:
        def step(self, action):
            next_observation = {"policy": {key: value.clone() for key, value in observation["policy"].items()},
                                "critic": reset_critic.clone()}
            return next_observation, torch.tensor([1.0]), torch.tensor([False]), torch.tensor([True]), {"pre_reset": {"critic_h1": terminal_critic.clone()}}
    transition, _ = collect_h1_transition(Environment(), trainer, observation, RolloutConfig(1, 1))
    assert torch.equal(transition.timeout_privileged, terminal_critic)
    assert not torch.equal(transition.timeout_privileged, transition.next_privileged)
    assert torch.allclose(transition.timeout_value, trainer.critic(terminal_critic))


def test_unified_environment_source_retains_a0004_ray_evidence_and_explicit_next_command():
    source = (Path(__file__).parents[1] / "instinctlab/envs/unified_h1_direct_env.py").read_text()
    assert "evaluation_ray_source" in source and "accepted_ray_hits_w" in source
    assert "prepare_h1_step" in source and "_h1_next_command_b" in source
