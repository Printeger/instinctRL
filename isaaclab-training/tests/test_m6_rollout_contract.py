"""Pure adapter contract tests for the bounded M6 learning pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from instinctlab.training import (
    HistoryActor, NetworkConfig, PPOConfig, PPOTrainer, PrivilegedCritic,
    RolloutConfig, RolloutTransition, assemble_rollout, collect_rollout,
    collect_transition, load_checkpoint, save_checkpoint,
)


NETWORK = NetworkConfig(lidar_channels=(1, 1, 1), state_hidden=4, actor_hidden=8, critic_hidden=(4, 4))
ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_SOURCE = ROOT / "instinctlab/training/rollout.py"


def _trainer(seed: int = 7) -> PPOTrainer:
    torch.manual_seed(seed)
    return PPOTrainer(
        HistoryActor(NETWORK), PrivilegedCritic(NETWORK),
        PPOConfig(gamma=0.9, gae_lambda=0.8, minibatch_size=2, epoch_count=1, permutation_seed=19),
    )


def _observation(step: float = 0.0, *, privileged_shift: float = 0.0) -> dict[str, Any]:
    lidar = torch.full((2, 12, 59, 360), step, dtype=torch.float32)
    state = torch.full((2, 52), step / 10.0, dtype=torch.float32)
    critic = torch.zeros(2, 18, dtype=torch.float32)
    critic[:, 0] = privileged_shift
    critic[:, 2] = 2.0
    critic[:, 3] = 1.0
    critic[:, 16] = 4.0
    return {"policy": {"lidar_grid": lidar, "state_vec": state}, "critic": critic}


def _extras(terminated: torch.Tensor, truncated: torch.Tensor, *, shift: float = 0.0) -> dict[str, Any]:
    terminated = terminated.reshape(2, 1)
    truncated = truncated.reshape(2, 1)
    done = terminated | truncated
    root = torch.zeros(2, 13, dtype=torch.float32)
    root[:, 0] = shift
    root[:, 2] = 2.0
    root[:, 3] = 1.0
    task = {
        "root_state_w": root.clone(),
        "actual_velocity_b": torch.full((2, 3), shift, dtype=torch.float32),
        "exact_clearance_m": torch.full((2, 1), 4.0, dtype=torch.float32),
        "collision": terminated.clone(),
        "termination": {"terminated": terminated.clone(), "truncated": truncated.clone(), "done": done.clone()},
    }
    return {"pre_reset": {"reset_mask": done.flatten().clone(), "root_state_w": root, "task": task}}


class FakeEnv:
    def __init__(self, cases: list[tuple[list[bool], list[bool]]], *, mutation: str | None = None, shift: float = 0.0):
        self.cases = cases
        self.mutation = mutation
        self.shift = shift
        self.index = 0
        self.action_identity = []
        self.actions: list[torch.Tensor] = []
        self.current: dict[str, Any] | None = None

    def step(self, action: torch.Tensor):
        self.action_identity.append(action)
        self.actions.append(action.detach().clone())
        terminated = torch.tensor(self.cases[self.index][0], dtype=torch.bool)
        truncated = torch.tensor(self.cases[self.index][1], dtype=torch.bool)
        reward = torch.tensor([1.0 + self.index, 2.0 + self.index], dtype=torch.float32)
        next_observation = _observation(float(self.index + 1), privileged_shift=self.shift + self.index + 1)
        extras = _extras(terminated, truncated, shift=self.shift + self.index)
        if self.mutation == "reward_dtype": reward = reward.double()
        elif self.mutation == "mask_shape": terminated = terminated[:1]
        elif self.mutation == "mask_dtype": truncated = truncated.to(torch.int64)
        elif self.mutation == "extras_missing": extras = {}
        elif self.mutation == "reset_mismatch": extras["pre_reset"]["reset_mask"].logical_not_()
        elif self.mutation == "task_mask": extras["pre_reset"]["task"]["termination"]["truncated"].logical_not_()
        elif self.mutation == "root_mismatch": extras["pre_reset"]["task"]["root_state_w"][0, 0] += 1.0
        elif self.mutation == "terminal_nan": extras["pre_reset"]["task"]["actual_velocity_b"][0, 0] = float("nan")
        elif self.mutation == "returned_order": next_observation = {"critic": next_observation["critic"], "policy": next_observation["policy"]}
        elif self.mutation == "returned_alias": next_observation = self.current
        elif self.mutation == "action_mutation": action[0, 0] = 0.0 if action[0, 0] != 0.0 else 1.0
        self.index += 1
        self.current = next_observation
        return next_observation, reward, terminated, truncated, extras


def _snapshot(trainer: PPOTrainer) -> dict[str, torch.Tensor]:
    return {f"actor.{name}": value.detach().clone() for name, value in trainer.actor.named_parameters()} | {
        f"critic.{name}": value.detach().clone() for name, value in trainer.critic.named_parameters()
    }


def _assert_snapshot(trainer: PPOTrainer, before: dict[str, torch.Tensor]) -> None:
    after = _snapshot(trainer)
    assert tuple(after) == tuple(before)
    assert all(torch.equal(after[key], value) for key, value in before.items())
    assert trainer.update_counter == 0
    assert not trainer.actor_optimizer.state and not trainer.critic_optimizer.state


def _nested_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right): return False
    if isinstance(left, torch.Tensor): return torch.equal(left, right)
    if isinstance(left, dict): return list(left) == list(right) and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)): return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def test_collection_stores_exact_detached_action_identity_and_time_major_flattening():
    trainer = _trainer()
    observation = _observation()
    environment = FakeEnv([([False, False], [False, False])] * 4)
    environment.current = observation
    result = collect_rollout(environment, trainer, observation, RolloutConfig(4, 2))
    segment = result.segment
    assert segment.policy["lidar_grid"].shape == (4, 2, 12, 59, 360)
    assert segment.privileged.shape == (4, 2, 18)
    assert segment.actions.shape == (4, 2, 4)
    assert segment.batch.actions.shape == (8, 4)
    assert torch.equal(segment.batch.actions, segment.actions.reshape(8, 4))
    assert torch.equal(segment.batch.privileged, segment.privileged.reshape(8, 18))
    assert torch.equal(segment.batch.policy["state_vec"], segment.policy["state_vec"].reshape(8, 52))
    assert all(environment.action_identity[index] is not segment.transitions[index].action for index in range(4))
    assert all(torch.equal(environment.actions[index], segment.transitions[index].action) for index in range(4))
    for transition in segment.transitions:
        assert not transition.old_log_probability.requires_grad
        with torch.no_grad():
            expected = trainer.actor.log_probability(transition.policy, transition.action)
        assert torch.equal(expected, transition.old_log_probability)
    observation["policy"]["lidar_grid"].fill_(99.0)
    assert torch.count_nonzero(segment.policy["lidar_grid"][0]) == 0


def test_static_actor_call_and_sampled_action_boundary_are_exact_and_fallback_free():
    source = ROLLOUT_SOURCE.read_text(encoding="utf-8")
    assert "trainer.actor.sample(policy)" in source
    assert "environment.step(action)" in source
    assert "trainer.critic(privileged)" in source
    assert "build_critic_state(" in source
    assert "reset_boundary = terminated | truncated" in source
    for forbidden in ("nan_to_num", "SyncDataCollector", "ReplayBuffer", "mean_action", "action.clamp"):
        assert forbidden not in source


def test_pure_timeout_uses_only_pre_reset_terminal_critic_and_cuts_post_reset_value():
    trainer = _trainer()
    observation = _observation(privileged_shift=1.0)
    environment = FakeEnv([([False, False], [True, False])], shift=7.0)
    environment.current = observation
    transition, _ = collect_transition(environment, trainer, observation, RolloutConfig(1, 2))
    expected_terminal = torch.cat((
        environment.current["critic"].new_tensor([[7.0, 0.0, 2.0, 1.0] + [0.0] * 9]),
        torch.full((1, 3), 7.0), torch.tensor([[4.0, 0.0]]),
    ), dim=1)
    assert torch.equal(transition.timeout_privileged[0:1], expected_terminal)
    assert torch.count_nonzero(transition.timeout_privileged[1]) == 0
    first = assemble_rollout((transition,), RolloutConfig(1, 2), trainer.config)
    changed = RolloutTransition(**{**transition.__dict__, "next_value": transition.next_value + 1000.0})
    second = assemble_rollout((changed,), RolloutConfig(1, 2), trainer.config)
    assert torch.equal(first.advantages[0, 0], second.advantages[0, 0])
    expected = transition.reward[0] + trainer.config.gamma * transition.timeout_value[0] - transition.value[0]
    torch.testing.assert_close(first.advantages[0, 0], expected, rtol=0.0, atol=1.0e-6)


def _manual_transition(step: int, rewards, values, next_values, timeout, terminated, truncated, reset):
    policy = _observation(float(step))["policy"]
    return RolloutTransition(
        policy=policy, privileged=_observation(float(step))["critic"], action=torch.zeros(2, 4),
        old_log_probability=torch.zeros(2, 1), value=torch.tensor(values[step]).reshape(2, 1),
        reward=torch.tensor(rewards[step]).reshape(2, 1), terminated=torch.tensor(terminated[step]).reshape(2, 1),
        truncated=torch.tensor(truncated[step]).reshape(2, 1), reset_boundary=torch.tensor(reset[step]).reshape(2, 1),
        next_privileged=_observation(float(step + 1))["critic"], next_value=torch.tensor(next_values[step]).reshape(2, 1),
        timeout_privileged=torch.zeros(2, 18), timeout_value=torch.tensor(timeout[step]).reshape(2, 1),
    )


def test_adapter_mixed_hand_table_distinguishes_continuation_all_done_causes_and_recursion_cuts():
    rewards = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    values = [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]]
    next_values = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
    timeout = [[100.0, 7.0], [200.0, 300.0], [400.0, 500.0]]
    terminated = [[False, False], [True, False], [True, False]]
    truncated = [[False, True], [True, False], [False, False]]
    reset = [[False, True], [True, False], [True, False]]
    transitions = tuple(_manual_transition(i, rewards, values, next_values, timeout, terminated, truncated, reset) for i in range(3))
    segment = assemble_rollout(transitions, RolloutConfig(3, 2), PPOConfig(gamma=0.9, gae_lambda=0.8))
    expected = torch.tensor([[[10.58], [7.3]], [[1.5], [79.04]], [[2.5], [57.0]]])
    torch.testing.assert_close(segment.advantages, expected, rtol=0.0, atol=1.0e-5)
    assert torch.equal(segment.batch.advantages, segment.advantages.reshape(6, 1))
    changed = list(transitions)
    changed[1] = RolloutTransition(**{
        **changed[1].__dict__, "next_value": changed[1].next_value + torch.tensor([[900.0], [0.0]]),
        "timeout_value": changed[1].timeout_value + torch.tensor([[-700.0], [0.0]]),
    })
    repeated = assemble_rollout(tuple(changed), RolloutConfig(3, 2), PPOConfig(gamma=0.9, gae_lambda=0.8))
    assert torch.equal(segment.advantages[:, 0], repeated.advantages[:, 0])
    assert torch.equal(segment.returns[:, 0], repeated.returns[:, 0])


@pytest.mark.parametrize("kind", [
    "reward_dtype", "mask_shape", "extras_missing", "reset_mismatch", "task_mask",
    "mask_dtype", "root_mismatch", "terminal_nan", "returned_order", "returned_alias", "action_mutation",
])
def test_returned_contract_failures_precede_any_optimizer_mutation(kind):
    trainer = _trainer()
    observation = _observation()
    environment = FakeEnv([([False, False], [True, False])], mutation=kind)
    environment.current = observation
    before = _snapshot(trainer)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        collect_transition(environment, trainer, observation, RolloutConfig(1, 2))
    _assert_snapshot(trainer, before)


@pytest.mark.parametrize("kind", ["order", "critic_shape", "policy_dtype", "policy_nonfinite", "batch"])
def test_pre_action_contract_failures_do_not_step_or_mutate_optimizers(kind):
    trainer = _trainer()
    observation = _observation()
    if kind == "order": observation = {"critic": observation["critic"], "policy": observation["policy"]}
    elif kind == "critic_shape": observation["critic"] = observation["critic"][:, :-1]
    elif kind == "policy_dtype": observation["policy"]["state_vec"] = observation["policy"]["state_vec"].double()
    elif kind == "policy_nonfinite": observation["policy"]["lidar_grid"][0, 0, 0, 0] = float("nan")
    else: observation["policy"]["lidar_grid"] = observation["policy"]["lidar_grid"][:1]
    environment = FakeEnv([([False, False], [False, False])])
    before = _snapshot(trainer)
    with pytest.raises((TypeError, ValueError)):
        collect_transition(environment, trainer, observation, RolloutConfig(1, 2))
    assert environment.index == 0
    _assert_snapshot(trainer, before)


@pytest.mark.parametrize("kind", ["top_missing", "top_extra", "policy_missing", "policy_extra", "policy_order"])
def test_missing_extra_and_reordered_actor_boundaries_fail_before_step(kind):
    trainer = _trainer(); observation = _observation()
    if kind == "top_missing": observation.pop("critic")
    elif kind == "top_extra": observation["extras"] = {}
    elif kind == "policy_missing": observation["policy"].pop("state_vec")
    elif kind == "policy_extra": observation["policy"]["root_state_w"] = torch.zeros(2, 13)
    else:
        observation["policy"] = {"state_vec": observation["policy"]["state_vec"],
                                 "lidar_grid": observation["policy"]["lidar_grid"]}
    environment = FakeEnv([([False, False], [False, False])]); before = _snapshot(trainer)
    with pytest.raises((TypeError, ValueError)):
        collect_transition(environment, trainer, observation, RolloutConfig(1, 2))
    assert environment.index == 0
    _assert_snapshot(trainer, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device mismatch requires the frozen GPU runtime")
def test_cross_device_actor_critic_boundary_fails_before_step():
    trainer = _trainer(); observation = _observation(); observation["critic"] = observation["critic"].cuda()
    environment = FakeEnv([([False, False], [False, False])]); before = _snapshot(trainer)
    with pytest.raises(ValueError):
        collect_transition(environment, trainer, observation, RolloutConfig(1, 2))
    assert environment.index == 0
    _assert_snapshot(trainer, before)


def test_incomplete_or_inconsistent_horizon_fails_closed():
    trainer = _trainer()
    observation = _observation()
    environment = FakeEnv([([False, False], [False, False])])
    environment.current = observation
    transition, _ = collect_transition(environment, trainer, observation, RolloutConfig(1, 2))
    with pytest.raises(ValueError): assemble_rollout((transition,), RolloutConfig(2, 2), trainer.config)
    bad = RolloutTransition(**{**transition.__dict__, "reset_boundary": ~transition.reset_boundary})
    with pytest.raises(ValueError): assemble_rollout((bad,), RolloutConfig(1, 2), trainer.config)


def test_actor_is_independent_of_privilege_root_reward_masks_and_extras_but_policy_is_discriminating():
    trainer = _trainer()
    first_observation = _observation(privileged_shift=0.0)
    second_observation = _observation(privileged_shift=99.0)
    first_env = FakeEnv([([False, False], [False, False])], shift=0.0); first_env.current = first_observation
    second_env = FakeEnv([([True, False], [True, False])], shift=50.0); second_env.current = second_observation
    rng = torch.get_rng_state().clone()
    first, _ = collect_transition(first_env, trainer, first_observation, RolloutConfig(1, 2))
    torch.set_rng_state(rng)
    second, _ = collect_transition(second_env, trainer, second_observation, RolloutConfig(1, 2))
    assert torch.equal(first.action, second.action)
    assert torch.equal(first.old_log_probability, second.old_log_probability)
    assert not torch.equal(first.value, second.value)
    changed_policy = _observation()
    changed_policy["policy"]["state_vec"][:, 0] = 10.0
    with torch.no_grad():
        assert not torch.equal(trainer.actor(first_observation["policy"]).alpha, trainer.actor(changed_policy["policy"]).alpha)


def test_checkpoint_boundary_restores_outputs_optimizers_counter_and_rng_for_continuation(tmp_path: Path):
    trainer = _trainer()
    observation = _observation()
    environment = FakeEnv([([False, False], [True, False])] + [([False, False], [False, False])] * 3)
    environment.current = observation
    first = collect_rollout(environment, trainer, observation, RolloutConfig(4, 2))
    trainer.update(first.segment.batch)
    with torch.no_grad():
        expected_actor = trainer.actor(first.next_observation["policy"])
        expected_value = trainer.critic(first.next_observation["critic"])
    path = save_checkpoint((tmp_path / "pipeline.pt").resolve(), trainer)
    expected_rng = torch.get_rng_state().clone()
    actor_optimizer = deepcopy(trainer.actor_optimizer.state_dict())
    critic_optimizer = deepcopy(trainer.critic_optimizer.state_dict())
    restored = _trainer(999)
    load_checkpoint(path, restored)
    with torch.no_grad():
        actual_actor = restored.actor(first.next_observation["policy"])
        actual_value = restored.critic(first.next_observation["critic"])
    assert torch.equal(expected_actor.alpha, actual_actor.alpha)
    assert torch.equal(expected_actor.beta, actual_actor.beta)
    assert torch.equal(expected_value, actual_value)
    assert restored.update_counter == 1
    assert _nested_equal(actor_optimizer, restored.actor_optimizer.state_dict())
    assert _nested_equal(critic_optimizer, restored.critic_optimizer.state_dict())
    assert torch.equal(expected_rng, torch.get_rng_state())
