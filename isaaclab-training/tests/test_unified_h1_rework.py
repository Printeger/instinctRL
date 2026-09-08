from copy import deepcopy
from pathlib import Path

import pytest
import torch

from instinctlab.task.scene_h1 import H1_RAY_MESH_PATH, combined_scene_triangles
from instinctlab.training import HistoryActor, NetworkConfig, PPOConfig, PPOTrainer
from instinctlab.training.checkpoint import CheckpointError
from instinctlab.training.checkpoint_h1 import load_checkpoint_h1, save_checkpoint_h1
from instinctlab.training.checkpoint_h1 import _apply_rng_state, _restore_rng_state
from instinctlab.training.networks_h1 import PrivilegedCriticH1
from instinctlab.training.ppo import PPOBatch
from instinctlab.task.rewards_h1 import compute_reward_h1
from instinctlab.training.ppo import compute_gae


METHOD = {"method": "a" * 64}
SOURCE = {"product": "b" * 40}


def _trainer():
    network = NetworkConfig(lidar_channels=(2, 2, 2), state_hidden=4, actor_hidden=8, critic_hidden=(8, 8))
    return PPOTrainer(HistoryActor(network), PrivilegedCriticH1(network), PPOConfig(minibatch_size=64, epoch_count=1))


def _args(generator, update=0):
    return dict(method_hashes=METHOD, source_identity=SOURCE, transition_counter=512 * update,
                global_policy_step=16 * update,
                curriculum={"version": "curriculum-h1-v1", "completed_policy_steps": 16 * update},
                training_seed=20260908, reset_number=3, curriculum_generator=generator)


def _load(path, trainer):
    return load_checkpoint_h1(path, trainer, expected_method_hashes=METHOD,
                              expected_source_identity=SOURCE, curriculum_generator=torch.Generator())


def test_combined_scene_has_three_finite_translated_surfaces_per_lane():
    origins = torch.zeros(32, 3); origins[:, 0] = torch.arange(32) * 100
    points, indices, identities = combined_scene_triangles(origins)
    assert len(points) == 32 * 3 * 4 and len(indices) == 32 * 3 * 6
    assert identities.count("env0:floor") == identities.count("env31:x_wall") == 2
    assert points[12][0] == 100 - 42 and points[-4][0] == 3100 - 6 and points[-4][1] == 6
    source = (Path(__file__).parents[1] / "instinctlab/envs/unified_h1_direct_env.py").read_text()
    assert "mesh_prim_paths=[H1_RAY_MESH_PATH]" in source and "_spawn_unified_h1_combined_scene" in source


def test_checkpoint_rejects_counter_curriculum_and_malformed_metadata_transactionally(tmp_path):
    source = _trainer(); path = (tmp_path / "cold.pt").resolve()
    save_checkpoint_h1(path, source, **_args(torch.Generator().manual_seed(1)))
    payload = torch.load(path, weights_only=True)
    mutations = []
    wrong = deepcopy(payload); wrong["counters"]["transition"] = 512; mutations.append(wrong)
    wrong = deepcopy(payload); wrong["curriculum"]["completed_policy_steps"] = 999; mutations.append(wrong)
    wrong = deepcopy(payload); wrong["curriculum"] = [1]; mutations.append(wrong)
    wrong = deepcopy(payload); wrong["training_seed"] = 7; mutations.append(wrong)
    wrong = deepcopy(payload); wrong["version"] = "checkpoint-h1-v1"; mutations.append(wrong)
    for index, wrong in enumerate(mutations):
        corrupt = (tmp_path / f"bad_{index}.pt").resolve(); torch.save(wrong, corrupt)
        target = _trainer(); before = deepcopy(target.actor.state_dict()); before_rng = torch.get_rng_state().clone()
        with pytest.raises(CheckpointError): _load(corrupt, target)
        assert all(torch.equal(value, target.actor.state_dict()[key]) for key, value in before.items())
        assert torch.equal(before_rng, torch.get_rng_state())


def test_checkpoint_rejects_actual_adam_step_disagreement_partial_and_nonfinite(tmp_path):
    source = _trainer(); lidar = torch.zeros(1, 12, 59, 360).expand(512, -1, -1, -1)
    policy = {"lidar_grid": lidar, "state_vec": torch.zeros(1, 52).expand(512, -1)}
    with torch.no_grad(): actions, logp, _ = source.actor.sample(policy)
    source.update(PPOBatch(policy, torch.zeros(512, 21), actions, logp,
                           torch.linspace(-1, 1, 512).unsqueeze(1), torch.zeros(512, 1)))
    path = (tmp_path / "updated.pt").resolve(); save_checkpoint_h1(path, source, **_args(torch.Generator(), 1))
    payload = torch.load(path, weights_only=True)
    variants = []
    wrong = deepcopy(payload)
    for state in wrong["actor_optimizer"]["state"].values(): state["step"].fill_(7)
    variants.append(wrong)
    wrong = deepcopy(payload); wrong["actor_optimizer"]["state"].pop(next(iter(wrong["actor_optimizer"]["state"]))); variants.append(wrong)
    wrong = deepcopy(payload); next(iter(wrong["critic_optimizer"]["state"].values()))["step"].fill_(float("nan")); variants.append(wrong)
    wrong = deepcopy(payload); next(iter(wrong["critic_optimizer"]["state"].values()))["step"].fill_(7.5); variants.append(wrong)
    for index, wrong in enumerate(variants):
        corrupt = (tmp_path / f"adam_bad_{index}.pt").resolve(); torch.save(wrong, corrupt)
        with pytest.raises(CheckpointError): _load(corrupt, _trainer())


def test_actual_reward_short_trajectories_and_gae_match_hand_calculation():
    steps = 40; zero3 = torch.zeros(steps, 3); zero1 = torch.zeros(steps, 1)
    lidar = torch.zeros(steps, 12, 59, 360); lidar[:, 0::3] = 40.; lidar[:, 1::3] = 1.; lidar[:, 2::3] = 1.
    base = dict(lidar_grid=lidar,
        command_b=zero3, v_gov_b=zero3, v_final_b=zero3, previous_issued_v_final_b=zero3,
        actual_velocity_b=torch.full((steps, 3), .0), anchor_loss=zero1,
        anchor_active=zero1, anchor_usable_fraction=torch.ones_like(zero1), ics_beta=torch.ones_like(zero1),
        exact_clearance_m=torch.ones_like(zero1), collision=torch.zeros(steps, 1, dtype=torch.bool),
        below_height=torch.zeros(steps, 1, dtype=torch.bool), above_height=torch.zeros(steps, 1, dtype=torch.bool))
    safe = compute_reward_h1(**base).total
    collision_args = dict(base); collision_args["collision"] = torch.zeros(steps, 1, dtype=torch.bool); collision_args["collision"][-1] = True
    failed = compute_reward_h1(**collision_args).total
    expected_safe = sum(float(safe[index]) * .99 ** index for index in range(steps))
    expected_failed = sum(float(failed[index]) * .99 ** index for index in range(steps))
    assert expected_failed < expected_safe and failed[-1].item() == pytest.approx(safe[-1].item() - 300.)
    rewards = failed.reshape(steps, 1, 1); values = torch.zeros_like(rewards)
    terminated = torch.zeros_like(rewards, dtype=torch.bool); terminated[-1] = True
    result = compute_gae(rewards, values, values, terminated, torch.zeros_like(terminated), terminated,
                         timeout_bootstrap_values=torch.zeros_like(rewards), gamma=.99, gae_lambda=1.)
    assert result.returns[0].item() == pytest.approx(expected_failed, abs=2e-4)


def test_cuda_rng_apply_and_rollback_use_no_gpu_substitute(monkeypatch):
    calls=[]
    monkeypatch.setattr(torch.cuda,"set_rng_state_all",lambda values:calls.append([value.clone() for value in values]))
    generator=torch.Generator().manual_seed(4)
    before={"torch":torch.get_rng_state().clone(),"torch_cuda":[torch.tensor([1],dtype=torch.uint8)],
            "python":__import__("random").getstate(),"numpy":__import__("numpy").random.get_state(),
            "curriculum":generator.get_state().clone()}
    numpy_state=before["numpy"]
    rng={"torch_cpu":before["torch"],"torch_cuda":[torch.tensor([2],dtype=torch.uint8)],
         "curriculum":before["curriculum"]}
    _apply_rng_state(rng,(before["python"],numpy_state),generator,cuda_active=True)
    _restore_rng_state(before,generator)
    assert calls[0][0].item()==2 and calls[1][0].item()==1
