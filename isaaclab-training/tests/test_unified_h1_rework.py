from copy import deepcopy
from pathlib import Path

import pytest
import torch

from instinctlab.task.scene_h1 import H1_RAY_MESH_PATH, combined_scene_triangles
from instinctlab.training import HistoryActor, NetworkConfig, PPOConfig, PPOTrainer
from instinctlab.training.checkpoint import CheckpointError
from instinctlab.training.checkpoint_h1 import load_checkpoint_h1, save_checkpoint_h1
from instinctlab.training.networks_h1 import PrivilegedCriticH1
from instinctlab.training.ppo import PPOBatch


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
