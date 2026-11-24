import pytest
import torch

from instinctlab.training.checkpoint import CheckpointError
from instinctlab.training.checkpoint_h1 import attach_hover_reflex, checkpoint_actor, load_checkpoint_h1, save_checkpoint_h1
from instinctlab.training.hover_reflex import HOVER_REFLEX_FIELD, HOVER_REFLEX_METHOD
from instinctlab.training.networks import HistoryActor, NetworkConfig
from instinctlab.training.networks_h1 import PrivilegedCriticH1
from instinctlab.training.ppo import PPOConfig, PPOTrainer
from tools.r6_training_checkpoint_audit import audit
from training_workloads.unified_h1.robustness_training_method import HOVER_REFLEX_CANDIDATE, bound_method_hashes
from training_workloads.unified_h1.robustness_training_plan import BASELINE_SHA


def inputs():
    lidar = torch.ones(3, 12, 59, 360)
    lidar[:, 0::3] = 3.
    lidar[:, 0::3, 18:42, 165:195] = 1.6
    lidar[[0, 2], 9, 18:42, 165:195] = 1.25
    state = torch.zeros(3, 52)
    state[2, 45] = .2
    return dict(lidar_grid=lidar, state_vec=state)


def test_gain_zero_is_exact_and_keeps_gradient_without_changing_other_states():
    actor = HistoryActor()
    policy = inputs()
    baseline = actor(policy)
    gain = actor.add_hover_reflex()
    zero_gain = actor(policy)
    assert torch.equal(zero_gain.alpha, baseline.alpha)
    assert torch.equal(zero_gain.beta, baseline.beta)
    shift = actor.hover_reflex(policy['lidar_grid'], policy['state_vec'])
    assert not shift.any()
    direction = actor.hover_reflex.retreat_direction(policy['lidar_grid'], policy['state_vec'])
    assert direction[0, 0] < -.5
    assert torch.count_nonzero(direction[1:]) == 0
    shift[0, 1].backward()
    assert gain.grad < -.5
    with torch.no_grad():
        gain.fill_(5.)
        changed = actor(policy)
        assert not torch.equal(changed.mean_action[0], baseline.mean_action[0])
        assert torch.equal(changed.mean_action[1:], baseline.mean_action[1:])


def test_checkpoint_marker_and_adam_attachment_preserve_existing_parameters(tmp_path):
    config = NetworkConfig()
    method = PPOConfig(permutation_seed=13)
    trainer = PPOTrainer(HistoryActor(config), PrivilegedCriticH1(config), method)
    trainer.update_counter = 640
    optimizer = trainer.actor_optimizer
    previous_params = list(optimizer.param_groups[0]['params'])
    for param in previous_params:
        optimizer.state[param] = dict(step=torch.tensor(5120.), exp_avg=torch.ones_like(param),
                                      exp_avg_sq=torch.ones_like(param))
    stored = optimizer.state_dict()
    expected_clock = attach_hover_reflex(trainer)
    assert expected_clock['inherited_adam_clock'] == 5120
    assert all(x is y for x, y in zip(previous_params, optimizer.param_groups[0]['params'][:-1]))
    gain = optimizer.param_groups[0]['params'][-1]
    assert gain is trainer.actor.hover_reflex.gain
    assert optimizer.state[gain]['exp_avg'] == optimizer.state[gain]['exp_avg_sq'] == 0.
    for key in stored['state']:
        assert torch.equal(stored['state'][key]['exp_avg'], optimizer.state_dict()['state'][key]['exp_avg'])

    payload = dict(actor=trainer.actor.state_dict(), network_config=config.manifest(),
                   method_hashes={HOVER_REFLEX_FIELD: HOVER_REFLEX_METHOD})
    restored = checkpoint_actor(payload)
    restored.load_state_dict(payload['actor'], strict=True)
    assert tuple(restored.state_dict()) == tuple(trainer.actor.state_dict())
    with pytest.raises(CheckpointError, match='hover reflex method/state mismatch'):
        checkpoint_actor(dict(payload, method_hashes={}))

    for param in trainer.critic.parameters():
        trainer.critic_optimizer.state[param] = dict(step=torch.tensor(5120.), exp_avg=torch.zeros_like(param),
                                                      exp_avg_sq=torch.zeros_like(param))
    plan = dict(initialization=dict(method_hashes={'original': 'h1'}), recipe={'sha256': 'a' * 64})
    recipe = dict(candidate_id=HOVER_REFLEX_CANDIDATE, method={'sha256': 'b' * 64})
    hashes = bound_method_hashes(plan, recipe)
    assert hashes[HOVER_REFLEX_FIELD] == HOVER_REFLEX_METHOD
    source = {'test': 'fixed'}
    generator = torch.Generator().manual_seed(13)
    file = tmp_path / 'u640-reflex.pt'
    save_checkpoint_h1(file, trainer, method_hashes=hashes, source_identity=source,
                       transition_counter=640 * 512, global_policy_step=640 * 16,
                       curriculum=dict(version='curriculum-h1-r6-v1', completed_policy_steps=640 * 16,
                                       recipe_sha256='a' * 64, origin_checkpoint_sha256=BASELINE_SHA, branch_seed=13),
                       training_seed=13, reset_number=1, curriculum_generator=generator)
    saved = torch.load(file, map_location='cpu', weights_only=True)
    assert audit(saved)['adam_step'] == 5120
    reloaded = PPOTrainer(checkpoint_actor(saved), PrivilegedCriticH1(config), method)
    load_checkpoint_h1(file, reloaded, expected_method_hashes=hashes, expected_source_identity=source,
                       curriculum_generator=generator)
    assert reloaded.update_counter == 640
    assert torch.equal(reloaded.actor.hover_reflex.gain, gain)
