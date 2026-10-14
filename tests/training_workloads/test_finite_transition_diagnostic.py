import pytest
import torch

from instinctlab.control.anchor import MeasurementAnchor
from instinctlab.control.governor import decode_governor
from training_workloads.unified_h1.transition_diagnostic import (
    transition_commands, validate_transition_step, TRANSITION_SAMPLES,
)


def policy(command):
    state = torch.zeros(32, 52)
    state[:, 45:48] = command
    return {'lidar_grid': torch.zeros(1, 12, 59, 360).expand(32, -1, -1, -1), 'state_vec': state}


def row(command, following, metrics):
    current = policy(command)
    action = torch.full((32, 4), .5)
    root = torch.zeros(32, 13)
    root[:, 2] = 2
    root[:, 3] = 1
    pre = {'root_state_w': root, 'control': {
        'v_corr_b': decode_governor(action, current['state_vec']).v_corr_b,
        'v_final_b': torch.zeros(32, 3), 'v_final_w': torch.zeros(32, 3),
        'anchor_metrics': metrics}}
    return dict(current_policy=current, next_policy=policy(following), command=command,
                following=following, action=action, pre_reset=pre,
                reward=torch.zeros(32), reset_mask=torch.zeros(32, dtype=torch.bool))


def test_temporal_zero_positive_negative_zero_has_exact_budget_and_no_rng():
    before = torch.get_rng_state().clone()
    commands = transition_commands()
    assert torch.equal(before, torch.get_rng_state())
    assert commands.shape == (112, 32, 3) and TRANSITION_SAMPLES == 3584
    assert commands[:16].eq(0).all() and commands[80:].eq(0).all()
    assert commands[16:48].norm(dim=-1).gt(.15).all()
    assert torch.equal(commands[16:48], -commands[48:80])
    assert commands[16].unique(dim=0).shape[0] == 8


def test_actual_anchor_exits_and_recaptures_during_temporal_reversal():
    commands = transition_commands()
    anchor = MeasurementAnchor()
    ranges = torch.full((32, 59, 360), 3.)
    valid = torch.ones_like(ranges, dtype=torch.bool)
    reliable = torch.ones_like(ranges)
    no_reset = torch.zeros(32, dtype=torch.bool)
    for step, expected_active in [(0, 32), (16, 0), (48, 0), (80, 32)]:
        metrics = anchor.update(ranges, valid, reliable, commands[step], no_reset, no_reset).metrics
        result = validate_transition_step(**row(commands[step], commands[step + 1], metrics))
        assert result['active_anchor_lanes'] == expected_active


@pytest.fixture
def zero_row():
    zero = torch.zeros(32, 1)
    metrics = {k: zero.clone() for k in ['active', 'reset_code', 'usable_fraction', 'mean_absolute_error_m', 'huber_loss']}
    commands = transition_commands()
    return row(commands[15], commands[16], metrics)


def test_rejects_command_switch_one_observation_too_early(zero_row):
    zero_row['current_policy']['state_vec'][:, 45:48] = zero_row['following']
    with pytest.raises(RuntimeError, match='obs_t current command'):
        validate_transition_step(**zero_row)


def test_rejects_next_command_one_observation_too_late(zero_row):
    zero_row['next_policy']['state_vec'][:, 45:48] = 0
    with pytest.raises(RuntimeError, match=r'obs_t\+1 command'):
        validate_transition_step(**zero_row)


def test_rejects_wrong_vertical_correction_semantics(zero_row):
    zero_row['pre_reset']['control']['v_corr_b'][:, 2] = .1
    with pytest.raises(RuntimeError, match='actual xyz decoder'):
        validate_transition_step(**zero_row)


def test_nonzero_reference_cannot_remain_active(zero_row):
    commands = transition_commands()
    metrics = zero_row['pre_reset']['control']['anchor_metrics']
    metrics['active'][:] = 1
    with pytest.raises(RuntimeError, match='reference remains active'):
        validate_transition_step(**row(commands[16], commands[17], metrics))
