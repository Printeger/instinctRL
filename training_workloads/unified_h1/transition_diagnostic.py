"""Finite implementation check; no replacement for any original task metric."""
import torch

from .schedule import DIRECTIONS

TRANSITION_STEPS = 112
TRANSITION_SAMPLES = TRANSITION_STEPS * 32


def transition_commands(*, device='cpu', dtype=torch.float32):
    """Zero, existing positive directions, their temporal reverse, zero."""
    if dtype != torch.float32:
        raise ValueError('finite transition uses production float32 commands')
    commands = torch.zeros(TRANSITION_STEPS, 32, 3, device=device, dtype=dtype)
    directions = torch.tensor(DIRECTIONS, device=device, dtype=dtype)[torch.arange(32, device=device) % 8]
    commands[16:48] = directions
    commands[48:80] = -directions
    return commands


def validate_transition_step(*, current_policy, next_policy, command, following,
                             action, pre_reset, reward, reset_mask):
    """Check the actual issued/returned pair without sampling or filling data."""
    from instinctlab.training.networks import validate_policy
    from instinctlab.control.governor import decode_governor
    for policy in [current_policy, next_policy]:
        validate_policy(policy)
        if policy['state_vec'].shape != (32, 52):
            raise ValueError('transition actor batch/schema mismatch')
    if not torch.equal(current_policy['state_vec'][:, 45:48], command):
        raise RuntimeError('transition obs_t current command is misindexed')
    if not torch.equal(next_policy['state_vec'][:, 45:48], following):
        raise RuntimeError('transition obs_t+1 command is misindexed')
    decoded = decode_governor(action, current_policy['state_vec'])
    tensors = {'root_state_w': (pre_reset['root_state_w'], (32, 13)),
               'v_corr_b': (pre_reset['control']['v_corr_b'], (32, 3)),
               'v_final_b': (pre_reset['control']['v_final_b'], (32, 3)),
               'v_final_w': (pre_reset['control']['v_final_w'], (32, 3))}
    for name, (value, shape) in tensors.items():
        if not isinstance(value, torch.Tensor) or value.shape != shape or not torch.isfinite(value).all():
            raise ValueError('transition finite/shape failure: ' + name)
    if not torch.equal(decoded.v_corr_b, pre_reset['control']['v_corr_b']):
        raise RuntimeError('transition action does not match actual xyz decoder')
    if (not isinstance(reward, torch.Tensor) or reward.shape not in [(32,), (32, 1)]
            or not torch.isfinite(reward).all()):
        raise ValueError('transition reward invalid')
    if not isinstance(reset_mask, torch.Tensor) or reset_mask.shape != (32,) or reset_mask.dtype != torch.bool:
        raise ValueError('transition reset mask invalid')
    anchor = pre_reset['control']['anchor_metrics']
    for name in ['active', 'reset_code', 'usable_fraction', 'mean_absolute_error_m', 'huber_loss']:
        value = anchor[name]
        if not isinstance(value, torch.Tensor) or value.shape != (32, 1) or not torch.isfinite(value).all():
            raise ValueError('transition anchor metric invalid: ' + name)
    nonzero = command.norm(dim=-1) >= .15
    if torch.any(anchor['active'].reshape(32)[nonzero] != 0):
        raise RuntimeError('transition reference remains active after command exit')
    return {'samples': 32, 'actual_reset_lanes': int(reset_mask.sum().item()),
            'zero_command_lanes': int((command.norm(dim=-1) <= .001).sum().item()),
            'active_anchor_lanes': int((anchor['active'] > 0).sum().item()),
            'shape_command_decoder_reference_checks': 'PASS',
            'performance_scoring': 'NOT_AN_ORIGINAL_TASK_ACCEPTANCE'}
