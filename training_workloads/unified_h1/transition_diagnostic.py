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


def execute_finite_transition(backend):
    """Execute only inside a separately authorized transition-mode lifecycle.

    Backend creation, strict checkpoint loading and probe recalculation precede
    this call. Original Z evaluation and its scoring function are unchanged.
    """
    from .logging import LearningLog
    from .pilot_diagnostic import _digest, _state
    from training_workloads.scale_003.persistence import atomic_json_replace
    auth = backend.authorization
    if auth['mode'] != 'research-transition':
        raise PermissionError('finite transition requires its exact authorized mode')
    budget = auth['budgets']
    if (budget['training_transitions'] != 0 or budget['optimizer_updates'] != 0
            or budget['checkpoint_writes'] != 0 or budget['nontraining_transitions'] != TRANSITION_SAMPLES):
        raise PermissionError('finite transition budget differs from112x32 nontraining only')
    if backend.trainer is None:
        raise RuntimeError('strictly loaded checkpoint actor required')
    env, trainer = backend.environment, backend.trainer
    if not env._h1_pure_hover_training:
        raise RuntimeError('finite transition must retain reviewed center reset distribution')
    trainer.actor.eval()
    trainer.critic.eval()
    original_state = _digest(_state(trainer))
    commands = transition_commands(device=env.device)
    env.set_h1_evaluation_case('Z')
    env.set_h1_initial_command(commands[0], 0)
    backend.resource_check('before_transition_reset', backend.counters.explicit_resets + 1)
    observation, _ = backend._preserving_reset(seed=auth['parameters']['evaluation_seed'])
    env._refresh_task_snapshot()
    log = LearningLog(backend.output / 'finite_transition')
    log.append('trace.jsonl', {'kind': 'T0', 'seed': auth['parameters']['evaluation_seed'],
               'root_state_w': env._robot.data.root_state_w,
               'sensor_position_w_m': env._evaluation_ray_snapshot.ray_starts_w[:, 0],
               'env_origins_w_m': env.scene.env_origins,
               'scope': 'IMPLEMENTATION_NOT_ORIGINAL_TASK_METRIC'})
    checks = []
    for step in range(TRANSITION_STEPS):
        backend.resource_check('before_transition_step', step)
        backend.counters.require_step_capacity(kind='nontraining', lanes=32, maximum=TRANSITION_SAMPLES)
        backend.counters.require_reset_capacity(automatic=True, lanes=32, maximum=budget['automatic_resets'])
        current, following = commands[step], commands[min(step + 1, TRANSITION_STEPS - 1)]
        # The environment may reuse observation buffers across step/reset.
        # Freeze the real actor input before advancing physics, without a new
        # sensor read, history advance or RNG draw.
        policy = {key: value.detach().clone() for key, value in observation['policy'].items()}
        if not torch.equal(policy['state_vec'][:, 45:48], current):
            raise RuntimeError('finite transition stale command before actor')
        env.prepare_h1_step(global_policy_step=step, command_b=current, next_command_b=following)
        with torch.no_grad():
            action = trainer.actor(policy).mean_action.detach()
        backend.counters.attempt_step()
        observation, reward, terminated, truncated, extras = env.step(action)
        backend.counters.complete_step(32, kind='nontraining')
        reset = terminated | truncated
        backend.counters.reserve_reset(automatic=True, lanes=int(reset.sum()), maximum=budget['automatic_resets'])
        pre = extras['pre_reset']
        # Write actual returned data before interpreting it; a validation failure
        # must preserve the consumed physical step instead of dropping its row.
        log.append('trace.jsonl', {'kind': 'STEP', 'step': step, 'command_b': current,
                   'following_command_b': following, 'actor_state_vec': policy['state_vec'],
                   'returned_state_vec': observation['policy']['state_vec'],
                   'issued_action': action, 'reward': reward, 'reset_mask': reset,
                   'root_state_w': pre['root_state_w'], 'control': pre['control'],
                   'actual_velocity_b': pre['task']['actual_velocity_b'],
                   'termination': pre['task']['termination']})
        check = validate_transition_step(current_policy=policy, next_policy=observation['policy'],
                   command=current, following=following, action=action, pre_reset=pre,
                   reward=reward, reset_mask=reset)
        checks.append(check)
    if _digest(_state(trainer)) != original_state:
        raise RuntimeError('finite transition changed model or optimizer state')
    for start, end in [(0, 16), (80, 112)]:
        if not any(c['active_anchor_lanes'] > 0 for c in checks[start:end]):
            raise RuntimeError('finite transition reference lifecycle has no observed active zero segment')
    summary = {'status': 'PASS_IMPLEMENTATION_ONLY', 'steps': TRANSITION_STEPS,
               'nontraining_transitions': TRANSITION_SAMPLES,
               'actual_reset_lanes': sum(c['actual_reset_lanes'] for c in checks),
               'checks': checks, 'model_optimizer_unchanged': True, 'optimizer_calls': 0,
               'original_task_performance': 'NOT_ASSERTED',
               'protocol': 'zero16/existingdirections32/temporalreverse32/zero32'}
    atomic_json_replace(backend.output, 'finite_transition_summary.json', summary)
    return summary
