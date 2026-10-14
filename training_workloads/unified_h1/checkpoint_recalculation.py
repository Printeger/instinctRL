"""Read-only Beta actor recalculation against an actual saved first-update probe.

The caller binds and strictly loads the checkpoint/input/probe identities. This
function performs no loading, simulator action, sampling or optimizer step.
"""
from collections.abc import Mapping
import torch

from .pilot_diagnostic import (
    FLOAT32_ATOL, FLOAT32_RTOL, SAMPLES, _digest, _state,
    _rng_snapshot, _rng_equal, _restore_rng,
)


def recalculate_actor_probe(trainer, policy, actions, reference):
    """Compare the u1 actor to its actual after-minibatch-8 probe boundary.

    Critic values at that observer boundary precede the last critic minibatch;
    comparing them to the completed-u1 critic would use the wrong boundary.
    """
    from instinctlab.training.networks import validate_policy
    validate_policy(policy)
    if any(value.shape[0] != SAMPLES for value in policy.values()):
        raise ValueError('recalculation requires all512 frozen actor inputs')
    device = next(trainer.actor.parameters()).device
    if (not isinstance(actions, torch.Tensor) or actions.shape != (SAMPLES, 4)
            or actions.dtype != torch.float32 or actions.device != device
            or not torch.isfinite(actions).all()):
        raise ValueError('recalculation actions must be finite float32 [512,4] on actor device')
    expected = {'action_mean': (SAMPLES, 4), 'action_std': (SAMPLES, 4),
                'log_probability': (SAMPLES, 1)}
    if not isinstance(reference, Mapping):
        raise ValueError('actual saved probe reference required')
    for key, shape in expected.items():
        value = reference.get(key)
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError('saved probe shape/dtype/finite mismatch: ' + key)
    before = _digest(_state(trainer))
    rng = _rng_snapshot()
    try:
        with torch.no_grad():
            output = trainer.actor(policy)
            alpha, beta = output.alpha, output.beta
            values = {
                'action_mean': alpha / (alpha + beta),
                'action_std': torch.sqrt(alpha * beta / ((alpha + beta).square() * (alpha + beta + 1))),
                'log_probability': trainer.actor.log_probability_from_output(output, actions),
            }
        differences = {}
        for key, value in values.items():
            if value.shape != expected[key] or not torch.isfinite(value).all():
                raise ValueError('actor recalculation output invalid: ' + key)
            target = reference[key].to(device=device)
            if not torch.allclose(value, target, atol=FLOAT32_ATOL, rtol=FLOAT32_RTOL):
                raise RuntimeError('saved actor probe recalculation mismatch: ' + key)
            differences[key] = float((value - target).abs().max().item())
        return {'status': 'PASS', 'samples': SAMPLES, 'device': str(device),
                'atol': FLOAT32_ATOL, 'rtol': FLOAT32_RTOL,
                'maximum_absolute_errors': differences,
                'reference_boundary': 'ACTOR_AFTER_FIRST_UPDATE_MINIBATCH8',
                'critic_value_comparison': 'EXCLUDED_DIFFERENT_OBSERVER_BOUNDARY',
                'model_optimizer_unchanged': True, 'policy_rng_consumed': False,
                'optimizer_calls': 0}
    finally:
        rng_unchanged = _rng_equal(rng, _rng_snapshot())
        if not rng_unchanged:
            _restore_rng(rng)
        if _digest(_state(trainer)) != before:
            raise RuntimeError('actor recalculation changed model or optimizer state')
        if not rng_unchanged:
            raise RuntimeError('actor recalculation consumed RNG')
