"""Registered soft zero-command loss using only the original actor boundary."""
import math
import torch


def zero_command_mean_prior(policy, output, weight):
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError('mean prior weight must be finite and nonnegative')
    state = policy['state_vec']
    mean = output.mean_action
    if (state.ndim != 2 or state.shape[1] != 52 or mean.shape != (state.shape[0], 4)
            or state.dtype != torch.float32 or mean.dtype != torch.float32
            or state.device != mean.device or state.shape[0] == 0
            or not bool(torch.isfinite(state).all().item())
            or not bool(torch.isfinite(mean).all().item())):
        raise ValueError('mean prior requires finite original actor tensors')
    zero = (state[:, 45:48].norm(dim=-1) <= .001).to(mean.dtype)
    return weight * (zero * ((mean[:, 1:4] - .5) / .1).square().sum(-1)).mean()


def configure_trainer(trainer, profile):
    """Only validated exact research/formal profiles opt in; legacy remains None."""
    if profile and 'mean_prior_weight' in profile:
        weight = profile['mean_prior_weight']
        trainer.actor_auxiliary = lambda policy, output: zero_command_mean_prior(policy, output, weight)
