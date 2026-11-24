"""PPO reward for an opt-in physical obstacle, and a sensor-only hover prior."""

import torch

from instinctlab.control.lee_velocity_controller import rotate_world_to_body_wxyz
from instinctlab.control.command_adapter import rotate_body_to_world
from instinctlab.task.rewards_h1 import RewardH1Result
from .mean_prior import zero_command_mean_prior


CANDIDATE = 'R6-CANDIDATE-INSTINCT-GPU-R2'


def choose_velocity(command, velocity, relative, object_velocity, active):
    """Local action comparison for rewards. Every argument stays off the actor path."""
    horizontal = relative.clone()
    horizontal[:, 2] = 0.
    forward = horizontal / horizontal.norm(dim=-1, keepdim=True).clamp_min(.01)
    side = torch.stack((-forward[:, 1], forward[:, 0], torch.zeros_like(forward[:, 0])), dim=-1)
    candidates = torch.stack((command, torch.zeros_like(command), -.3 * forward, .27 * side,
                              -.27 * side, .45 * command + .19 * side, .45 * command - .19 * side), dim=1)
    time = torch.linspace(.1, 2., 12, device=command.device, dtype=command.dtype)
    lag = .24 * (1 - (-time / .24).exp())
    own = candidates[:, :, None] * (time - lag)[None, None, :, None] + velocity[:, None, None] * lag[None, None, :, None]
    path = relative[:, None] + time[None, :, None] * object_velocity[:, None]
    separation = (path[:, None] - own).norm(dim=-1)
    shortfall = (.80 - separation).clamp_min(0.).square()
    deviation = (candidates - command[:, None]).square().sum(dim=2)
    # The worst separation captures the danger, while its duration rewards
    # escaping even when every candidate starts too close to avoid shortfall.
    cost = deviation + 12. * shortfall.amax(dim=2) + 6. * shortfall.mean(dim=2)
    risk = active & (shortfall[:, 0].amax(dim=1) > 0.)
    choice = torch.where(risk, cost.argmin(dim=1), 0)
    return candidates[torch.arange(len(command), device=command.device), choice], risk


def amend_reward(reward, env, obstacle):
    trace = obstacle.pre_reset
    root = env._task.root_state_w
    attitude = root[:, 3:7]
    command_world = rotate_body_to_world(env._command_b, attitude)
    target_w, hazard = choose_velocity(command_world, root[:, 7:10],
                                       trace['sphere_position_w'] - root[:, :3], trace['sphere_velocity_w'], trace['active'])
    mask = hazard[:, None]
    target = rotate_world_to_body_wxyz(attitude, target_w)
    values = reward.components.copy()
    errors = {
        'reward_velocity': -(env._task.actual_velocity_b - target).norm(dim=1, keepdim=True),
        'reward_governor_proxy': -.1 * (env._control.v_gov_b - target).norm(dim=1, keepdim=True),
        'reward_final_proxy': -.25 * (env._control.v_final_b - target).norm(dim=1, keepdim=True),
    }
    for key in ('reward_anchor','reward_zero_extra','reward_preservation'):
        errors[key] = torch.zeros_like(values[key])
    for key, value in errors.items():
        values[key] = torch.where(mask, value, values[key])
    total = sum(value for key, value in values.items() if key != 'reward_total')
    values['reward_total'] = total
    return RewardH1Result(total, values, reward.diagnostics, reward.risk)


@torch.no_grad()
def plausible_approach(policy, *, period_s):
    state = policy['state_vec'].reshape(-1, 4, 13)
    scan = policy['lidar_grid'][:, :, ::3]
    approach = torch.zeros_like(scan[:, 9], dtype=torch.bool)
    # Use distinct acquisitions separated long enough for the range change to
    # stand out from measurement noise. Wrap azimuth around the scan seam.
    for k in range(3):
        gap = (3 - k) * period_s + state[:, k, 12] - state[:, -1, 12]
        delta = (scan[:, k * 3] - scan[:, 9]) / gap[:, None, None].clamp_min(.01)
        good = ((gap >= .16) & (gap <= .33))[:, None, None]
        good = good & (delta > .20) & (delta < 1.5) & (scan[:, 9] < 1.9)
        good &= (scan[:, k * 3 + 2] > .55) & (scan[:, 11] > .55)
        approach |= good & good.roll(1, dims=-1) & good.roll(-1, dims=-1) & good.roll(2, dims=-1) & good.roll(-2, dims=-1)
    stable = (state[:, :, :3].norm(dim=2).amax(dim=1) < .22) & (state[:, -1, 12] <= .22)
    return approach.flatten(1).any(1) & stable


def install_prior(trainer, *, weight, period_s, dynamic_block, reflex=None, ics=None):
    if reflex is not None:
        from instinctlab.control.ics import attenuate_command, mounted_mid360_rays_body

        if ics is None:
            raise ValueError('Hover targets require the active ICS configuration')
        parameter = next(trainer.actor.parameters())
        rays = mounted_mid360_rays_body(device=parameter.device, dtype=parameter.dtype)

    def gated(policy, output):
        zero = policy['state_vec'][:, 45:48].norm(dim=1) <= .001
        if not dynamic_block():
            gated.last_metrics = dict(hover_prior_removed=0, hover_prior_retained=int(zero.sum().item()))
            return zero_command_mean_prior(policy, output, weight)
        if reflex is None:
            approaching = plausible_approach(policy, period_s=period_s)
        else:
            direction = reflex.retreat_direction(policy['lidar_grid'], policy['state_vec'])
            approaching = direction.norm(dim=1) > .1
        safe = zero & ~approaching
        correction = output.mean_action[:, 1:4] - .5
        penalty = (correction / .1).square().sum(dim=1)
        gated.last_metrics = dict(hover_prior_removed=int((zero & approaching).sum().item()),
                                  hover_prior_retained=int(safe.sum().item()))
        prior = weight * (safe * penalty).mean()
        if reflex is None or not bool(approaching.any()):
            return prior
        with torch.no_grad():
            target = .16 * direction[approaching]
            authority = attenuate_command(policy['lidar_grid'][approaching], rays, target, period_s, ics)
            feasible = authority.metrics['beta'].flatten() >= .25
        error = ((correction[approaching] - target) / .16).square().sum(dim=1)
        gated.last_metrics['hover_reflex_targets'] = int(feasible.sum().item())
        return prior + .15 * (feasible * error).sum() / len(policy['state_vec'])

    trainer.actor_auxiliary = gated
    binding = dict(weight=weight, policy_period_s=period_s, inputs=['lidar_grid', 'state_vec'])
    if reflex is not None:
        binding['direct_hover_target'] = dict(speed_mps=.16, weight=.15, minimum_ics_beta=.25)
    return binding
