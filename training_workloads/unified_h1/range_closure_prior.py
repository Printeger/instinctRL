"""A training-only command gain prior based on causal local range closure."""

import torch

from instinctlab.control.ics import mounted_mid360_rays_body


CANDIDATE_ID = 'R6-CANDIDATE-INSTINCT-NEXT-R3'
WEIGHT = 0.5
LOOKAHEAD_S = 0.75
MIN_GAIN = 0.45


def sampled_rays(device):
    return mounted_mid360_rays_body(device=device).reshape(59, 360, 3)[::3, ::3].reshape(-1, 3)


@torch.no_grad()
def command_gain_ceiling(policy, rays, *, period_s, clearance_m):
    """A ceiling below one requires coherent closure in the current command cone."""
    scan = policy['lidar_grid'][:, 6:12, ::3, ::3]
    state = policy['state_vec'].reshape(-1, 4, 13)
    previous, latest = state[:, -2], state[:, -1]
    age = latest[:, 12, None]
    previous_age = previous[:, 12, None]
    acquisition_gap = period_s + previous_age - age
    # Angular rates are sampled on policy ticks. Delayed measurements can span
    # earlier ticks than the last two, so use the whole available gyro history.
    slow_rotation = state[:, :, :3].norm(dim=2).amax(dim=1) < .12
    causal_pair = ((acquisition_gap > .01) & (acquisition_gap <= 2.5 * period_s)
                   & (previous_age <= 2.01 * period_s) & (age <= 2.01 * period_s))

    command = latest[:, 6:9]
    speed = command.norm(dim=1, keepdim=True)
    heading = command / speed.clamp_min(.03)
    forward = (heading @ rays.T).reshape_as(scan[:, 0]) >= .80
    near, far = scan[:, 3], scan[:, 0]
    closing = (far - near) / acquisition_gap.clamp_min(.01)[:, :, None]
    credible = forward & (scan[:, 1] > .5) & (scan[:, 4] > .5) & (scan[:, 2] >= .55) & (scan[:, 5] >= .55)
    credible &= (closing >= .08) & (closing <= 3.0)

    time_to_margin = (near - clearance_m) / closing.clamp_min(.08) - age[:, :, None]
    urgency = torch.where(credible, (1.0 - time_to_margin / LOOKAHEAD_S).clamp(0., 1.), 0.)
    # Three nearby returns along one elevation reject an isolated range jump.
    coherent = torch.minimum(urgency, torch.minimum(urgency.roll(1, -1), urgency.roll(-1, -1)))
    active_pair = causal_pair[:, 0] & slow_rotation & (speed[:, 0] >= .03)
    risk = torch.where(active_pair, coherent.amax(dim=(1, 2)), 0.)
    return 1.0 - (1.0 - MIN_GAIN) * risk


def add_to_trainer(trainer, *, period_s, clearance_m):
    rays = sampled_rays(next(trainer.actor.parameters()).device)
    previous = trainer.actor_auxiliary

    def combined(policy, output):
        limit = command_gain_ceiling(policy, rays, period_s=period_s, clearance_m=clearance_m)
        excess = (output.mean_action[:, 0] - limit).clamp_min(0.)
        closure_loss = WEIGHT * excess.square().mean()
        with torch.no_grad():
            active, above_ceiling, loss = torch.stack(
                ((limit < 1).sum(), (excess > 0).sum(), closure_loss)).tolist()
        combined.last_metrics = dict(range_closure_active=int(active), range_closure_above_ceiling=int(above_ceiling),
                                     range_closure_loss=loss)
        return previous(policy, output) + closure_loss

    trainer.actor_auxiliary = combined
    return dict(sensor_inputs=['lidar_grid', 'state_vec'], actor_gain_only=True,
                policy_period_s=period_s, clearance_m=clearance_m, lookahead_s=LOOKAHEAD_S,
                minimum_prior_gain=MIN_GAIN, weight=WEIGHT)
