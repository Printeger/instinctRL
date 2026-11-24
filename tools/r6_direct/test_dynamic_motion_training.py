from types import SimpleNamespace

import torch

from instinctlab.observations.moving_sphere import travel_clock
from training_workloads.unified_h1.dynamic_motion_training import choose_velocity, install_prior, plausible_approach
from training_workloads.unified_h1.mean_prior import zero_command_mean_prior


def test_training_obstacle_returns_without_teleporting_and_evaluation_stays_linear():
    age = torch.tensor([0., 8.99, 9., 9.01, 17.99, 18.])
    travel, direction = travel_clock(age, half_period_s=9.)
    torch.testing.assert_close(travel, torch.tensor([0., 8.99, 9., 8.99, .01, 0.]))
    torch.testing.assert_close(direction, torch.tensor([1., 1., -1., -1., -1., 1.]))
    linear, direction = travel_clock(age, half_period_s=None)
    torch.testing.assert_close(linear, age)
    torch.testing.assert_close(direction, torch.ones_like(age))


def test_forecast_retains_safe_command_and_rewards_escaping_an_immediate_threat():
    command = torch.tensor([[.30, 0., 0.], [.30, 0., 0.], [.30, 0., 0.], [0., 0., 0.]])
    position = torch.tensor([[1.45, 0., 0.], [1.45, 0., 0.], [1.45, 0., 0.], [.38, 0., 0.]])
    velocity = torch.tensor([[-.42, 0., 0.], [.36, 0., 0.], [-.42, 0., 0.], [-.12, 0., 0.]])
    active = torch.tensor([True, True, False, True])
    target, hazard = choose_velocity(command, command, position, velocity, active)
    assert hazard.tolist() == [True, False, False, True]
    assert target[0, 0] < command[0, 0] and target[0, 1].abs() > .15
    torch.testing.assert_close(target[1:3], command[1:3])
    assert target[3, 0] < -.15

    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotated, hazard_rotated = choose_velocity(command @ rotation.T, command @ rotation.T,
                                              position @ rotation.T, velocity @ rotation.T, active)
    torch.testing.assert_close(rotated, target @ rotation.T)
    torch.testing.assert_close(hazard_rotated, hazard)


def test_hover_relief_uses_distinct_coherent_hits_and_preserves_original_static_loss():
    lidar = torch.ones(5, 12, 59, 360)
    lidar[:, 0::3] = 1.2
    seam = [358, 359, 0, 1, 2]
    lidar[:2, 3, 30, seam] = 1.28
    lidar[2, 3, 30, 0] = 1.28  # Single beam impulse.
    lidar[3:, 3, 30, seam] = 1.28
    lidar[3, 5, 30, seam] = 0.  # Unreliable earlier return.
    state = torch.zeros(5, 4, 13)
    state[1, -1, 12] = .2  # Repeated acquisition under repeated frame drops.
    state[4, -2, 2] = .30  # Angular motion can create apparent range closure.
    policy = dict(lidar_grid=lidar, state_vec=state.flatten(1))
    assert plausible_approach(policy, period_s=.1).tolist() == [True, False, False, False, False]

    mean = torch.full((5, 4), .8, requires_grad=True)
    output = SimpleNamespace(mean_action=mean)
    phase = SimpleNamespace(enabled=False)
    trainer = SimpleNamespace()
    install_prior(trainer, weight=.012, period_s=.1, dynamic_block=lambda: phase.enabled)
    torch.testing.assert_close(trainer.actor_auxiliary(policy, output), zero_command_mean_prior(policy, output, .012))
    phase.enabled = True
    trainer.actor_auxiliary(policy, output).backward()
    assert trainer.actor_auxiliary.last_metrics == dict(hover_prior_removed=1, hover_prior_retained=4)
    assert not mean.grad[0].any() and mean.grad[1:, 1:].all()


def test_stationary_near_wall_noise_keeps_hover_regularization():
    count = 16
    scans = torch.ones(count, 12, 59, 360)
    scans[:, 0::3] = 3.
    scans[:, 0::3, :, 100:190] = 1.4
    scans[:, 0::3] += .01 * torch.randn(count, 4, 59, 360, generator=torch.Generator().manual_seed(7212))
    state = torch.zeros(count, 52)
    # A broad locally closing return among otherwise matched wall inputs.
    scans[-1, 3, 30:37, 125:138] += .09
    approach = plausible_approach(dict(lidar_grid=scans, state_vec=state), period_s=.1)
    assert approach.tolist() == [False] * (count - 1) + [True]
