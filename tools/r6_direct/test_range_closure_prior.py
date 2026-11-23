from types import SimpleNamespace

import torch

from training_workloads.unified_h1.range_closure_prior import (
    add_to_trainer, command_gain_ceiling, sampled_rays,
)


def test_matching_latest_scans_require_causal_and_coherent_approach():
    grid = torch.ones(8, 12, 59, 360)
    grid[:, 9] = 1.02
    grid[:, 6] = 1.07
    state = torch.zeros(8, 4, 13)
    state[:, -1, 6] = .2
    grid[1, 6] = .97  # Receding with the same latest ranges.
    state[2, -1, 12] = .1  # The two observations refer to one acquisition.
    state[3, -2:, 12] = .1
    state[3, -3, 2] = .29  # Motion during the delayed acquisition interval.
    grid[4, 7] = 0.  # Invalid previous measurements.
    grid[5, 6] = 1.02
    grid[5, 6, 30, 180] = 1.07  # An isolated positive sample.
    state[6, -1, 6] = 0.
    state[7, -2:, 12] = .1  # A delayed but distinct pair, closer now.
    policy = dict(lidar_grid=grid, state_vec=state.flatten(1))
    limit = command_gain_ceiling(policy, sampled_rays('cpu'), period_s=.1, clearance_m=.95)
    assert .45 < limit[0] < .7
    torch.testing.assert_close(limit[1:7], torch.ones(6))
    assert .45 <= limit[7] < limit[0]

    mean = torch.full((8, 4), .9, requires_grad=True)
    actor = torch.nn.Linear(1, 1)
    trainer = SimpleNamespace(actor=actor, actor_auxiliary=lambda _policy, out: out.mean_action[:, 1].mean())
    add_to_trainer(trainer, period_s=.1, clearance_m=.95)
    loss = trainer.actor_auxiliary(policy, SimpleNamespace(mean_action=mean))
    loss.backward()
    assert torch.isfinite(loss) and mean.grad[0, 0] > 0. and mean.grad[7, 0] > 0.
    assert trainer.actor_auxiliary.last_metrics["range_closure_active"] == 2
    assert trainer.actor_auxiliary.last_metrics["range_closure_above_ceiling"] == 2
    torch.testing.assert_close(mean.grad[1:7, 0], torch.zeros(6))
    torch.testing.assert_close(mean.grad[:, 1], torch.full((8,), 1 / 8))
    assert not mean.grad[:, 2:].any()
