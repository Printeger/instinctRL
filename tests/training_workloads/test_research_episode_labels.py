"""Pure-hover episodes must not inherit mixed-curriculum wall labels."""
from types import SimpleNamespace

import pytest
import torch

from training_workloads.unified_h1.driver import UnifiedH1Driver
from training_workloads.unified_h1.logging import LearningLog
from training_workloads.unified_h1.schedule import ScheduleClock


@pytest.mark.parametrize('pure_hover', [False, True])
def test_driver_exposes_actual_clock_mode(pure_hover):
    clock = ScheduleClock(1600, pure_hover=pure_hover)
    state = torch.zeros(32, 52)
    state[:, 45:48] = clock.plan().command_b
    observation = {'policy': {'state_vec': state}}
    class Environment:
        def prepare_h1_step(self, **plan): self.plan = plan
        def step(self, action):
            state[:, 45:48] = self.plan['next_command_b']
            return observation, torch.zeros(32), torch.zeros(32, dtype=torch.bool), torch.zeros(32, dtype=torch.bool), {}
    driver = UnifiedH1Driver(Environment(), observation, clock=clock)
    driver.step(torch.full((32, 4), .5))
    assert driver.take_evidence()[0]['_h1_driver']['pure_hover'] is pure_hover


@pytest.mark.parametrize('pure_hover,expected', [(True, ['center', 'center']),
                                               (False, ['near_x_wall', 'near_y_wall']),
                                               (None, ['near_x_wall', 'near_y_wall'])])
def test_reset_episode_labels_respect_explicit_research_distribution(tmp_path, pure_hover, expected):
    journal = LearningLog(tmp_path)
    driver = {'global_policy_step': 1599}
    if pure_hover is not None: driver['pure_hover'] = pure_hover
    root = torch.zeros(32, 13)
    root[:, 2] = 2
    transition = SimpleNamespace(reward=torch.zeros(32), reset_boundary=torch.ones(32, dtype=torch.bool))
    evidence = {'_h1_driver': driver, 'pre_reset': {'root_state_w': root,
                'task': {'actual_velocity_b': torch.zeros(32, 3), 'exact_clearance_m': torch.ones(32, 1)}}}
    journal._record_blocks_and_episodes(transition, evidence, torch.zeros(32, 3), update=100, step=0)
    assert journal._episode_start_types[6:8] == expected
