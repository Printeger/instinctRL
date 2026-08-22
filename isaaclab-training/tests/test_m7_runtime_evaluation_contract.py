"""Pure tests for exact M6-to-M7 runtime evaluation wiring."""

from __future__ import annotations

from collections import OrderedDict
import copy

import pytest
import torch

from instinctlab.evaluation import EVALUATION_INPUT_KEYS, RuntimeEvaluationCollector, RuntimeEvaluationConfig
from instinctlab.evaluation.runtime import (
    ANCHOR_KEYS,
    CONTROL_KEYS,
    ICS_KEYS,
    PRE_RESET_KEYS,
    SOURCE_MANIFEST,
    TASK_KEYS,
    TERMINATION_KEYS,
    TERMINATION_MASK_KEYS,
)


def _policy() -> OrderedDict[str, torch.Tensor]:
    lidar = torch.empty(2, 12, 59, 360, dtype=torch.float32)
    lidar[:, 0::3] = 40.0
    lidar[:, 1::3] = 1.0
    lidar[:, 2::3] = 1.0
    state = torch.zeros(2, 52, dtype=torch.float32)
    state[:, 5::13] = 1.0
    state[:, -7:-4] = torch.tensor([[0.1, 0.2, 0.3], [-0.3, 0.0, 0.4]])
    return OrderedDict((('lidar_grid', lidar), ('state_vec', state)))


def _critic(x: float = 0.0) -> torch.Tensor:
    value = torch.zeros(2, 18, dtype=torch.float32)
    value[:, 0] = torch.tensor([x, x + 10.0])
    value[:, 3] = 1.0
    return value


def _f(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _pre_reset(*, timeout: bool = False) -> tuple[OrderedDict[str, object], torch.Tensor, torch.Tensor]:
    root = _critic(1.0)[:, :13].clone()
    done = torch.tensor([[timeout], [False]], dtype=torch.bool)
    false = torch.zeros(2, 1, dtype=torch.bool)
    timeout_mask = done.clone()
    masks = OrderedDict((
        ('collision', false.clone()), ('below_height', false.clone()),
        ('above_height', false.clone()), ('timeout', timeout_mask),
    ))
    assert tuple(masks) == TERMINATION_MASK_KEYS
    termination = OrderedDict((
        ('terminated', false.clone()), ('truncated', done.clone()), ('done', done.clone()),
        ('reason_code', torch.tensor([[1 if timeout else 0], [0]], dtype=torch.long)), ('masks', masks),
    ))
    assert tuple(termination) == TERMINATION_KEYS
    anchor = OrderedDict((
        ('active', _f([[1.0], [0.0]])), ('reset_code', _f([[0.0], [0.0]])),
        ('usable_fraction', _f([[0.8], [0.0]])),
        ('mean_absolute_error_m', _f([[0.2], [0.4]])), ('huber_loss', _f([[0.02], [0.08]])),
    ))
    assert tuple(anchor) == ANCHOR_KEYS
    ics = OrderedDict((
        ('beta', _f([[0.0], [1.0]])), ('active_beam_count', _f([[4.0], [5.0]])),
        ('minimum_clearance_m', _f([[0.9], [1.1]])),
        ('worst_margin_m_s', _f([[-0.1], [0.2]])), ('emergency', _f([[1.0], [0.0]])),
    ))
    assert tuple(ics) == ICS_KEYS
    control = OrderedDict((
        ('alpha', _f([[1.0], [1.0]])), ('v_corr_b', _f([[0.0] * 3, [0.0] * 3])),
        ('v_gov_b', _f([[0.1, 0.2, 0.3], [0.0, 0.0, 0.1]])),
        ('v_final_b', _f([[0.05, 0.1, 0.15], [0.0, 0.0, 0.1]])),
        ('v_final_w', _f([[0.05, 0.1, 0.15], [0.0, 0.0, 0.1]])),
        ('anchor_metrics', anchor), ('ics_metrics', ics),
    ))
    assert tuple(control) == CONTROL_KEYS
    task = OrderedDict((
        ('root_state_w', root.clone()), ('actual_velocity_b', _f([[0.02, 0.1, 0.2], [0.0, 0.0, 0.0]])),
        ('exact_clearance_m', _f([[0.8], [1.2]])), ('collision', false.clone()),
        ('termination', termination),
    ))
    assert tuple(task) == TASK_KEYS
    episode = torch.tensor([8 if timeout else 1, 1], dtype=torch.long)
    pre = OrderedDict((
        ('reset_mask', done[:, 0].clone()), ('episode_length', episode), ('root_state_w', root.clone()),
        ('joint_pos', torch.zeros(2, 4)), ('joint_vel', torch.zeros(2, 4)), ('control', control),
        ('lee', OrderedDict()), ('wrench', OrderedDict()), ('reward', OrderedDict()), ('task', task),
    ))
    assert tuple(pre) == PRE_RESET_KEYS
    returned = torch.zeros(2, 18, dtype=torch.float32)
    returned[:, :13] = root
    returned[:, 3] = 1.0
    if timeout:
        returned[0] = _critic(100.0)[0]
    returned_episode = torch.where(done[:, 0], torch.zeros_like(episode), episode)
    return pre, returned, returned_episode


def _record(collector: RuntimeEvaluationCollector, policy: OrderedDict[str, torch.Tensor], *, timeout: bool) -> None:
    audited = collector.prepare_policy(policy)
    pre, returned, episode = _pre_reset(timeout=timeout)
    collector.record_transition(
        pre_action_policy=audited,
        action=torch.full((2, 4), 0.5),
        pre_reset=pre,
        returned_critic=returned,
        returned_episode_length=episode,
    )


def _collector() -> RuntimeEvaluationCollector:
    return RuntimeEvaluationCollector(_critic(), RuntimeEvaluationConfig(expected_policy_steps=2))


def test_exact_source_mapping_reset_lifecycle_and_canonical_record() -> None:
    collector = _collector()
    policy = _policy()
    _record(collector, policy, timeout=False)
    _record(collector, policy, timeout=True)
    tree = collector.finalize(OrderedDict((('product_commit', '0' * 40), ('fixture_id', 'runtime-v1'), ('generator_id', 'pytest'), ('seed', 7))))
    values = tree['evaluation_inputs']
    assert tuple(SOURCE_MANIFEST) == EVALUATION_INPUT_KEYS == tuple(values)
    assert values['commanded_velocity_b_m_s'].shape == (2, 2, 3)
    assert torch.equal(values['commanded_velocity_b_m_s'][0], policy['state_vec'][:, -7:-4])
    assert torch.equal(values['position_w_m'][1, :, 0], torch.tensor([1.0, 11.0]))
    assert tree['reset_masks'].tolist() == [[False, False], [True, False]]
    assert tree['returned_episode_lengths'].tolist() == [[1, 1], [0, 1]]
    assert tree['station_reference_before_w_m'][0, :, 0].tolist() == [0.0, 10.0]
    assert tree['station_reference_after_w_m'][1, :, 0].tolist() == [100.0, 10.0]
    assert tree['gates']['reset_count'] == 1
    assert '"actor_audit":{"status":"PASS"' in tree['canonical_record']


def test_spy_actor_receives_exactly_one_unchanged_audited_policy_mapping() -> None:
    class Spy:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def __call__(self, value: object) -> torch.Tensor:
            self.calls.append(value)
            assert tuple(value) == ('lidar_grid', 'state_vec')
            return torch.full((2, 4), 0.5)

    collector, policy, spy = _collector(), _policy(), Spy()
    audited = collector.prepare_policy(policy)
    action = spy(audited)
    pre, returned, episode = _pre_reset()
    collector.record_transition(pre_action_policy=audited, action=action, pre_reset=pre,
                                returned_critic=returned, returned_episode_length=episode)
    assert spy.calls == [policy] and audited is policy


@pytest.mark.parametrize('mutation', [
    'missing', 'reordered', 'policy_changed', 'action_grad', 'action_nonfinite', 'alias_policy',
    'alias_sources', 'dtype', 'device', 'reset_mismatch', 'episode_mismatch',
    'stale_live', 'postreset_terminal', 'nonfinite',
])
def test_runtime_sources_fail_closed(mutation: str) -> None:
    collector, policy = _collector(), _policy()
    audited = collector.prepare_policy(policy)
    pre, returned, episode = _pre_reset(timeout=mutation == 'postreset_terminal')
    action = torch.full((2, 4), 0.5)
    if mutation == 'missing':
        pre['control'].pop('alpha')
    elif mutation == 'reordered':
        pre.move_to_end('reset_mask')
    elif mutation == 'policy_changed':
        policy['state_vec'][0, 0] = 0.25
    elif mutation == 'action_grad':
        action.requires_grad_()
    elif mutation == 'action_nonfinite':
        action[0, 0] = float('nan')
    elif mutation == 'alias_policy':
        pre['task']['actual_velocity_b'] = policy['state_vec'].as_strided((2, 3), (52, 1))
    elif mutation == 'alias_sources':
        pre['control']['v_final_b'] = pre['task']['actual_velocity_b']
    elif mutation == 'dtype':
        pre['task']['actual_velocity_b'] = pre['task']['actual_velocity_b'].double()
    elif mutation == 'device':
        if not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
        pre['task']['actual_velocity_b'] = pre['task']['actual_velocity_b'].cuda()
    elif mutation == 'reset_mismatch':
        pre['reset_mask'][0] = True
    elif mutation == 'episode_mismatch':
        episode[1] += 1
    elif mutation == 'stale_live':
        returned[1, 0] += 1.0
    elif mutation == 'postreset_terminal':
        pre['task']['root_state_w'][0] = returned[0, :13]
        pre['root_state_w'][0] = returned[0, :13]
    elif mutation == 'nonfinite':
        pre['task']['exact_clearance_m'][0, 0] = float('inf')
    with pytest.raises((TypeError, ValueError)):
        collector.record_transition(pre_action_policy=audited, action=action, pre_reset=pre,
                                    returned_critic=returned, returned_episode_length=episode)


def test_output_is_owned_and_incomplete_or_extra_coverage_fails() -> None:
    collector, policy = _collector(), _policy()
    _record(collector, policy, timeout=False)
    with pytest.raises(RuntimeError):
        collector.finalize(OrderedDict())
    _record(collector, policy, timeout=True)
    tree = collector.finalize(OrderedDict((('product_commit', '0' * 40), ('fixture_id', 'x'), ('generator_id', 'x'), ('seed', 1))))
    snapshot = copy.deepcopy(tree)
    policy['state_vec'].add_(1.0)
    assert torch.equal(tree['evaluation_inputs']['commanded_velocity_b_m_s'], snapshot['evaluation_inputs']['commanded_velocity_b_m_s'])
    with pytest.raises(RuntimeError):
        collector.prepare_policy(_policy())
