"""Pure tests for exact M6-to-M7 runtime evaluation wiring."""

from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json

import pytest
import torch

from instinctlab.evaluation import EVALUATION_INPUT_KEYS, RuntimeEvaluationCollector, RuntimeEvaluationConfig
from instinctlab.evaluation.runtime import (
    ANCHOR_KEYS,
    CONTROL_KEYS,
    ICS_KEYS,
    PRE_RESET_KEYS,
    SOURCE_MANIFEST,
    SOURCE_METADATA_KEYS,
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
        returned_policy=copy.deepcopy(policy),
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
    assert all(tuple(entry) == SOURCE_METADATA_KEYS for entry in tree['source_manifest'].values())
    assert tree['source_manifest'] == SOURCE_MANIFEST
    encoded_manifest = json.dumps(tree['source_manifest'], separators=(',', ':')).encode()
    assert hashlib.sha256(encoded_manifest).hexdigest() == '08c62ddb3a6ce6ce6df0f673f14b2826e2de85107d59eb79d4eeed2530f64af0'
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
                                returned_policy=copy.deepcopy(policy), returned_critic=returned,
                                returned_episode_length=episode)
    assert spy.calls == [policy] and audited is policy


@pytest.mark.parametrize('mutation', [
    'missing', 'extra', 'reordered', 'malformed_nested', 'wrong_shape', 'policy_changed',
    'action_grad', 'action_nonfinite', 'alias_policy', 'alias_sources', 'alias_returned_policy',
    'dtype', 'device', 'reset_mismatch', 'episode_mismatch', 'stale_live',
    'postreset_terminal', 'nonfinite', 'anchor_active_domain', 'anchor_usable_domain',
    'negative_anchor_error', 'negative_clearance', 'beta_domain', 'emergency_beta',
    'collision_mismatch', 'terminated_masks', 'truncated_mask', 'reason_precedence',
])
def test_runtime_sources_fail_closed(mutation: str) -> None:
    collector, policy = _collector(), _policy()
    audited = collector.prepare_policy(policy)
    pre, returned, episode = _pre_reset(timeout=mutation == 'postreset_terminal')
    returned_policy = copy.deepcopy(policy)
    action = torch.full((2, 4), 0.5)
    if mutation == 'missing':
        pre['control'].pop('alpha')
    elif mutation == 'extra':
        pre['unexpected'] = torch.zeros(2)
    elif mutation == 'reordered':
        pre.move_to_end('reset_mask')
    elif mutation == 'malformed_nested':
        pre['control'] = torch.zeros(2)
    elif mutation == 'wrong_shape':
        pre['task']['actual_velocity_b'] = torch.zeros(2, 2)
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
    elif mutation == 'alias_returned_policy':
        returned_policy['lidar_grid'] = policy['lidar_grid']
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
    elif mutation == 'anchor_active_domain':
        pre['control']['anchor_metrics']['active'][0, 0] = 0.5
    elif mutation == 'anchor_usable_domain':
        pre['control']['anchor_metrics']['usable_fraction'][0, 0] = 1.1
    elif mutation == 'negative_anchor_error':
        pre['control']['anchor_metrics']['mean_absolute_error_m'][0, 0] = -0.1
    elif mutation == 'negative_clearance':
        pre['task']['exact_clearance_m'][0, 0] = -0.1
    elif mutation == 'beta_domain':
        pre['control']['ics_metrics']['beta'][0, 0] = 1.1
    elif mutation == 'emergency_beta':
        pre['control']['ics_metrics']['beta'][0, 0] = 0.5
    elif mutation == 'collision_mismatch':
        pre['task']['collision'][0, 0] = True
    elif mutation == 'terminated_masks':
        pre['task']['termination']['terminated'][0, 0] = True
        pre['task']['termination']['done'][0, 0] = True
        pre['reset_mask'][0] = True
        episode[0] = 0
    elif mutation == 'truncated_mask':
        pre['task']['termination']['truncated'][0, 0] = True
        pre['task']['termination']['done'][0, 0] = True
        pre['reset_mask'][0] = True
        episode[0] = 0
    elif mutation == 'reason_precedence':
        pre['task']['termination']['reason_code'][0, 0] = 4
    with pytest.raises((TypeError, ValueError)):
        collector.record_transition(pre_action_policy=audited, action=action, pre_reset=pre,
                                    returned_policy=returned_policy, returned_critic=returned,
                                    returned_episode_length=episode)


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
    pointers = [value.untyped_storage().data_ptr() for value in tree['evaluation_inputs'].values()]
    assert len(pointers) == len(set(pointers))
    with pytest.raises(RuntimeError):
        collector.prepare_policy(_policy())


def test_returned_policy_is_owned_and_binds_exactly_to_next_audited_policy() -> None:
    collector, first = _collector(), _policy()
    audited = collector.prepare_policy(first)
    pre, returned_critic, episode = _pre_reset(timeout=True)
    returned_policy = _policy()
    expected_next = copy.deepcopy(returned_policy)
    collector.record_transition(
        pre_action_policy=audited,
        action=torch.full((2, 4), 0.5),
        pre_reset=pre,
        returned_policy=returned_policy,
        returned_critic=returned_critic,
        returned_episode_length=episode,
    )
    returned_policy['state_vec'].add_(1.0)
    assert collector.prepare_policy(expected_next) is expected_next


@pytest.mark.parametrize('mutation', ['keys', 'shape', 'dtype', 'device', 'values'])
def test_returned_policy_next_audited_binding_fails_closed(mutation: str) -> None:
    collector, first = _collector(), _policy()
    audited = collector.prepare_policy(first)
    pre, returned_critic, episode = _pre_reset(timeout=True)
    returned_policy = _policy()
    collector.record_transition(
        pre_action_policy=audited,
        action=torch.full((2, 4), 0.5),
        pre_reset=pre,
        returned_policy=returned_policy,
        returned_critic=returned_critic,
        returned_episode_length=episode,
    )
    next_policy = copy.deepcopy(returned_policy)
    if mutation == 'keys':
        next_policy.move_to_end('lidar_grid')
    elif mutation == 'shape':
        next_policy['state_vec'] = next_policy['state_vec'][:, :-1]
    elif mutation == 'dtype':
        next_policy['state_vec'] = next_policy['state_vec'].double()
    elif mutation == 'device':
        if not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
        next_policy = OrderedDict((name, value.cuda()) for name, value in next_policy.items())
    elif mutation == 'values':
        next_policy['state_vec'][0, 0] = 0.25
    with pytest.raises((TypeError, ValueError)):
        collector.prepare_policy(next_policy)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'reordered', 'malformed', 'field_value'])
def test_structured_source_manifest_fails_closed(monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    import instinctlab.evaluation.runtime as runtime

    manifest = copy.deepcopy(SOURCE_MANIFEST)
    if mutation == 'missing':
        manifest.pop('observability_source')
    elif mutation == 'extra':
        manifest['unexpected'] = copy.deepcopy(next(iter(manifest.values())))
    elif mutation == 'reordered':
        manifest.move_to_end('commanded_velocity_b_m_s')
    elif mutation == 'malformed':
        manifest['commanded_velocity_b_m_s'] = 'not metadata'
    elif mutation == 'field_value':
        manifest['commanded_velocity_b_m_s']['units'] = 'km/h'
    monkeypatch.setattr(runtime, 'SOURCE_MANIFEST', manifest)
    collector, policy = _collector(), _policy()
    _record(collector, policy, timeout=False)
    _record(collector, policy, timeout=True)
    with pytest.raises(RuntimeError):
        collector.finalize(OrderedDict((('product_commit', '0' * 40), ('fixture_id', 'x'), ('generator_id', 'x'), ('seed', 1))))


def test_emitted_source_manifest_is_deeply_owned() -> None:
    first_collector, policy = _collector(), _policy()
    _record(first_collector, policy, timeout=False)
    _record(first_collector, policy, timeout=True)
    provenance = OrderedDict((('product_commit', '0' * 40), ('fixture_id', 'x'), ('generator_id', 'x'), ('seed', 1)))
    first = first_collector.finalize(provenance)
    module_snapshot = copy.deepcopy(SOURCE_MANIFEST)
    first['source_manifest']['commanded_velocity_b_m_s']['source'] = 'mutated'
    first['source_manifest']['commanded_velocity_b_m_s'].move_to_end('source')
    first['source_manifest'].move_to_end('commanded_velocity_b_m_s')
    second_collector = _collector()
    _record(second_collector, _policy(), timeout=False)
    _record(second_collector, _policy(), timeout=True)
    second = second_collector.finalize(provenance)
    assert SOURCE_MANIFEST == module_snapshot
    assert second['source_manifest'] == module_snapshot
