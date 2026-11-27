from copy import deepcopy

import pytest

from training_workloads.unified_h1.command_replay import sampled_events
from training_workloads.unified_h1.command_replay_audit import body_to_world, score_lane


def test_replay_causally_holds_timestamped_events():
    stream = {'duration_ms': 500, 'events': [{'t_ms': t} for t in (0, 60, 100, 215)]}
    assert sampled_events(stream) == [0, 2, 2, 3, 3, 3]
    stream['events'][2]['t_ms'] = 101
    assert sampled_events(stream) == [0, 1, 2, 3, 3, 3]


def test_audit_projects_actor_command_to_world():
    half = 2**-.5
    assert body_to_world([half, 0, 0, half], [0, -1, 0]) == pytest.approx([1, 0, 0])


def test_recovery_uses_new_intent_and_stops_at_first_reset():
    sample = {
        'raw_event': {'velocity_w_m_s': [-.2, 0., 0.]}, 'event_index': 1,
        'root_state_after_w': [[0.]*7+[-.2, 0., 0.]], 'mover_position_w': [[1., 0., 0.]],
        'mover_position_after_w': [[1., 0., 0.]], 'reset': [False], 'collision': [False],
        'ray_minimum_m': [.5], 'mover_winning_beams': [0], 'v_final_w': [[-.2, 0., 0.]],
        'ics_emergency': [False], 'ics_beta': [1.], 'escape_active': [False],
        'escape_recovering': [False], 'termination_reason': [0],
    }
    rows = [dict(deepcopy(sample), step=i) for i in range(11)]
    for row in rows[2:]:
        row['mover_position_w'][0][1] = 1.
    rows[8]['reset'][0] = rows[8]['collision'][0] = True
    rows[8]['termination_reason'][0] = 1
    rows[9]['ray_minimum_m'][0] = .01
    report = score_lane(rows, {'path_reference_w': [[0., 0., 0.]]}, 0, True)
    assert report['actual_resume_after_corridor_s'] == .1
    assert report['issued_resume_after_corridor_s'] == 0.
    assert report['first_collision_step'] == 8
    assert report['elapsed_s'] == .9
    assert report['minimum_sampled_ray_m'] == .5
