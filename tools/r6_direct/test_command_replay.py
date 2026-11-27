from copy import deepcopy

import pytest

from training_workloads.unified_h1.command_replay import sampled_events
from training_workloads.unified_h1.command_replay_audit import body_to_world, score_lane


def test_command_pair_is_admitted_as_normal_system_run(tmp_path, monkeypatch):
    from tools.r6_direct import storage

    monkeypatch.setattr(storage, 'SYSTEM_ROOT', tmp_path)
    run_id = 'R6-DIRECT-INSTINCT-GPU-COMMAND-PAIR-01'
    root = tmp_path/run_id
    plan = dict(run_id=run_id, attempt_dir=str(root), output_root=str(root/'control'))
    assert storage.system_paths(plan) == root
    with pytest.raises(ValueError, match='DIRECT_SYSTEM_RUN_ID'):
        storage.system_paths(plan, diagnostic=True)


def test_pair_constructs_the_shared_research_backend(tmp_path, monkeypatch):
    import json
    from tools.r6_evidence import file_pin
    from training_workloads.unified_h1 import command_replay as replay
    from training_workloads.unified_h1.backend import UnifiedH1LiveBackend

    models = {}
    for label in replay.MODELS:
        path = tmp_path/f'{label}.pt'
        path.write_bytes(label.encode())
        models[label] = file_pin(path)
    monkeypatch.setattr(replay, 'MODELS', {name: p['sha256'] for name, p in models.items()})
    stream = tmp_path/'stream.json'
    stream.write_text(json.dumps(dict(schema_version='instinctrl-world-command-stream/1', source='scripted_development',
        duration_ms=18000, events=[{'t_ms': 0, 'velocity_w_m_s': [.2, 0, 0]},
                                  {'t_ms': 6400, 'velocity_w_m_s': [-.2, 0, 0]}])))
    contract = replay.contract(dict(experiment_id=replay.PREFIX+'PAIR-01', seed=3, training_seed=20260908,
                                    models=models, execution=replay.EXECUTION, stream=file_pin(stream),
                                    wall_seconds=120, output_cap_bytes=512*2**20))
    (tmp_path/'results'/contract['mode']).mkdir(parents=True)
    backend = UnifiedH1LiveBackend(run_contract=contract, attempt=tmp_path, app=None,
                                  environment=None, resource_guard=None)
    assert backend.curriculum_generator.initial_seed() == 20260908
    assert backend.runtime_contract['budgets']['training_transitions'] == 0


def test_replay_causally_holds_timestamped_events():
    stream = {'duration_ms': 500, 'events': [{'t_ms': t} for t in (0, 60, 100, 215)]}
    assert sampled_events(stream) == [0, 2, 2, 3, 3, 3]
    stream['events'][2]['t_ms'] = 101
    assert sampled_events(stream) == [0, 1, 2, 3, 3, 3]


def test_route_schedule_selects_the_lane_layout_stream():
    from pathlib import Path
    from tools.r6_evidence import file_pin
    from training_workloads.unified_h1.command_replay import ROUTE_PREFIX, event_schedule

    root = Path(__file__).resolve().parents[2]
    stream = file_pin(root/'docs/experiments/instinct-continuous-command/routes/STREAMS.json')
    layouts, ids, events = event_schedule({'experiment_id': ROUTE_PREFIX+'01', 'stream': stream})
    assert len(ids) == len(events) == 421
    assert [layouts[i] for i in (0, 8, 16, 24, 28)] == ['room_corner', 'trunk_aisle', 'street_gap', 'corridor', 'open_tube']
    assert events[64][16]['velocity_w_m_s'] == [-.25, 0., 0.]
    assert events[64][0]['velocity_w_m_s'] == [.28, 0., 0.]
    assert events[100][24]['velocity_w_m_s'] == [.15, .19, 0.]


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
    assert report['after_corridor']['actual_resume_s'] == .1
    assert report['after_corridor']['issued_resume_s'] == 0.
    assert report['first_contact_step'] == 8
    assert report['elapsed_s'] == .9
    assert report['minimum_sampled_ray_m'] == .5


def test_crazy_generator_preserves_prefix_and_freezes_all_modes(tmp_path):
    import json
    from pathlib import Path
    from tools.r6_direct.generate_crazy_commands import generate
    from tools.r6_evidence import file_pin
    from training_workloads.unified_h1.command_replay import CRAZY_MODES, load_route_streams

    root = Path(__file__).resolve().parents[2]
    prefix = file_pin(root/'docs/experiments/instinct-continuous-command/routes/STREAMS.json')
    recorded = generate(prefix, 725, 62000)
    assert generate(prefix, 725, 62000) == recorded
    assert generate(prefix, 726, 62000) != recorded
    path = tmp_path/'streams.json'
    path.write_text(json.dumps(recorded))
    readback = load_route_streams(file_pin(path))
    for stream in readback.values():
        assert {e['mode'] for e in stream['events'] if 'mode' in e} == set(CRAZY_MODES)


def test_motion_tube_keeps_forward_hazards_and_unexplained_closure():
    import torch
    from instinctlab.control.ics import ICSConfig, attenuate_command, mounted_mid360_rays_body

    rays = mounted_mid360_rays_body()
    side = (rays - torch.tensor([.109, .994, 0.])).square().sum(dim=1).argmin()
    ahead = rays[:, 0].argmax()
    cmd = torch.tensor([[.24, 0., 0.]])
    radial = ICSConfig(clearance_margin_m=.15)
    tube = ICSConfig(clearance_margin_m=.15, motion_tube_radius_m=.60)
    confirms = ICSConfig(clearance_margin_m=.15, motion_tube_radius_m=.60, motion_tube_confirm_closure=True)

    def output(beam, history, config):
        grid = torch.zeros(1, 4, 3, len(rays))
        grid[:, :, 0] = 40.
        grid[0, :, 0, beam] = torch.tensor(history)
        grid[0, :, 1:, beam] = 1.
        return attenuate_command(grid.reshape(1, 12, 59, 360), rays, cmd, .1, config).v_final_b[0, 0].item()

    assert output(side, [.92]*4, radial) == 0.
    assert output(side, [.92]*4, tube) == pytest.approx(.24)
    assert output(ahead, [.92]*4, tube) == 0.
    assert output(side, [1.1, 1.00, .94, .88], tube) == 0.
    assert output(side, [1.1, 1.00, .94, .88], confirms) == 0.
    one_spike = [1.10, .99, .94, .93]
    assert output(side, one_spike, tube) == 0.
    assert output(side, one_spike, confirms) == pytest.approx(.24)


def test_clearance_uses_the_nearest_finite_wall_surface():
    import numpy as np
    from tools.r6_direct.replay_clearance import triangle_distance_squared

    triangle = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    queries = np.array([[.5,.5,1.], [1.5,1.5,0.], [-1.,-1.,0.]])
    assert triangle_distance_squared(queries, triangle).tolist() == pytest.approx([1., .5, 2.])


def test_early_brake_replaces_only_an_accelerating_escape_at_its_trigger(monkeypatch):
    from types import SimpleNamespace
    import torch
    from instinctlab.control.escape_option import DURATION_STEPS, EscapeOption
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.governor import GovernorConfig, decode_governor

    policy = {'lidar_grid': torch.empty(2, 1), 'state_vec': torch.zeros(2, 52)}
    policy['state_vec'][:, 45] = .25
    action = torch.tensor([[0., .8, .5, .5], [0., .5, .8, .5]])
    option = EscapeRecovery(lanes=2, device='cpu', dtype=torch.float32, recenter=False,
                            brake_forward_escape=True)
    option.applied.fill_(True)
    option.direction[:, 0] = 1.
    option.remaining.fill_(DURATION_STEPS-1)
    monkeypatch.setattr(EscapeOption, 'apply', lambda self, policy, action: action.clone())
    governor = GovernorConfig(v_corr_limit=.5)
    monkeypatch.setattr(option, 'issued', lambda lidar, state, action, cfg: SimpleNamespace(
        v_final_b=decode_governor(action, state, governor).v_gov_b))

    def selected():
        return option.apply_recovery(policy, action, None)

    assert selected()[0].tolist() == [0., .5, .5, .5]
    assert option.braking.tolist() == [True, False]
    option.remaining.sub_(1)
    policy['state_vec'][:, 45] = -.25
    assert selected()[0].tolist() == [0., .5, .5, .5]
    option.applied.zero_()
    assert torch.equal(selected(), action)
    assert not option.braking.any()
