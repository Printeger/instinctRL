from copy import deepcopy

import pytest

from training_workloads.unified_h1.command_replay import sampled_events
from training_workloads.unified_h1.command_replay_audit import body_to_world, score_lane


def test_long_trace_gzip_preserves_the_original_json_lines(tmp_path):
    from training_workloads.unified_h1.command_replay import LONG_PREFIX, encode_trace_row, open_trace, trace_path

    rows = [{'kind': 'initial'}, {'kind': 'transition', 'measured': .22177478671073914}]
    raw = b''.join(encode_trace_row(row, compressed=False) for row in rows)
    path = trace_path(tmp_path, 'candidate', 'layout_routes', run_id=LONG_PREFIX+'01')
    path.write_bytes(b''.join(encode_trace_row(row, compressed=True) for row in rows))
    with open_trace(path) as replay:
        assert replay.read().encode() == raw


def test_street_sweep_checks_each_physical_tick_and_restarts_the_lane_clock(tmp_path):
    import numpy as np
    import torch
    from instinctlab.observations.moving_sphere import street_sweep_pose
    from training_workloads.unified_h1.command_replay import CLEARANCE_SHAPE, LANES, authored_street_sweep, command_layout
    from training_workloads.unified_h1.command_replay_audit import verify_street_sweep

    layouts = [command_layout(i) for i in range(LANES)]
    street = [i for i, layout in enumerate(layouts) if layout == 'street_gap']
    origins = torch.arange(LANES * 3, dtype=torch.float64).reshape(LANES, 3)
    sweep = authored_street_sweep(829377162)
    phases = torch.as_tensor(sweep['phase_m'], dtype=torch.float64)
    rows = [{'reset': [i == street[0] for i in range(LANES)]}, {'reset': [False] * LANES}]
    starts = torch.zeros((2, len(street)), dtype=torch.float64)
    starts[1, 1:] = .10
    substep = torch.arange(21, dtype=torch.float64)[None, :, None] * .005
    locations = street_sweep_pose(origins[street][None, None], starts[:, None] + substep, phases[None, None])
    initial_positions = origins.clone()
    initial_positions[street] = locations[0, 0]
    initial = {'scene_origins_w': origins.tolist(), 'mover_start_w': initial_positions.tolist()}
    samples = np.zeros((2, *CLEARANCE_SHAPE), dtype='<f4')
    for index, lane in enumerate(street):
        samples[:, :, lane, 3:6] = locations[:, :, index].numpy()
    path = tmp_path / 'clearance.f32'
    samples.tofile(path)
    verify_street_sweep(path, initial, rows, layouts, sweep)
    samples[1, 5, street[0], 4] += .01
    samples.tofile(path)
    with pytest.raises(ValueError, match='COMMAND_REPLAY_SWEEP_CLEARANCE_PATH'):
        verify_street_sweep(path, initial, rows, layouts, sweep)


def test_shallow_steering_requires_two_observed_paths_and_respects_protection():
    import torch
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.ics import ICSConfig
    from instinctlab.control.pipeline import ControlCoreConfig

    option = EscapeRecovery(lanes=5, device='cpu', dtype=torch.float32, recenter=True,
        align_intercept=True, intercept=True, intercept_reserve=True, commit_intercept=True,
        retreat_priority=True, command_handoff=True, wall_tangent=True, finite_edge=True, rank_return=True,
        intercept_choice=True, reserve_direction=True, along_intercept=True, finite_intercept=True,
        crossing_stop=True, crossing_stop_continuation=True, crossing_stop_shallow=True,
        crossing_stop_early=True, crossing_reference=True, shallow_steering=True)
    state = torch.zeros((5, 4, 13))
    state[:, :, 5] = -1.
    state[:, :, 6] = .28
    state[-1, :, 6] = 0.
    lidar = torch.ones((5, 12, 59, 360))
    lidar[:, 0::3] = 3.
    lidar[1, 5::3] = .4
    lidar[2, 0::3] = .2
    selected = torch.tensor([[0., .5, .5, .5]]).repeat(5, 1)
    actor = torch.tensor([[1., .5, .5, .5]]).repeat(5, 1)
    option.applied[:] = True
    option.remaining[:] = 12
    config = ControlCoreConfig(ics=ICSConfig(motion_tube_radius_m=.60, confirmed_tube_radius_m=.41))
    option.steer_along_observed_opening(lidar, state.flatten(1), selected, actor, config,
                                      torch.tensor([True, True, True, False, True]))

    assert option.shallow_used.tolist() == [True, False, False, False, False]
    assert option.applied.tolist() == [False, True, True, True, True]
    assert option.remaining.tolist() == [0, 12, 12, 12, 12]
    assert option.issued(lidar, state.flatten(1), selected, config).v_final_b[0, 0] > .26


def test_tracked_resume_is_reproved_without_canceling_the_held_escape():
    import torch
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.pipeline import ControlCoreConfig

    option = EscapeRecovery(lanes=3, device='cpu', dtype=torch.float32, recenter=True,
        align_intercept=True, intercept=True, intercept_reserve=True, commit_intercept=True, clear_pass=True,
        retreat_priority=True, command_handoff=True, rank_return=True, intercept_choice=True,
        reserve_direction=True, along_intercept=True, finite_intercept=True, crossing_stop=True,
        crossing_stop_continuation=True, crossing_stop_shallow=True, crossing_stop_early=True,
        crossing_reference=True, static_scan_stop=True, tracked_resume=True)
    state = torch.zeros((3, 4, 13))
    state[:, :, 5] = -1.
    state[:, :, 6] = .28
    lidar = torch.ones((3, 12, 59, 360))
    lidar[:, 0::3] = 3.
    selected = torch.tensor([[0., .5, .5, .5]]).repeat(3, 1)
    actor = torch.tensor([[1., .5, .5, .5]]).repeat(3, 1)
    option.applied[:] = True
    option.remaining[:] = 12
    tracks = (torch.arange(3), torch.tensor([[2.1, 1.2], [.20, .3], [2.1, 1.2]]),
              torch.tensor([[0., .58]]).repeat(3, 1), torch.zeros((3, 2)), torch.full((3,), .34))
    args = lidar, state.flatten(1), selected, actor, ControlCoreConfig(), option.applied.clone(), tracks
    option.prove_tracked_resume(*args)
    assert not option.tracked_resume_used.any()
    option.prove_tracked_resume(*args)
    option.intercepted[2] = True
    option.resume_after_interception(selected, actor)

    assert option.tracked_resume_used.tolist() == [True, False, False]
    assert option.tracked_resume_witness.tolist() == [True, False, False]
    assert option.applied.tolist() == [True, True, True]
    assert option.remaining.tolist() == [12, 12, 12]
    assert selected[:, 0].tolist() == [1., 0., 0.]


def test_bilateral_stopping_retains_unknown_near_and_emergency_constraints():
    import torch
    from instinctlab.control.bilateral_stopping import BilateralStoppingPipeline, apply_bilateral_stopping
    from instinctlab.control.ics import ICSConfig, attenuate_command, mounted_mid360_rays_body
    from instinctlab.control.pipeline import ControlCoreConfig, ControlCorePipeline

    rays = mounted_mid360_rays_body()
    lateral = rays[:, 1]
    front = torch.tensor([.82, .50, .82, .82, .82])[:, None]
    behind = torch.tensor([1.16, 1.48, 8., 1.16, 1.16])[:, None]
    distance = torch.where(lateral > 0., front, behind)/lateral.abs().clamp_min(1.e-6)
    depth = distance.clamp_max(8.)[:, None].expand(-1, 4, -1).clone()
    closest = (rays-torch.tensor([0., 1., 0.])).square().sum(dim=1).argmin()
    depth[3, :, closest] = .43
    depth[4, :, closest] = .30
    lidar = torch.stack((depth, torch.ones_like(depth), torch.ones_like(depth)), dim=2).reshape(5, 12, 59, 360)
    history = torch.zeros(5, 4, 13)
    history[:, :, 5] = -1.
    history[:, :, 7] = .28
    target = history[:, -1, 6:9]
    config = ICSConfig(clearance_margin_m=.15, motion_tube_radius_m=.60, confirmed_tube_radius_m=.41)
    ordinary = attenuate_command(lidar, rays, target, .1, config, state_vec=history.flatten(1))
    certified = apply_bilateral_stopping(lidar, history.flatten(1), rays, target, .1, config, ordinary)

    assert not ordinary.v_final_b.any()
    assert certified.v_final_b[0, 1] > .25
    assert torch.equal(certified.v_final_b[1:], ordinary.v_final_b[1:])
    assert torch.equal(certified.metrics['emergency'], ordinary.metrics['emergency'])
    assert torch.equal(certified.cache['active_mask'][1:], ordinary.cache['active_mask'][1:])

    pipeline = BilateralStoppingPipeline(ControlCorePipeline(ControlCoreConfig(ics=config)))
    reset = torch.zeros(1, dtype=torch.bool)
    args = (lidar[:1], history[:1].flatten(1), torch.tensor([[1., .5, .5, .5]]), .1,
            reset, reset, torch.tensor([[1., 0., 0., 0.]]))
    assert not pipeline(*args).v_final_b.any()
    pipeline.allow_bilateral_stopping = True
    assert pipeline(*args).v_final_b[0, 1] > .25
    assert pipeline.bilateral_stopping_used


def test_tube_stopping_retains_independent_and_near_constraints():
    import math

    import torch
    from instinctlab.control.ics import ICSConfig, attenuate_command, mounted_mid360_rays_body
    from instinctlab.control.tube_stopping import certify_tube_request

    rays = mounted_mid360_rays_body()
    angles = torch.arange(12).float()*math.tau/12
    normals = torch.stack((torch.zeros_like(angles), angles.cos(), angles.sin()), dim=1)
    cosines = rays@normals.T
    state = torch.zeros(1,4,13)
    state[:,:,5] = -1.
    state[:,:,7] = -.24
    target = torch.tensor([[0.,-.215,0.]])
    config = ICSConfig(clearance_margin_m=.15, motion_tube_radius_m=.60, motion_tube_confirm_closure=True,
        motion_tube_neighbor_history=True, confirmed_tube_radius_m=.41, confirmed_tube_align_rotation=True)
    for offset, obstruction in ((0.,False),(.36,False),(0.,True),(.49,False)):
        distance = .86+offset*normals[:,1]
        ranges = (distance/cosines.masked_fill(cosines <= 0.,1.e-10)).amin(dim=1)
        valid = ranges < 40.
        ranges = ranges.clamp_max(40.)
        if obstruction:
            foreign_surface = (rays[:,1] < -.95) & (rays[:,2].abs() < .12) & ((rays[:,0]-.3).abs() < .08)
            ranges[foreign_surface] = .50
        scan = torch.stack((ranges,valid.float(),valid.float())).reshape(1,3,59,360).repeat(1,4,1,1)
        original = attenuate_command(scan,rays,target,.10,config,state_vec=state.flatten(1))
        assert original.metrics['beta'].item() == 0.
        assert original.metrics['emergency'].item() == (1. if offset == .49 else 0.)
        verified = certify_tube_request(scan,state.flatten(1),rays,target,config,original,previous_speed_bound=.30)
        if offset == 0. and not obstruction:
            assert verified is not None
            assert verified.result.metrics['beta'].item() == 1.
            assert torch.equal(verified.result.v_final_b,verified.target)
            assert .28 < float(verified.target.norm()) < .30
            assert verified.hold_s == .60
            assert verified.observed_gap_m >= verified.required_gap_m and verified.cleared_limiting_beams > 0
            stale = state.clone()
            stale[:,0,12] = .12
            assert certify_tube_request(scan,stale.flatten(1),rays,target,config,original,
                                        previous_speed_bound=.30) is None
        else:
            assert verified is None


def test_finite_wall_proof_keeps_the_radial_stop_at_an_isolated_wall():
    import torch
    from instinctlab.control.ics import ICSConfig, attenuate_command, mounted_mid360_rays_body
    from instinctlab.control.local_wall_stopping import FacetedWallEdgeReproofPipeline, SupportedWallEdgeReproofPipeline

    rays = mounted_mid360_rays_body()
    lateral = rays[:, 1]
    history = torch.zeros(1, 4, 13)
    history[:, :, 5] = -1.
    history[:, :, 7] = .20
    state = history.flatten(1)
    target = torch.tensor([[0., .174, 0.]])
    config = ICSConfig(clearance_margin_m=.15, motion_tube_radius_m=.60, motion_tube_confirm_closure=True,
        motion_tube_neighbor_history=True, confirmed_tube_radius_m=.41, confirmed_tube_align_rotation=True)

    for reverse_distance in (40., 1.30, 1.95):
        ranges = torch.where(lateral > 0., .90, reverse_distance)/lateral.abs().clamp_min(1.e-6)
        valid = (ranges < 40.).float()
        lidar = torch.stack((ranges.clamp_max(40.), valid, valid)).reshape(1, 3, 59, 360).repeat(1, 4, 1, 1)
        base = attenuate_command(lidar, rays, target, .10, config, state_vec=state)
        assert not base.v_final_b.any()
        assert not base.metrics['emergency'].any()
        previous = FacetedWallEdgeReproofPipeline.certify(lidar, state, rays, target, config, base,
                                                         previous_speed_bound=.30)
        updated = SupportedWallEdgeReproofPipeline.certify(lidar, state, rays, target, config, base,
                                                           previous_speed_bound=.30)

        assert previous is not None
        assert torch.equal(previous.result.v_final_b, target)
        if reverse_distance == 1.30:
            assert updated is not None
            assert torch.equal(updated.result.v_final_b, target)
        else:
            assert updated is None


def test_crossing_stop_proof_keeps_the_vehicle_clear_after_it_brakes():
    import torch
    from instinctlab.control.escape_passage import crossing_stop_clearance

    lidar = torch.zeros((1, 12, 1, 1))
    rays = torch.tensor([[1., 0., 0.]])
    position = torch.tensor([[-.075, -2.185]])
    motion = torch.tensor([[0., .58]])
    coast = torch.tensor([[.23, .04]])
    radius = torch.tensor([.34])
    issued = torch.tensor([[[0., .30, 0.], [.293, -.064, 0.], [-.293, -.064, 0.]]])

    clearance = crossing_stop_clearance(lidar, rays, position, motion, coast, radius, issued, hold_s=2.6)
    assert (clearance >= .39).tolist() == [[False, True, False]]

    lidar[:, 9] = .7
    lidar[:, 10:12] = 1.
    clearance = crossing_stop_clearance(lidar, rays, position, motion, coast, radius, issued, hold_s=2.6)
    assert (clearance < .39).all()


def test_crossing_brake_renewal_keeps_the_identity_on_the_shared_scan(monkeypatch):
    import torch
    from instinctlab.control import brake_renewal

    clock = torch.tensor([-.30, -.20, -.10, 0.]).repeat(4, 1)
    centers = torch.zeros(4, 4, 3)
    centers[:, :, 0] = 1.10
    centers[:, :, 1] = .5*clock
    radii = torch.full_like(clock, .34)
    history = torch.zeros(4, 52)
    anchors = torch.cat((centers[:, -2], radii[:, -2, None], torch.tensor([[0., .50, -1.1]]).repeat(4, 1)), dim=1)
    anchors[1, 0] += .04
    anchors[2, 5] = -.50
    anchors[3, 6] = 1.1

    def nearest(lidar, state, rays, *, require_nearest):
        assert require_nearest
        return torch.arange(4), centers, radii, clock

    monkeypatch.setattr(brake_renewal, '_crossing_shapes', nearest)
    lanes, position, motion, coast, radius = brake_renewal.renew_crossing(
        None, history, None, torch.zeros(4, 4, 2), torch.zeros(4, 2), anchors)

    assert lanes.tolist() == [0]
    torch.testing.assert_close(position, torch.tensor([[1.10, 0.]]))
    torch.testing.assert_close(motion, torch.tensor([[0., .50]]))
    torch.testing.assert_close(coast, torch.zeros(1, 2))
    torch.testing.assert_close(radius, torch.tensor([.34]))


def test_crossing_clearance_checks_a_release_before_the_planned_stop():
    import torch
    from instinctlab.control.escape_passage import intercept_scores

    position = torch.tensor([[.20, -1.]])
    motion = torch.tensor([[0., .50]])
    coast = torch.zeros_like(motion)
    radius = torch.tensor([.34])
    issued = torch.tensor([[[.40, 0., 0.]]])
    args = position, motion, coast, radius, issued
    kept = intercept_scores(*args, hold_s=2.4, preview_steps=121)
    releasing = intercept_scores(*args, hold_s=2.4, preview_steps=121, release_early=True)
    sampled = torch.stack([intercept_scores(*args, hold_s=step*.01, preview_steps=121)
                           for step in range(241)]).amin(dim=0)
    assert kept.item() > releasing.item()+.25
    assert torch.allclose(releasing, -radius[:, None], atol=1.e-6)
    assert torch.allclose(releasing, sampled, atol=.004)


def test_extended_crossing_range_keeps_the_original_seed_gate():
    import torch
    from instinctlab.control.escape_passage import _closing_seed

    elapsed = torch.tensor([[-.3, -.2, -.1, 0.]]).expand(2, -1)
    distance = torch.tensor([2.4, 1.5])[:, None]-.6*elapsed
    ranges = distance[:, :, None, None].expand(-1, -1, 17, 19)
    reliable = torch.ones_like(ranges, dtype=torch.bool)

    near, _ = _closing_seed(ranges, reliable, elapsed)
    far, _ = _closing_seed(ranges, reliable, elapsed, minimum_range_m=1.9, maximum_range_m=2.65)
    assert near.tolist() == [False, True]
    assert far.tolist() == [True, False]


def test_early_crossing_direction_uses_the_far_return_seed(monkeypatch):
    import torch
    from instinctlab.control import escape_passage, escape_reference
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.pipeline import ControlCoreConfig

    config = ControlCoreConfig()
    state = torch.zeros((1, 4, 13))
    state[:, :, 5] = -1.
    state[:, :, 6] = -.24
    lidar = torch.ones((1, 12, 59, 360))
    lidar[:, 0::3] = torch.tensor([2.57, 2.51, 2.45, 2.39])[None, :, None, None]
    actor = torch.tensor([[1., .5, .5, .5]])
    eligible = torch.ones(1, dtype=torch.bool)
    response = torch.zeros(1, 4, 2), torch.zeros(1, 2)

    def moving_return(*args, **kwargs):
        return (torch.tensor([0]), torch.tensor([[.025, -2.60]]), torch.tensor([[-.0414, .5861]]),
                torch.zeros(1, 2), torch.tensor([.34]))

    def refine(scan, history, rays, motion, *, seeded):
        far = escape_passage._return_seed(scan, history, minimum_range_m=1.9, maximum_range_m=2.65)
        assert not escape_passage._return_seed(scan, history)[0].any()
        assert far[0].all() and all(torch.equal(actual, expected) for actual, expected in zip(seeded, far))
        return motion.new_tensor([[0., .5861]])

    monkeypatch.setattr(escape_passage, 'moving_intercepts', moving_return)
    monkeypatch.setattr(escape_reference, 'crossing_direction_from_planes', refine)
    for use_plane in (False, True):
        option = EscapeRecovery(lanes=1, device='cpu', dtype=torch.float32, recenter=True,
            align_intercept=True, intercept=True, intercept_reserve=True, commit_intercept=True,
            intercept_choice=True, reserve_direction=True, along_intercept=True, finite_intercept=True,
            crossing_stop=True, crossing_stop_continuation=True, crossing_stop_shallow=True,
            crossing_stop_early=True, crossing_reference=True, far_crossing_reference=use_plane)
        lanes, _, _ = option.anticipate_crossing(lidar, state.flatten(1), actor, config, eligible, response)
        assert lanes.tolist() == ([] if use_plane else [0])
        if use_plane:
            assert option.far_crossing_refined.tolist() == [True]


def test_opening_crossing_stop_checks_four_scans_and_other_close_returns():
    import torch
    from instinctlab.control.escape_passage import clear_opening_crossing_at_stop
    from instinctlab.control.ics import mounted_mid360_rays_body

    rays = mounted_mid360_rays_body()
    centers = torch.tensor([[.90, value, 0.] for value in (.26, .32, .38, .44)]).expand(6, -1, -1).clone()
    centers[4] = centers[4].flip(0)
    centers[5, :, 0] = .73
    projection = centers @ rays.T
    discriminant = .34**2 - centers.square().sum(dim=2, keepdim=True) + projection.square()
    first = projection-discriminant.clamp_min(0.).sqrt()
    depth = torch.where((discriminant >= 0.) & (first > 0.), first, 3.)
    separate_return = (rays - torch.tensor([-1., 0., 0.])).square().sum(dim=1).argmin()
    depth[3, -1, separate_return] = .77
    lidar = torch.stack((depth, torch.ones_like(depth), torch.ones_like(depth)), dim=2).reshape(6, 12, 59, 360)
    histories = torch.zeros(6, 4, 13)
    histories[1, 0, 0] = .26
    histories[2, 0, 12] = .05
    response = torch.zeros(6, 4, 2), torch.zeros(6, 2)
    clear = clear_opening_crossing_at_stop(lidar, histories.flatten(1), rays, response, .4)
    assert clear.tolist() == [True, False, False, False, False, False]


def test_settled_crossing_requires_a_reversal_before_continuing_that_proof():
    import torch
    from instinctlab.control.escape_passage import clear_reversing_crossing_at_hover
    from instinctlab.control.ics import mounted_mid360_rays_body

    rays = mounted_mid360_rays_body()
    centers = torch.tensor([[.80, value, .05] for value in (-1.83, -1.888, -1.838, -1.780)]).expand(6, -1, -1).clone()
    centers[1:3, :, 1] = torch.tensor([-1.952, -1.895, -1.838, -1.780])
    centers[4, :, 0] = .65
    projection = centers @ rays.T
    discriminant = .34**2 - centers.square().sum(dim=2, keepdim=True) + projection.square()
    first = projection-discriminant.clamp_min(0.).sqrt()
    depth = torch.where((discriminant >= 0.) & (first > 0.), first, 3.)
    other = (rays - torch.tensor([-1., 0., 0.])).square().sum(dim=1).argmin()
    depth[5, -1, other] = .82
    scans = torch.stack((depth, torch.ones_like(depth), torch.ones_like(depth)), dim=2).reshape(6, 12, 59, 360)
    histories = torch.zeros(6, 4, 13)
    histories[3, -1, 6] = .01
    response = torch.zeros(6, 4, 2), torch.zeros(6, 2)
    previously_verified = torch.tensor([False, False, True, True, False, False])
    clear = clear_reversing_crossing_at_hover(scans, histories.flatten(1), rays, response,
                                              (0., 0., .05), .40, previously_verified)
    assert clear.tolist() == [True, False, True, False, False, False]


def test_settled_crossing_recognizes_a_turn_between_samples():
    import torch
    from instinctlab.control.escape_passage import clear_reversing_crossing_at_hover
    from instinctlab.control.ics import mounted_mid360_rays_body

    rays = mounted_mid360_rays_body()
    centers = torch.tensor([[.866, value, .10] for value in (-1.970, -2.028, -2.024, -1.966)])
    centers = centers.expand(6, -1, -1).clone()
    centers[1:3, :, 1] = torch.tensor([-2.028, -2.024, -1.966, -1.908])
    centers[3, :, 0] = .72
    projection = centers @ rays.T
    discriminant = .34**2-centers.square().sum(dim=2, keepdim=True)+projection.square()
    first = projection-discriminant.clamp_min(0.).sqrt()
    depth = torch.where((discriminant >= 0.) & (first > 0.), first, 3.)
    other = (rays-torch.tensor([-1., 0., 0.])).square().sum(dim=1).argmin()
    depth[5, -1, other] = .82
    scans = torch.stack((depth, torch.ones_like(depth), torch.ones_like(depth)), dim=2).reshape(6, 12, 59, 360)
    histories = torch.zeros(6, 4, 13)
    histories[4, -1, 6] = .01
    response = torch.zeros(6, 4, 2), torch.zeros(6, 2)
    previously_verified = torch.tensor([False, True, False, False, False, False])
    args = scans, histories.flatten(1), rays, response, (0., 0., .05), .40, previously_verified
    assert not clear_reversing_crossing_at_hover(*args).any()
    result = clear_reversing_crossing_at_hover(*args, ordinary_onset=torch.ones(6, dtype=torch.bool))
    assert result.tolist() == [True, False, False, False, False, False]


def test_static_stop_rejects_shifted_near_returns_at_the_same_clearance():
    import torch
    from instinctlab.control.escape_passage import clear_static_scan_at_stop
    from instinctlab.control.ics import mounted_mid360_rays_body

    rays = mounted_mid360_rays_body()
    wall = torch.where(rays[:, 0] > 0., 1.8/rays[:, 0], 3.).clamp(max=3.)
    centers = torch.tensor([[[1.57, value, 0.] for value in (-.24, -.16, -.08, 0.)]]).expand(2, -1, -1).clone()
    centers[0, :, 1] = 0.
    projection = centers @ rays.T
    discriminant = .34**2-centers.square().sum(dim=2, keepdim=True)+projection.square()
    depth = projection-discriminant.clamp_min(0.).sqrt()
    depths = torch.where((discriminant >= 0.) & (depth > 0.), torch.minimum(depth, wall), wall)
    scans = torch.stack((depths, torch.ones_like(depths), torch.ones_like(depths)), dim=2).reshape(2, 12, 59, 360)
    state = torch.zeros(2, 52)
    response = torch.zeros(2, 4, 2), torch.zeros(2, 2)
    assert clear_static_scan_at_stop(scans, state, rays, response).tolist() == [True, False]


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
    from training_workloads.unified_h1.command_replay import CRAZY_MODES, LANES, load_lane_streams, load_route_streams

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
    independent = generate(prefix, 725, 62000, independent_lanes=True)
    assert len({row['seed'] for row in independent['lanes']}) == LANES
    path.write_text(json.dumps(independent))
    layouts, lanes = load_lane_streams(file_pin(path))
    assert layouts[0] == layouts[1] and lanes[0] != lanes[1]


def test_long_generator_bounds_planned_position_without_reading_the_plant(tmp_path):
    import json
    from pathlib import Path
    from tools.r6_direct.generate_crazy_commands import generate
    from tools.r6_evidence import file_pin
    from training_workloads.unified_h1.command_replay import PLANNED_POSITION_LIMITS, load_route_streams

    documents = Path(__file__).resolve().parents[2]/'docs/experiments/instinct-continuous-command'
    prefix = file_pin(documents/'corner/RETIMED-PREFIX.json')
    bounded = generate(prefix, 725, 1800000, limit_planned_position=True)
    path = tmp_path/'commands.json'
    path.write_text(json.dumps(bounded))
    streams = load_route_streams(file_pin(path))
    assert len(streams) == 5
    assert bounded['generator']['planned_position_limits'] == PLANNED_POSITION_LIMITS

    unbounded = generate(prefix, 725, 1800000)
    unbounded['generator']['planned_position_limits'] = PLANNED_POSITION_LIMITS
    path.write_text(json.dumps(unbounded))
    with pytest.raises(ValueError, match='GENERATOR_POSITION_LIMIT'):
        load_route_streams(file_pin(path))


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


def test_motion_tube_can_use_neighbor_history_for_a_consistent_off_path_return():
    from dataclasses import replace
    import torch
    from instinctlab.control.ics import ICSConfig, attenuate_command, mounted_mid360_rays_body

    rays = mounted_mid360_rays_body()
    side = int((rays-torch.tensor([.109, .994, 0.])).square().sum(dim=1).argmin())
    ahead = int(rays[:, 0].argmax())
    velocity = torch.tensor([[.24, 0., 0.]]*4)
    base = ICSConfig(clearance_margin_m=.15, motion_tube_radius_m=.60, motion_tube_confirm_closure=True)
    beams = torch.zeros(4, 4, 3, len(rays))
    beams[:, :, 0] = 40.
    old_neighbors = [side-2, side-1, side+1, side+2]
    beams[:, 1:3, 0, old_neighbors] = .92
    beams[:, 1:3, 1:, old_neighbors] = 1.
    beams[:, -1, 0, side] = .92
    beams[:, -1, 1:, side] = 1.
    beams[1, 2, 0, old_neighbors] = .75
    beams[2, -1, 0, ahead] = .92
    beams[2, -1, 1:, ahead] = 1.
    beams[3, -1, 0, ahead] = .39
    beams[3, -1, 1:, ahead] = 1.
    scan = beams.reshape(4, 12, 59, 360)

    assert not attenuate_command(scan, rays, velocity, .1, base).v_final_b.any()
    test = replace(base, motion_tube_neighbor_history=True)
    issued = attenuate_command(scan, rays, velocity, .1, test, state_vec=torch.zeros(4,52))
    assert issued.v_final_b[:, 0].tolist() == pytest.approx([.24, 0., 0., 0.])
    assert issued.metrics['emergency'][:, 0].tolist() == [0., 0., 0., 1.]
    stamped = torch.zeros(4,52)
    stamped[0, 38] = .2
    stale = attenuate_command(scan, rays, velocity, .1, test, state_vec=stamped)
    assert not stale.v_final_b.any()


def test_confirmed_standoff_keeps_emergency_and_unconfirmed_closing_guards():
    import torch
    from instinctlab.control.ics import ICSConfig, attenuate_command, mounted_mid360_rays_body

    rays = mounted_mid360_rays_body()
    ahead = int(rays[:, 0].argmax())
    velocity = torch.tensor([[.24, 0., 0.]]*4+[[.35, 0., 0.]])
    beams = torch.zeros(5, 4, 3, len(rays))
    beams[:, :, 0] = 40.
    beams[:, :, 0, ahead] = .80
    beams[:, :, 1:, ahead] = 1.
    beams[2, 1:3, 0, ahead] = torch.tensor([.92, .86])
    beams[3, -1, 0, ahead] = .39
    state = torch.zeros(5, 52)
    state[1, 51] = .20
    config = ICSConfig(clearance_margin_m=.15, confirmed_standoff_m=.45)
    scan = beams.reshape(5, 12, 59, 360)
    issued = attenuate_command(scan, rays, velocity, .1, config, state_vec=state)
    assert issued.v_final_b[:, 0].tolist() == pytest.approx([.24, 0., 0., 0., 0.])
    assert issued.metrics['emergency'][:, 0].tolist() == [0., 0., 0., 1., 0.]


def test_clearance_uses_the_nearest_finite_wall_surface():
    import numpy as np
    from tools.r6_direct.replay_clearance import triangle_distance_squared

    triangle = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    queries = np.array([[.5,.5,1.], [1.5,1.5,0.], [-1.,-1.,0.]])
    assert triangle_distance_squared(queries, triangle).tolist() == pytest.approx([1., .5, 2.])


def test_dense_clearance_preserves_an_interior_physics_tick(tmp_path):
    import numpy as np
    from tools.r6_direct.replay_clearance import measured_clearance
    from training_workloads.unified_h1.command_replay import CLEARANCE_SHAPE, LANES
    from training_workloads.unified_h1.command_replay_audit import dense_clearance, verify_clearance

    samples = np.zeros((1, *CLEARANCE_SHAPE), dtype='<f4')
    samples[..., 2] = 1.
    samples[..., 6] = .2
    samples[0, 11, 0, 2] = .5
    samples[0, 11, 0, 6] = .23
    raw = tmp_path/'clearance.f32'
    samples.tofile(raw)
    initial = {'root_state_w': samples[0, 0, :, :3].tolist()}
    end = {'root_state_after_w': samples[0, -1, :, :3].tolist(), 'mover_position_w': [[0.]*3]*LANES,
           'mover_position_after_w': [[0.]*3]*LANES}
    verify_clearance(raw, initial, [end])
    result = measured_clearance(dense_clearance(raw, 1), 0, 1, np.empty((0, 3, 3)), crossing=False)
    assert result['elapsed_physics_tick'] == 11
    assert result['minimum_clearance_lower_bound_m'] == pytest.approx(.27)
    samples[0, 0, 0, 0] = .1
    samples.tofile(raw)
    with pytest.raises(ValueError, match='COMMAND_REPLAY_CLEARANCE_ENDPOINT'):
        verify_clearance(raw, initial, [end])


def test_frozen_opportunities_keep_missed_ticks_in_the_denominator():
    import numpy as np
    from tools.r6_direct.replay_clearance import frozen_opportunity

    commands = np.tile([.2, 0., 0.], (100, 1))
    measurements = commands.copy()
    measurements[14:30] = 0.
    result = frozen_opportunity(measurements, commands, first_flight_steps=30, start=0, end=100)
    assert result['scheduled_moving_steps'] == 100
    assert result['first_flight_moving_steps'] == 30
    assert result['met_fraction'] == pytest.approx(.14)
    assert result['measured_progress_m'] == pytest.approx(.28)


def test_frozen_stop_rejects_a_short_stationary_spell_and_a_reset():
    import numpy as np
    from tools.r6_direct.replay_clearance import frozen_stop

    commands = np.zeros((31, 3))
    commands[0, 0] = .3
    velocities = commands.copy()
    velocities[8, 0] = .1
    before = np.zeros_like(commands)
    after = np.cumsum(velocities*.1, axis=0)
    passed = frozen_stop(before, after, velocities, commands, 31, 1, 31)
    assert passed['stopped_after_s'] == pytest.approx(.9)
    assert passed['max_drift_m'] == pytest.approx(.04)
    failed = frozen_stop(before, after, velocities, commands, 15, 1, 31)
    assert not failed['first_flight_completed_window']
    assert failed['stopped_after_s'] is None


def test_hazard_recovery_tracks_the_latest_direction():
    import numpy as np
    from tools.r6_direct.replay_clearance import hazard_responses

    nominal = np.full(20, .6)
    nominal[2:5] = .1
    protected = nominal.copy()
    protected[3:5] = .6
    intent = np.zeros((20, 3))
    intent[5:, 0] = -.25
    observed = intent.copy()
    observed[5:8, 0] = .25
    result = hazard_responses(nominal, protected, intent, observed)
    assert result['risk_windows'] == 1
    event, = result['events']
    assert event['intervention_after_s'] == pytest.approx(.1)
    assert event['first_feasible_request_step'] == 5
    assert event['recovery_after_s'] == pytest.approx(.4)
    confirmed = hazard_responses(nominal, protected, intent, observed, sustained_opportunity=True)['events'][0]
    assert confirmed['recovery_onset_step'] == 8
    assert confirmed['recovery_confirmed_step'] == 12
    assert confirmed['recovery_after_s'] == pytest.approx(.8)

    nominal[7] = .1
    mover = np.zeros(20, dtype=bool)
    mover[2:5] = True
    visible = np.zeros(20, dtype=bool)
    visible[3] = True
    grouped = hazard_responses(nominal, protected, intent, observed, sustained_opportunity=True,
                               exposure=(mover, visible, np.full(20, .3)))
    assert grouped['risk_windows'] == 1
    assert grouped['events'][0]['first_clear_step'] == 8
    assert grouped['events'][0]['moving_hazard_steps'] == 3
    assert grouped['events'][0]['visible_moving_hazard_steps'] == 1


def test_mover_clock_preserves_three_decimal_places_at_thirty_minutes():
    from types import SimpleNamespace
    import torch
    from instinctlab.observations.moving_sphere import MovingSphere

    asset = SimpleNamespace(root_physx_view=SimpleNamespace(get_transforms=lambda: torch.zeros(1, 7)),
                            write_root_velocity_to_sim=lambda value: None)
    mover = MovingSphere(asset, torch.zeros(1, 3))
    mover.enabled = True
    mover.start = mover.velocity = torch.zeros(1, 3)
    mover.motion_ticks = torch.tensor([359999])
    mover.push(.005)
    assert mover.age.item() == pytest.approx(1800., abs=.0002)


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


def test_recovery_retreat_uses_the_existing_limiter_and_holds_briefly(monkeypatch):
    from types import SimpleNamespace
    import torch
    from instinctlab.control.escape_option import EscapeOption
    from instinctlab.control.escape_recovery import EscapeRecovery

    policy = {'lidar_grid': torch.empty(1, 1), 'state_vec': torch.zeros(1, 52)}
    policy['state_vec'][0, 45] = .26
    action = torch.tensor([[.8, .5, .5, .5]])
    option = EscapeRecovery(lanes=1, device='cpu', dtype=torch.float32, recenter=True, recovery_retreat=True)
    option.since_escape.fill_(30)
    monkeypatch.setattr(EscapeOption, 'apply', lambda self, policy, action: action.clone())
    monkeypatch.setattr(option.detector, 'retreat_direction', lambda *args, **kw: torch.tensor([[-.2, .98, 0.]]))

    def issued(lidar, state, action, cfg):
        reverse = action[:, 1] < .4
        final = torch.zeros_like(action[:, 1:])
        final[reverse, 0] = -.24
        return SimpleNamespace(v_final_b=final, metrics={'emergency': torch.zeros(len(action), 1)})

    monkeypatch.setattr(option, 'issued', issued)
    result = option.apply_recovery(policy, action, None)
    assert result[0].tolist() == pytest.approx([0., .2, .5, .5])
    assert option.retreating.item()
    assert not option.recovering.item()
    assert torch.equal(option.apply_recovery(policy, action, None), result)
    option.reset(torch.tensor([0]))
    assert not option.retreat_hold_steps.any()


def test_side_hover_prefers_an_admissible_retreat_only_on_zero_intent(monkeypatch):
    from types import SimpleNamespace
    import torch
    from instinctlab.control.escape_recovery import EscapeRecovery

    option = EscapeRecovery(lanes=3, device='cpu', dtype=torch.float32, recenter=True, side_hover_retreat=True)
    state = torch.zeros(3, 52)
    state[1,45] = .2
    retreat = torch.tensor([[0., 1., 0.], [0., 1., 0.], [1., 0., 0.]])
    tangent = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    monkeypatch.setattr(option, 'issued', lambda lidar, state, action, cfg: SimpleNamespace(
        v_final_b=action[:, 1:]-.5, metrics={'emergency': torch.zeros(len(action), 1)}))
    chosen = option.choose_side_hover_retreat(torch.empty(3, 1), state, retreat, tangent.clone(), None)
    assert torch.equal(chosen[0], retreat[0])
    assert torch.equal(chosen[1:], tangent[1:])


def test_clear_pass_keeps_the_intercept_and_cannot_use_stale_or_moving_history():
    import torch
    from instinctlab.control.escape_passage import clear_hover_passage
    from instinctlab.control.ics import mounted_mid360_rays_body
    from instinctlab.observations.moving_sphere import sphere_range

    rays = mounted_mid360_rays_body()
    lanes = 4
    directions = rays[None].expand(lanes, -1, -1)
    centres = torch.tensor([[1.28, -1.82, .10], [.31, -1.50, .10], [1.28, -1.82, .10], [1.28, -1.82, .10]])
    history = []
    for index in range(4):
        centres_frame = centres+torch.tensor([0., -.058*(3-index), 0.])
        distance = sphere_range(torch.zeros_like(centres), directions, centres_frame)
        valid = distance.isfinite()
        history.append(torch.stack((torch.where(valid, distance, 40.), valid, valid), dim=1))
    scan = torch.cat(history, dim=1).reshape(lanes, 12, 59, 360)
    state = torch.zeros(lanes, 4, 13)
    state[2, 1, 12] = .20
    state[3, 1, 9] = .10
    assert clear_hover_passage(scan, state.reshape(lanes, 52), rays).tolist() == [True, False, False, False]
    assert clear_hover_passage(scan, state.reshape(lanes, 52), rays, curved=True).tolist() == [True, False, False, False]


def test_recenter_rearms_one_second_after_credible_emergency():
    from types import SimpleNamespace
    import torch
    from instinctlab.control.escape_recovery import EscapeRecovery, RECENTER_REARM_STEPS
    from instinctlab.control.ics import ICSConfig

    option = EscapeRecovery(lanes=2, device='cpu', dtype=torch.float32, recenter=True, recenter_rearm=True)
    config = SimpleNamespace(ics=ICSConfig(clearance_margin_m=.15))
    lidar = torch.zeros(2, 12, 1, 1)
    lidar[:, -3] = .38
    lidar[:, -2] = 1.
    lidar[0, -1] = 1.
    option.update_recenter_rearm(lidar, config)
    assert option.rearm_steps.tolist() == [RECENTER_REARM_STEPS, 0]
    lidar[:, -3] = .42
    for remaining in range(RECENTER_REARM_STEPS-1, -1, -1):
        option.update_recenter_rearm(lidar, config)
        assert option.rearm_steps.tolist() == [remaining, 0]


def test_ranked_escape_only_switches_to_an_admissible_improvement(monkeypatch):
    import torch
    from types import SimpleNamespace
    from instinctlab.control import escape_passage
    from instinctlab.control.escape_recovery import EscapeRecovery

    option = EscapeRecovery(lanes=2, device='cpu', dtype=torch.float32, recenter=True,
                            recenter_rearm=True, rank_return=True)
    option.rearm_steps[0] = 2
    state = torch.zeros(2, 52)
    state[:, 45] = .26
    retreat = torch.tensor([[-.54, -.84, 0.]]*2)
    selected = torch.tensor([[.55, -.83, 0.]]*2)
    monkeypatch.setattr(escape_passage, 'approach_scores', lambda *args: (
        torch.tensor([True]), torch.tensor([[.14, .45, .47]])))
    monkeypatch.setattr(option, 'issued', lambda scan, history, action, config:
        SimpleNamespace(v_final_b=action[:, 1:]-.5))
    direction = option.choose_after_return(torch.zeros(2, 1), state, torch.arange(2), retreat, selected, None)
    assert direction[0].tolist() == [-1., 0., 0.]
    assert direction[1].tolist() == pytest.approx([.55, -.83, 0.])
    assert option.ranked.tolist() == [True, False]


def test_hover_path_stops_only_when_the_legacy_projection_loses_clearance(monkeypatch):
    import torch
    from types import SimpleNamespace
    from instinctlab.control import escape_passage
    from instinctlab.control.escape_option import DURATION_STEPS
    from instinctlab.control.escape_recovery import EscapeRecovery

    option = EscapeRecovery(lanes=2, device='cpu', dtype=torch.float32, recenter=True, clear_pass=True,
                            recenter_rearm=True, rank_return=True, hover_path=True)
    option.remaining.fill_(DURATION_STEPS-1)
    option.applied.fill_(True)
    monkeypatch.setattr(option.detector, 'retreat_direction', lambda *args, **kw: torch.tensor([[0., 1., 0.]]*2))
    monkeypatch.setattr(escape_passage, 'hover_escape_scores', lambda *args, **kwargs: (
        torch.tensor([True, True]), torch.tensor([[0., .66, .7, .8, .2], [.69, .68, .7, .7, .7]])))
    monkeypatch.setattr(option, 'issued', lambda scan, history, action, config:
        SimpleNamespace(v_final_b=action[:, 1:]-.5))
    selected = torch.tensor([[0., .8, .5, .5]]*2)
    option.choose_hover_path(torch.zeros(2, 1), torch.zeros(2, 52), selected, None)
    assert selected[0].tolist() == [0., .5, .5, .5]
    assert selected[1].tolist() == pytest.approx([0., .8, .5, .5])
    assert option.holding.tolist() == [True, False]
    assert option.applied.tolist() == [False, True]


def test_releasing_old_escape_requires_clear_fresh_scan_evidence():
    import torch
    from instinctlab.control.escape_passage import clear_to_end_escape

    lidar = torch.ones(4, 12, 59, 360)
    lidar[:, 0::3] *= 1.2
    lidar[1, 9, 30, 180] = .7
    lidar[2, 11] = 0.
    lidar[3, 9] = .95
    assert clear_to_end_escape(lidar, torch.zeros(4, 52)).tolist() == [True, False, False, False]


def test_turn_fit_requires_two_agreeing_intervals_after_reversal():
    import torch
    from instinctlab.control.escape_passage import _latest_turn

    elapsed = torch.tensor([[-.3, -.2, -.1, 0.]]*2)
    centers = torch.tensor([[[0., 2.15], [0., 2.20], [0., 2.16], [0., 2.11]],
                            [[0., 2.15], [0., 2.20], [0., 2.16], [0., 2.21]]])
    valid, velocity = _latest_turn(centers, elapsed)
    assert valid.tolist() == [True, False]
    assert velocity[0].tolist() == pytest.approx([0., -.47], abs=2.e-6)


def test_intercept_checks_contact_before_waiting_for_range_crop():
    from types import SimpleNamespace
    import torch
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.governor import GovernorConfig
    from instinctlab.control.ics import ICSConfig
    from instinctlab.observations.moving_sphere import sphere_range

    option = EscapeRecovery(lanes=4, device='cpu', dtype=torch.float32, recenter=False, intercept=True)
    directions = option.detector._full_rays[None].expand(4, -1, -1)
    center = torch.tensor([[.38, -1.50, .10], [-1.28, -1.50, .10], [.38, -1.50, .10], [.38, -1.50, .10]])
    history = []
    for index in range(4):
        movement = torch.tensor([[0., .58, 0.], [0., .58, 0.], [0., .58, 0.], [0., .30, 0.]])
        distance = sphere_range(torch.zeros_like(center), directions, center-(3-index)*.1*movement)
        valid = distance.isfinite()
        history.append(torch.stack((torch.where(valid, distance, 40.), valid, valid), dim=1))
    lidar = torch.cat(history, dim=1).reshape(4, 12, 59, 360)
    state = torch.zeros(4, 4, 13)
    state[2, 1, 12] = .20
    state[3, :, 10] = -.30
    action = torch.tensor([[0., .69, .50, .50]]*4)
    config = SimpleNamespace(governor=GovernorConfig(), ics=ICSConfig(
        clearance_margin_m=.15, motion_tube_radius_m=.60, motion_tube_confirm_closure=True))

    option.avoid_interception(lidar, state.reshape(4, 52), action, config)
    assert option.intercepted.tolist() == [True, False, False, False]
    assert action[0].tolist() == pytest.approx([0., .20, .50, .50], abs=2.e-5)
    assert action[1, 1].item() == pytest.approx(.69)


def test_displacement_is_time_aligned_during_a_command_reversal():
    import math
    import torch
    from instinctlab.control.escape_passage import advance_command_response

    offsets = torch.tensor([[[-.06, 0.], [-.04, 0.], [-.02, 0.], [0., 0.]]])
    coast = torch.tensor([[.2, 0.]])
    state = torch.zeros(1, 52)
    state[0, 35] = .2
    state[0, 48] = -.2
    earlier, latest = advance_command_response(offsets, coast, state)
    assert torch.allclose(earlier, offsets)
    assert torch.allclose(latest, coast)

    state[0, 35] = -.2
    earlier, latest = advance_command_response(earlier, latest, state)
    expected_distance = -.02+.35*.4*(1.-math.exp(-.1/.35))
    assert earlier[0, 2, 0].item() == pytest.approx(-expected_distance, abs=1.e-6)
    assert earlier[0, -1].tolist() == [0., 0.]
    assert 0. < latest[0, 0] < .2


def test_stopping_reserve_only_strengthens_withdrawal_that_is_already_underway():
    from types import SimpleNamespace
    import torch
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.governor import GovernorConfig
    from instinctlab.control.ics import ICSConfig
    from instinctlab.observations.moving_sphere import sphere_range

    option = EscapeRecovery(lanes=2, device='cpu', dtype=torch.float32, recenter=False,
                            intercept=True, align_intercept=True, intercept_reserve=True)
    rays = option.detector._full_rays[None].expand(2, -1, -1)
    center = torch.tensor([[.38, -1.62, .1], [1.28, -1.62, .1]])
    history = []
    for offset in (.174, .116, .058, 0.):
        distance = sphere_range(torch.zeros_like(center), rays, center-torch.tensor([0., offset, 0.]))
        valid = distance.isfinite()
        history.append(torch.stack((torch.where(valid, distance, 40.), valid, valid), dim=1))
    lidar = torch.cat(history, dim=1).reshape(2, 12, 59, 360)
    action = torch.tensor([[0., .30, .50, .50]]*2)
    config = SimpleNamespace(governor=GovernorConfig(), ics=ICSConfig(
        clearance_margin_m=.15, motion_tube_radius_m=.60, motion_tube_confirm_closure=True))

    option.avoid_interception(lidar, torch.zeros(2, 52), action, config)
    assert option.intercepted.tolist() == [True, False]
    assert action[0].tolist() == pytest.approx([0., .20, .50, .50], abs=2.e-5)
    assert action[1, 1].item() == pytest.approx(.30)


def test_command_handoff_ends_obsolete_escape_and_preserves_current_threat(monkeypatch):
    from types import SimpleNamespace
    import torch
    from instinctlab.control.escape_recovery import EscapeRecovery

    option = EscapeRecovery(lanes=3, device='cpu', dtype=torch.float32, recenter=True,
                            rank_return=True, command_handoff=True)
    option.remaining[:] = 12
    option.cooldown[:] = 10
    option.applied[:] = True
    option.retreat_basis[:, 0] = -1.
    state = torch.zeros(3, 52)
    state[:, 32] = .25
    state[:, 46] = .25
    scans = torch.ones(3, 12, 59, 360)
    scans[:, 0::3] = 1.2
    scans[1, 9, 30, 180] = .70
    actor = torch.tensor([[0., .5, .73, .5]]*3)
    held = torch.tensor([[0., .5, .20, .5]]*3)
    held[2] = actor[2]
    monkeypatch.setattr(option, 'issued', lambda scan, history, action, config: SimpleNamespace(
        v_final_b=action[:, 1:]-.5, metrics={'beta': torch.ones(len(action), 1),
                                           'emergency': torch.zeros(len(action), 1)}))

    option.handoff_changed_command(scans, state, held, actor, None)
    assert option.handed_off.tolist() == [True, False, False]
    assert option.cooldown.tolist() == [0, 10, 10]
    assert option.applied.tolist() == [False, True, True]
    assert torch.equal(held[0], actor[0])
    assert held[1, 2].item() == pytest.approx(.20)


def test_steady_handoff_requires_consecutive_proofs_in_the_current_episode(monkeypatch):
    from types import SimpleNamespace
    import torch
    from instinctlab.control import escape_passage
    from instinctlab.control.escape_recovery import EscapeRecovery

    option = EscapeRecovery(lanes=3, device='cpu', dtype=torch.float32, recenter=True,
                            rank_return=True, command_handoff=True, clear_pass=True, static_scan_stop=True,
                            intercept=True, align_intercept=True, commit_intercept=True, finite_edge=True,
                            retry_command_handoff=True, verify_composed_crossing=True, verify_recent_handoff=True,
                            verify_steady_handoff=True)
    option.applied[:] = True
    option.remaining[:] = 12
    state = torch.zeros(3, 4, 13)
    state[:, :, 6] = .25
    scans = torch.ones(3, 12, 59, 360)
    scans[:, 0::3] = 1.2
    actor = torch.tensor([[0., .73, .50, .50]]*3)
    retreat = torch.tensor([[0., .29, .50, .50]]*3)
    monkeypatch.setattr(escape_passage, 'clear_to_end_escape',
                        lambda scan, *args, **kwargs: torch.ones(len(scan), dtype=torch.bool))
    monkeypatch.setattr(escape_passage, 'moving_intercepts',
                        lambda scan, *args, **kwargs: ((scan[:, 9, 0, 0] < 1.).nonzero().flatten(),))
    monkeypatch.setattr(option, 'issued', lambda scan, history, action, config: SimpleNamespace(
        v_final_b=action[:, 1:]-.5, metrics={'beta': torch.ones(len(action), 1),
                                           'emergency': torch.zeros(len(action), 1)}))

    selected = retreat.clone()
    option.handoff_changed_command(scans, state.flatten(1), selected, actor, None)
    assert option.steady_handoff_witness.tolist() == [True, True, True]
    assert torch.equal(selected, retreat)

    option.reset(torch.tensor([2]))
    option.applied[2] = True
    option.remaining[2] = 12
    scans[1, 9, 0, 0] = .7
    option.handoff_changed_command(scans, state.flatten(1), selected, actor, None)
    assert option.steady_handoff_witness.tolist() == [True, False, True]
    assert option.steady_handoff_used.tolist() == [True, False, False]
    assert torch.equal(selected[0], actor[0]) and torch.equal(selected[1:], retreat[1:])

    scans[1, 9, 0, 0] = 1.2
    option.handoff_changed_command(scans, state.flatten(1), selected, actor, None)
    assert option.steady_handoff_used.tolist() == [False, False, True]
    assert torch.equal(selected[1], retreat[1]) and torch.equal(selected[2], actor[2])


def test_intercept_continues_only_during_four_fresh_unobserved_frames(monkeypatch):
    from types import SimpleNamespace
    import torch
    from instinctlab.control.escape_recovery import EscapeRecovery

    option = EscapeRecovery(lanes=3, device='cpu', dtype=torch.float32, recenter=False,
                            intercept=True, commit_intercept=True)
    option.intercept_steps.fill_(4)
    option.intercept_direction[:, 0] = -1.
    state = torch.zeros(3, 52)
    state[2, 0] = 1.  # Current ICS emergency in the mock, regardless of the previous path.
    lidar = torch.zeros(3, 12)
    selected = torch.tensor([[.8, .5, .5, .5]]*3)
    monkeypatch.setattr(option, 'issued', lambda scan, history, action, cfg: SimpleNamespace(
        v_final_b=action[:, 1:]-.5, metrics={'emergency': history[:, :1]}))
    option.continue_intercept(lidar, state, selected, None, torch.tensor([1]))
    assert option.intercept_steps.tolist() == [3, 0, 0]
    assert option.intercept_continued.tolist() == [True, False, False]
    assert selected[0, 1].item() == pytest.approx(.20)
    for _ in range(3):
        option.continue_intercept(lidar, state, selected, None, torch.tensor([], dtype=torch.long))
        assert option.intercept_continued[0]
    option.continue_intercept(lidar, state, selected, None, torch.tensor([], dtype=torch.long))
    assert not option.intercept_continued.any()


def test_recovery_retreat_takes_priority_only_while_ics_issues_it(monkeypatch):
    from types import SimpleNamespace
    import torch
    from instinctlab.control.escape_recovery import EscapeRecovery

    option = EscapeRecovery(lanes=4, device='cpu', dtype=torch.float32, recenter=False, retreat_priority=True)
    option.retreating[:] = torch.tensor([True, True, True, False])
    option.retreat_hold_direction[:, 0] = -1.
    selected = torch.tensor([[0., .20, .50, .50], [0., .50, .50, .50],
                            [0., .20, .50, .50], [0., .20, .50, .50]])
    state = torch.zeros(4, 52)
    state[2, 0] = 1.
    monkeypatch.setattr(option, 'issued', lambda scan, history, action, cfg: SimpleNamespace(
        v_final_b=action[:, 1:]-.5, metrics={'emergency': history[:, :1]}))
    assert option.usable_recovery_retreat(torch.zeros(4, 1), state, selected, None).tolist() == [0]


def test_intercept_protects_retreat_direction_without_hiding_a_blocked_retreat(monkeypatch):
    from types import SimpleNamespace
    import torch
    from instinctlab.control import escape_passage
    from instinctlab.control.escape_recovery import EscapeRecovery

    option = EscapeRecovery(lanes=3, device='cpu', dtype=torch.float32, recenter=False,
                            intercept=True, retreat_priority=True)
    option.retreating[:] = True
    option.retreat_hold_direction[:, 0] = -1.
    state = torch.zeros(3, 52)
    state[1, 0] = .1
    selected = torch.tensor([[0., .30, .50, .50]]*3)

    def issued(scan, history, action, cfg):
        velocity = action[:, 1:]-.5
        blocked = (history[:, 0] > .0) & torch.isclose(velocity[:, 0], velocity.new_tensor(-.20))
        return SimpleNamespace(v_final_b=velocity.masked_fill(blocked[:, None], 0.),
                               metrics={'emergency': torch.zeros(len(action), 1)})

    monkeypatch.setattr(option, 'issued', issued)
    monkeypatch.setattr(escape_passage, 'moving_intercepts', lambda *args, **kwargs: (
        torch.arange(3), torch.zeros(3, 2), torch.tensor([[0., .58]]*3), torch.zeros(3, 2), torch.full((3,), .34)))
    monkeypatch.setattr(escape_passage, 'intercept_scores', lambda pos, motion, coast, radius, velocity:
        torch.tensor([[.328], [.328], [.14]]) if velocity.shape[1] == 1 else torch.tensor([[.49, .51, .70]]*3))
    option.avoid_interception(torch.zeros(3, 1), state, selected, None)
    assert selected[0].tolist() == pytest.approx([0., .20, .50, .50])
    assert selected[1].tolist() == pytest.approx([0., .50, .50, .50])
    assert selected[2].tolist() == pytest.approx([0., .50, .50, .50])


def test_zero_input_stays_put_only_with_clearance_from_the_whole_pass_line():
    import torch
    from instinctlab.control.escape_recovery import EscapeRecovery

    option = EscapeRecovery(lanes=4, device='cpu', dtype=torch.float32, recenter=False,
                            clear_pass=True, rank_return=True, offset_stop=True)
    option.applied[:] = True
    option.remaining[:] = 12
    lidar = torch.ones(4, 12, 2, 2)
    lidar[:, 0::3] = 1.3
    lidar[2, 9, 0, 0] = .75
    state = torch.zeros(4, 52)
    state[3, 45] = .1
    selected = torch.tensor([[0., .8, .5, .5]]*4)
    position = torch.tensor([[1.3, -1.6], [.97, -1.6], [1.3, -1.6], [1.3, -1.6]])
    coast = torch.tensor([[.28, 0.]]*4)
    motion = torch.tensor([[0., .58]]*4)
    mask = option.stop_for_offset_pass(lidar, state, selected, torch.arange(4), position,
                                       motion, coast, torch.full((4,), .34))
    assert mask.tolist() == [True, False, False, False]
    assert option.offset_stopped.tolist() == mask.tolist()
    assert option.applied.tolist() == [False, True, True, True]
    assert selected[0].tolist() == pytest.approx([0., .50, .50, .50])


def test_settled_zero_onset_requires_two_clearance_fits(monkeypatch):
    import torch
    from instinctlab.control import escape_passage
    from instinctlab.control.escape_option import DURATION_STEPS
    from instinctlab.control.escape_recovery import EscapeRecovery

    option = EscapeRecovery(lanes=7, device='cpu', dtype=torch.float32, recenter=False,
                            clear_pass=True, rank_return=True, offset_stop=True, settled_offset_stop=True)
    option.applied[:] = True
    option.remaining[:] = DURATION_STEPS-1
    option.remaining[3] -= 1
    lidar = torch.ones(7, 12, 2, 2)
    lidar[:, 0::3] = 1.25
    lidar[6, 10] = 0.
    state = torch.zeros(7, 52)
    state[4, 9] = .1
    state[5, 25] = .05
    selected = torch.tensor([[0., .8, .5, .5]]*7)
    position = torch.tensor([[.90, -1.6], [.90, -1.6], [.75, -1.6]]+[[.90, -1.6]]*4)
    motion = torch.tensor([[0., .58]]*7)

    def confirm(scans, history, rays, *, curved, require_curvature):
        assert curved and require_curvature and len(scans) == len(history) == 2
        return torch.tensor([True, False])

    monkeypatch.setattr(escape_passage, 'clear_hover_passage', confirm)
    cancel = option.stop_for_offset_pass(lidar, state, selected, torch.arange(7), position,
                                         motion, torch.zeros(7, 2), torch.full((7,), .34))
    assert cancel.tolist() == [True, False, False, False, False, False, False]
    assert selected[0].tolist() == pytest.approx([0., .50, .50, .50])


@pytest.mark.parametrize('use_raw_speed', [False, True])
def test_tangent_preserves_diagonal_progress_and_keeps_the_frontal_wall_stop(use_raw_speed):
    import math

    import torch
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.ics import ICSConfig
    from instinctlab.control.pipeline import ControlCoreConfig

    guard = EscapeRecovery(lanes=2, device='cpu', dtype=torch.float32, recenter=False, wall_tangent=True,
                           tangent_magnitude=use_raw_speed)
    direction = guard.detector._full_rays[:, 1]
    hit = direction >= .78/40.
    depth = torch.where(hit, .78/direction.clamp_min(.01), 40.)
    scan = torch.stack((depth, hit, hit)).reshape(3, 59, 360)
    lidars = scan.repeat(2, 4, 1, 1)
    states = torch.zeros(2, 4, 13)
    states[:, :, 5] = -1.
    states[0, :, 6:9] = torch.tensor([.15, .19, 0.])
    states[1, :, 6:9] = torch.tensor([0., .24, 0.])
    states = states.reshape(2, 52)
    config = ControlCoreConfig(ics=ICSConfig(clearance_margin_m=.15, motion_tube_radius_m=.60,
                                            motion_tube_confirm_closure=True))
    actions = torch.tensor([[1., .5, .5, .5]]*2)
    assert guard.issued(lidars, states, actions, config).v_final_b.tolist() == [[0.,0.,0.], [0.,0.,0.]]
    guard.retain_wall_tangent(lidars, states, actions, config)
    result = guard.issued(lidars, states, actions, config)
    assert guard.tangent_used.tolist() == [True, False]
    axial_speed = math.hypot(.15, .19) if use_raw_speed else .15
    assert result.v_final_b[0].tolist() == pytest.approx([axial_speed, 0., 0.], abs=1.e-6)
    assert result.v_final_b[1].tolist() == [0.,0.,0.]


def test_narrow_tube_needs_clear_axis_and_return_support_on_both_sides():
    from dataclasses import replace

    import torch
    from instinctlab.control.ics import ICSConfig, attenuate_command, mounted_mid360_rays_body

    rays = mounted_mid360_rays_body()
    forward = rays[:, 0] > .0
    near = .80/rays[:, 0].clamp_min(.001)
    far = 2.25/rays[:, 0].clamp_min(.001)
    jamb = near*rays[:, 1] >= .42
    side = torch.where(jamb, near, far)
    unobserved = torch.where(jamb, near, torch.inf)
    depths = torch.stack((side, near, unobserved, side)).masked_fill(~forward, torch.inf)
    valid = depths <= 40.
    scans = torch.stack((depths.clamp_max(40.), valid, valid), dim=1).reshape(4, 3, 59, 360).repeat(1, 4, 1, 1)
    state = torch.zeros(4, 4, 13)
    state[:, :, 5] = -1.
    state[3, 1, 12] = .05
    command = torch.tensor([[.24,0.,0.]]*4)
    original = ICSConfig(clearance_margin_m=.15, motion_tube_radius_m=.60, motion_tube_confirm_closure=True)
    base = attenuate_command(scans,rays,command,.1,original,state_vec=state.reshape(4, 52))
    candidate = attenuate_command(scans,rays,command,.1,replace(original,confirmed_tube_radius_m=.41),
                                  state_vec=state.reshape(4, 52))
    assert base.metrics['beta'].flatten().tolist() == [0.]*4
    assert candidate.metrics['beta'].flatten().tolist() == [1.,0.,0.,0.]


def test_inner_tube_aligns_small_turns_using_only_imu_and_command_history():
    import torch
    from instinctlab.control.ics import ICSConfig, attenuate_command, mounted_mid360_rays_body

    rays = mounted_mid360_rays_body()
    angle = torch.tensor([.084, .056, .028, 0.])
    heading = torch.stack((angle.cos(), torch.zeros_like(angle), -angle.sin()), dim=1)
    gravity = torch.stack((-angle.sin(), torch.zeros_like(angle), -angle.cos()), dim=1)
    forward = heading@rays.T
    side = (.46/rays[:, 1].clamp_min(.001)).masked_fill(rays[:, 1] <= 0., torch.inf)
    far = (2.25/forward.clamp_min(.001)).masked_fill(forward <= 0., torch.inf)
    depth = torch.minimum(side, far)
    frontal = (.80/forward.clamp_min(.001)).masked_fill(forward <= 0., torch.inf)
    ranges = torch.stack((depth, frontal, depth, depth, depth))
    valid = ranges <= 40.
    scans = torch.stack((ranges.clamp_max(40.), valid, valid), dim=2).reshape(5,12,59,360)
    state = torch.zeros(5,4,13)
    state[:, :, 1] = .28
    state[:, :, 3:6] = gravity
    state[:, :, 6:9] = .24*heading
    state[2, :, 1] = .45
    state[3, 1, 12] = .05
    state[4, 1, 7] = -.10
    command = state[:, -1, 6:9]
    config = ICSConfig(clearance_margin_m=.15, motion_tube_radius_m=.60, motion_tube_confirm_closure=True,
                       confirmed_tube_radius_m=.41, confirmed_tube_align_rotation=True)
    result = attenuate_command(scans, rays, command, .1, config, state_vec=state.reshape(5,52))
    assert result.metrics['beta'].flatten().tolist() == [1.,0.,0.,0.,0.]


def test_verified_resume_requires_two_clear_predictions_for_the_latest_request(monkeypatch):
    import torch
    from instinctlab.control import escape_passage
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.pipeline import ControlCoreConfig

    guard = EscapeRecovery(lanes=3, device='cpu', dtype=torch.float32, recenter=False, rank_return=True,
                           verified_resume=True)
    guard.applied[:] = True
    guard.braking[:] = True
    guard.remaining[:] = 17
    lidar = torch.ones(3,12,59,360)
    lidar[:, 0::3] = 2.
    state = torch.zeros(3,4,13)
    state[:, :, 5] = -1.
    state[:, :, 6] = .20
    command = torch.tensor([[1., .5, .5, .5]]*3)
    selected = torch.tensor([[0., .5, .5, .5]]*3)
    positions = torch.tensor([[0.,1.2],[0.,1.2],[1.2,1.2]])
    zeros = torch.zeros_like(positions)
    tracks = (torch.arange(3), positions, zeros, zeros, torch.full((3,), .34))
    config = ControlCoreConfig()
    monkeypatch.setattr(escape_passage, 'clear_to_end_escape',
                        lambda lidar, state, **kwargs: torch.zeros(len(lidar),dtype=torch.bool))

    guard.resume_latest_request(lidar, state.reshape(3,52), selected, command, config, guard.applied, tracks)
    assert guard.resumed.tolist() == [False, False, False]
    positions[1] = torch.tensor([.3,0.])
    state[2, -1, 6:9] = torch.tensor([0.,.20,0.])
    guard.resume_latest_request(lidar, state.reshape(3,52), selected, command, config, guard.applied, tracks)
    assert guard.resumed.tolist() == [True, False, False]
    assert guard.applied.tolist() == [False, True, True]
    assert guard.braking.tolist() == [False, True, True]
    assert torch.equal(selected[0], command[0])


def test_distant_stop_release_still_needs_reliable_range_and_no_closing_seed():
    import torch
    from instinctlab.control.escape_passage import clear_to_end_escape

    lidar = torch.zeros(3,12,59,360)
    lidar[:, 1::3, :, :213] = 1.
    lidar[:, 2::3, :, :213] = 1.
    lidar[:, 0::3] = 1.46
    lidar[1, 0::3] = .90
    lidar[2, 0:9:3] = 1.60
    state = torch.zeros(3,52)
    assert not bool(clear_to_end_escape(lidar,state).any())
    assert clear_to_end_escape(lidar,state,allow_distant_returns=True).tolist() == [True,False,False]


def test_settled_turn_fallback_requires_the_independent_surface_margin(monkeypatch):
    import torch
    from instinctlab.control import escape_passage

    center = torch.tensor([[1.90,.90],[1.87,.958],[1.881,.911],[1.894,.856]])
    paths = torch.stack((center, center-torch.tensor([1.8,0.])))
    elapsed = torch.tensor([[-.30,-.20,-.10,0.]]*2)
    shapes = [(point, torch.full((2,),.34), torch.ones(2,dtype=torch.bool))
              for point in paths.unbind(dim=1)]
    monkeypatch.setattr(escape_passage, 'track_nearby_return',
                        lambda *args, **kwargs:(torch.ones(2,dtype=torch.bool),paths,elapsed,shapes))
    baseline = escape_passage.clear_hover_passage(None,None,None,curved=True,require_curvature=True)
    revised = escape_passage.clear_hover_passage(None,None,None,curved=True,require_curvature=True,turn_fallback=True)
    assert baseline.tolist() == [False,False]
    assert revised.tolist() == [True,False]


@pytest.mark.parametrize('use_early_diagonal',[False,True])
def test_fresh_intercept_can_keep_the_prior_choice_or_use_a_diagonal(monkeypatch, use_early_diagonal):
    import torch
    from instinctlab.control import escape_passage
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.pipeline import ControlCoreConfig

    guard = EscapeRecovery(lanes=2, device='cpu', dtype=torch.float32, recenter=False, intercept=True,
                           commit_intercept=True, intercept_choice=True, early_diagonal=use_early_diagonal)
    prior = torch.tensor([-.8,.6,0.])
    guard.intercept_direction[0] = prior
    guard.intercept_continued[0] = True
    lidar = torch.ones(2,12,59,360)
    lidar[:, 0::3] = 3.
    states = torch.zeros(2,52)
    states[:, 45] = .24
    selected = torch.tensor([[1.,.5,.5,.5]]*2)
    positions = torch.ones(2,2)
    motion = torch.tensor([[0.,.58]]*2)
    tracks = (torch.arange(2), positions, motion, torch.zeros_like(positions), torch.full((2,),.34))
    monkeypatch.setattr(escape_passage, 'moving_intercepts', lambda *args, **kwargs: tracks)

    def closest(pos, velocity, coast, radius, candidates):
        if candidates.shape[1] == 1:
            return torch.full((2,1),.05)
        return torch.tensor([[.09,.437,.25,.454,.61,.1],[.09,.25,.22,.64,.297,.1]])

    monkeypatch.setattr(escape_passage,'intercept_scores',closest)
    guard.avoid_interception(lidar,states,selected,ControlCoreConfig())
    assert guard.intercepted.tolist() == [True,True]
    side = torch.tensor([-1.,.60,0.])
    expected = side/side.norm() if use_early_diagonal else prior
    assert torch.allclose(selected[0,1:], .5+.30*expected)
    assert selected[1,1] < .5 and selected[1,2] > .5


def test_reserve_choice_is_reused_only_on_the_next_fresh_exposure(monkeypatch):
    import torch
    from instinctlab.control import escape_passage
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.pipeline import ControlCoreConfig

    guard = EscapeRecovery(lanes=2, device='cpu', dtype=torch.float32, recenter=False,
                           intercept=True, intercept_reserve=True, commit_intercept=True,
                           intercept_choice=True, early_diagonal=True, reserve_direction=True)
    scans = torch.ones(2, 12, 59, 360)
    scans[:, 0::3] = 3.
    states = torch.zeros(2, 52)
    states[:, 45] = -.16
    actor = torch.tensor([[1., .5, .5, .5]]*2)
    lanes = torch.arange(2)
    base_score = .37

    def moving(*args, **kwargs):
        n = len(lanes)
        return lanes, torch.zeros(n, 2), torch.tensor([[0., .58]]*n), torch.zeros(n, 2), torch.full((n,), .34)

    def closest(pos, velocity, coast, radius, candidates):
        if candidates.shape[1] == 1:
            return torch.where(candidates.norm(dim=-1) > .01, base_score, .05)
        scores = [.05, .29, .19, .50, .70, .71]
        if base_score < .36:
            scores[3:] = [.61, .61, .62]
        return candidates.new_tensor(scores).expand(len(pos), -1)

    monkeypatch.setattr(escape_passage, 'moving_intercepts', moving)
    monkeypatch.setattr(escape_passage, 'intercept_scores', closest)
    first = actor.clone()
    guard.avoid_interception(scans, states, first, ControlCoreConfig())
    assert (first[:, 1] < .5).all()
    assert guard.reserve_direction_seen.tolist() == [True, True]
    assert guard.intercept_steps.tolist() == [0, 0]

    lanes = torch.tensor([0])
    base_score = .30
    second = actor.clone()
    guard.avoid_interception(scans, states, second, ControlCoreConfig())
    assert torch.allclose(second[0], first[0])
    assert torch.equal(second[1], actor[1])
    assert guard.intercept_steps.tolist() == [4, 0]
    assert guard.reserve_direction_seen.tolist() == [False, False]


@pytest.mark.parametrize('recheck', [False, True])
def test_zero_release_rechecks_only_while_the_same_stop_continues(monkeypatch, recheck):
    import torch
    from instinctlab.control import escape_passage
    from instinctlab.control.escape_recovery import DURATION_STEPS, EscapeRecovery

    guard = EscapeRecovery(lanes=2, device='cpu', dtype=torch.float32, recenter=False,
                           clear_pass=True, rank_return=True, stop_release=True, zero_recheck=recheck)
    guard.applied[:] = True
    guard.remaining[:] = DURATION_STEPS-2
    lidar = torch.zeros(2, 1)
    states = torch.zeros(2, 52)
    states[:, 32] = .20
    escape = torch.tensor([[0., .8, .5, .5]]*2)
    selected = escape.clone()
    proof_available = False
    monkeypatch.setattr(escape_passage, 'clear_to_end_escape',
                        lambda scan, *args, **kwargs: torch.full((len(scan),), proof_available))

    guard.release_older_escape(lidar, states, selected)
    states[:, 32] = 0.
    for _ in range(8):
        guard.release_older_escape(lidar, states, selected)
    states[1, 45] = .20
    proof_available = True
    guard.release_older_escape(lidar, states, selected)
    assert guard.released.tolist() == [recheck, False]
    assert torch.equal(selected[1], escape[1])
    assert torch.equal(selected[0], torch.tensor([0., .5, .5, .5]) if recheck else escape[0])


def test_reverse_diagonal_requires_imminent_contact_and_no_standard_choice(monkeypatch):
    import torch
    from instinctlab.control import escape_passage
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.pipeline import ControlCoreConfig

    guard = EscapeRecovery(lanes=3, device='cpu', dtype=torch.float32, recenter=False,
                           intercept=True, commit_intercept=True, intercept_choice=True,
                           early_diagonal=True, reverse_diagonal=True)
    scans = torch.ones(3, 12, 59, 360)
    scans[:, 0::3] = 3.
    states = torch.zeros(3, 52)
    states[:, 45] = .20
    actor = torch.tensor([[1., .5, .5, .5]]*3)
    tracks = (torch.arange(3), torch.zeros(3, 2), torch.tensor([[0., .58]]*3),
              torch.zeros(3, 2), torch.full((3,), .34))
    monkeypatch.setattr(escape_passage, 'moving_intercepts', lambda *args, **kwargs: tracks)

    def closest(pos, velocity, coast, radius, candidates):
        if candidates.shape[1] == 1:
            return torch.tensor([[.10], [.10], [.22]])
        if candidates.shape[1] == 6:
            return torch.tensor([[.10, .25, .05, .18, .27, .12],
                                 [.10, .25, .05, .18, .50, .12],
                                 [.22, .25, .23, .22, .27, .21]])
        assert candidates.shape[:2] == (1, 2)
        return torch.tensor([[.62, .05]])

    monkeypatch.setattr(escape_passage, 'intercept_scores', closest)
    selected = actor.clone()
    guard.avoid_interception(scans, states, selected, ControlCoreConfig())
    assert guard.intercepted.tolist() == [True, True, False]
    assert selected[0, 1] < .5 and selected[0, 2] < .5
    assert selected[1, 1] < .5 and selected[1, 2] > .5
    assert torch.equal(selected[2], actor[2])


@pytest.mark.parametrize('finite', [False, True])
def test_along_intercept_requires_zero_intent_and_a_measured_clearance_gain(monkeypatch, finite):
    import torch
    from instinctlab.control import escape_passage
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.pipeline import ControlCoreConfig

    guard = EscapeRecovery(lanes=3, device='cpu', dtype=torch.float32, recenter=False, intercept=True,
                           intercept_reserve=True, commit_intercept=True, intercept_choice=True,
                           early_diagonal=True, reserve_direction=True, along_intercept=True, finite_intercept=finite)
    scans = torch.ones(3, 12, 59, 360)
    scans[:, 0::3] = 3.
    states = torch.zeros(3, 52)
    states[2, 45] = .24
    selected = torch.tensor([[0., .74, .5, .5]]*3)
    tracks = (torch.arange(3), torch.zeros(3, 2), torch.tensor([[0., .58]]*3),
              torch.zeros(3, 2), torch.full((3,), .34))
    guard.intercept_direction[1, 1] = 1.
    guard.reserve_direction_seen[1] = True
    monkeypatch.setattr(escape_passage, 'moving_intercepts', lambda *args, **kwargs: tracks)

    def closest(pos, velocity, coast, radius, candidates, *, hold_s=None):
        if hold_s is not None:
            assert hold_s == .60
            return candidates.new_tensor([[.32, .44], [.32, .44]])
        if candidates.shape[1] == 1:
            return torch.where(candidates.norm(dim=2) > .01, .40, .10)
        if candidates.shape[1] == 6:
            return candidates.new_tensor([.10, .50, .85, .05, .60, .80]).expand(len(pos), -1)
        return candidates.new_tensor([[.85, .70], [.85, .70]])

    def observed(scan, rays, coast, candidates):
        assert len(scan) == 2
        return candidates.new_tensor([[.383, .80], [.55, .80]])

    monkeypatch.setattr(escape_passage, 'intercept_scores', closest)
    monkeypatch.setattr(escape_passage, 'observed_path_clearance', observed)
    guard.avoid_interception(scans, states, selected, ControlCoreConfig())
    torch.testing.assert_close(selected[0, 1:], torch.tensor([.5, .7999, .5]))
    torch.testing.assert_close(selected[1, 1:], torch.tensor([.5, .7999, .5] if finite else [.80, .5, .5]))
    torch.testing.assert_close(selected[2, 1:], torch.tensor([.80, .5, .5]))
    assert guard.intercept_speed.tolist() == pytest.approx([.2999, .2999 if finite else .30, .30])
    assert guard.intercept_steps.tolist() == ([4, 4, 0] if finite else [0, 0, 0])
    if finite:
        monkeypatch.setattr(escape_passage, 'moving_intercepts',
                            lambda *args, **kwargs: (torch.empty(0, dtype=torch.long), None, None, None, None))
        braking = torch.tensor([[0., .5, .5, .5]]*3)
        guard.avoid_interception(scans, states, braking, ControlCoreConfig())
        torch.testing.assert_close(braking[:2, 1:], torch.tensor([[.5, .7999, .5]]*2))
        assert guard.intercept_steps.tolist() == [3, 3, 0]


def test_opening_steering_requires_a_single_repeated_open_flank():
    import torch
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.escape_tangent import wall_opening_direction

    option = EscapeRecovery(lanes=3, device='cpu', dtype=torch.float32, recenter=False)
    rays = option.detector._full_rays
    forwards, sides, _ = rays.unbind(dim=1)
    positive_open = sides*1.4/forwards.clamp_min(.20) > .70
    negative_open = sides*1.4/forwards.clamp_min(.20) < -.70
    distant = torch.stack((positive_open, positive_open | negative_open, positive_open))
    depth = torch.where(distant, 2.2, 1.4)/forwards.clamp_min(.20)
    scans = torch.ones(3, 12, 59, 360)
    scans[:, 0::3] = depth.reshape(3, 1, 59, 360)
    scans[2, 7].flatten()[positive_open] = 0.
    state = torch.zeros(3, 4, 13)
    state[:, :, 5] = -1.
    state[:, :, 6] = .28

    confirmed, heading, opening = wall_opening_direction(scans, state.reshape(3, 52), rays)
    assert confirmed.tolist() == [True, False, False]
    assert torch.equal(heading[0], torch.tensor([1., 0., 0.]))
    assert torch.equal(opening[0], torch.tensor([0., 1., 0.]))


def test_finite_edge_advances_only_toward_an_observed_flank(monkeypatch):
    import torch
    from instinctlab.control import escape_tangent
    from instinctlab.control.escape_recovery import EscapeRecovery
    from instinctlab.control.ics import ICSConfig
    from instinctlab.control.pipeline import ControlCoreConfig

    guard = EscapeRecovery(lanes=3, device='cpu', dtype=torch.float32, recenter=False,
                           finite_edge=True, edge_continuation=True)
    rays = guard.detector._full_rays
    forward = rays[:, 0] > 0.
    near = .80/rays[:, 0].clamp_min(.001)
    far = 2.25/rays[:, 0].clamp_min(.001)
    jamb = near*rays[:, 1] <= -.08
    side = torch.where(jamb, near, far)
    missing = torch.where(jamb, near, torch.inf)
    ranges = torch.stack((side, near, missing)).masked_fill(~forward, torch.inf)
    valid = ranges <= 40.
    lidar = torch.stack((ranges.clamp_max(40.), valid, valid), dim=1).reshape(3, 3, 59, 360).repeat(1, 4, 1, 1)
    state = torch.zeros(3, 4, 13)
    state[:, :, 5] = -1.
    state[:, :, 6] = .25
    state = state.reshape(3, 52)
    config = ControlCoreConfig(ics=ICSConfig(clearance_margin_m=.15, motion_tube_radius_m=.60,
        motion_tube_confirm_closure=True, confirmed_tube_radius_m=.41))
    action = torch.tensor([[1., .5, .5, .5]]*3)
    guard.progress_around_edge(lidar, state, action, config)
    actual = guard.issued(lidar, state, action, config).v_final_b
    assert guard.detour_used.tolist() == [True, False, False]
    assert actual[0, 0] >= .15 and actual[0, 1] > 0.
    assert actual[1:].tolist() == [[0.,0.,0.], [0.,0.,0.]]
    targets = escape_tangent.finite_edge_targets

    def scanned_without_flank(*args):
        trial, support = targets(*args)
        return trial, torch.zeros_like(support)

    monkeypatch.setattr(escape_tangent, 'finite_edge_targets', scanned_without_flank)
    neutral = torch.tensor([[1., .5, .5, .5]]*3)
    guard.progress_around_edge(lidar, state, neutral, config)
    assert guard.edge_continued.tolist() == [True, False, False]
    assert guard.edge_steps.tolist() == [8,0,0]
    state[0, 45] = -.25
    guard.progress_around_edge(lidar, state, neutral, config)
    assert not guard.edge_continued.any()
    assert guard.edge_steps.tolist() == [0,0,0]


def test_crossing_phase_is_checked_before_a_lanes_early_reset(tmp_path):
    import json
    from training_workloads.unified_h1.command_replay_audit import verify_pulse_and_crossing
    from training_workloads.unified_h1.robustness import lane_role

    mass = 1.181517923
    block = {'task': 'pulse_hold'}
    force = [[axis*mass*.5 for axis in lane_role(block, lane)[1]] for lane in range(32)]
    pulse = [dict(block_tick=1000+i, sim_tick=1002+i, force_n=force, mass_kg=[mass]*32) for i in range(40)]
    path = tmp_path/'pulse.json'
    path.write_text(json.dumps(pulse))
    initial = dict(departure_delay_s=[9.-2.2/.58]+[0.]*31, path_reference_w=[[0.,0.,0.]]*32,
                   mover_start_w=[[0.,2.2,0.]]*32)
    rows = [dict(physics_tick=2+20*i, reset=[i == 80]+[False]*31,
                 mover_position_w=[[0.,.58 if i == 80 else 2.2,0.]]*32) for i in range(100)]
    args = (path, initial, rows, ['street_gap']+['room_corner']*31, [9000], .5)
    verify_pulse_and_crossing(*args)
    rows[80]['mover_position_w'][0] = [0., 2.2, 0.]
    with pytest.raises(ValueError, match='CROSSING_READBACK'):
        verify_pulse_and_crossing(*args)


def test_long_opportunities_match_each_lanes_frozen_commands():
    from training_workloads.unified_h1.command_replay import verify_long_opportunities

    layouts = ['room_corner', 'room_corner']
    streams = [dict(duration_ms=4000, events=[dict(t_ms=0, velocity_w_m_s=[.2,0.,0.]),
                                            dict(t_ms=release, velocity_w_m_s=[0.,0.,0.])])
               for release in (1000, 1300)]
    plan = dict(windows=[dict(id='early', layout='room_corner', start_step=0, end_step=8),
                        dict(id='late', layout='room_corner', lane=1, start_step=10, end_step=12)],
                stop_windows=[dict(id=f'stop-{lane}', lane=lane, start_step=start, end_step=start+20)
                              for lane,start in enumerate((10,13))])
    verify_long_opportunities(plan, layouts, streams)

    plan['windows'][1]['lane'] = 0
    with pytest.raises(ValueError, match='LONG_PROGRESS_WITHOUT_INPUT'):
        verify_long_opportunities(plan, layouts, streams)

    plan['windows'][1]['lane'] = 1
    plan['stop_windows'][1]['start_step'] = 12
    with pytest.raises(ValueError, match='LONG_STOP_WITHOUT_RELEASE'):
        verify_long_opportunities(plan, layouts, streams)


def test_long_gates_fail_lost_exposure_and_unrecovered_latest_command():
    from copy import deepcopy
    from tools.r6_direct.replay_clearance import long_gate_results

    gate = dict(minimum_clearance_m=.15, maximum_tracking_rmse_m_s=.15, minimum_progress_fraction=.6,
                maximum_slow_feasible_s=2., maximum_stop_s=1., maximum_stop_drift_m=.2,
                minimum_hazard_events=1, maximum_intervention_s=.5, minimum_recovery_opportunities=1,
                maximum_recovery_s=1.5)
    flight = dict(layout='street_gap', elapsed_s=1800., initial_termination_reason=None,
                  first_reset_step=None, first_contact_step=None, route_progress={})
    measure = dict(lane=0, physics_tick_clearance=dict(samples=360001, minimum_clearance_lower_bound_m=.2),
                   externally_scheduled_progress={'go': dict(met_fraction=.8, measured_progress_m=.4,
                                                        requested_progress_m=.5, observed_only_rmse_m_s=.09)},
                   tracking_in_feasible_moving_windows=dict(rmse_m_s=.08, fraction_at_least_60_percent_of_intent=.75,
                                                           longest_slow_feasible_window_s=1.2),
                   externally_scheduled_stops={'stop': dict(first_flight_completed_window=True,
                                                             stopped_after_s=.8, max_drift_m=.16)},
                   hazard_response_preview=dict(events=[dict(first_feasible_request_step=70,
                                                             intervention_after_s=.2, recovery_after_s=1.2,
                                                             visible_moving_hazard_steps=6)]))
    logs = [dict(deepcopy(measure), lane=i) for i in range(3)]
    lanes = [dict(flight) for i in range(3)]
    assert long_gate_results(logs, lanes, {'street_gap': gate})['passed']
    assert not long_gate_results(logs[:2], lanes[:2], {'street_gap': gate})['passed']
    lanes[0].update(elapsed_s=300., first_contact_step=2999, first_reset_step=2999)
    logs[1]['hazard_response_preview']['events'][0]['recovery_after_s'] = None
    results = long_gate_results(logs, lanes, {'street_gap': gate})
    assert not results['passed']
    assert results['by_layout']['street_gap']['lanes'][0]['failed'] == ['uninterrupted']
    assert results['by_layout']['street_gap']['lanes'][1]['failed'] == ['latest_command_recovery']
    logs[2]['hazard_response_preview']['events'][0]['visible_moving_hazard_steps'] = 0
    assert long_gate_results(logs, lanes, {'street_gap': gate})['by_layout']['street_gap']['lanes'][2]['failed'] == [
        'intervention']


def test_safe_progress_can_brake_before_a_sustained_request_would_reach_a_wall():
    import numpy as np
    from tools.r6_direct.replay_clearance import projected_clearance

    wall = np.array([[[.63,-2.,0.], [.63,2.,0.], [.63,-2.,4.]],
                     [[.63,2.,4.], [.63,2.,0.], [.63,-2.,4.]]])
    initial = np.tile([0.,0.,2.], (30, 1))
    velocity = np.zeros((30, 3))
    target = np.tile([.28,0.,0.], (30, 1))
    held, _ = projected_clearance(initial, velocity, target, None, wall, .21)
    braking, _ = projected_clearance(initial, velocity, target, None, wall, .21, hold_s=.6)
    assert held[0] < .15 < braking[0]
    assert np.isnan(braking[-1])
