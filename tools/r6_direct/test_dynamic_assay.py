from types import SimpleNamespace

import torch
import pytest


def test_clearance_check_override_is_limited_to_frozen_origin():
    from instinctlab.control.escape_recovery import FALLBACK_25_VERSION, REAR_CLEARANCE_25_VERSION
    from training_workloads.unified_h1.dynamic_assay import U1792_STUDY_PREFIX, approved_execution_override
    from training_workloads.unified_h1.robustness_training_plan import U1792_CHECKPOINT_SHA

    run_id = U1792_STUDY_PREFIX+'CHECK-CLEARANCE-FRESH-02'
    exact = (run_id, 'CHECK', 'control', {'sha256':U1792_CHECKPOINT_SHA}, FALLBACK_25_VERSION, REAR_CLEARANCE_25_VERSION)
    assert approved_execution_override(*exact)
    for index, value in ((0, 'R6-DIRECT-OLD'), (1, 'HOLDOUT'), (2, 'candidate'), (2, 'reference'),
                         (3, {'sha256':'0'*64}), (4, REAR_CLEARANCE_25_VERSION), (5, FALLBACK_25_VERSION)):
        perturbed = [*exact]
        perturbed[index] = value
        assert not approved_execution_override(*perturbed)

from instinctlab.observations.moving_sphere import MAX_SPEED_M_S, PARK_HEIGHT_M, MovingSphere, sphere_range, trajectories
from training_workloads.unified_h1.dynamic_assay import LANES, contract, score, visibility_reset


def test_pipe_probe_freezes_one_origin_and_three_paired_physical_modes(tmp_path, monkeypatch):
    from instinctlab.control.escape_recovery import FALLBACK_25_VERSION
    from tools.r6_evidence import file_pin
    from training_workloads.unified_h1 import robustness_training_plan as plans
    from training_workloads.unified_h1.dynamic_assay import groups_for
    from training_workloads.unified_h1.pipe_authority_probe import PIPE_PROBE_RUN, mirror_reset
    from training_workloads.unified_h1.robustness_scenarios import reset_spec

    checkpoint = tmp_path/'u1792.pt'
    checkpoint.write_bytes(b'fixture')
    origin = file_pin(checkpoint)
    monkeypatch.setattr(plans,'U1792_CHECKPOINT_SHA',origin['sha256'])
    inputs = dict(experiment_id=PIPE_PROBE_RUN,training_seed=20260908,seed=31,wall_seconds=900,
                  models={'candidate':origin},arm_options={'candidate':FALLBACK_25_VERSION},output_cap_bytes=2**30)
    result = contract(inputs)
    assert result['budgets']['nontraining_transitions'] == 11520
    assert result['budgets']['explicit_resets'] == 3
    assert [group[-1] for group in groups_for(PIPE_PROBE_RUN)] == [0,9,7]
    with pytest.raises(ValueError,match='FIXED_ORIGIN'):
        contract(dict(inputs,arm_options={'candidate':None}))
    spec = dict(id='pipe-test',task='stop',speed_m_s=.25,steps=120)
    left,right = (reset_spec(spec,seed=31,split='development',lane=lane,mount_position_m=(.02,0.,.05))
                  for lane in (24,25))
    mirrored = mirror_reset(left,right)
    assert mirrored['lane'] == 25 and mirrored['yaw_degrees'] == -left['yaw_degrees']
    x,y,z = left['root_position_local_m']
    assert mirrored['root_position_local_m'] == [x,-y,z]


def test_rays_choose_a_physical_near_surface_and_preserve_misses():
    origin = torch.tensor([[0., 0., 2.]])
    directions = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]]])
    center = torch.tensor([[1., 0., 2.]])
    depths = sphere_range(origin, directions, center)
    torch.testing.assert_close(depths, torch.tensor([[.66, torch.inf, torch.inf]]))
    hits = torch.tensor([[[2.,0.,2.], [0.,2.,2.], [-2.,0.,2.]]])
    transforms = torch.cat((center, torch.zeros(1,4)), dim=1)
    asset = SimpleNamespace(root_physx_view=SimpleNamespace(get_transforms=lambda: transforms,
                                                             get_velocities=lambda: torch.zeros(1, 6)))
    scene = MovingSphere(asset, origin)
    scene.start, scene.velocity, scene.active = center, torch.zeros_like(center), torch.tensor([True])
    out = scene.merge(hits, origin, directions, capture=True, tick=20)
    torch.testing.assert_close(out[0,0],torch.tensor([.66,0.,2.]))
    torch.testing.assert_close(out[0,1:],hits[0,1:])
    assert scene.pre_reset['winning_beams'].tolist() == [1]


def test_approach_and_crossing_paths_are_directional():
    expected = torch.tensor([[-.42,0.,0.],[.36,0.,0.],[0.,.58,0.],[0.,-.58,0.]])
    origin = torch.zeros(4,3)
    axis = torch.tensor([[1.,0.,0.]]).expand(4,-1)
    _, velocity, _ = trajectories(origin,axis,torch.arange(4))
    torch.testing.assert_close(velocity,expected)


def test_dynamic_servo_bounds_velocity_and_only_writes_pose_at_boundaries():
    pose = torch.tensor([[0., 0., PARK_HEIGHT_M, 0., 0., 0., 1.]])
    calls = []

    def place(value, env_ids=None):
        pose.copy_(value)
        calls.append(('pose', value.clone()))

    asset = SimpleNamespace(
        root_physx_view=SimpleNamespace(get_transforms=lambda: pose, get_masses=lambda: torch.tensor([[250.]])),
        write_root_pose_to_sim=place,
        write_root_velocity_to_sim=lambda value, env_ids=None: calls.append(('velocity', value.clone())),
    )
    scene = MovingSphere(asset, torch.zeros(1, 3))
    scene.configure(torch.zeros(1, 3), torch.tensor([[1., 0., 0.]]), torch.tensor([0]))
    pose[:, 0] -= .05  # Deliberate tracking error large enough to exceed the cap.
    scene.push(.005)
    assert [kind for kind, _ in calls] == ['pose', 'velocity', 'velocity']
    torch.testing.assert_close(calls[-1][1], torch.tensor([[MAX_SPEED_M_S, 0., 0., 0., 0., 0.]]))
    scene.finish()
    assert [kind for kind, _ in calls[-2:]] == ['pose', 'velocity']
    torch.testing.assert_close(calls[-1][1], torch.zeros(1, 6))


def test_wall_approach_ends_before_the_ball_reaches_the_wall_plane():
    world = torch.zeros(1, 3)
    parked = torch.tensor([[0., 0., PARK_HEIGHT_M, 0., 0., 0., 1.]])
    asset = SimpleNamespace(
        root_physx_view=SimpleNamespace(get_transforms=lambda: parked, get_masses=lambda: torch.tensor([[250.]])),
        write_root_pose_to_sim=lambda value, env_ids=None: None,
        write_root_velocity_to_sim=lambda value, env_ids=None: None,
    )
    scene = MovingSphere(asset, world)
    scene.configure(torch.tensor([[4.6, 0., 2.]]), torch.tensor([[-1., 0., 0.]]), torch.tensor([5]))
    scene.age.fill_(12.)
    assert 4.6 < scene._pose()[0, 0] < 6. - .34


def test_dynamic_result_budget_is_separate_and_exact(tmp_path):
    from tools.r6_evidence import file_pin

    first = tmp_path / 'candidate.pt'
    second = tmp_path / 'reference.pt'
    first.write_bytes(b'candidate')
    second.write_bytes(b'reference')
    params = dict(experiment_id='R6-DIRECT-INSTINCT-GPU-R2-DYNAMIC',training_seed=20260908,seed=1,
                  models=dict(candidate=file_pin(first),reference=file_pin(second)),
                  wall_seconds=720,output_cap_bytes=2**30)
    budgets = contract(params)['budgets']
    assert budgets['nontraining_transitions'] == 15360
    assert budgets['explicit_resets'] == 4
    assert budgets['optimizer_updates'] == 0

    visibility = contract(dict(params, experiment_id='R6-DIRECT-INSTINCT-GPU-R2-VISIBILITY'))['budgets']
    assert visibility['nontraining_transitions'] == 7680
    assert visibility['explicit_resets'] == 2
    assert visibility['optimizer_updates'] == 0


def test_persistence_is_bounded_and_reset_per_lane():
    from training_workloads.unified_h1.escape_hold_probe import EscapeHoldProbe, HOLD_STEPS

    state = torch.zeros(3, 52)
    probe = EscapeHoldProbe(state)
    cue = torch.tensor([[0., 1., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.]])
    quiet = torch.zeros_like(cue)
    assert torch.equal(probe(None, (None, state), cue), cue)
    state[1, -1] = .3
    state[0, 41] = 1.
    held = probe(None, (None, state), quiet)
    torch.testing.assert_close(held[0, 1:3], torch.tensor([.9950042, -.0998334]))
    assert torch.count_nonzero(held[1:]) == 0
    state.zero_()
    for _ in range(HOLD_STEPS - 2):
        assert torch.count_nonzero(probe(None, (None, state), quiet)) == 2
    assert torch.equal(probe(None, (None, state), quiet), quiet)
    probe.reset(torch.tensor([True, False, False]))
    probe(None, (None, state), cue)
    assert probe.remaining.tolist() == [HOLD_STEPS - 1, 0, 0]


def test_option_preserves_samples_outside_threat_and_stops_on_stale_measurement():
    from instinctlab.control.escape_option import EscapeOption

    option = EscapeOption(lanes=3, device='cpu', dtype=torch.float32)
    state = torch.zeros(3, 52)
    state[:, 45] = torch.tensor([.25, -.25, 0.])
    option.detector = SimpleNamespace(
        retreat_direction=lambda *args, **kwargs: torch.tensor([[-1., 0., 0.], [-1., 0., 0.], [0., 0., 0.]]),
        choose_escape=lambda *args: (torch.tensor([[0., 1., 0.]]), torch.tensor([1.])),
    )
    policy = dict(lidar_grid=torch.zeros(3, 1), state_vec=state)
    sample = torch.full((3, 4), .5)
    expected = sample.clone()
    expected[0] = torch.tensor([0., .5, .8, .5])
    torch.testing.assert_close(option.apply(policy, sample), expected)
    assert option.applied.tolist() == [True, False, False]
    assert torch.equal(sample, torch.full((3, 4), .5))
    state[:, -1] = .3
    assert torch.equal(option.apply(policy, sample), sample)
    assert option.remaining.tolist() == [0, 0, 0]


def test_recovery_keeps_the_original_ics_and_leaves_quiet_commands_alone():
    from instinctlab.control.escape_recovery import EscapeRecovery, RECOVERY_DELAY
    from instinctlab.control.ics import ICSConfig
    from instinctlab.control.pipeline import ControlCoreConfig

    option = EscapeRecovery(lanes=3, device='cpu', dtype=torch.float32, recenter=True)
    config = ControlCoreConfig(ics=ICSConfig(clearance_margin_m=.15))
    state = torch.zeros((3, 52))
    state[:2, 45] = .25
    lidar = torch.ones((3, 12, 59, 360))
    lidar[:, 0::3] = 10.
    rays = option.detector._full_rays
    beam = (rays @ torch.tensor([.7071068, .7071068, 0.])).argmax()
    lidar[0, 0::3].reshape(4, -1)[:, beam] = .7
    lidar[1, 0::3].reshape(4, -1)[:, beam] = .3
    action = torch.full((3, 4), .5)
    action[:, 0] = 1.
    option.since_escape[:] = RECOVERY_DELAY
    original = option.issued(lidar, state, action, config)
    selection = option.apply_recovery(dict(lidar_grid=lidar, state_vec=state), action, config)
    issued = option.issued(lidar, state, selection, config)
    assert original.v_final_b[0, 0] == 0
    assert issued.v_final_b[0, 0] > .1
    assert issued.metrics['emergency'][1, 0] == 1
    assert option.recovering.tolist() == [True, False, False]
    torch.testing.assert_close(selection[1:], action[1:])


def test_holdout_collision_triangles_are_shared_with_the_sensor_mesh():
    from training_workloads.unified_h1.escape_holdout_scene import scene_geometry

    origins = torch.zeros((32, 3))
    origins[:, 0] = 100 * torch.arange(32)
    points, faces, solids, solid_faces = scene_geometry(origins, 77162057)
    assert scene_geometry(origins, 77162057) == (points, faces, solids, solid_faces)
    assert scene_geometry(origins, 77162058) != (points, faces, solids, solid_faces)
    assert len(faces) == len(solid_faces) + 32 * 6
    assert {tuple(solids[i]) for i in solid_faces} <= {tuple(points[i]) for i in faces}


def test_visibility_reset_changes_only_sensor_translation_and_half_turn():
    block = dict(id='dynamic-0', task='stop', speed_m_s=.25, pulse_acceleration_m_s2=0.,
                 sensor_to_wall_distance_m=None, steps=120)
    mount = (.12, .025, .05)
    front, rear, w_front, w_rear = [
        visibility_reset(block, seed=123, lane=i * 8 + 3, mode=8 + i, mount=mount) for i in range(4)]
    assert front['sensor_position_local_m'] == rear['sensor_position_local_m']
    assert w_front['sensor_position_local_m'] == w_rear['sensor_position_local_m']
    assert w_front['sensor_position_local_m'] == [4.6, *front['sensor_position_local_m'][1:]]
    assert front['yaw_degrees'] - rear['yaw_degrees'] == 180
    assert w_front['root_quaternion_wxyz'] == front['root_quaternion_wxyz']
    assert w_rear['root_quaternion_wxyz'] == rear['root_quaternion_wxyz']
    assert front['root_position_local_m'] != rear['root_position_local_m']


def test_visible_reflex_reports_directional_action_on_live_samples():
    zero = [[0., 0., 0.]] * LANES
    base = dict(axis_w=zero, initial_position_w=zero, command_w=zero, velocity_w=zero,
                vehicle_position_w=zero, modes=[4] * LANES, collision=[False] * LANES,
                post=dict(active=[False] * LANES), terminal=[False] * LANES)
    first = dict(base, step=0, pre=dict(winning_beams=[0] * LANES),
                 v_gov_b=zero, v_final_b=zero, reflex_direction_b=zero)
    direction = [[-1., 0., 0.]] + zero[1:]
    second = dict(first, step=1, reflex_direction_b=direction)
    third = dict(second, step=2, pre=dict(winning_beams=[2] * LANES),
                 v_gov_b=[[-.16, .02, 0.]] + zero[1:], v_final_b=[[-.04, .005, 0.]] + zero[1:])
    first['terminal'] = [False, True] + [False] * (LANES - 2)
    lanes = score([first, second, third])
    assert lanes[0]['reflex_steps'] == 2
    assert lanes[0]['first_visible_reflex'] == dict(step=2, governor_mps=.16, final_mps=.04)
    assert lanes[1]['first_visible_reflex'] is None


def test_lateral_option_selects_issued_lateral_only_on_existing_trigger():
    from instinctlab.control.escape_option import DURATION_STEPS
    from instinctlab.control.escape_recovery import EscapeRecovery

    option = EscapeRecovery(lanes=2, device='cpu', dtype=torch.float32, recenter=True, lateral=True)
    option.detector = SimpleNamespace(retreat_direction=lambda lidar, state, **kwargs: state.new_tensor([[1., 0., 0.]]))
    # The obstacle clips pure lateral candidates. Diagonal commands remain feasible.
    option.issued = lambda lidar, state, action, config: SimpleNamespace(
        v_final_b=(action[:, 1:] - .5) * (action[:, 1:2] > .55))
    option.applied[:] = True
    option.remaining[:] = torch.tensor([DURATION_STEPS-1, DURATION_STEPS-2])
    option.direction[:, 0] = 1.
    actions = torch.tensor([[0., .8, .5, .5], [0., .8, .5, .5]])
    option.select_lateral(torch.zeros(2, 1), torch.zeros(2, 52), actions, None)
    assert actions[0, 2] > .75
    torch.testing.assert_close(actions[1], torch.tensor([0., .8, .5, .5]))
    torch.testing.assert_close(option.direction[0].norm(), torch.tensor(1.))


def test_calibration_requires_the_actual_ray_table_for_the_declared_mount(tmp_path):
    import math
    import pytest

    from instinctlab.control.ics import mounted_mid360_rays_body
    from tools.r6_evidence import file_pin
    from training_workloads.unified_h1.lidar_calibration import verify

    nominal = mounted_mid360_rays_body()
    angle = math.radians(-5.)
    calibrated = nominal.clone()
    calibrated[:, 0] = math.cos(angle)*nominal[:, 0] + math.sin(angle)*nominal[:, 2]
    calibrated[:, 2] = -math.sin(angle)*nominal[:, 0] + math.cos(angle)*nominal[:, 2]
    path = tmp_path/'execution-lidar-rays.f32'
    path.write_bytes(calibrated.numpy().astype('<f4').tobytes())
    half = math.radians(25.)/2
    value = dict(pitch_degrees=25, rotation_wxyz=[math.cos(half),0.,math.sin(half),0.], body_rays=file_pin(path))
    assert verify(value, tmp_path, 'execution', 25) == file_pin(path)
    path.write_bytes(nominal.numpy().astype('<f4').tobytes())
    value['body_rays'] = file_pin(path)
    with pytest.raises(ValueError, match='LIDAR_CALIBRATION_RAY_READBACK'):
        verify(value, tmp_path, 'execution', 25)


def test_redirect_recovers_nonemergency_lateral_release_with_same_side():
    from instinctlab.control.escape_recovery import EscapeRecovery

    option = EscapeRecovery(lanes=2, device='cpu', dtype=torch.float32, recenter=True,
                            lateral=True, fast_lateral=True, redirect=True)
    option.retreat_basis[:, 0] = 1.
    option.direction[:, 1] = -1.
    option.escape_speed.fill_(.45)
    option.applied.fill_(True)
    state = torch.zeros(2, 52)
    state[1,0] = 1.
    chosen = torch.tensor([[0., .5, .05, .5], [0., .5, .05, .5]])
    original = chosen.clone()

    def issue(lidar, state, action, config):
        emergency = state[:, :1] > .5
        beta = (~emergency & (action[:,1:2] > .55)).float()
        return SimpleNamespace(metrics=dict(beta=beta, emergency=emergency),
                               v_final_b=beta*(action[:,1:]-.5))

    option.issued = issue
    option.redirect_clipped(torch.zeros(2,1), state, chosen, None)
    assert option.redirected.tolist() == [True, False]
    assert chosen[0,1] > .55 and chosen[0,2] < .5
    torch.testing.assert_close(chosen[1], original[1])
