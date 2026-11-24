from types import SimpleNamespace

import torch

from instinctlab.observations.moving_sphere import MAX_SPEED_M_S, PARK_HEIGHT_M, MovingSphere, sphere_range, trajectories
from training_workloads.unified_h1.dynamic_assay import LANES, contract, score, visibility_reset


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
