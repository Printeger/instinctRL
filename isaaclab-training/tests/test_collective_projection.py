import math

import torch

from instinctlab.assets.taslab_uav.config import TASLAB_UAV
from instinctlab.control.collective_projection import restore_vertical_projection
from instinctlab.control.lee_velocity_controller import GRAVITY_M_S2, taslab_velocity_control


def test_small_tilt_correction_preserves_mixer_torque_and_respects_reserve():
    root = torch.zeros(5, 13, dtype=torch.float64)
    angles = torch.tensor([3., 8., 3., 3., 0.], dtype=root.dtype)*math.pi/180.
    root[:, 3], root[:, 5] = (angles/2.).cos(), (angles/2.).sin()
    target = root.new_zeros((5, 3))
    target[2, 0] = 1.8
    mass = root.new_full((5, 1), 1.18)
    allocation = root.new_tensor([
        [-.08, .08, .08, -.08],
        [-.08, -.08, .08, .08],
        [-.015, .015, -.015, .015],
        [1., 1., 1., 1.],
    ]).expand(5, -1, -1)
    stock = taslab_velocity_control(root, target, measured_mass_kg=mass, measured_allocation=allocation)
    commands = stock.normalized_rotor_commands.clone()
    commands[3, 0] = .99
    corrected, delta = restore_vertical_projection(commands, root, target, stock.collective_thrust_n,
                                                    mass, allocation)

    assert 0. < delta[0, 0] <= .03
    assert torch.equal(delta[1:], torch.zeros_like(delta[1:]))
    assert torch.equal(corrected[1:], commands[1:])
    rotors = TASLAB_UAV.rotors
    max_force = root.new_tensor(rotors.force_constants)*root.new_tensor(rotors.max_angular_velocities).square()
    force_delta = .5*(corrected[0]-commands[0])*max_force
    wrench_delta = allocation[0] @ force_delta
    torch.testing.assert_close(wrench_delta[:3], root.new_zeros(3), atol=1.e-12, rtol=0.)
    realized_vertical = (stock.collective_thrust_n[0, 0]+wrench_delta[3])*angles[0].cos()/mass[0, 0]
    torch.testing.assert_close(realized_vertical, root.new_tensor(GRAVITY_M_S2), atol=1.e-12, rtol=0.)


def test_tilt_gate_uses_the_same_normalized_attitude_as_lee():
    root = torch.zeros(2, 13, dtype=torch.float64)
    angles = root.new_tensor([-5.000010, -4.999990])*math.pi/180.
    scale = root.new_tensor([.999991, 1.000009])
    root[:, 3], root[:, 5] = scale*(angles/2.).cos(), scale*(angles/2.).sin()
    target = root.new_zeros((2, 3))
    mass = root.new_full((2, 1), 1.18)
    allocation = torch.eye(4, dtype=root.dtype).expand(2, -1, -1)
    normalized = root.new_zeros((2, 4))
    stock = mass*((GRAVITY_M_S2-.015)/angles.cos())[:, None]

    actual_commands, delta = restore_vertical_projection(normalized, root, target, stock, mass, allocation)
    assert torch.equal(actual_commands[0], normalized[0])
    assert delta[0, 0] == 0.
    assert 0. < delta[1, 0] <= .03
