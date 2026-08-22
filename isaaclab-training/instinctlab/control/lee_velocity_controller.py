# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Strict TASLAB velocity-mode Lee controller adapted from reviewed MIT code."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from instinctlab.assets.taslab_uav.config import TASLAB_UAV

GRAVITY_M_S2 = 9.81
QUATERNION_NORM_TOLERANCE = 1.0e-5


@dataclass(frozen=True)
class LeeControllerResult:
    normalized_rotor_commands: torch.Tensor
    held_position_w_m: torch.Tensor
    held_yaw_rad: torch.Tensor
    desired_rotation_w: torch.Tensor
    collective_thrust_n: torch.Tensor


def quaternion_to_rotation_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Return active body-to-world rotation matrices for scalar-first quaternions."""
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
            2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
            2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[0], 3, 3)


def rotate_world_to_body_wxyz(quaternion: torch.Tensor, vectors_w: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into body coordinates."""
    return torch.bmm(quaternion_to_rotation_matrix_wxyz(quaternion).transpose(1, 2), vectors_w.unsqueeze(-1)).squeeze(-1)


def _normalize(vector: torch.Tensor, name: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    if bool((norm <= torch.finfo(vector.dtype).eps).any().item()):
        raise ValueError(f"degenerate {name} in Lee controller")
    return vector / norm


def _constant_tensor(values: tuple[float, ...], reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=reference.dtype, device=reference.device)


def taslab_mixer(reference: torch.Tensor) -> torch.Tensor:
    """Return the reviewed rotor-to-acceleration/thrust pseudoinverse mixer."""
    rotor = TASLAB_UAV.rotors
    angles = _constant_tensor(rotor.angles, reference)
    arms = _constant_tensor(rotor.arm_lengths, reference)
    directions = _constant_tensor(rotor.directions, reference)
    force_constants = _constant_tensor(rotor.force_constants, reference)
    moment_constants = _constant_tensor(rotor.moment_constants, reference)
    allocation = torch.stack(
        (
            torch.sin(angles) * arms,
            -torch.cos(angles) * arms,
            -directions * moment_constants / force_constants,
            torch.ones_like(angles),
        )
    )
    inertia4 = torch.diag(
        torch.tensor(
            (TASLAB_UAV.inertia.xx, TASLAB_UAV.inertia.yy, TASLAB_UAV.inertia.zz, 1.0),
            dtype=reference.dtype,
            device=reference.device,
        )
    )
    return allocation.T @ torch.linalg.inv(allocation @ allocation.T) @ inertia4


def _validate_inputs(root_state_w: torch.Tensor, target_velocity_w: torch.Tensor) -> torch.Tensor:
    if not isinstance(root_state_w, torch.Tensor) or not isinstance(target_velocity_w, torch.Tensor):
        raise TypeError("Lee inputs must be torch.Tensor instances")
    if root_state_w.ndim != 2 or root_state_w.shape[0] <= 0 or root_state_w.shape[1] != 13:
        raise ValueError("root_state_w must have shape (N, 13)")
    if target_velocity_w.shape != (root_state_w.shape[0], 3):
        raise ValueError("target_velocity_w must have shape (N, 3)")
    if root_state_w.dtype not in (torch.float32, torch.float64) or target_velocity_w.dtype != root_state_w.dtype:
        raise TypeError("Lee inputs must share float32 or float64 dtype")
    if target_velocity_w.device != root_state_w.device:
        raise ValueError("Lee inputs must share one device")
    if not bool(torch.isfinite(root_state_w).all().item()) or not bool(torch.isfinite(target_velocity_w).all().item()):
        raise ValueError("Lee inputs must be finite")
    quaternion = root_state_w[:, 3:7]
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool((torch.abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE).any().item()):
        raise ValueError("root quaternion must be unit length within tolerance")
    return quaternion / norm


def taslab_velocity_control(
    root_state_w: torch.Tensor, target_velocity_w: torch.Tensor
) -> LeeControllerResult:
    """Hold current position/yaw and map world velocity target to raw rotor coordinates."""
    quaternion = _validate_inputs(root_state_w, target_velocity_w)
    position_w = root_state_w[:, :3]
    velocity_w = root_state_w[:, 7:10]
    angular_velocity_w = root_state_w[:, 10:13]
    rotation = quaternion_to_rotation_matrix_wxyz(quaternion)
    angular_velocity_b = torch.bmm(rotation.transpose(1, 2), angular_velocity_w.unsqueeze(-1)).squeeze(-1)
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0]).unsqueeze(-1)

    velocity_gain = _constant_tensor(TASLAB_UAV.lee_gains.velocity, root_state_w)
    acceleration = (velocity_w - target_velocity_w) * velocity_gain
    acceleration = acceleration - torch.tensor(
        (0.0, 0.0, GRAVITY_M_S2), dtype=root_state_w.dtype, device=root_state_w.device
    )
    b1_desired = torch.cat((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), dim=-1)
    b3_desired = -_normalize(acceleration, "desired body-z")
    b2_desired = _normalize(torch.linalg.cross(b3_desired, b1_desired, dim=-1), "desired body-y")
    desired_rotation = torch.stack(
        (torch.linalg.cross(b2_desired, b3_desired, dim=-1), b2_desired, b3_desired), dim=-1
    )
    attitude_error_matrix = 0.5 * (
        torch.bmm(desired_rotation.transpose(1, 2), rotation)
        - torch.bmm(rotation.transpose(1, 2), desired_rotation)
    )
    attitude_error = torch.stack(
        (
            attitude_error_matrix[:, 2, 1],
            attitude_error_matrix[:, 0, 2],
            attitude_error_matrix[:, 1, 0],
        ),
        dim=-1,
    )
    inertia = _constant_tensor(
        (TASLAB_UAV.inertia.xx, TASLAB_UAV.inertia.yy, TASLAB_UAV.inertia.zz), root_state_w
    )
    attitude_gain = _constant_tensor(TASLAB_UAV.lee_gains.attitude, root_state_w) / inertia
    angular_rate_gain = _constant_tensor(TASLAB_UAV.lee_gains.angular_rate, root_state_w) / inertia
    angular_acceleration = -attitude_error * attitude_gain - angular_velocity_b * angular_rate_gain
    collective_thrust = -TASLAB_UAV.mass * (acceleration * rotation[:, :, 2]).sum(dim=-1, keepdim=True)
    acceleration_thrust = torch.cat((angular_acceleration, collective_thrust), dim=-1)
    command = torch.matmul(taslab_mixer(root_state_w), acceleration_thrust.T).T
    rotor = TASLAB_UAV.rotors
    maximum_thrust = _constant_tensor(
        tuple(kf * omega * omega for kf, omega in zip(rotor.force_constants, rotor.max_angular_velocities, strict=True)),
        root_state_w,
    )
    normalized_command = command / maximum_thrust * 2.0 - 1.0
    outputs = (normalized_command, desired_rotation, collective_thrust)
    if not all(bool(torch.isfinite(value).all().item()) for value in outputs):
        raise FloatingPointError("Lee controller produced a non-finite result")
    return LeeControllerResult(
        normalized_command,
        position_w.clone(),
        yaw,
        desired_rotation,
        collective_thrust,
    )


__all__ = [
    "GRAVITY_M_S2", "LeeControllerResult", "QUATERNION_NORM_TOLERANCE",
    "quaternion_to_rotation_matrix_wxyz", "rotate_world_to_body_wxyz",
    "taslab_mixer", "taslab_velocity_control",
]
