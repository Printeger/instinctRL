"""Preserve Lee's requested vertical acceleration inside the rotor boundary."""

import math

import torch

from instinctlab.assets.taslab_uav.config import TASLAB_UAV
from .lee_velocity_controller import GRAVITY_M_S2, quaternion_to_rotation_matrix_wxyz
from .runtime_adapter import measured_control_plant


MAX_VERTICAL_CORRECTION_M_S2 = .03
MIN_ROTOR_FRACTION = .02
MIN_UP_PROJECTION = math.cos(math.radians(5.))
_ROTORS = TASLAB_UAV.rotors
_FORCE_LIMITS = tuple(
    coefficient*speed**2 for coefficient, speed in zip(
        _ROTORS.force_constants, _ROTORS.max_angular_velocities, strict=True))


def restore_vertical_projection(normalized, root, target_w, stock_collective, mass, allocation):
    """Adjust collective without requesting a change in mixer torque."""
    attitude = root[:, 3:7]
    attitude = attitude/torch.linalg.vector_norm(attitude, dim=1, keepdim=True)
    up = quaternion_to_rotation_matrix_wxyz(attitude)[:, 2, 2:3]
    requested = GRAVITY_M_S2 + TASLAB_UAV.lee_gains.velocity[2]*(target_w[:, 2:3]-root[:, 9:10])
    delta = requested-(stock_collective/mass)*up
    admitted = (up >= MIN_UP_PROJECTION) & (delta.abs() <= MAX_VERTICAL_CORRECTION_M_S2)
    change = torch.where(admitted, mass*delta/up.clamp_min(MIN_UP_PROJECTION), 0.)
    wrench = torch.cat((torch.zeros_like(root[:, :3]), change), dim=1)
    changed_force = torch.linalg.solve(allocation, wrench.unsqueeze(-1)).squeeze(-1)
    stock = (normalized+1.)*.5
    candidate = stock + changed_force/normalized.new_tensor(_FORCE_LIMITS)
    reserve = ((stock >= MIN_ROTOR_FRACTION) & (stock <= 1.-MIN_ROTOR_FRACTION)
               & (candidate >= MIN_ROTOR_FRACTION) & (candidate <= 1.-MIN_ROTOR_FRACTION))
    admitted &= reserve.all(dim=1, keepdim=True) & (delta != 0.)
    return torch.where(admitted, 2.*candidate-1., normalized), torch.where(admitted, delta, 0.)


class CollectiveProjectionAdapter:
    """Apply the trim immediately before the existing rotor adapter."""

    def __init__(self, baseline, control_state):
        self.baseline = baseline
        self.control_state = control_state
        self.rotors = baseline.rotors
        self.peak_correction = torch.zeros((self.rotors.num_envs,), device=self.rotors.device, dtype=self.rotors.dtype)
        self.eligible = None

    def apply(self, articulation, normalized):
        root, target_w, stock_collective = self.control_state()
        mass, allocation = measured_control_plant(articulation, root)
        corrected, delta = restore_vertical_projection(normalized, root, target_w, stock_collective, mass, allocation)
        if self.eligible is not None:
            selected = self.eligible[:, None]
            corrected = torch.where(selected, corrected, normalized)
            delta = torch.where(selected, delta, 0.)
        result = self.baseline.apply(articulation, corrected)
        torch.maximum(self.peak_correction, delta[:, 0].abs(), out=self.peak_correction)
        return result

    def pending_wrenches(self):
        return self.baseline.pending_wrenches()

    def reset(self, articulation, indices=None):
        self.baseline.reset(articulation, indices)
        self.peak_correction[indices] = 0.
        if self.eligible is not None:
            self.eligible[indices] = False
