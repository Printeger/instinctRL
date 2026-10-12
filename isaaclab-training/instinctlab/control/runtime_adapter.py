"""Topology-strict public Isaac Lab wrench adapter for TASLAB rotor output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .rotor_actuation import ROTOR_NAMES, RotorResult, TaslabRotorModel

EXPECTED_BODY_NAMES = ("base_link", *ROTOR_NAMES)
BASE_BODY_INDEX = 0
ROTOR_BODY_INDICES = (1, 2, 3, 4)


@dataclass(frozen=True)
class WrenchResult:
    rotor: RotorResult
    forces_body_n: torch.Tensor
    torques_body_nm: torch.Tensor


def measured_control_plant(articulation: Any, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure total mass and the actual rotor wrench allocation about system COM.

    All inputs are public plant measurements for the privileged flight control
    boundary. No body mass, transform, rotor state or actor input is modified.
    """
    from .lee_velocity_controller import quaternion_to_rotation_matrix_wxyz
    from instinctlab.assets.taslab_uav.config import TASLAB_UAV
    if tuple(articulation.body_names) != EXPECTED_BODY_NAMES:
        raise ValueError("control mass/geometry requires exact five-body topology without duplicates")
    mass = articulation.root_physx_view.get_masses()
    body = articulation.data.body_link_state_w
    local = articulation.data.body_com_pose_b
    n = reference.shape[0]
    if not all(isinstance(x,torch.Tensor) for x in (mass,body,local)):
        raise RuntimeError("actual control plant mass/geometry unavailable")
    if mass.shape != (n,5) or body.shape != (n,5,13) or local.shape != (n,5,7):
        raise ValueError("actual control plant mass/geometry shape mismatch")
    if any(not bool(torch.isfinite(x).all()) for x in (mass,body,local)) or bool((mass<=0).any()):
        raise ValueError("actual control plant measurements must be finite with positive per-body mass")
    mass = mass.to(device=reference.device,dtype=torch.float64)
    body = body.to(device=reference.device,dtype=torch.float64)
    local = local.to(device=reference.device,dtype=torch.float64)
    q = body[:,:,3:7]
    norms = torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    if bool(((norms-1).abs()>1e-5).any()): raise ValueError("actual body orientations must be normalized")
    rotation = quaternion_to_rotation_matrix_wxyz((q/norms).reshape(-1,4)).reshape(n,5,3,3)
    coms = body[:,:,:3]+torch.matmul(rotation,local[:,:,:3].unsqueeze(-1)).squeeze(-1)
    total = mass.sum(dim=-1,keepdim=True)
    combined = (coms*mass.unsqueeze(-1)).sum(dim=1)/total
    base_inverse = rotation[:,0].transpose(1,2)
    arms = torch.matmul(base_inverse[:,None],(body[:,1:,:3]-combined[:,None]).unsqueeze(-1)).squeeze(-1)
    directions = torch.matmul(base_inverse[:,None],rotation[:,1:,:,2].unsqueeze(-1)).squeeze(-1)
    torque = torch.linalg.cross(arms,directions,dim=-1)
    # Runtime adapter applies each reaction moment to base local Z.
    rotors = TASLAB_UAV.rotors
    reaction = torch.tensor([-d*km/kf for d,km,kf in zip(rotors.directions,rotors.moment_constants,rotors.force_constants,strict=True)],device=reference.device,dtype=torch.float64)
    torque[:,:,2] += reaction
    allocation = torch.cat((torque.transpose(1,2),directions[:,:,2].unsqueeze(1)),dim=1)
    if not bool(torch.isfinite(allocation).all()) or bool((torch.linalg.svdvals(allocation)[:,-1]<=1e-8).any()):
        raise ValueError("actual rotor wrench allocation must be finite and nonsingular")
    return total.to(reference.dtype), allocation.to(reference.dtype)


class TaslabArticulationWrenchAdapter:
    """Validate complete topology, stage local wrenches, then commit rotor state."""

    def __init__(self, num_envs: int, device: torch.device | str, dtype: torch.dtype) -> None:
        self.rotors = TaslabRotorModel(num_envs, device, dtype)
        self._forces = torch.zeros(num_envs, 5, 3, device=self.rotors.device, dtype=dtype)
        self._torques = torch.zeros_like(self._forces)

    def validate_articulation(self, articulation: Any) -> None:
        if tuple(getattr(articulation, "body_names", ())) != EXPECTED_BODY_NAMES:
            raise ValueError(f"TASLAB body topology must be exactly {EXPECTED_BODY_NAMES}")
        if getattr(articulation, "num_bodies", None) != len(EXPECTED_BODY_NAMES):
            raise ValueError("TASLAB articulation must contain exactly five bodies")
        if getattr(articulation, "num_instances", None) != self.rotors.num_envs:
            raise ValueError("articulation instance count must match rotor environments")
        articulation_device = getattr(articulation, "device", None)
        if articulation_device is None or torch.device(articulation_device) != self.rotors.device:
            raise ValueError("articulation device must match rotor environments")
        if not callable(getattr(articulation, "set_external_force_and_torque", None)):
            raise TypeError("articulation must provide set_external_force_and_torque")
        if not callable(getattr(articulation, "write_data_to_sim", None)):
            raise TypeError("articulation must provide write_data_to_sim")

    def _wrench_buffers(self, rotor: RotorResult) -> tuple[torch.Tensor, torch.Tensor]:
        forces = torch.zeros_like(self._forces)
        torques = torch.zeros_like(self._torques)
        forces[:, ROTOR_BODY_INDICES, 2] = rotor.thrusts_n
        torques[:, BASE_BODY_INDEX, 2] = rotor.reaction_moments_nm.sum(dim=-1)
        if not bool(torch.isfinite(forces).all().item()) or not bool(torch.isfinite(torques).all().item()):
            raise FloatingPointError("rotor wrench buffers must be finite")
        return forces, torques

    def apply(self, articulation: Any, normalized_commands: torch.Tensor) -> WrenchResult:
        """Preflight topology/commands, stage local +Z wrenches, then commit state."""
        self.validate_articulation(articulation)
        rotor = self.rotors.preview(normalized_commands)
        forces, torques = self._wrench_buffers(rotor)
        articulation.set_external_force_and_torque(forces, torques, is_global=False)
        self.rotors.commit(rotor)
        self._forces.copy_(forces)
        self._torques.copy_(torques)
        return WrenchResult(rotor, forces.clone(), torques.clone())

    def write_data_to_sim(self, articulation: Any) -> None:
        self.validate_articulation(articulation)
        articulation.write_data_to_sim()

    def reset(self, articulation: Any, env_indices: torch.Tensor | None = None) -> None:
        """Clear selected rotor state and pending simulator wrench buffers."""
        self.validate_articulation(articulation)
        if env_indices is None:
            indices = torch.arange(self.rotors.num_envs, device=self.rotors.device, dtype=torch.long)
        else:
            if not isinstance(env_indices, torch.Tensor) or env_indices.ndim != 1 or env_indices.dtype != torch.long:
                raise TypeError("env_indices must be a rank-one torch.long tensor")
            if env_indices.device != self.rotors.device:
                raise ValueError("env_indices must share the adapter device")
            if bool(((env_indices < 0) | (env_indices >= self.rotors.num_envs)).any().item()):
                raise IndexError("env_indices contains an out-of-range environment")
            if env_indices.unique().numel() != env_indices.numel():
                raise ValueError("env_indices must not contain duplicates")
            indices = env_indices
        zero_forces = torch.zeros(indices.numel(), 5, 3, device=self.rotors.device, dtype=self.rotors.dtype)
        zero_torques = torch.zeros_like(zero_forces)
        articulation.set_external_force_and_torque(zero_forces, zero_torques, env_ids=indices, is_global=False)
        self.rotors.reset(indices)
        self._forces[indices] = 0.0
        self._torques[indices] = 0.0

    def pending_wrenches(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forces.detach().clone(), self._torques.detach().clone()


__all__ = [
    "BASE_BODY_INDEX", "EXPECTED_BODY_NAMES", "ROTOR_BODY_INDICES",
    "TaslabArticulationWrenchAdapter", "WrenchResult", "measured_control_plant",
]
