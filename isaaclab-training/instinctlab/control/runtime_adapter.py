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
    "TaslabArticulationWrenchAdapter", "WrenchResult",
]
