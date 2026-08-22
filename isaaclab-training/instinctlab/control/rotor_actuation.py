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

"""Exact deterministic TASLAB four-rotor response adapted from reviewed MIT code."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from instinctlab.assets.taslab_uav.config import TASLAB_UAV

TAU_UP = 0.43
TAU_DOWN = 0.43
ROTOR_NAMES = ("rotor_0", "rotor_1", "rotor_2", "rotor_3")


@dataclass(frozen=True)
class RotorResult:
    normalized_commands: torch.Tensor
    target_throttle: torch.Tensor
    throttle: torch.Tensor
    thrusts_n: torch.Tensor
    reaction_moments_nm: torch.Tensor


class TaslabRotorModel:
    """Per-environment rotor throttle state with explicit preview/commit atomicity."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype,
        dt_s: float = 0.005,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("rotor dtype must be float32 or float64")
        if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or not math.isfinite(float(dt_s)) or dt_s <= 0.0:
            raise ValueError("dt_s must be a finite positive scalar")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.dt_s = float(dt_s)
        self._throttle = torch.zeros(num_envs, 4, device=self.device, dtype=dtype)
        rotor = TASLAB_UAV.rotors
        self._maximum_thrust_n = torch.tensor(
            tuple(kf * omega * omega for kf, omega in zip(rotor.force_constants, rotor.max_angular_velocities, strict=True)),
            device=self.device,
            dtype=dtype,
        )
        self._maximum_moment_nm = torch.tensor(
            tuple(km * omega * omega for km, omega in zip(rotor.moment_constants, rotor.max_angular_velocities, strict=True)),
            device=self.device,
            dtype=dtype,
        )
        self._directions = torch.tensor(rotor.directions, device=self.device, dtype=dtype)

    @property
    def throttle(self) -> torch.Tensor:
        return self._throttle.detach().clone()

    def _validate_commands(self, normalized_commands: torch.Tensor) -> None:
        if not isinstance(normalized_commands, torch.Tensor):
            raise TypeError("normalized_commands must be a torch.Tensor")
        if normalized_commands.shape != (self.num_envs, 4):
            raise ValueError("normalized_commands must have shape (N, 4)")
        if normalized_commands.device != self.device or normalized_commands.dtype != self.dtype:
            raise ValueError("normalized_commands must share rotor device and dtype")
        if not bool(torch.isfinite(normalized_commands).all().item()):
            raise ValueError("normalized_commands must be finite")

    def preview(self, normalized_commands: torch.Tensor) -> RotorResult:
        """Compute the next response without mutating throttle."""
        self._validate_commands(normalized_commands)
        target = torch.sqrt(torch.clamp((normalized_commands + 1.0) / 2.0, 0.0, 1.0))
        tau = torch.where(target > self._throttle, TAU_UP, TAU_DOWN)
        next_throttle = self._throttle + tau * (target - self._throttle)
        squared = next_throttle.square().clamp(0.0, 1.0)
        thrusts = squared * self._maximum_thrust_n
        moments = squared * self._maximum_moment_nm * -self._directions
        return RotorResult(normalized_commands.clone(), target, next_throttle, thrusts, moments)

    def commit(self, result: RotorResult) -> None:
        if not isinstance(result, RotorResult):
            raise TypeError("result must be RotorResult")
        if result.throttle.shape != self._throttle.shape or result.throttle.device != self.device or result.throttle.dtype != self.dtype:
            raise ValueError("result throttle does not match rotor state")
        if not bool(torch.isfinite(result.throttle).all().item()) or bool(
            ((result.throttle < 0.0) | (result.throttle > 1.0)).any().item()
        ):
            raise ValueError("result throttle must be finite and in [0, 1]")
        self._throttle.copy_(result.throttle)

    def step(self, normalized_commands: torch.Tensor) -> RotorResult:
        result = self.preview(normalized_commands)
        self.commit(result)
        return result

    def reset(self, env_indices: torch.Tensor | None = None) -> None:
        if env_indices is None:
            self._throttle.zero_()
            return
        if not isinstance(env_indices, torch.Tensor) or env_indices.ndim != 1 or env_indices.dtype != torch.long:
            raise TypeError("env_indices must be a rank-one torch.long tensor")
        if env_indices.device != self.device:
            raise ValueError("env_indices must share the rotor device")
        if bool(((env_indices < 0) | (env_indices >= self.num_envs)).any().item()):
            raise IndexError("env_indices contains an out-of-range environment")
        if env_indices.unique().numel() != env_indices.numel():
            raise ValueError("env_indices must not contain duplicates")
        self._throttle[env_indices] = 0.0


__all__ = ["ROTOR_NAMES", "RotorResult", "TAU_DOWN", "TAU_UP", "TaslabRotorModel"]
