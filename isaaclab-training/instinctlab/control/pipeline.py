"""Atomic composition of the M5 actor-clean pure-tensor control core."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from instinctlab.observations.audit import audit_actor_observation
from .anchor import AnchorConfig, MeasurementAnchor
from .command_adapter import rotate_body_to_world, validate_adapter_inputs
from .governor import GovernorConfig, b0_governor, decode_governor, extract_latest_commands
from .ics import ICSConfig, attenuate_command, extract_lidar_history, mounted_mid360_rays_body

PUBLIC_RESULT_KEYS = frozenset(
    ("alpha", "v_corr_b", "v_gov_b", "v_final_b", "v_final_w", "anchor_metrics", "ics_metrics")
)
INTERNAL_SNAPSHOT_KEYS = frozenset(("anchor", "ics"))


@dataclass(frozen=True)
class ControlCoreConfig:
    governor: GovernorConfig = GovernorConfig()
    anchor: AnchorConfig = AnchorConfig()
    ics: ICSConfig = ICSConfig()
    use_b0: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.governor, GovernorConfig):
            raise TypeError("governor must be GovernorConfig")
        if not isinstance(self.anchor, AnchorConfig):
            raise TypeError("anchor must be AnchorConfig")
        if not isinstance(self.ics, ICSConfig):
            raise TypeError("ics must be ICSConfig")
        if not isinstance(self.use_b0, bool):
            raise TypeError("use_b0 must be boolean")


@dataclass(frozen=True)
class ControlCoreResult:
    alpha: torch.Tensor
    v_corr_b: torch.Tensor
    v_gov_b: torch.Tensor
    v_final_b: torch.Tensor
    v_final_w: torch.Tensor
    anchor_metrics: dict[str, torch.Tensor]
    ics_metrics: dict[str, torch.Tensor]


class ControlCorePipeline:
    """M4 audit through ICS, with attitude accepted only at the last boundary."""

    def __init__(self, config: ControlCoreConfig = ControlCoreConfig()) -> None:
        if not isinstance(config, ControlCoreConfig):
            raise TypeError("config must be ControlCoreConfig")
        self.config = config
        self._anchor = MeasurementAnchor(config.anchor)
        self._last_ics_cache: dict[str, torch.Tensor] = {}

    def __call__(
        self,
        lidar_grid: torch.Tensor,
        state_vec: torch.Tensor,
        governor_action: torch.Tensor,
        history_dt_s: float,
        episode_reset: torch.Tensor,
        explicit_reset: torch.Tensor,
        controller_quaternion_wxyz: torch.Tensor,
    ) -> ControlCoreResult:
        observation = {"lidar_grid": lidar_grid, "state_vec": state_vec}
        audit_actor_observation(observation)
        if not isinstance(episode_reset, torch.Tensor) or not isinstance(explicit_reset, torch.Tensor):
            raise TypeError("reset masks must be torch.Tensor instances")
        if episode_reset.shape != (lidar_grid.shape[0],) or explicit_reset.shape != (lidar_grid.shape[0],):
            raise ValueError("reset masks must have shape (N,)")
        if episode_reset.dtype != torch.bool or explicit_reset.dtype != torch.bool:
            raise TypeError("reset masks must have boolean dtype")
        if episode_reset.device != lidar_grid.device or explicit_reset.device != lidar_grid.device:
            raise ValueError("reset masks must share the actor tensor device")

        if self.config.use_b0:
            if not isinstance(governor_action, torch.Tensor) or governor_action.shape != (lidar_grid.shape[0], 4):
                raise ValueError("governor_action must retain shape (N, 4) in B0 mode")
            if governor_action.dtype != state_vec.dtype or governor_action.device != state_vec.device:
                raise ValueError("governor_action must share actor dtype and device in B0 mode")
            if not bool(torch.isfinite(governor_action).all().item()) or bool(
                ((governor_action < 0.0) | (governor_action > 1.0)).any().item()
            ):
                raise ValueError("governor_action must remain finite and normalized in B0 mode")
            governor = b0_governor(state_vec, self.config.governor.velocity_limit)
        else:
            governor = decode_governor(governor_action, state_vec, self.config.governor)

        ranges, validity, reliability = extract_lidar_history(lidar_grid)
        latest_ranges = ranges[:, -1]
        latest_validity = validity[:, -1]
        latest_reliability = reliability[:, -1]
        v_cmd_b, _ = extract_latest_commands(state_vec)
        self._anchor.validate_update(
            latest_ranges, latest_validity, latest_reliability, v_cmd_b, episode_reset, explicit_reset
        )
        rays = mounted_mid360_rays_body(device=lidar_grid.device, dtype=lidar_grid.dtype)
        validate_adapter_inputs(governor.v_gov_b, controller_quaternion_wxyz)
        ics = attenuate_command(lidar_grid, rays, governor.v_gov_b, history_dt_s, self.config.ics)

        anchor = self._anchor.update(
            latest_ranges, latest_validity, latest_reliability, v_cmd_b, episode_reset, explicit_reset
        )
        v_final_w = rotate_body_to_world(ics.v_final_b, controller_quaternion_wxyz)
        self._last_ics_cache = {key: value.detach().clone() for key, value in ics.cache.items()}
        return ControlCoreResult(
            governor.alpha, governor.v_corr_b, governor.v_gov_b, ics.v_final_b,
            v_final_w, anchor.metrics, ics.metrics,
        )

    def internal_snapshot(self) -> dict[str, dict[str, torch.Tensor]]:
        """Detached evidence-only dense state, separate from public output."""
        return {
            "anchor": self._anchor.cache_snapshot(),
            "ics": {key: value.detach().clone() for key, value in self._last_ics_cache.items()},
        }


__all__ = [
    "ControlCoreConfig", "ControlCorePipeline", "ControlCoreResult",
    "INTERNAL_SNAPSHOT_KEYS", "PUBLIC_RESULT_KEYS",
]
