"""Live environment selection for the frozen unified-h1-v1 method.

The accepted v1 environment remains unchanged.  The workload stages ``c_t``
and ``c_(t+1)`` before every step so reward and terminal bootstrap use the
commands from the frozen single-step contract.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from instinctlab.task.critic_h1 import CRITIC_H1_DIMENSION, build_critic_h1_state
from instinctlab.task.rewards_h1 import compute_reward_h1
from .taslab_direct_env import TaslabDirectEnv, TaslabDirectEnvCfg, _detached_snapshot


@configclass
class UnifiedH1DirectEnvCfg(TaslabDirectEnvCfg):
    state_space = CRITIC_H1_DIMENSION
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=100.0, replicate_physics=True)
    task_contract_version: str = "unified-h1-v1"


class UnifiedH1DirectEnv(TaslabDirectEnv):
    """H1 reward, critic and mixed-stage reset wiring."""

    cfg: UnifiedH1DirectEnvCfg

    def __init__(self, cfg: UnifiedH1DirectEnvCfg, render_mode: str | None = None, **kwargs):
        self._h1_global_policy_step = 0
        self._h1_next_command_b: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)

    def set_h1_initial_command(self, command_b: torch.Tensor, global_policy_step: int) -> None:
        if not isinstance(global_policy_step, int) or isinstance(global_policy_step, bool) or global_policy_step < 0:
            raise ValueError("global_policy_step must be a nonnegative integer")
        self._validate_h1_command("command_b", command_b)
        self._command_b.copy_(command_b)
        self._h1_global_policy_step = global_policy_step

    def _validate_h1_command(self, name: str, value: torch.Tensor) -> None:
        if (
            not isinstance(value, torch.Tensor) or value.shape != self._command_b.shape
            or value.dtype != self._command_b.dtype or value.device != self._command_b.device
            or not bool(torch.isfinite(value).all().item())
        ):
            raise TypeError(f"{name} violates the environment command contract")

    def prepare_h1_step(self, *, global_policy_step: int, command_b: torch.Tensor, next_command_b: torch.Tensor) -> None:
        if global_policy_step != self._h1_global_policy_step:
            raise ValueError("global command clock mismatch")
        self._validate_h1_command("command_b", command_b)
        self._validate_h1_command("next_command_b", next_command_b)
        if not torch.equal(command_b, self._command_b):
            raise ValueError("actor observation and staged c_t differ")
        self._h1_next_command_b = next_command_b.detach().clone()

    def _get_rewards(self) -> torch.Tensor:
        if self._task is None or self._control is None or self._h1_next_command_b is None:
            raise RuntimeError("H1 task/control/next-command state is unavailable")
        masks = self._task.termination.masks
        reward = compute_reward_h1(
            lidar_grid=self._policy["lidar_grid"], command_b=self._command_b,
            v_gov_b=self._control.v_gov_b, v_final_b=self._control.v_final_b,
            previous_issued_v_final_b=self._reward_previous_v_final_b,
            actual_velocity_b=self._task.actual_velocity_b,
            anchor_loss=self._control.anchor_metrics["huber_loss"],
            anchor_active=self._control.anchor_metrics["active"],
            anchor_usable_fraction=self._control.anchor_metrics["usable_fraction"],
            ics_beta=self._control.ics_metrics["beta"], exact_clearance_m=self._task.exact_clearance_m,
            collision=masks["collision"], below_height=masks["below_height"], above_height=masks["above_height"],
        )
        self._reward = reward
        timeout_critic = build_critic_h1_state(
            root_state_w=self._task.root_state_w, env_origin_w=self.scene.env_origins,
            actual_velocity_b=self._task.actual_velocity_b, exact_clearance_m=self._task.exact_clearance_m,
            collision=self._task.collision, command_b=self._h1_next_command_b,
        ).privileged
        self.extras["reward_components"] = {key: value.detach().clone() for key, value in reward.components.items()}
        self.extras["reward_diagnostics"] = {key: value.detach().clone() for key, value in reward.diagnostics.items()}
        self.extras["pre_reset"] = {
            "reset_mask": self.reset_buf.detach().clone(), "episode_length": self.episode_length_buf.detach().clone(),
            "root_state_w": self._robot.data.root_state_w.detach().clone(), "critic_h1": timeout_critic.detach().clone(),
            "control": _detached_snapshot(self._control), "reward": _detached_snapshot(reward),
            "task": _detached_snapshot(self._task),
        }
        self._command_b.copy_(self._h1_next_command_b)
        self._h1_global_policy_step += 1
        self._h1_next_command_b = None
        return reward.total[:, 0].clone()

    def _get_observations(self):
        legacy = super()._get_observations()
        old = legacy["critic"]
        critic = build_critic_h1_state(
            root_state_w=old[:, :13], env_origin_w=self.scene.env_origins,
            actual_velocity_b=old[:, 13:16], exact_clearance_m=old[:, 16:17],
            collision=old[:, 17:18].to(torch.bool), command_b=self._command_b,
        ).privileged
        return {"policy": legacy["policy"], "critic": critic.detach().clone()}

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        indices = self._validate_reset_indices(env_ids)
        super()._reset_idx(indices)
        if indices.numel() == 0 or self._h1_global_policy_step < 1024:
            return
        near = indices[(indices % 8) >= 6]
        if near.numel() == 0:
            return
        root = self._robot.data.root_state_w[near].clone()
        local = root[:, :3] - self.scene.env_origins[near]
        x_mask = (near % 8) == 6
        local[x_mask, 0] = 4.6; local[x_mask, 1] = 0.0
        local[~x_mask, 0] = 0.0; local[~x_mask, 1] = 4.6
        local[:, 2] = 2.0
        root[:, :3] = local + self.scene.env_origins[near]
        root[:, 7:13] = 0.0
        self._robot.write_root_pose_to_sim(root[:, :7], near)
        self._robot.write_root_velocity_to_sim(root[:, 7:], near)


__all__ = ["UnifiedH1DirectEnv", "UnifiedH1DirectEnvCfg"]
