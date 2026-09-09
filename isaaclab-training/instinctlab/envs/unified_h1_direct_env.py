"""Live environment selection for the frozen unified-h1-v1 method.

The accepted v1 environment remains unchanged.  The workload stages ``c_t``
and ``c_(t+1)`` before every step so reward and terminal bootstrap use the
commands from the frozen single-step contract.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import importlib

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from pxr import UsdGeom, UsdPhysics

from instinctlab.task.critic_h1 import CRITIC_H1_DIMENSION, build_critic_h1_state
from instinctlab.task.rewards_h1 import compute_reward_h1
from instinctlab.task.scene_h1 import H1_RAY_MESH_PATH, H1_RAY_SCENE_PATH, combined_scene_triangles
from instinctlab.control.pipeline import ControlCoreResult
from .taslab_direct_env import TaslabDirectEnv, TaslabDirectEnvCfg, _detached_snapshot

def _define_triangle_mesh(path: str, points: list[tuple[float, float, float]], indices: list[int], *, collision: bool) -> None:
    mesh = UsdGeom.Mesh.Define(sim_utils.get_current_stage(), path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())


def _spawn_unified_h1_combined_scene(env_origins: torch.Tensor) -> None:
    """Author the actual 32-lane finite ray mesh and matching wall collision mesh."""
    points, indices, _ = combined_scene_triangles(env_origins)
    sim_utils.create_prim(H1_RAY_SCENE_PATH, "Xform")
    _define_triangle_mesh(H1_RAY_MESH_PATH, points, indices, collision=False)
    # The physical ground remains Isaac's plane.  Only wall triangles are
    # duplicated into the collision mesh, avoiding a co-planar collision floor.
    wall_points: list[tuple[float, float, float]] = []
    wall_indices: list[int] = []
    for offset in range(0, len(points), 12):
        for surface_offset in (4, 8):
            start = len(wall_points)
            wall_points.extend(points[offset + surface_offset:offset + surface_offset + 4])
            wall_indices.extend((start, start + 1, start + 2, start, start + 2, start + 3))
    sim_utils.create_prim("/World/H1CollisionWalls", "Xform")
    _define_triangle_mesh("/World/H1CollisionWalls/Walls", wall_points, wall_indices, collision=True)


@configclass
class UnifiedH1DirectEnvCfg(TaslabDirectEnvCfg):
    state_space = CRITIC_H1_DIMENSION
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=100.0, replicate_physics=True)
    ray_caster = TaslabDirectEnvCfg.ray_caster.replace(mesh_prim_paths=[H1_RAY_MESH_PATH])
    task_contract_version: str = "unified-h1-v1"


class UnifiedH1DirectEnv(TaslabDirectEnv):
    """H1 reward, critic and mixed-stage reset wiring."""

    cfg: UnifiedH1DirectEnvCfg

    def __init__(self, cfg: UnifiedH1DirectEnvCfg, render_mode: str | None = None, **kwargs):
        self._h1_global_policy_step = 0
        self._h1_next_command_b: torch.Tensor | None = None
        self._b0_command_b: torch.Tensor | None = None
        self._h1_evaluation_case: str | None = None
        super().__init__(cfg, render_mode, **kwargs)
        # Bind Isaac Sim's own timeline index at the construction boundary,
        # before this environment performs any explicit reset or policy step.
        # An unavailable/non-integral counter is retained as None so the future
        # workload fails closed instead of claiming zero initialization cost.
        timeline_index = getattr(self.sim, "current_time_step_index", None)
        self.h1_initialization_physics_ticks = timeline_index if type(timeline_index) is int else None
        self.h1_initialization_tick_source = "SimulationContext.current_time_step_index"

    def _setup_scene(self) -> None:
        _spawn_unified_h1_combined_scene(self.scene.env_origins)
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["taslab"] = self._robot
        self.scene.clone_environments(copy_from_source=False)
        self._ray_caster = RayCaster(self.cfg.ray_caster)
        self.scene.sensors["mid360"] = self._ray_caster
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact"] = self._contact_sensor
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground", "/World/H1CollisionWalls"])

    def set_h1_initial_command(self, command_b: torch.Tensor, global_policy_step: int) -> None:
        if not isinstance(global_policy_step, int) or isinstance(global_policy_step, bool) or global_policy_step < 0:
            raise ValueError("global_policy_step must be a nonnegative integer")
        self._validate_h1_command("command_b", command_b)
        self._command_b.copy_(command_b)
        # A diagnostic clock jump does not call ``_get_observations``.  Keep
        # the control pipeline's private actor snapshot on the same c_t as the
        # externally returned observation without advancing actor history.
        policy = getattr(self, "_policy", None)
        if isinstance(policy, dict) and isinstance(policy.get("state_vec"), torch.Tensor):
            state = policy["state_vec"]
            if state.shape != (self._command_b.shape[0], 52):
                raise RuntimeError("internal H1 policy state shape mismatch")
            state[:, 45:48].copy_(command_b)
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

    def prepare_b0_step(self, *, command_b: torch.Tensor, next_command_b: torch.Tensor) -> None:
        """Stage the reviewed direct B0 path; actor/governor/anchor/ICS are bypassed."""
        self._validate_h1_command("command_b", command_b)
        self._validate_h1_command("next_command_b", next_command_b)
        self._command_b.copy_(command_b)
        self._h1_next_command_b = next_command_b.detach().clone()
        self._b0_command_b = command_b.detach().clone()

    def set_h1_evaluation_case(self, case: str | None) -> None:
        if case not in {None, "Z", "T", "S", "X", "W"}:
            raise ValueError("unknown metric-h1 evaluation case")
        self._h1_evaluation_case = case

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._b0_command_b is None:
            super()._pre_physics_step(actions)
            return
        b0 = importlib.import_module("instinctlab.control." + "baseline_b0")
        direct = b0.execute_b0_direct_velocity(self._b0_command_b, self._robot.data.root_quat_w, b0.B0DirectConfig())
        zeros3 = torch.zeros_like(direct.issued_body_velocity_m_s)
        zeros1 = zeros3[:, :1]
        ones1 = torch.ones_like(zeros1)
        self._actions.zero_()
        self._reward_previous_v_final_b.copy_(self._previous_issued_v_final_b)
        self._control = ControlCoreResult(
            ones1, zeros3, direct.issued_body_velocity_m_s, direct.issued_body_velocity_m_s,
            direct.issued_world_velocity_m_s,
            {"active": zeros1, "usable_fraction": zeros1, "mean_absolute_error_m": zeros1, "huber_loss": zeros1},
            {"beta": ones1, "active_beam_count": zeros1, "minimum_clearance_m": zeros1,
             "worst_margin_m_s": zeros1, "emergency": zeros1},
        )
        self._previous_issued_v_final_b.copy_(direct.issued_body_velocity_m_s)
        self._episode_reset_pending.zero_()
        self._explicit_reset_pending.zero_()
        self._b0_command_b = None

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
        if self._evaluation_ray_snapshot is None:
            raise RuntimeError("same-state evaluation ray source is unavailable")
        source = self._evaluation_ray_snapshot
        self.extras["evaluation_ray_source"] = OrderedDict((
            ("ray_starts_w", source.ray_starts_w.detach().clone()),
            ("ray_directions_w", source.ray_directions_w.detach().clone()),
            ("accepted_ray_hits_w", source.accepted_ray_hits_w.detach().clone()),
            ("accepted_beam_valid", source.accepted_beam_valid.detach().clone()),
            ("mesh_path", self.cfg.ray_caster.mesh_prim_paths[0]),
            ("maximum_distance_m", float(self.cfg.ray_caster.max_distance)),
            ("source_step", source.source_step.detach().clone()),
            ("reset_mask", self.reset_buf.detach().clone()),
        ))
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
        if indices.numel() and self._h1_evaluation_case is not None:
            configured = torch.zeros(self.num_envs, 3, device=self.device, dtype=self._actions.dtype)
            configured[:, 2] = 2.0
            if self._h1_evaluation_case == "W":
                lane = torch.arange(self.num_envs, device=self.device)
                tangent = (lane % 4 - 1.5).to(self._actions.dtype) * .05
                configured[:16, 0] = 4.6; configured[:16, 1] = tangent[:16]
                configured[16:, 1] = 4.6; configured[16:, 0] = tangent[16:]
            else:
                offsets = torch.tensor([[-.1, -.1], [-.1, .1], [.1, -.1], [.1, .1]], device=self.device, dtype=self._actions.dtype)
                configured[:, :2] = offsets[torch.arange(self.num_envs, device=self.device) // 8]
            root = self._robot.data.root_state_w[indices].clone()
            root[:, :3] = configured[indices] + self.scene.env_origins[indices]
            root[:, 3:7] = 0.0; root[:, 3] = 1.0; root[:, 7:13] = 0.0
            self._robot.write_root_pose_to_sim(root[:, :7], indices)
            self._robot.write_root_velocity_to_sim(root[:, 7:], indices)
            return
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


__all__ = [
    "H1_RAY_MESH_PATH", "UnifiedH1DirectEnv", "UnifiedH1DirectEnvCfg",
]
