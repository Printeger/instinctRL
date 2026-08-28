"""Deterministic DirectRLEnv composition of the accepted TASLAB task blocks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from pxr import UsdGeom

from instinctlab.assets.taslab_uav.articulation import TASLAB_ARTICULATION_CFG
from instinctlab.control.lee_velocity_controller import (
    rotate_world_to_body_wxyz,
    taslab_velocity_control,
)
from instinctlab.control.pipeline import ControlCoreConfig, ControlCorePipeline, ControlCoreResult
from instinctlab.control.runtime_adapter import TaslabArticulationWrenchAdapter, WrenchResult
from instinctlab.observations.actor import ActorObservationHistory, RAY_COUNT
from instinctlab.sensors.mid360 import (
    MAX_RANGE_M,
    MID360_RAY_CASTER_CFG,
    UPDATE_PERIOD_S,
    compose_mounted_sensor_pose,
    ranges_and_validity,
)
from instinctlab.task.critic import CRITIC_DIMENSION, build_critic_state
from instinctlab.task.rewards import RewardResult, compute_reward
from instinctlab.task.termination import TerminationConfig, TerminationResult, compute_termination

PHYSICS_DT_S = 0.005
DECIMATION = 20
POLICY_DT_S = PHYSICS_DT_S * DECIMATION
DEVICE = "cuda:0"
ACTION_DIMENSION = 4
CONTACT_FORCE_THRESHOLD_N = 1.0
STATIC_PLANE_HALF_EXTENT_M = 42
STATIC_PLANE_SPACING_M = 2
POLICY_KEYS = ("lidar_grid", "state_vec")
OBSERVATION_SPACE = {
    "lidar_grid": gym.spaces.Box(low=0.0, high=MAX_RANGE_M, shape=(12, 59, 360), dtype=np.float32),
    "state_vec": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(52,), dtype=np.float32),
}


@configclass
class TaslabDirectEnvCfg(DirectRLEnvCfg):
    """Frozen M6-002 environment constants and public Gym spaces."""

    seed = 0
    decimation = DECIMATION
    episode_length_s = 100.0
    action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(ACTION_DIMENSION,), dtype=np.float32)
    observation_space = OBSERVATION_SPACE
    state_space = CRITIC_DIMENSION
    sim: SimulationCfg = SimulationCfg(dt=PHYSICS_DT_S, render_interval=DECIMATION, device=DEVICE)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=4.0, replicate_physics=True)
    robot = TASLAB_ARTICULATION_CFG.replace(
        prim_path="/World/envs/env_.*/TASLAB",
        spawn=TASLAB_ARTICULATION_CFG.spawn.replace(activate_contact_sensors=True),
    )
    ray_caster = MID360_RAY_CASTER_CFG.replace(
        prim_path="/World/envs/env_.*/TASLAB/base_link",
        mesh_prim_paths=["/World/MID360StaticScene"],
    )
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/TASLAB/.*",
        update_period=PHYSICS_DT_S,
        history_length=1,
        debug_vis=False,
    )
    commanded_body_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    contact_force_threshold_n: float = CONTACT_FORCE_THRESHOLD_N


@dataclass(frozen=True)
class _TaskSnapshot:
    root_state_w: torch.Tensor
    actual_velocity_b: torch.Tensor
    exact_clearance_m: torch.Tensor
    collision: torch.Tensor
    termination: TerminationResult


@dataclass(frozen=True)
class _EvaluationRaySnapshot:
    """Owned raw rays from the sole task-snapshot sensor update."""

    ray_starts_w: torch.Tensor
    ray_directions_w: torch.Tensor
    accepted_ray_hits_w: torch.Tensor
    accepted_beam_valid: torch.Tensor
    source_step: torch.Tensor


def _detached_snapshot(value: Any) -> Any:
    """Copy nested diagnostic state without exposing a live control reference."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _detached_snapshot(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _detached_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detached_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_detached_snapshot(item) for item in value]
    return value


def _zero_selected_batch(value: Any, indices: torch.Tensor, batch_size: int) -> None:
    """Clear selected leading-batch tensor slices in one nested cache object."""
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            value[indices] = 0
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _zero_selected_batch(getattr(value, field.name), indices, batch_size)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _zero_selected_batch(item, indices, batch_size)


def _spawn_static_ray_plane() -> None:
    """Create the same deterministic z=0 triangle mesh qualified by M3."""
    coordinates = list(
        range(-STATIC_PLANE_HALF_EXTENT_M, STATIC_PLANE_HALF_EXTENT_M + STATIC_PLANE_SPACING_M,
              STATIC_PLANE_SPACING_M)
    )
    points = [(float(x), float(y), 0.0) for y in coordinates for x in coordinates]
    width = len(coordinates)
    indices: list[int] = []
    for row in range(width - 1):
        for column in range(width - 1):
            lower_left = row * width + column
            lower_right = lower_left + 1
            upper_left = lower_left + width
            upper_right = upper_left + 1
            indices.extend((lower_left, lower_right, upper_right, lower_left, upper_right, upper_left))
    sim_utils.create_prim("/World/MID360StaticScene", "Xform")
    mesh = UsdGeom.Mesh.Define(sim_utils.get_current_stage(), "/World/MID360StaticScene/PlaneMesh")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")


class TaslabDirectEnv(DirectRLEnv):
    """Actor-clean vector environment; privilege is confined to task/critic surfaces."""

    cfg: TaslabDirectEnvCfg

    def __init__(self, cfg: TaslabDirectEnvCfg, render_mode: str | None = None, **kwargs: Any) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        dtype = self._robot.data.root_state_w.dtype
        self._history = ActorObservationHistory()
        self._history_initialized = False
        self._control_pipelines = [ControlCorePipeline(ControlCoreConfig(use_b0=False)) for _ in range(self.num_envs)]
        self._wrench_adapter = TaslabArticulationWrenchAdapter(self.num_envs, self.device, dtype)
        self._wrench_adapter.validate_articulation(self._robot)
        self._command_b = torch.tensor(cfg.commanded_body_velocity_m_s, device=self.device, dtype=dtype).repeat(
            self.num_envs, 1
        )
        if not bool(torch.isfinite(self._command_b).all().item()):
            raise ValueError("commanded_body_velocity_m_s must be finite")
        self._actions = torch.zeros(self.num_envs, ACTION_DIMENSION, device=self.device, dtype=dtype)
        self._previous_issued_v_final_b = torch.zeros(self.num_envs, 3, device=self.device, dtype=dtype)
        self._reward_previous_v_final_b = torch.zeros_like(self._previous_issued_v_final_b)
        self._episode_reset_pending = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._explicit_reset_pending = torch.ones_like(self._episode_reset_pending)
        self._policy = {
            "lidar_grid": torch.zeros(self.num_envs, 12, 59, 360, device=self.device, dtype=dtype),
            "state_vec": torch.zeros(self.num_envs, 52, device=self.device, dtype=dtype),
        }
        self._control: ControlCoreResult | None = None
        self._lee = None
        self._wrench: WrenchResult | None = None
        self._reward: RewardResult | None = None
        self._task: _TaskSnapshot | None = None
        self._evaluation_source_step = 0
        self._evaluation_ray_snapshot: _EvaluationRaySnapshot | None = None

    def _setup_scene(self) -> None:
        _spawn_static_ray_plane()
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["taslab"] = self._robot
        self.scene.clone_environments(copy_from_source=False)
        self._ray_caster = RayCaster(self.cfg.ray_caster)
        self.scene.sensors["mid360"] = self._ray_caster
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact"] = self._contact_sensor
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def step(self, action: torch.Tensor):
        """Reject device/dtype/domain errors before DirectRLEnv can move or mutate action."""
        self._validate_action(action)
        return super().step(action)

    def _validate_action(self, action: torch.Tensor) -> None:
        if not isinstance(action, torch.Tensor):
            raise TypeError("action must be a torch.Tensor")
        if action.shape != (self.num_envs, ACTION_DIMENSION):
            raise ValueError(f"action must have shape ({self.num_envs}, {ACTION_DIMENSION})")
        if action.device != self._actions.device:
            raise ValueError("action must already share the environment device")
        if action.dtype != self._actions.dtype:
            raise TypeError("action must share the environment floating dtype")
        if not bool(torch.isfinite(action).all().item()):
            raise ValueError("action must be finite")
        if bool(((action < 0.0) | (action > 1.0)).any().item()):
            raise ValueError("action must remain in the closed normalized [0,1] domain")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._validate_action(actions)
        staged_action = actions.clone()
        results = []
        for index, pipeline in enumerate(self._control_pipelines):
            selection = slice(index, index + 1)
            results.append(
                pipeline(
                    self._policy["lidar_grid"][selection],
                    self._policy["state_vec"][selection],
                    staged_action[selection],
                    POLICY_DT_S,
                    self._episode_reset_pending[selection],
                    self._explicit_reset_pending[selection],
                    self._robot.data.root_quat_w[selection],
                )
            )
        self._actions.copy_(staged_action)
        self._reward_previous_v_final_b.copy_(self._previous_issued_v_final_b)
        self._control = self._join_control_results(results)
        self._previous_issued_v_final_b.copy_(self._control.v_final_b)
        self._episode_reset_pending.zero_()
        self._explicit_reset_pending.zero_()

    @staticmethod
    def _join_control_results(results: list[ControlCoreResult]) -> ControlCoreResult:
        if not results:
            raise RuntimeError("at least one control result is required")
        join = lambda name: torch.cat([getattr(value, name) for value in results], dim=0)
        anchor_keys = tuple(results[0].anchor_metrics)
        ics_keys = tuple(results[0].ics_metrics)
        return ControlCoreResult(
            join("alpha"), join("v_corr_b"), join("v_gov_b"), join("v_final_b"), join("v_final_w"),
            {key: torch.cat([value.anchor_metrics[key] for value in results], dim=0) for key in anchor_keys},
            {key: torch.cat([value.ics_metrics[key] for value in results], dim=0) for key in ics_keys},
        )

    def _apply_action(self) -> None:
        if self._control is None:
            raise RuntimeError("control output is unavailable before action preprocessing")
        root_state_w = self._validated_root_state()
        self._lee = taslab_velocity_control(root_state_w, self._control.v_final_w)
        self._wrench = self._wrench_adapter.apply(self._robot, self._lee.normalized_rotor_commands)

    def _validated_root_state(self) -> torch.Tensor:
        root = self._robot.data.root_state_w
        if root.shape != (self.num_envs, 13) or root.dtype != self._actions.dtype or root.device != self._actions.device:
            raise RuntimeError("public articulation root_state_w contract mismatch")
        if not bool(torch.isfinite(root).all().item()):
            raise FloatingPointError("public articulation root_state_w is non-finite")
        return root

    def _read_sensor_producers(
        self, *, capture_evaluation_source: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ray_caster.update(UPDATE_PERIOD_S, force_recompute=True)
        hits_w = self._ray_caster.data.ray_hits_w
        parent_pos_w = self._ray_caster.data.pos_w
        parent_quat_w = self._ray_caster.data.quat_w
        if hits_w.shape != (self.num_envs, RAY_COUNT, 3):
            raise RuntimeError("MID360 public ray_hits_w shape mismatch")
        mount_pos = torch.tensor(self.cfg.ray_caster.offset.pos, device=self.device, dtype=hits_w.dtype)
        mount_quat = torch.tensor(self.cfg.ray_caster.offset.rot, device=self.device, dtype=hits_w.dtype)
        sensor_origins_w, _ = compose_mounted_sensor_pose(parent_pos_w, parent_quat_w, mount_pos, mount_quat)
        _, ranges_m, validity = ranges_and_validity(hits_w, sensor_origins_w)
        if capture_evaluation_source:
            mounted_starts = self._ray_caster.ray_starts
            mounted_directions = self._ray_caster.ray_directions
            expected_ray_shape = (self.num_envs, RAY_COUNT, 3)
            if mounted_starts.shape != expected_ray_shape or mounted_directions.shape != expected_ray_shape:
                raise RuntimeError("RayCaster public local ray geometry shape mismatch")
            ray_starts_w = quat_apply(
                parent_quat_w.repeat(1, RAY_COUNT), mounted_starts
            )
            ray_starts_w = ray_starts_w + parent_pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(
                parent_quat_w.repeat(1, RAY_COUNT), mounted_directions
            )
            self._evaluation_source_step += 1
            self._evaluation_ray_snapshot = _EvaluationRaySnapshot(
                ray_starts_w.detach().clone(),
                ray_directions_w.detach().clone(),
                hits_w.detach().clone(),
                validity.detach().clone(),
                torch.full(
                    (self.num_envs,), self._evaluation_source_step,
                    dtype=torch.long, device=self.device,
                ),
            )
        exact_clearance_m = ranges_m.min(dim=1, keepdim=True).values
        forces_w = self._contact_sensor.data.net_forces_w
        if forces_w is None or forces_w.ndim != 3 or forces_w.shape[0] != self.num_envs or forces_w.shape[2] != 3:
            raise RuntimeError("ContactSensor public net_forces_w contract mismatch")
        if not bool(torch.isfinite(forces_w).all().item()):
            raise FloatingPointError("ContactSensor public net_forces_w is non-finite")
        collision = (torch.linalg.vector_norm(forces_w, dim=-1) > self.cfg.contact_force_threshold_n).any(
            dim=1, keepdim=True
        )
        return ranges_m, validity, exact_clearance_m, collision

    def _refresh_task_snapshot(self) -> _TaskSnapshot:
        root = self._validated_root_state()
        actual_velocity_b = rotate_world_to_body_wxyz(root[:, 3:7], root[:, 7:10])
        _, _, clearance, collision = self._read_sensor_producers(capture_evaluation_source=True)
        termination = compute_termination(
            height_w_m=root[:, 2:3], collision=collision,
            elapsed_steps=self.episode_length_buf.unsqueeze(-1),
            config=TerminationConfig(maximum_episode_steps=self.max_episode_length),
        )
        snapshot = _TaskSnapshot(root.clone(), actual_velocity_b, clearance, collision, termination)
        self._task = snapshot
        return snapshot

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task = self._refresh_task_snapshot()
        self.extras["task"] = {
            "reason_code": task.termination.reason_code.detach().clone(),
            "done": task.termination.done.detach().clone(),
            "exact_clearance_m": task.exact_clearance_m.detach().clone(),
            "collision": task.collision.detach().clone(),
        }
        return task.termination.terminated[:, 0], task.termination.truncated[:, 0]

    def _get_rewards(self) -> torch.Tensor:
        if self._task is None or self._control is None:
            raise RuntimeError("same-state task/control outputs are unavailable")
        reward = compute_reward(
            v_cmd_b=self._command_b,
            v_gov_b=self._control.v_gov_b,
            v_final_b=self._control.v_final_b,
            previous_issued_v_final_b=self._reward_previous_v_final_b,
            actual_velocity_b=self._task.actual_velocity_b,
            anchor_loss=self._control.anchor_metrics["huber_loss"],
            anchor_active=self._control.anchor_metrics["active"],
            anchor_usable_fraction=self._control.anchor_metrics["usable_fraction"],
            ics_beta=self._control.ics_metrics["beta"],
            ics_emergency=self._control.ics_metrics["emergency"],
            exact_clearance_m=self._task.exact_clearance_m,
            collision=self._task.collision,
        )
        self._reward = reward
        self.extras["reward_components"] = {
            key: value.detach().clone() for key, value in reward.components.items()
        }
        self.extras["reward_diagnostics"] = {
            key: value.detach().clone() for key, value in reward.diagnostics.items()
        }
        self.extras["pre_reset"] = {
            "reset_mask": self.reset_buf.detach().clone(),
            "episode_length": self.episode_length_buf.detach().clone(),
            "root_state_w": self._robot.data.root_state_w.detach().clone(),
            "joint_pos": self._robot.data.joint_pos.detach().clone(),
            "joint_vel": self._robot.data.joint_vel.detach().clone(),
            "control": _detached_snapshot(self._control),
            "lee": _detached_snapshot(self._lee),
            "wrench": _detached_snapshot(self._wrench),
            "reward": _detached_snapshot(reward),
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
        return reward.total[:, 0].clone()

    def _get_observations(self) -> dict[str, Any]:
        root = self._validated_root_state()
        ranges_m, validity, clearance, collision = self._read_sensor_producers()
        policy_time = self.episode_length_buf.to(root.dtype).unsqueeze(-1) * POLICY_DT_S
        policy = self._history.update(
            ranges_m=ranges_m,
            validity=validity,
            sensor_frame_time_s=policy_time,
            policy_time_s=policy_time,
            body_angular_velocity_rad_s=self._robot.data.root_ang_vel_b,
            gravity_direction_body=self._robot.data.projected_gravity_b,
            commanded_body_velocity_m_s=self._command_b,
            previous_issued_body_velocity_m_s=self._previous_issued_v_final_b,
        )
        self._history_initialized = True
        if tuple(policy) != POLICY_KEYS:
            raise RuntimeError("policy branch must contain exactly the M4 keys")
        actual_velocity_b = rotate_world_to_body_wxyz(root[:, 3:7], root[:, 7:10])
        critic = build_critic_state(
            root_state_w=root,
            actual_velocity_b=actual_velocity_b,
            exact_clearance_m=clearance,
            collision=collision,
        ).privileged
        self._policy = policy
        return {"policy": {key: value.detach().clone() for key, value in policy.items()}, "critic": critic.detach().clone()}

    def _validate_reset_indices(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            indices = env_ids
        else:
            if isinstance(env_ids, (str, bytes)):
                raise TypeError("reset indices must be an integer sequence")
            indices = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        if indices.ndim != 1 or indices.dtype != torch.long or indices.device != self._actions.device:
            raise TypeError("reset indices must be rank-one torch.long on the environment device")
        if bool(((indices < 0) | (indices >= self.num_envs)).any().item()):
            raise IndexError("reset indices contain an out-of-range environment")
        if indices.unique().numel() != indices.numel():
            raise ValueError("reset indices must not contain duplicates")
        return indices

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        indices = self._validate_reset_indices(env_ids)
        if indices.numel() == 0:
            return
        root = self._robot.data.default_root_state[indices].clone()
        root[:, :3] += self.scene.env_origins[indices]
        joint_pos = self._robot.data.default_joint_pos[indices].clone()
        joint_vel = self._robot.data.default_joint_vel[indices].clone()
        self._wrench_adapter.reset(self._robot, indices)
        self._robot.write_root_pose_to_sim(root[:, :7], indices)
        self._robot.write_root_velocity_to_sim(root[:, 7:], indices)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=indices)
        self._robot.reset(indices)
        super()._reset_idx(indices)
        if self._history_initialized:
            self._history.reset(indices)
        else:
            self._history.reset()
        for index in indices.detach().cpu().tolist():
            self._control_pipelines[index] = ControlCorePipeline(ControlCoreConfig(use_b0=False))
        self._actions[indices] = 0.0
        self._previous_issued_v_final_b[indices] = 0.0
        self._reward_previous_v_final_b[indices] = 0.0
        self._episode_reset_pending[indices] = True
        self._explicit_reset_pending[indices] = True
        self._policy["lidar_grid"][indices] = 0.0
        self._policy["state_vec"][indices] = 0.0
        caches = (self._control, self._lee, self._wrench, self._reward, self._task)
        if indices.numel() == self.num_envs:
            self._control = None
            self._lee = None
            self._wrench = None
            self._reward = None
            self._task = None
        else:
            for cache in caches:
                if cache is not None:
                    _zero_selected_batch(cache, indices, self.num_envs)


__all__ = [
    "ACTION_DIMENSION", "CONTACT_FORCE_THRESHOLD_N", "DECIMATION", "DEVICE",
    "OBSERVATION_SPACE", "PHYSICS_DT_S", "POLICY_DT_S", "POLICY_KEYS",
    "TaslabDirectEnv", "TaslabDirectEnvCfg",
]
