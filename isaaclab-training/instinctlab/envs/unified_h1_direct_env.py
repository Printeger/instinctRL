"""Live environment selection for the frozen unified-h1-v1 method.

The accepted v1 environment remains unchanged.  The workload stages ``c_t``
and ``c_(t+1)`` before every step so reward and terminal bootstrap use the
commands from the frozen single-step contract.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib
import json
from typing import Any

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from pxr import UsdGeom, UsdPhysics

from instinctlab.task.critic_h1 import CRITIC_H1_DIMENSION, build_critic_h1_state
from instinctlab.task.rewards_h1 import compute_reward_h1
from instinctlab.task.scene_h1 import H1_RAY_MESH_PATH, H1_RAY_SCENE_PATH, combined_scene_triangles
from instinctlab.control.pipeline import ControlCoreResult
from instinctlab.control.lee_velocity_controller import rotate_world_to_body_wxyz
from instinctlab.control.rotor_actuation import ROTOR_NAMES
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
    # Isaac Lab's configclass turns mutable defaults into instance-only
    # dataclass fields.  Read the inherited sensor configuration from a fresh
    # base instance instead of assuming that the field remains on the class.
    ray_caster = TaslabDirectEnvCfg().ray_caster.replace(mesh_prim_paths=[H1_RAY_MESH_PATH])
    task_contract_version: str = "unified-h1-v1"


class UnifiedH1DirectEnv(TaslabDirectEnv):
    """H1 reward, critic and mixed-stage reset wiring."""

    cfg: UnifiedH1DirectEnvCfg

    def __init__(self, cfg: UnifiedH1DirectEnvCfg, render_mode: str | None = None, **kwargs):
        self._h1_global_policy_step = 0
        self._h1_next_command_b: torch.Tensor | None = None
        self._b0_command_b: torch.Tensor | None = None
        self._h1_evaluation_case: str | None = None
        # The enhanced-B0 protocol is dormant in every normal environment.  It
        # becomes observational only after an exact, one-shot registration.
        self._enhanced_b0_observer: Callable[[list[Mapping[str, Any]]], None] | None = None
        self._enhanced_b0_registered = False
        self._enhanced_b0_active = False
        self._enhanced_b0_coordinate: tuple[str, int, int] | None = None
        self._enhanced_b0_batches = 0
        self._enhanced_b0_previous_system: dict[tuple[str, int], Mapping[str, Any]] = {}
        self._enhanced_b0_previous_velocity: dict[tuple[str, int], tuple[float, float, float]] = {}
        self._enhanced_b0_accounting: Mapping[str, Callable[..., None]] | None = None
        self._enhanced_b0_episode_ids: dict[tuple[str, int], int] = {}
        self._enhanced_b0_reset_boundaries: dict[tuple[str, int], Mapping[str, Any]] = {}
        self._enhanced_b0_last_snapshots: dict[tuple[str, int], Mapping[str, Any]] = {}
        self._enhanced_b0_explicit_reset = False
        self._enhanced_b0_cache_generation = 0
        self._enhanced_b0_cache_unavailable: set[tuple[str, int]] = set()
        super().__init__(cfg, render_mode, **kwargs)
        # Bind Isaac Sim's own timeline index at the construction boundary,
        # before this environment performs any explicit reset or policy step.
        # An unavailable/non-integral counter is retained as None so the future
        # workload fails closed instead of claiming zero initialization cost.
        timeline_index = getattr(self.sim, "current_time_step_index", None)
        self.h1_initialization_physics_ticks = timeline_index if type(timeline_index) is int else None
        self.h1_initialization_tick_source = "SimulationContext.current_time_step_index"

    @staticmethod
    def _diagnostic_tensor_digest(value: Any) -> str:
        """Hash a detached tensor without exposing or retaining its storage."""
        digest = hashlib.sha256()
        def update(item: Any) -> None:
            if isinstance(item, torch.Tensor):
                snapshot = item.detach().contiguous().cpu()
                digest.update(json.dumps({"shape": list(snapshot.shape), "dtype": str(snapshot.dtype)}, sort_keys=True).encode())
                digest.update(snapshot.numpy().tobytes())
            elif isinstance(item, Mapping):
                for key, nested in item.items(): digest.update(repr(key).encode()); update(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item: update(nested)
            else: digest.update(repr(item).encode())
        update(value)
        return digest.hexdigest()

    def read_only_diagnostic_state_fingerprints(self) -> OrderedDict[str, Mapping[str, Any]]:
        """Return JSON-only fingerprints for the exact protected live state."""
        digest = self._diagnostic_tensor_digest
        control = [self._actions, self._command_b, self._previous_issued_v_final_b]
        dynamics = [self._robot.data.root_state_w, self._robot.data.body_state_w,
                    self._robot.data.joint_pos, self._robot.data.joint_vel]
        pending_force, pending_torque = self._wrench_adapter.pending_wrenches()
        reset = [self.reset_buf, self.episode_length_buf, self._episode_reset_pending,
                 self._explicit_reset_pending]
        actor = [self._policy["state_vec"], self._policy["lidar_grid"],
                 getattr(self._history, "_state", None)]
        timeline = getattr(self.sim, "current_time_step_index", None)
        return OrderedDict((
            ("control", {"digest": digest(control)}),
            ("dynamics", {"digest": digest([*dynamics, pending_force, pending_torque])}),
            ("rng", {"torch_cpu": digest(torch.random.get_rng_state())}),
            ("reset", {"digest": digest(reset)}),
            ("actor_history", {"digest": digest(actor)}),
            ("simulator", {"time_step_index": timeline if type(timeline) is int else "UNAVAILABLE"}),
        ))

    def register_read_only_post_physics_observer(
        self, callback: Callable[[list[Mapping[str, Any]]], None], **binding: Any
    ) -> dict[str, Any]:
        """Bind once after ``scene.update`` and before DirectRLEnv reset logic."""
        expected = {"ordering": "after_physics_before_reset", "exactly_once": True, "lanes": 32}
        if binding != expected or not callable(callback):
            raise ValueError("enhanced B0 observer binding changed")
        if self._enhanced_b0_registered:
            raise RuntimeError("enhanced B0 observer may be registered exactly once")
        if self.num_envs != 32:
            raise RuntimeError("enhanced B0 observer requires exactly 32 lanes")
        original_update = self.scene.update
        def update_then_observe(dt: float) -> None:
            original_update(dt)
            if self._enhanced_b0_active:
                if self._enhanced_b0_coordinate is None:
                    raise RuntimeError("enhanced B0 physics coordinate is unavailable")
                if self._lee is None or self._wrench is None:
                    raise RuntimeError("enhanced B0 caches were not rebuilt by a real control/physics step")
                callback(self._enhanced_b0_physics_rows())
                self._enhanced_b0_batches += 1
                case, policy_step, physics_step = self._enhanced_b0_coordinate
                self._enhanced_b0_coordinate = (case, policy_step, physics_step + 1)
        self.scene.update = update_then_observe
        self._enhanced_b0_observer = callback
        self._enhanced_b0_registered = True
        return {"boundary": "after_physics_before_reset", "ordering_unchanged": True,
                "read_only": True, "exactly_once": True, "lanes": 32}

    def configure_enhanced_b0_execution_accounting(
        self, accounting: Mapping[str, Callable[..., None]]
    ) -> None:
        """Bind the workload-owned pre-action guards before the one live run."""
        required = ("before_reset", "after_reset", "before_step", "after_automatic_reset", "after_step")
        if self._enhanced_b0_accounting is not None or tuple(accounting) != required:
            raise RuntimeError("enhanced B0 execution accounting binding changed")
        if any(not callable(accounting[name]) for name in required):
            raise TypeError("enhanced B0 execution accounting callbacks must be callable")
        self._enhanced_b0_accounting = dict(accounting)

    def enhanced_b0_read_only_protocol(self) -> dict[str, Any]:
        """Expose only the adapter's exact supported production protocol."""
        return {"identity": "post-physics-before-reset/v1",
                "register": self.register_read_only_post_physics_observer,
                "state_probe": self.read_only_diagnostic_state_fingerprints,
                "accounting": self.configure_enhanced_b0_execution_accounting,
                "run": self.run_frozen_enhanced_b0_zts}

    @staticmethod
    def _cross3(a: Sequence[float], b: Sequence[float]) -> list[float]:
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

    @staticmethod
    def _tolist(value: torch.Tensor) -> list[Any]:
        return value.detach().cpu().tolist()

    def _enhanced_b0_physics_rows(self) -> list[Mapping[str, Any]]:
        """Snapshot articulation/controller/rotor state; never synthesize absent wrenches."""
        if self._enhanced_b0_coordinate is None or self._lee is None or self._wrench is None:
            raise RuntimeError("enhanced B0 state is unavailable at the post-physics boundary")
        case, policy_step, physics_step = self._enhanced_b0_coordinate
        if physics_step >= 20:
            raise RuntimeError("more than one callback was observed for a physics integration")
        data = self._robot.data
        required = ("body_state_w", "body_com_pose_b", "default_mass", "default_inertia")
        if any(not isinstance(getattr(data, name, None), torch.Tensor) for name in required):
            raise RuntimeError("public articulation body mechanics are unavailable")
        body = data.body_state_w
        com_pose = data.body_com_pose_b
        mass = data.default_mass
        inertia = data.default_inertia
        if body.shape[:2] != (32, 5) or com_pose.shape[:2] != (32, 5) or mass.shape[-1] != 5:
            raise RuntimeError("public articulation body topology changed")
        root = data.root_state_w
        actual_b = rotate_world_to_body_wxyz(root[:, 3:7], root[:, 7:10])
        dt = float(self.cfg.sim.dt)
        rows: list[Mapping[str, Any]] = []
        # Build a complete 32-lane batch against private copies.  A snapshot
        # failure must not consume a reset boundary or partially advance lanes.
        next_previous_system = dict(self._enhanced_b0_previous_system)
        next_previous_velocity = dict(self._enhanced_b0_previous_velocity)
        next_episode_ids = dict(self._enhanced_b0_episode_ids)
        next_last_snapshots = dict(self._enhanced_b0_last_snapshots)
        consumed_boundaries: list[tuple[str, int]] = []
        unavailable = {"availability": "UNAVAILABLE"}
        for lane in range(32):
            bodies = []
            total_mass = 0.0
            weighted = [0.0, 0.0, 0.0]
            momentum = [0.0, 0.0, 0.0]
            body_coms = []
            for index, name in enumerate(self._robot.body_names):
                state = self._tolist(body[lane, index]); local = self._tolist(com_pose[lane, index])
                m = float(mass[lane, index] if mass.ndim == 2 else mass[index])
                q = body[lane, index, 3:7]
                local_world = quat_apply(q.unsqueeze(0), com_pose[lane, index, :3].unsqueeze(0))[0]
                com = self._tolist(body[lane, index, :3] + local_world)
                com_velocity = self._tolist(body[lane, index, 7:10] + torch.linalg.cross(body[lane, index, 10:13], local_world))
                tensor_inertia = inertia[lane, index] if inertia.ndim == 3 else inertia[index]
                matrix = self._tolist(tensor_inertia.reshape(3, 3))
                bodies.append({"name": str(name), "mass_kg": m, "local_com_m": local[:3],
                    "local_com_quaternion_wxyz": local[3:7], "inertia_at_com_body_kg_m2": matrix,
                    "world_position_m": state[:3], "world_quaternion_wxyz": state[3:7],
                    "world_linear_velocity_m_s": state[7:10], "world_angular_velocity_rad_s": state[10:13]})
                total_mass += m; body_coms.append((m, com, com_velocity))
                for axis in range(3): weighted[axis] += m*com[axis]; momentum[axis] += m*com_velocity[axis]
            combined = [value/total_mass for value in weighted]
            angular = [0.0, 0.0, 0.0]
            # Preserve the full body-derived angular momentum schema.  The
            # analyzer independently recomputes this from the raw body rows.
            for index, (m, com, velocity) in enumerate(body_coms):
                orbital = self._cross3([com[i]-combined[i] for i in range(3)], [m*x for x in velocity])
                omega = body[lane, index, 10:13]
                q = body[lane, index, 3:7]
                w, x, y, z = q.unbind()
                rotation = torch.stack((
                    torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w))),
                    torch.stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w))),
                    torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))),
                ))
                local_inertia = inertia[lane, index] if inertia.ndim == 3 else inertia[index]
                world_inertia = rotation @ local_inertia.reshape(3, 3) @ rotation.T
                spin = self._tolist(world_inertia @ omega)
                angular = [angular[i]+orbital[i]+spin[i] for i in range(3)]
            system = {"total_mass_kg": total_mass, "combined_com_world_m": combined,
                "linear_momentum_kg_m_s": momentum, "angular_momentum_at_com_kg_m2_s": angular}
            key = (case, lane); previous = next_previous_system.get(key)
            if previous is None:
                inferred_force = [0.0, 0.0, 0.0]; inferred_torque = [0.0, 0.0, 0.0]
            else:
                inferred_force = [(momentum[i]-previous["linear_momentum_kg_m_s"][i])/dt for i in range(3)]
                inferred_torque = [(angular[i]-previous["angular_momentum_at_com_kg_m2_s"][i])/dt for i in range(3)]
            next_previous_system[key] = system
            rotor_rows=[]; rotor_force=[0.0,0.0,0.0]; rotor_torque=[0.0,0.0,0.0]
            rotor = self._wrench.rotor
            for rotor_index, name in enumerate(ROTOR_NAMES):
                body_index=rotor_index+1; direction=self._tolist(quat_apply(body[lane,body_index,3:7].unsqueeze(0), torch.tensor([[0.,0.,1.]],device=self.device,dtype=body.dtype))[0])
                point=self._tolist(body[lane,body_index,:3]); thrust=float(rotor.thrusts_n[lane,rotor_index]); reaction=float(rotor.reaction_moments_nm[lane,rotor_index])
                force=[thrust*x for x in direction]; arm=[point[i]-combined[i] for i in range(3)]; torque=self._cross3(arm,force)
                torque=[torque[i]+reaction*direction[i] for i in range(3)]
                rotor_force=[rotor_force[i]+force[i] for i in range(3)]; rotor_torque=[rotor_torque[i]+torque[i] for i in range(3)]
                rotor_rows.append({"name":name,"command":float(rotor.normalized_commands[lane,rotor_index]),"target_throttle":float(rotor.target_throttle[lane,rotor_index]),"actual_throttle":float(rotor.throttle[lane,rotor_index]),"thrust_n":thrust,"reaction_moment_nm":reaction,"application_point_world_m":point,"direction_world":direction})
            gravity_force=[0.0,0.0,-total_mass*9.81]; gravity_torque=[0.0,0.0,0.0]
            for m, com, _ in body_coms:
                part=self._cross3([com[i]-combined[i] for i in range(3)],[0.0,0.0,-m*9.81]); gravity_torque=[gravity_torque[i]+part[i] for i in range(3)]
            velocity=self._tolist(root[lane,7:10]); prior_velocity=next_previous_velocity.get(key)
            acceleration=[0.0,0.0,0.0] if prior_velocity is None else [(velocity[i]-prior_velocity[i])/dt for i in range(3)]
            next_previous_velocity[key]=tuple(velocity)
            available=lambda force,torque:{"availability":"AVAILABLE","force_n":force,"torque_at_com_nm":torque}
            controller={"target_velocity_b_m_s":self._tolist(self._command_b[lane]),"velocity_error_b_m_s":self._tolist(self._command_b[lane]-actual_b[lane]),"desired_rotation_w":self._tolist(self._lee.desired_rotation_w[lane]),"collective_n":float(self._lee.collective_thrust_n[lane,0])}
            current_snapshot={"controller":controller,"rotors":rotor_rows}
            boundary=self._enhanced_b0_reset_boundaries.get(key)
            if key in self._enhanced_b0_cache_unavailable:
                raise RuntimeError("post-reset cache remains explicitly unavailable")
            if boundary is not None and self._enhanced_b0_cache_generation <= int(boundary["cache_generation"]):
                raise RuntimeError("post-reset cache was not rebuilt by a new control/physics step")
            boundary=self._enhanced_b0_reset_boundaries.get(key)
            if boundary is None:
                episode_id=next_episode_ids.get(key, 0)
                reset={"mask":False,"reason":"","initial_boundary":False,"last_pre_reset_state":None,"post_reset_states":[]}
            else:
                episode_id=int(boundary["episode_id"])
                reset={"mask":True,"reason":str(boundary["reason"]),"initial_boundary":bool(boundary["initial_boundary"]),"last_pre_reset_state":boundary["last_pre_reset_state"],"post_reset_states":[current_snapshot]}
                next_episode_ids[key]=episode_id+1; consumed_boundaries.append(key)
            next_last_snapshots[key]=current_snapshot
            rows.append(OrderedDict((("schema_version","instinctrl-enhanced-b0-physics-row/1"),("experiment_id","UNIFIED-H1-R5-STAGE1-B0-RESET-AVAILABILITY-CLEANUP-R1-20260929"),("case",case),("policy_step",policy_step),("physics_step",physics_step),("lane",lane),("episode_id",episode_id),("reset",reset),("root_state",{"pose":self._tolist(root[lane,:7]),"linear_velocity":velocity,"angular_velocity":self._tolist(root[lane,10:13])}),("system",system),("bodies",bodies),("controller",controller),("rotors",rotor_rows),("wrenches",OrderedDict((("gravity",available(gravity_force,gravity_torque)),("rotor",available(rotor_force,rotor_torque)),("drag_aero",dict(unavailable)),("contact",dict(unavailable)),("joint_constraint",dict(unavailable)),("other",dict(unavailable))))),("momentum",{"dt_s":dt,"inferred_wrench":{"force_n":inferred_force,"torque_at_com_nm":inferred_torque}}),("response",{"position_world_m":self._tolist(root[lane,:3]),"velocity_world_m_s":velocity,"acceleration_world_m_s2":acceleration}))))
        self._enhanced_b0_previous_system = next_previous_system
        self._enhanced_b0_previous_velocity = next_previous_velocity
        self._enhanced_b0_episode_ids = next_episode_ids
        self._enhanced_b0_last_snapshots = next_last_snapshots
        for key in consumed_boundaries:
            self._enhanced_b0_reset_boundaries.pop(key)
        return rows

    def run_frozen_enhanced_b0_zts(self, **parameters: Any) -> dict[str, Any]:
        expected = {"seed": 20260909, "cases": ("Z", "T", "S"),
                    "target_velocity_b_m_s": (0.0, 0.0, 0.0), "retries": 0}
        if parameters != expected or not self._enhanced_b0_registered or self._enhanced_b0_active or self._enhanced_b0_accounting is None:
            raise RuntimeError("frozen enhanced B0 invocation changed or observer/accounting absent")
        horizons = {"Z": 200, "T": 100, "S": 330}
        zero_command = torch.zeros_like(self._command_b)
        zero_action = torch.zeros_like(self._actions)
        all_lanes = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        accounting = self._enhanced_b0_accounting
        self._enhanced_b0_active = True
        try:
            for case in ("Z", "T", "S"):
                self.set_h1_evaluation_case(case)
                accounting["before_reset"](case)
                self._enhanced_b0_explicit_reset = True
                try:
                    if case == "Z": self.reset(seed=20260909)
                    else: self._reset_idx(all_lanes)
                finally:
                    self._enhanced_b0_explicit_reset = False
                for policy_step in range(horizons[case]):
                    accounting["before_step"](case, policy_step)
                    self._enhanced_b0_coordinate = (case, policy_step, 0)
                    self.prepare_b0_step(command_b=zero_command, next_command_b=zero_command)
                    _, _, terminated, truncated, _ = self.step(zero_action)
                    accounting["after_step"](case, policy_step, 0)
                    if self._enhanced_b0_coordinate != (case, policy_step, 20):
                        raise RuntimeError("enhanced B0 callback count changed within policy step")
        finally:
            self._enhanced_b0_active = False
            self._enhanced_b0_coordinate = None
            self.set_h1_evaluation_case(None)
        if self._enhanced_b0_batches != 12600:
            raise RuntimeError("enhanced B0 callback coverage is not exact")
        return {"physics_callback_batches": 12600, "physics_rows": 403200,
                "control_physics_reset_observation_order": "UNCHANGED"}

    def _setup_scene(self) -> None:
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["taslab"] = self._robot
        self.scene.clone_environments(copy_from_source=False)
        # InteractiveScene initializes env_origins during cloning when
        # replicate_physics=True.  Author the global translated H1 geometry
        # only after that lifecycle boundary, while keeping the source robot
        # available before cloning and the mesh available before RayCaster.
        _spawn_unified_h1_combined_scene(self.scene.env_origins)
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

    def _apply_action(self) -> None:
        """Mark only controller/rotor caches rebuilt by the real action path."""
        super()._apply_action()
        if self._enhanced_b0_active:
            if self._lee is None or self._wrench is None:
                raise RuntimeError("enhanced B0 action path did not rebuild controller/rotor caches")
            self._enhanced_b0_cache_generation += 1
            self._enhanced_b0_cache_unavailable.clear()

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

    def _enhanced_b0_snapshot(self, lane: int) -> Mapping[str, Any]:
        if self._lee is None or self._wrench is None:
            raise RuntimeError("enhanced B0 controller/rotor reset state is unavailable")
        body=self._robot.data.body_state_w; root=self._robot.data.root_state_w
        actual_b=rotate_world_to_body_wxyz(root[:,3:7],root[:,7:10])
        controller={"target_velocity_b_m_s":self._tolist(self._command_b[lane]),"velocity_error_b_m_s":self._tolist(self._command_b[lane]-actual_b[lane]),"desired_rotation_w":self._tolist(self._lee.desired_rotation_w[lane]),"collective_n":float(self._lee.collective_thrust_n[lane,0])}
        rotor=self._wrench.rotor; rotors=[]
        for rotor_index,name in enumerate(ROTOR_NAMES):
            body_index=rotor_index+1
            direction=self._tolist(quat_apply(body[lane,body_index,3:7].unsqueeze(0),torch.tensor([[0.,0.,1.]],device=self.device,dtype=body.dtype))[0])
            rotors.append({"name":name,"command":float(rotor.normalized_commands[lane,rotor_index]),"target_throttle":float(rotor.target_throttle[lane,rotor_index]),"actual_throttle":float(rotor.throttle[lane,rotor_index]),"thrust_n":float(rotor.thrusts_n[lane,rotor_index]),"reaction_moment_nm":float(rotor.reaction_moments_nm[lane,rotor_index]),"application_point_world_m":self._tolist(body[lane,body_index,:3]),"direction_world":direction})
        return {"controller":controller,"rotors":rotors}

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        indices=self._validate_reset_indices(env_ids)
        lanes=[int(lane) for lane in indices.detach().cpu().tolist()]
        capture = bool(lanes) and self._enhanced_b0_active and self._h1_evaluation_case in {"Z","T","S"}
        before={}
        if capture:
            # Resolve the fallback explicitly: dict.get(default) would evaluate
            # a cache snapshot even when a valid prior row already exists.
            for lane in lanes:
                key=(self._h1_evaluation_case,lane)
                if key in self._enhanced_b0_reset_boundaries:
                    raise RuntimeError("reset boundary is already awaiting a rebuilt cache")
                last=self._enhanced_b0_last_snapshots.get(key)
                initial = not self._enhanced_b0_last_snapshots and self._enhanced_b0_cache_generation == 0
                if last is None and not initial:
                    last=self._enhanced_b0_snapshot(lane)
                if not initial and (not isinstance(last,Mapping) or set(last)!={"controller","rotors"}):
                    raise RuntimeError("last valid pre-reset cache snapshot is unavailable")
                before[lane]=(last,initial)
            # Invalidate first.  The inherited reset clears full-lane caches,
            # and no exception may reopen them or lose the lane/episode map.
            reason="explicit_case_reset" if self._enhanced_b0_explicit_reset else "automatic_environment_reset"
            for lane,(last,initial) in before.items():
                key=(self._h1_evaluation_case,lane)
                self._enhanced_b0_previous_system.pop(key,None); self._enhanced_b0_previous_velocity.pop(key,None)
                self._enhanced_b0_cache_unavailable.add(key)
                self._enhanced_b0_reset_boundaries[key]={"episode_id":self._enhanced_b0_episode_ids.get(key,0),"reason":reason,"initial_boundary":initial,"last_pre_reset_state":last,"cache_generation":self._enhanced_b0_cache_generation}
        try:
            self._reset_idx_impl(indices)
        finally:
            # Entering the physical reset path settles its reset exactly once,
            # at this boundary, even if reset internals or step diagnostics fail.
            if capture:
                if self._enhanced_b0_accounting is None:
                    raise RuntimeError("enhanced B0 reset accounting is unavailable")
                if self._enhanced_b0_explicit_reset:
                    self._enhanced_b0_accounting["after_reset"](self._h1_evaluation_case)
                else:
                    case, policy_step, _ = self._enhanced_b0_coordinate
                    self._enhanced_b0_accounting["after_automatic_reset"](
                        case, policy_step, len(lanes))

    def _reset_idx_impl(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
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
