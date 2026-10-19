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
        self._h1_residual_neutrality_weight = 0.0
        self._h1_pure_hover_training = False
        self._h1_anchor_weight = 4.0
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

    def _diagnostic_cached_actor_digest(self, value: Any) -> str:
        """Reuse leaf SHA only after exact device byte equality to private clones.

        Actor history is unchanged between physics substeps. Copying/hashing its
        entire lidar history twice per substep transfers hundreds of GB.
        Equality is over uint8 views, including signed zero and NaN payloads;
        tensor version counters or approximate checksums are insufficient.
        """
        cache = getattr(self, "_enhanced_b0_actor_digest_cache", {})
        next_cache = {}; digest = hashlib.sha256()
        def update(item: Any, path: tuple) -> None:
            if isinstance(item, torch.Tensor):
                current = item.detach().contiguous()
                metadata = {"shape":list(current.shape),"dtype":str(current.dtype)}
                old = cache.get(path)
                if (old is not None and old[0] == metadata and old[1].device == current.device
                        and torch.equal(old[1].view(torch.uint8), current.view(torch.uint8))):
                    entry = old
                else:
                    snapshot = current.clone()
                    leaf = hashlib.sha256(snapshot.cpu().numpy().tobytes()).hexdigest()
                    entry = (metadata, snapshot, leaf)
                next_cache[path] = entry
                digest.update(json.dumps(metadata,sort_keys=True).encode()); digest.update(entry[2].encode())
            elif isinstance(item, Mapping):
                for key,nested in item.items():
                    digest.update(repr(key).encode()); update(nested,(*path,("key",key)))
            elif isinstance(item,(list,tuple)):
                for index,nested in enumerate(item): update(nested,(*path,("index",index)))
            else: digest.update(repr(item).encode())
        update(value,())
        self._enhanced_b0_actor_digest_cache = next_cache
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
            ("actor_history", {"digest": self._diagnostic_cached_actor_digest(actor),
                "algorithm":"sha256-structure-leafsha256-exact-cached-bytes/1"}),
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
        required = ("body_link_state_w", "root_link_state_w", "root_state_w", "body_com_pose_b")
        if any(not isinstance(getattr(data, name, None), torch.Tensor) for name in required):
            raise RuntimeError("public articulation body mechanics are unavailable")
        # body_state_w already stores COM velocity; reconstruct_com needs link velocity.
        body = data.body_link_state_w
        com_pose = data.body_com_pose_b
        view = self._robot.root_physx_view
        # Measure the current controlled plant, rather than initialization defaults.
        mass = view.get_masses()
        inertia = view.get_inertias()
        if not isinstance(mass, torch.Tensor) or not isinstance(inertia, torch.Tensor):
            raise RuntimeError("actual articulation mass/inertia are unavailable")
        if body.shape[:2] != (32, 5) or com_pose.shape[:2] != (32, 5) or mass.shape[-1] != 5:
            raise RuntimeError("public articulation body topology changed")
        # Response pose/velocity refer to the same link point. Controller keeps
        # its existing root COM velocity feedback; this observer changes neither.
        root = data.root_link_state_w
        control_root = data.root_state_w
        actual_b = rotate_world_to_body_wxyz(control_root[:, 3:7], control_root[:, 7:10])
        # Detach each small batch once, rather than synchronizing hundreds of
        # individual GPU scalars per integration. The serialized source values
        # and subsequent double-precision derivations are unchanged.
        body_values=self._tolist(body);com_values=self._tolist(com_pose)
        mass_values=self._tolist(mass)
        inertia_values=self._tolist(inertia.reshape(*inertia.shape[:-1],3,3))
        root_values=self._tolist(root);control_root_values=self._tolist(control_root)
        command_values=self._tolist(self._command_b)
        error_values=self._tolist(self._command_b-actual_b)
        desired_values=self._tolist(self._lee.desired_rotation_w)
        collective_values=self._tolist(self._lee.collective_thrust_n)
        directions_values=self._tolist(quat_apply(body[:,:,3:7].reshape(-1,4),
            torch.tensor([0.,0.,1.],device=self.device,dtype=body.dtype).expand(160,3)).reshape(32,5,3))
        rotor=self._wrench.rotor
        rotor_values={name:self._tolist(getattr(rotor,name)) for name in
            ('normalized_commands','target_throttle','throttle','thrusts_n','reaction_moments_nm')}
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
            for index, name in enumerate(self._robot.body_names):
                state = body_values[lane][index]; local = com_values[lane][index]
                m = float(mass_values[lane][index] if mass.ndim == 2 else mass_values[index])
                matrix = inertia_values[lane][index] if inertia.ndim == 3 else inertia_values[index]
                bodies.append({"name": str(name), "mass_kg": m, "local_com_m": local[:3],
                    "local_com_quaternion_wxyz": local[3:7],
                    "inertia_at_com_body_kg_m2": matrix,
                    "world_position_m": state[:3], "world_quaternion_wxyz": state[3:7],
                    "world_linear_velocity_m_s": state[7:10], "world_angular_velocity_rad_s": state[10:13]})
            # Derive diagnostics from the same serialized primitives as the
            # verifier. Float32 world-position addition at 100m lane origins
            # otherwise disagrees with independent double-precision rebuilding.
            # These helpers own no simulator/controller/RNG and run only here.
            from training_workloads.unified_h1.b0_enhanced import reconstruct_com, rotor_wrench, quaternion_rotate
            try:
                system = reconstruct_com(bodies)
            except Exception as error:
                # This is an unvalidated input artifact, never a physics row.
                # Do not advance observer/reset state when deriving it fails.
                from training_workloads.unified_h1.b0_enhanced import safe_failure_primitives
                error.measurement_failure = safe_failure_primitives({
                    "schema_version":"instinctrl-enhanced-b0-measurement-failure/1",
                    "status":"UNVALIDATED", "normal_physics_row":False,
                    "coordinate":{"case":case,"policy_step":policy_step,"physics_step":physics_step,"lane":lane},
                    "failure":getattr(error,"measurement_failure",{"failure_branch":"BODY_DERIVATION","message":str(error)}),
                    "bodies":bodies,
                    "getter_sources":{"mass":"Articulation.root_physx_view.get_masses","inertia":"Articulation.root_physx_view.get_inertias","body":"data.body_link_state_w","local_com":"data.body_com_pose_b"},
                    "tensor_metadata":{name:{"dtype":str(value.dtype),"shape":list(value.shape)} for name,value in
                        (("mass",mass),("inertia",inertia),("body",body),("local_com",com_pose))},
                    "root_link_state":root_values[lane],"controller_root_state":control_root_values[lane],
                })
                raise
            total_mass = system["total_mass_kg"]
            combined = system["combined_com_world_m"]
            momentum = system["linear_momentum_kg_m_s"]
            angular = system["angular_momentum_at_com_kg_m2_s"]
            key = (case, lane); previous = next_previous_system.get(key)
            if previous is None:
                inferred_force = [0.0, 0.0, 0.0]; inferred_torque = [0.0, 0.0, 0.0]
            else:
                inferred_force = [(momentum[i]-previous["linear_momentum_kg_m_s"][i])/dt for i in range(3)]
                inferred_torque = [(angular[i]-previous["angular_momentum_at_com_kg_m2_s"][i])/dt for i in range(3)]
            next_previous_system[key] = system
            rotor_rows=[]
            reaction_direction=quaternion_rotate(bodies[0]["world_quaternion_wxyz"],(0.,0.,1.))
            for rotor_index, name in enumerate(ROTOR_NAMES):
                body_index=rotor_index+1; direction=directions_values[lane][body_index]
                point=body_values[lane][body_index][:3]; thrust=rotor_values['thrusts_n'][lane][rotor_index]; reaction=rotor_values['reaction_moments_nm'][lane][rotor_index]
                rotor_rows.append({"name":name,"command":rotor_values['normalized_commands'][lane][rotor_index],"target_throttle":rotor_values['target_throttle'][lane][rotor_index],"actual_throttle":rotor_values['throttle'][lane][rotor_index],"thrust_n":thrust,"reaction_moment_nm":reaction,"application_point_world_m":point,"direction_world":direction,"reaction_body":bodies[0]["name"],"reaction_direction_world":reaction_direction})
            rotor_total = rotor_wrench(rotor_rows, combined)
            rotor_force = rotor_total["force_n"]; rotor_torque = rotor_total["torque_at_com_nm"]
            gravity_force=[0.0,0.0,-total_mass*9.81]; gravity_torque=[0.0,0.0,0.0]
            for entry in bodies:
                m=entry["mass_kg"]
                offset=quaternion_rotate(entry["world_quaternion_wxyz"],entry["local_com_m"])
                com=[entry["world_position_m"][i]+offset[i] for i in range(3)]
                part=self._cross3([com[i]-combined[i] for i in range(3)],[0.0,0.0,-m*9.81])
                gravity_torque=[gravity_torque[i]+part[i] for i in range(3)]
            velocity=root_values[lane][7:10]; prior_velocity=next_previous_velocity.get(key)
            acceleration=[0.0,0.0,0.0] if prior_velocity is None else [(velocity[i]-prior_velocity[i])/dt for i in range(3)]
            next_previous_velocity[key]=tuple(velocity)
            available=lambda force,torque:{"availability":"AVAILABLE","force_n":force,"torque_at_com_nm":torque}
            controller={"target_velocity_b_m_s":command_values[lane],"velocity_error_b_m_s":error_values[lane],"desired_rotation_w":desired_values[lane],"collective_n":collective_values[lane][0]}
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
            rows.append(OrderedDict((("schema_version","instinctrl-enhanced-b0-physics-row/2"),("experiment_id","UNIFIED-H1-R5-READINESS-B0-PLANT-R2-20261012"),("case",case),("policy_step",policy_step),("physics_step",physics_step),("lane",lane),("episode_id",episode_id),("reset",reset),("root_state",{"pose":root_values[lane][:7],"linear_velocity":velocity,"angular_velocity":root_values[lane][10:13]}),("system",system),("bodies",bodies),("controller",controller),("rotors",rotor_rows),("wrenches",OrderedDict((("gravity",available(gravity_force,gravity_torque)),("rotor",available(rotor_force,rotor_torque)),("drag_aero",dict(unavailable)),("contact",dict(unavailable)),("joint_constraint",dict(unavailable)),("other",dict(unavailable))))),("momentum",{"dt_s":dt,"inferred_wrench":{"force_n":inferred_force,"torque_at_com_nm":inferred_torque}}),("response",{"position_world_m":root_values[lane][:3],"velocity_world_m_s":velocity,"acceleration_world_m_s2":acceleration}))))
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
        accounting = self._enhanced_b0_accounting
        self._enhanced_b0_active = True
        try:
            for case in ("Z", "T", "S"):
                self.set_h1_evaluation_case(case)
                accounting["before_reset"](case)
                self._enhanced_b0_explicit_reset = True
                try:
                    # Public reset flushes articulation kinematics via forward
                    # and rebuilds actual sensor/history observations. A bare
                    # _reset_idx leaves the cleared policy invalid at first
                    # reward. Only Z seeds; later resets do not reseed. The
                    # installed reset contract performs no physics integration.
                    self.reset(seed=20260909 if case == "Z" else None)
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

    def set_h1_research_profile(self, *, pure_hover: bool, anchor_weight: float) -> None:
        import math
        if type(pure_hover) is not bool or type(anchor_weight) not in (int, float) or not math.isfinite(anchor_weight) or anchor_weight < 0: raise ValueError("invalid research profile")
        self._h1_pure_hover_training = pure_hover
        self._h1_anchor_weight = float(anchor_weight)

    def set_h1_ics_clearance_margin(self, margin_m: float) -> None:
        """Explicit phase setting; retain measurement-only ICS and reset lifetime."""
        from dataclasses import replace
        from instinctlab.control.ics import ICSConfig
        validated = ICSConfig(clearance_margin_m=margin_m)
        self._h1_ics_clearance_margin_m = validated.clearance_margin_m
        for pipeline in self._control_pipelines:
            pipeline.config = replace(pipeline.config,
                ics=replace(pipeline.config.ics, clearance_margin_m=validated.clearance_margin_m))

    def set_h1_residual_neutrality_weight(self, weight: float) -> None:
        import math
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("residual neutrality weight must be finite and nonnegative")
        self._h1_residual_neutrality_weight = float(weight)

    def _get_rewards(self) -> torch.Tensor:
        if self._task is None or self._control is None or self._h1_next_command_b is None:
            raise RuntimeError("H1 task/control/next-command state is unavailable")
        masks = self._task.termination.masks
        reward = compute_reward_h1(
            lidar_grid=self._policy["lidar_grid"], command_b=self._command_b,
            v_gov_b=self._control.v_gov_b, v_final_b=self._control.v_final_b,
            v_corr_b=self._control.v_corr_b, residual_neutrality_weight=self._h1_residual_neutrality_weight,
            previous_issued_v_final_b=self._reward_previous_v_final_b,
            actual_velocity_b=self._task.actual_velocity_b,
            anchor_weight=self._h1_anchor_weight,
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
        from training_workloads.unified_h1.b0_enhanced import quaternion_rotate
        reaction_direction=quaternion_rotate(self._tolist(body[lane,0,3:7]),(0.,0.,1.))
        rotor=self._wrench.rotor; rotors=[]
        for rotor_index,name in enumerate(ROTOR_NAMES):
            body_index=rotor_index+1
            direction=self._tolist(quat_apply(body[lane,body_index,3:7].unsqueeze(0),torch.tensor([[0.,0.,1.]],device=self.device,dtype=body.dtype))[0])
            rotors.append({"name":name,"command":float(rotor.normalized_commands[lane,rotor_index]),"target_throttle":float(rotor.target_throttle[lane,rotor_index]),"actual_throttle":float(rotor.throttle[lane,rotor_index]),"thrust_n":float(rotor.thrusts_n[lane,rotor_index]),"reaction_moment_nm":float(rotor.reaction_moments_nm[lane,rotor_index]),"application_point_world_m":self._tolist(body[lane,body_index,:3]),"direction_world":direction,"reaction_body":str(self._robot.body_names[0]),"reaction_direction_world":reaction_direction})
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
        if hasattr(self, "_h1_ics_clearance_margin_m"):
            self.set_h1_ics_clearance_margin(self._h1_ics_clearance_margin_m)
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
        if indices.numel() == 0 or self._h1_global_policy_step < 1024 or self._h1_pure_hover_training:
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
