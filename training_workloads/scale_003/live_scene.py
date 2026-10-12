"""Authorized-runtime scene adapter. Import only after live authorization succeeds."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any

from .geometry import (
    GLOBAL_GROUND_PATH,
    MESH_PRIM_PATH,
    WALL_COLLIDER_ROOT,
    build_combined_geometry,
    validate_exact_observability_gate,
    validate_readback,
)
from .schedule import CommandClock
from .config import sha256_file
from .persistence import atomic_json_replace, atomic_torch_replace
from .runtime_support import configure_simulation_logging


def _author_triangle_mesh(stage: Any, path: str, points: Any, triangles: Any, *, collision: bool) -> None:
    from pxr import UsdGeom, UsdPhysics

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(list(points))
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([index for triangle in triangles for index in triangle])
    mesh.CreateSubdivisionSchemeAttr("none")
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())


def make_training_environment(progress: Any, attempt_dir: Path) -> Any:
    """Construct the process-local 32-copy scene from actual InteractiveScene origins."""
    from instinctlab.envs import TaslabDirectEnv, TaslabDirectEnvCfg
    from instinctlab.observations.audit import audit_actor_observation
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom
    import torch

    class Scale003Env(TaslabDirectEnv):
        def __init__(self, cfg: Any) -> None:
            self.scale003_clock = CommandClock()
            self.scale003_progress = progress
            self.scale003_geometry = None
            super().__init__(cfg)

        def _setup_scene(self) -> None:
            super()._setup_scene()
            origins = self.scene.env_origins.detach().cpu().tolist()
            geometry = build_combined_geometry(origins)
            stage = sim_utils.get_current_stage()
            UsdGeom.Xform.Define(stage, "/World/MID360StaticScene")
            _author_triangle_mesh(stage, geometry.mesh_prim_path, geometry.points, geometry.triangles, collision=False)
            UsdGeom.Xform.Define(stage, WALL_COLLIDER_ROOT)
            for collider in geometry.wall_colliders:
                _author_triangle_mesh(stage, collider.prim_path, collider.points, collider.triangles, collision=True)
            self.scale003_geometry = geometry

        def _get_observations(self) -> dict[str, Any]:
            command = torch.tensor(
                self.scale003_clock.command, device=self.device, dtype=self._command_b.dtype,
            ).repeat(self.num_envs, 1)
            self._command_b.copy_(command)
            observation = super()._get_observations()
            policy = observation.get("policy")
            if tuple(policy or ()) != ("lidar_grid", "state_vec"):
                raise ValueError("actor observation keys/order changed")
            audit_actor_observation(policy)
            if policy["lidar_grid"].shape != (32, 12, 59, 360) or policy["state_vec"].shape != (32, 52):
                raise ValueError("actor observation dimensions changed")
            return observation

        @property
        def schedule_index(self) -> int:
            return self.scale003_clock.index

        def prepare_schedule_segment(self, segment: Any, observation: Any) -> None:
            """Public adapter boundary used by the driver; private command state stays here."""
            if self.scale003_clock.index != segment.policy_step_start:
                raise ValueError("rollout segment is not aligned to the global command clock")
            command = torch.tensor(
                segment.command_b_m_s, device=self.device, dtype=self._command_b.dtype,
            ).repeat(self.num_envs, 1)
            if not torch.equal(self._command_b, command):
                raise ValueError("adapter command state is not aligned to the segment")
            policy = observation.get("policy") if isinstance(observation, dict) else None
            if not isinstance(policy, dict) or not torch.equal(policy["state_vec"][:, -7:-4], command):
                raise ValueError("returned actor state_vec is not aligned to the segment")

        def step(self, action: Any) -> Any:
            current = self.scale003_clock.index
            self.scale003_progress.write_step_attempt(current + 1)
            self.scale003_clock.advance_after_action()
            returned = super().step(action)
            self.scale003_progress.write_step_complete(self.scale003_clock.index)
            return returned

        def _reset_idx(self, env_ids: Any) -> None:
            before = self.scale003_clock.index
            super()._reset_idx(env_ids)
            self.scale003_clock.observe_reset()
            if self.scale003_clock.index != before:
                raise RuntimeError("environment reset restarted the process-global command clock")

    cfg = TaslabDirectEnvCfg()
    cfg.seed = 20260907
    cfg.scene.num_envs = 32
    cfg.scene.env_spacing = 100.0
    cfg.ray_caster.mesh_prim_paths = [MESH_PRIM_PATH]
    configure_simulation_logging(cfg, attempt_dir)
    if (cfg.sim.dt, cfg.decimation, cfg.ray_caster.max_distance) != (0.005, 20, 40.0):
        raise ValueError("frozen physics/MID360 range mismatch")
    return Scale003Env(cfg)


def live_scene_readback(env: Any) -> dict[str, Any]:
    """Bind the candidate to actual USD, Warp, collision, and env-origin state."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import RayCaster
    from pxr import UsdGeom, UsdPhysics

    expected = env.scale003_geometry
    if expected is None or env.cfg.ray_caster.mesh_prim_paths != [MESH_PRIM_PATH]:
        raise ValueError("live geometry was not bound to the explicit RayCaster prim")
    warp_mesh = RayCaster.meshes[MESH_PRIM_PATH]
    points = warp_mesh.points.numpy().tolist()
    raw_indices = warp_mesh.indices.numpy().tolist()
    indices = [int(item) for row in raw_indices for item in row] if raw_indices and isinstance(raw_indices[0], list) else raw_indices
    stage = sim_utils.get_current_stage()
    authored = UsdGeom.Mesh(stage.GetPrimAtPath(MESH_PRIM_PATH))
    authored_points = [list(value) for value in authored.GetPointsAttr().Get()]
    authored_indices = list(authored.GetFaceVertexIndicesAttr().Get())
    if points != authored_points or list(indices) != authored_indices:
        raise ValueError("Warp mesh and authored USD mesh differ")
    collider_rows = []
    for collider in expected.wall_colliders:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(collider.prim_path))
        collider_rows.append({
            "prim_path": collider.prim_path,
            "points": [list(value) for value in mesh.GetPointsAttr().Get()],
            "collision": bool(mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)),
        })
    ground_collision = any(
        str(prim.GetPath()).startswith(GLOBAL_GROUND_PATH) and prim.HasAPI(UsdPhysics.CollisionAPI)
        for prim in stage.Traverse()
    )
    validate_readback(
        expected=expected, mesh_prim_path=MESH_PRIM_PATH, points=points,
        triangle_indices=indices, collider_rows=collider_rows,
        ground_collision_present=ground_collision,
        sensed_mesh_collision=bool(authored.GetPrim().HasAPI(UsdPhysics.CollisionAPI)),
    )
    inventory = [
        {"path": str(prim.GetPath()), "type": str(prim.GetTypeName()),
         "collision": bool(prim.HasAPI(UsdPhysics.CollisionAPI))}
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(("/World/MID360StaticScene", WALL_COLLIDER_ROOT, GLOBAL_GROUND_PATH))
    ]
    payload = {
        "mesh_prim_path": MESH_PRIM_PATH, "points": len(points),
        "triangles": len(indices) // 3, "wall_colliders": len(collider_rows),
        "env_origins": env.scene.env_origins.detach().cpu().tolist(), "inventory": inventory,
        "coverage_limit": "local floor ends at 42m and walls span -42m..6m; visibility after drift outside each local fixture is not guaranteed",
    }
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def initial_source(env: Any) -> OrderedDict[str, Any]:
    """Acquire the actual sensor state without advancing policy or history."""
    import torch
    from isaaclab.utils.math import quat_apply

    _, validity, _, _ = env._read_sensor_producers()
    ray = env._ray_caster
    count = ray.ray_starts.shape[1]
    quat = ray.data.quat_w.repeat(1, count)
    return OrderedDict((
        ("ray_starts_w", (quat_apply(quat, ray.ray_starts) + ray.data.pos_w.unsqueeze(1)).detach().clone()),
        ("ray_directions_w", quat_apply(quat, ray.ray_directions).detach().clone()),
        ("accepted_ray_hits_w", ray.data.ray_hits_w.detach().clone()),
        ("accepted_beam_valid", validity.detach().clone()),
        ("mesh_path", MESH_PRIM_PATH), ("maximum_distance_m", float(env.cfg.ray_caster.max_distance)),
        ("source_step", torch.ones(32, dtype=torch.long, device=env.device)),
        ("reset_mask", torch.zeros(32, dtype=torch.bool, device=env.device)),
    ))


def visibility(source: Any, acquisition: Any, origins: Any) -> dict[str, list[int]]:
    """Count exact local fixture surfaces for each actual environment origin."""
    import torch

    hits = source["accepted_ray_hits_w"]
    valid = source["accepted_beam_valid"]
    normals = acquisition.raw_surface_normals_w
    origin = torch.as_tensor(origins, device=hits.device, dtype=hits.dtype).unsqueeze(1)
    local = hits - origin
    masks = (
        valid & (local[:, :, 2].abs() <= 1e-5) & (normals[:, :, 2] > 0.999),
        valid & ((local[:, :, 0] - 6.0).abs() <= 1e-5) & (normals[:, :, 0] < -0.999),
        valid & ((local[:, :, 1] - 6.0).abs() <= 1e-5) & (normals[:, :, 1] < -0.999),
    )
    return dict(zip(("floor", "x_wall", "y_wall"), [mask.sum(-1).cpu().tolist() for mask in masks], strict=True))


def persist_initial_source(evidence_root: Path, source: OrderedDict[str, Any]) -> dict[str, Any]:
    tensor_names = (
        "ray_starts_w", "ray_directions_w", "accepted_ray_hits_w", "accepted_beam_valid",
        "source_step", "reset_mask",
    )
    tensors = {name: source[name].detach().cpu().clone() for name in tensor_names}
    path = atomic_torch_replace(evidence_root, "initial_preflight/raw_ray_source.pt", tensors)
    metadata = {
        "schema_version": "scale-003-raw-ray-source.v1",
        "mesh_path": source["mesh_path"],
        "maximum_distance_m": source["maximum_distance_m"],
        "tensor_file": "raw_ray_source.pt",
        "tensor_file_sha256": sha256_file(path),
        "tensor_file_bytes": path.stat().st_size,
        "tensors": {
            name: {"shape": list(tensors[name].shape), "dtype": str(tensors[name].dtype), "device": "cpu"}
            for name in tensor_names
        },
        "actor_input": False,
        "purpose": "privileged preflight evidence only",
    }
    atomic_json_replace(evidence_root, "initial_preflight/raw_ray_source.json", metadata)
    return metadata


def persist_initial_acquisition(evidence_root: Path, acquisition: Any) -> dict[str, Any]:
    from instinctlab.evaluation.raycaster_observability import RAYCASTER_GEOMETRY_COMPARISON_STANDARD

    tensor_names = (
        "ray_hits_w", "ray_distances_m", "raw_surface_normals_w", "beam_valid",
        "hit_max_error_m", "distance_max_error_m", "reconstruction_max_error_m",
    )
    tensors = {name: getattr(acquisition, name).detach().cpu().clone() for name in tensor_names}
    path = atomic_torch_replace(evidence_root, "initial_preflight/raw_acquisition.pt", tensors)
    metadata = {
        "schema_version": "scale-004-raw-acquisition.v2",
        "geometry_comparison_standard": RAYCASTER_GEOMETRY_COMPARISON_STANDARD,
        "tensor_file": "raw_acquisition.pt",
        "tensor_file_sha256": sha256_file(path),
        "tensor_file_bytes": path.stat().st_size,
        "tensors": {
            name: {"shape": list(tensors[name].shape), "dtype": str(tensors[name].dtype), "device": "cpu"}
            for name in tensor_names
        },
        "status": "ACQUIRED_NOT_YET_GATE_VALIDATED",
    }
    atomic_json_replace(evidence_root, "initial_preflight/raw_acquisition.json", metadata)
    return metadata


def persist_initial_gate_failure(evidence_root: Path, stage: str, error: BaseException) -> None:
    atomic_json_replace(evidence_root, "initial_preflight/observability_failure.json", {
        "schema_version": "scale-003-initial-preflight-failure.v1",
        "status": "FAIL",
        "completed_stage": stage,
        "error_type": type(error).__name__,
        "error_module": type(error).__module__,
        "message": str(error),
        "unfinished": ["validated_observability_gate", "complete_initial_preflight"],
    })


def run_initial_gate(env: Any, evidence_root: Path | None = None) -> dict[str, Any]:
    """Run the authorized live gate from actual rays and exact normal Jacobians."""
    from instinctlab.evaluation.raycaster_observability import acquire_exact_observability_from_source

    source = initial_source(env)
    if evidence_root is not None:
        persist_initial_source(evidence_root, source)
    try:
        acquisition = acquire_exact_observability_from_source(source)
    except BaseException as error:
        if evidence_root is not None:
            persist_initial_gate_failure(evidence_root, "raw_ray_source", error)
        raise
    if evidence_root is not None:
        persist_initial_acquisition(evidence_root, acquisition)
    try:
        ranks = acquisition.observability.numerical_rank[0].detach().cpu().tolist()
        sigma = acquisition.observability.sigma_min[0].detach().cpu().tolist()
        hits = visibility(source, acquisition, env.scene.env_origins)
    except BaseException as error:
        if evidence_root is not None:
            persist_initial_gate_failure(evidence_root, "raw_acquisition", error)
        raise
    diagnostics = {
        "schema_version": "scale-003-observability-diagnostics.v1",
        "status": "ACQUIRED_NOT_YET_VALIDATED",
        "surface_hits": hits, "numerical_rank": ranks, "sigma_min": sigma,
        "singular_values": acquisition.observability.singular_values[0].detach().cpu().tolist(),
        "source": "actual_raw_raycaster_acquisition",
    }
    if evidence_root is not None:
        atomic_json_replace(evidence_root, "initial_preflight/observability_diagnostics.json", diagnostics)
    try:
        validated = validate_exact_observability_gate(
            surface_hits=hits, numerical_rank=ranks, sigma_min=[float(value) for value in sigma], num_envs=32,
        )
    except BaseException as error:
        if evidence_root is not None:
            persist_initial_gate_failure(evidence_root, "observability_diagnostics", error)
        raise
    result = {**validated, **{key: value for key, value in diagnostics.items() if key not in {"schema_version", "status"}}}
    if evidence_root is not None:
        atomic_json_replace(evidence_root, "initial_preflight/observability.json", result)
    return result
