"""Actual combined-scene readback and raw-ray observability evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from training_workloads.scale_003.persistence import atomic_json_replace, atomic_torch_replace

from .config import sha256_file


def _diagnostic_scalar(value: torch.Tensor) -> float | str:
    scalar = float(value.item())
    if math.isnan(scalar):
        return "nan"
    if math.isinf(scalar):
        return "positive_infinity" if scalar > 0.0 else "negative_infinity"
    return scalar


def _flat_indices(value: Any) -> list[int]:
    rows = value.numpy().tolist()
    return [int(item) for row in rows for item in row] if rows and isinstance(rows[0], list) else [int(item) for item in rows]


def live_geometry_readback(environment: Any) -> dict[str, Any]:
    """Compare the actual Warp/USD mesh and physical collision inventory to H1 construction."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import RayCaster
    from instinctlab.task.scene_h1 import H1_RAY_MESH_PATH, combined_scene_triangles
    from pxr import UsdGeom, UsdPhysics

    origins = environment.scene.env_origins.detach().cpu()
    expected_points, expected_indices, identities = combined_scene_triangles(origins)
    if environment.cfg.ray_caster.mesh_prim_paths != [H1_RAY_MESH_PATH]:
        raise ValueError("H1 RayCaster is not bound to the combined finite mesh")
    warp = RayCaster.meshes[H1_RAY_MESH_PATH]
    points = warp.points.numpy().tolist(); indices = _flat_indices(warp.indices)
    stage = sim_utils.get_current_stage()
    authored = UsdGeom.Mesh(stage.GetPrimAtPath(H1_RAY_MESH_PATH))
    authored_points = [list(item) for item in authored.GetPointsAttr().Get()]
    authored_indices = [int(item) for item in authored.GetFaceVertexIndicesAttr().Get()]
    expected_points_lists = [list(item) for item in expected_points]
    if points != authored_points or points != expected_points_lists or indices != authored_indices or indices != expected_indices:
        raise ValueError("actual Warp/USD combined mesh differs from the translated H1 fixture")
    collision = UsdGeom.Mesh(stage.GetPrimAtPath("/World/H1CollisionWalls/Walls"))
    collision_indices = [int(item) for item in collision.GetFaceVertexIndicesAttr().Get()]
    if len(collision_indices) != 32 * 4 * 3 or not collision.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError("actual H1 wall collision mesh is incomplete")
    ground_collision = any(
        str(prim.GetPath()).startswith("/World/ground") and prim.HasAPI(UsdPhysics.CollisionAPI)
        for prim in stage.Traverse()
    )
    if authored.GetPrim().HasAPI(UsdPhysics.CollisionAPI) or not ground_collision:
        raise ValueError("sensed mesh collision boundary differs from the H1 fixture")
    payload = {"schema_version": "unified-h1-geometry-readback.v1", "status": "PASS",
               "mesh_prim_path": H1_RAY_MESH_PATH, "envs": int(origins.shape[0]),
               "points": len(points), "triangles": len(indices) // 3,
               "surface_triangle_identities": len(identities),
               "collision_wall_triangles": len(collision_indices) // 3,
               "env_origins": origins.tolist()}
    payload["identity_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def initial_source(environment: Any) -> dict[str, Any]:
    environment._refresh_task_snapshot()
    source = environment._evaluation_ray_snapshot
    if source is None:
        raise RuntimeError("initial raw-ray snapshot is unavailable")
    return {
        "ray_starts_w": source.ray_starts_w.detach().clone(),
        "ray_directions_w": source.ray_directions_w.detach().clone(),
        "accepted_ray_hits_w": source.accepted_ray_hits_w.detach().clone(),
        "accepted_beam_valid": source.accepted_beam_valid.detach().clone(),
        "mesh_path": environment.cfg.ray_caster.mesh_prim_paths[0],
        "maximum_distance_m": float(environment.cfg.ray_caster.max_distance),
        "source_step": source.source_step.detach().clone(),
        "reset_mask": torch.zeros(32, dtype=torch.bool, device=source.ray_starts_w.device),
    }


def persist_initial_source(output: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    names = ("ray_starts_w", "ray_directions_w", "accepted_ray_hits_w",
             "accepted_beam_valid", "source_step", "reset_mask")
    tensors = {name: source[name].detach().cpu().clone() for name in names}
    path = atomic_torch_replace(output, "initial_preflight/raw_ray_source.pt", tensors)
    metadata = {"schema_version": "unified-h1-raw-ray-source.v1", "mesh_path": source["mesh_path"],
                "maximum_distance_m": source["maximum_distance_m"], "tensor_file": path.name,
                "tensor_file_sha256": sha256_file(path), "tensor_file_bytes": path.stat().st_size,
                "tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in tensors.items()},
                "actor_input": False, "purpose": "privileged preflight evidence only"}
    atomic_json_replace(output, "initial_preflight/raw_ray_source.json", metadata)
    return metadata


def validate_initial_observability(source: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Run the existing exact acquisition and require finite rank-3 local coverage."""
    from instinctlab.evaluation.raycaster_observability import (
        RaycasterAcquisitionError, acquire_exact_observability_from_source,
    )

    starts, directions, hits = (source[name] for name in ("ray_starts_w", "ray_directions_w", "accepted_ray_hits_w"))
    valid = source["accepted_beam_valid"]
    if starts.ndim != 3 or starts.shape[0] != 32 or starts.shape[-1] != 3:
        raise ValueError("raw-ray starts must be [32,R,3]")
    if directions.shape != starts.shape or hits.shape != starts.shape or valid.shape != starts.shape[:2]:
        raise ValueError("raw-ray tensor shapes disagree")
    if not bool(torch.isfinite(starts).all() and torch.isfinite(directions).all() and torch.isfinite(hits[valid]).all()):
        raise ValueError("required raw-ray values are nonfinite")
    try:
        acquisition = acquire_exact_observability_from_source(source)
    except RaycasterAcquisitionError as error:
        diagnostics = error.diagnostics
        tensors = {name: getattr(diagnostics, name).detach().cpu().clone() for name in (
            "ray_hits_w", "ray_parameters", "raw_surface_normals_w", "hit_max_error_m",
            "reconstruction_max_error_m")}
        try:
            tensor_path = atomic_torch_replace(output, "initial_preflight/raw_acquisition_failure.pt", tensors)
            atomic_json_replace(output, "initial_preflight/raw_acquisition_failure.json", {
                "schema_version": "unified-h1-acquisition-failure.v1", "status": "FAIL",
                "primary_error": {"error_type": type(error).__name__, "message": str(error)},
                "hit_max_error_m": _diagnostic_scalar(diagnostics.hit_max_error_m),
                "forward_reconstruction_max_error_m": _diagnostic_scalar(diagnostics.reconstruction_max_error_m),
                "hit_failing_beam": diagnostics.hit_failing_beam,
                "forward_reconstruction_failing_beam": diagnostics.reconstruction_failing_beam,
                "tensor_file": tensor_path.name, "tensor_file_sha256": sha256_file(tensor_path),
                "tensor_file_bytes": tensor_path.stat().st_size,
            })
        except BaseException as persistence_error:
            error.add_note(f"diagnostic persistence also failed: {type(persistence_error).__name__}: {persistence_error}")
        raise
    tensor_names = ("ray_hits_w", "ray_parameters", "ray_distances_m", "raw_surface_normals_w", "beam_valid",
                    "hit_max_error_m", "distance_max_error_m", "reconstruction_max_error_m")
    tensors = {name: getattr(acquisition, name).detach().cpu().clone() for name in tensor_names}
    tensor_path = atomic_torch_replace(output, "initial_preflight/raw_acquisition.pt", tensors)
    rank = acquisition.observability.numerical_rank.detach().cpu()
    sigma = acquisition.observability.sigma_min.detach().cpu()
    if rank.shape != (1, 32) or sigma.shape != (1, 32):
        raise ValueError("initial exact observability result must be [1,32]")
    rank = rank[0].clone()
    sigma = sigma[0].clone()
    if not bool((rank == 3).all()) or not bool(torch.isfinite(sigma).all() and (sigma > 0).all()):
        raise ValueError("initial exact observability is not finite rank three for all environments")
    reconstruction = float(acquisition.reconstruction_max_error_m.max().item())
    hit_identity = float(acquisition.hit_max_error_m.max().item())
    if reconstruction > 1e-5 or hit_identity > 1e-5:
        raise ValueError("initial ray forward reconstruction or hit identity failed")
    result = {"schema_version": "unified-h1-observability.v1", "status": "PASS",
              "environments": 32, "minimum_valid_beams": int(valid.sum(dim=1).min().item()),
              "numerical_rank": rank.tolist(), "sigma_min": sigma.tolist(),
              "forward_reconstruction_max_error_m": reconstruction,
              "same_state_hit_identity_max_error_m": hit_identity,
              "tensor_file": tensor_path.name, "tensor_file_sha256": sha256_file(tensor_path),
              "tensor_file_bytes": tensor_path.stat().st_size}
    atomic_json_replace(output, "initial_preflight/observability.json", result)
    return result


__all__ = ["initial_source", "live_geometry_readback", "persist_initial_source", "validate_initial_observability"]
