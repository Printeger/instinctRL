"""Evaluation-runtime-only exact RayCaster normal acquisition."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any

import torch

from .observability import ExactObservabilityConfig, ExactObservabilityResult, compute_exact_observability


@dataclass(frozen=True)
class RaycasterObservabilityConfig:
    """Immutable acquisition validation tolerances in world-frame SI units."""

    unit_tolerance: float = 1.0e-5
    distance_tolerance_m: float = 1.0e-5

    def __post_init__(self) -> None:
        for name in ("unit_tolerance", "distance_tolerance_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class RaycasterObservabilityResult:
    """Owned raw acquisition, qualified normal handoff, and exact result."""

    ray_hits_w: torch.Tensor
    ray_distances_m: torch.Tensor
    raw_surface_normals_w: torch.Tensor
    surface_normals_w: torch.Tensor
    beam_valid: torch.Tensor
    hit_max_error_m: torch.Tensor
    distance_max_error_m: torch.Tensor
    observability: ExactObservabilityResult


def _raycast_mesh_with_normals(
    ray_starts_w: torch.Tensor,
    ray_directions_w: torch.Tensor,
    mesh: Any,
    maximum_distance_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Import and invoke the frozen public operation only at runtime."""
    from isaaclab.utils.warp import raycast_mesh

    return raycast_mesh(
        ray_starts_w,
        ray_directions_w,
        mesh=mesh,
        max_dist=maximum_distance_m,
        return_distance=True,
        return_normal=True,
    )


def _owns_exact_storage(value: torch.Tensor) -> bool:
    return (
        value._base is None
        and value.storage_offset() == 0
        and value.untyped_storage().nbytes() == value.numel() * value.element_size()
    )


def _validate_owned_inputs(inputs: tuple[tuple[str, torch.Tensor], ...]) -> None:
    storage: dict[tuple[torch.device, int], str] = {}
    for name, value in inputs:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not _owns_exact_storage(value):
            raise ValueError(f"{name} must own its exact storage and must not be a view")
        key = (value.device, value.untyped_storage().data_ptr())
        if key in storage:
            raise ValueError(f"{storage[key]} and {name} alias storage")
        storage[key] = name


def _positive_infinity(value: torch.Tensor) -> torch.Tensor:
    return torch.isinf(value) & (value > 0.0)


def _owned(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def acquire_exact_raycaster_observability(
    ray_starts_w: torch.Tensor,
    ray_directions_w: torch.Tensor,
    mesh: Any,
    maximum_distance_m: float,
    accepted_ray_hits_w: torch.Tensor,
    accepted_beam_valid: torch.Tensor,
    config: RaycasterObservabilityConfig = RaycasterObservabilityConfig(),
) -> RaycasterObservabilityResult:
    """Acquire and qualify exact face normals for one same-state MID360 scan."""

    if not isinstance(config, RaycasterObservabilityConfig):
        raise TypeError("config must be RaycasterObservabilityConfig")
    inputs = (
        ("ray_starts_w", ray_starts_w),
        ("ray_directions_w", ray_directions_w),
        ("accepted_ray_hits_w", accepted_ray_hits_w),
        ("accepted_beam_valid", accepted_beam_valid),
    )
    _validate_owned_inputs(inputs)
    if ray_starts_w.ndim != 3 or ray_starts_w.shape[-1] != 3:
        raise ValueError("ray_starts_w must have exact shape [N,R,3]")
    environments, rays, _ = ray_starts_w.shape
    if environments <= 0 or rays <= 0:
        raise ValueError("ray dimensions N and R must be positive")
    if ray_directions_w.shape != ray_starts_w.shape or accepted_ray_hits_w.shape != ray_starts_w.shape:
        raise ValueError("directions and accepted hits must match ray_starts_w shape [N,R,3]")
    if accepted_beam_valid.shape != (environments, rays):
        raise ValueError("accepted_beam_valid must have exact shape [N,R]")
    if any(value.dtype != torch.float32 for value in (ray_starts_w, ray_directions_w, accepted_ray_hits_w)):
        raise TypeError("ray starts, directions, and accepted hits must use float32")
    if accepted_beam_valid.dtype != torch.bool:
        raise TypeError("accepted_beam_valid must use bool")
    devices = {value.device for _, value in inputs}
    if len(devices) != 1:
        raise ValueError("all tensor inputs must use the same device")
    if not bool(torch.isfinite(ray_starts_w).all().item()) or not bool(
        torch.isfinite(ray_directions_w).all().item()
    ):
        raise ValueError("ray starts and directions must be finite")
    direction_error = (torch.linalg.vector_norm(ray_directions_w, dim=-1) - 1.0).abs()
    if bool((direction_error > config.unit_tolerance).any().item()):
        raise ValueError("ray directions must be unit length within tolerance")
    if isinstance(maximum_distance_m, bool) or not isinstance(maximum_distance_m, (int, float)):
        raise TypeError("maximum_distance_m must be a real scalar")
    maximum_distance_m = float(maximum_distance_m)
    if not math.isfinite(maximum_distance_m) or maximum_distance_m <= 0.0:
        raise ValueError("maximum_distance_m must be finite and positive")
    if mesh is None or not hasattr(mesh, "id") or not hasattr(mesh, "device"):
        raise TypeError("mesh must be a frozen Warp mesh handle")

    valid_hits = accepted_ray_hits_w[accepted_beam_valid]
    invalid_hits = accepted_ray_hits_w[~accepted_beam_valid]
    if not bool(torch.isfinite(valid_hits).all().item()):
        raise ValueError("accepted valid hits must be finite")
    if invalid_hits.numel() and not bool(_positive_infinity(invalid_hits).all().item()):
        raise ValueError("accepted invalid hits must preserve the positive-infinity miss convention")
    accepted_distance = torch.linalg.vector_norm(accepted_ray_hits_w - ray_starts_w, dim=-1)
    if bool(((accepted_distance[accepted_beam_valid] < 0.0) |
             (accepted_distance[accepted_beam_valid] > maximum_distance_m)).any().item()):
        raise ValueError("accepted valid hit distance exceeds the declared cast interval")

    snapshots = tuple(value.detach().clone() for _, value in inputs)
    returned = _raycast_mesh_with_normals(ray_starts_w, ray_directions_w, mesh, maximum_distance_m)
    if not isinstance(returned, tuple) or len(returned) != 4:
        raise RuntimeError("raycast_mesh must return exactly four ordered results")
    ray_hits_w, ray_distances_m, raw_normals_w, face_ids = returned
    if face_ids is not None:
        raise RuntimeError("face ids must be unavailable when return_face_id is false")
    for name, value, shape in (
        ("ray hits", ray_hits_w, ray_starts_w.shape),
        ("ray distances", ray_distances_m, accepted_beam_valid.shape),
        ("ray normals", raw_normals_w, ray_starts_w.shape),
    ):
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise RuntimeError(f"returned {name} metadata mismatch")
        if value.dtype != torch.float32 or value.device != ray_starts_w.device:
            raise RuntimeError(f"returned {name} dtype/device mismatch")
    for (name, value), snapshot in zip(inputs, snapshots, strict=True):
        if not torch.equal(value, snapshot):
            raise RuntimeError(f"raycast_mesh mutated {name}")

    cast_valid = torch.isfinite(ray_hits_w).all(dim=-1)
    if not torch.equal(cast_valid, accepted_beam_valid):
        raise RuntimeError("returned hit validity disagrees with the accepted MID360 mask")
    if not torch.equal(ray_hits_w, accepted_ray_hits_w):
        raise RuntimeError("returned hits do not exactly match the accepted same-state sensor hits")
    if not bool(torch.isfinite(ray_distances_m[accepted_beam_valid]).all().item()):
        raise RuntimeError("returned valid distances must be finite")
    if not bool(torch.isfinite(raw_normals_w[accepted_beam_valid]).all().item()):
        raise RuntimeError("returned valid normals must be finite")
    for name, value in (("distances", ray_distances_m), ("normals", raw_normals_w)):
        invalid = value[~accepted_beam_valid]
        if invalid.numel() and not bool(_positive_infinity(invalid).all().item()):
            raise RuntimeError(f"returned invalid {name} must preserve the positive-infinity miss convention")
    distance_error = (ray_distances_m[accepted_beam_valid] - accepted_distance[accepted_beam_valid]).abs()
    distance_max_error = distance_error.max() if distance_error.numel() else torch.zeros((), device=ray_starts_w.device)
    if bool((distance_error > config.distance_tolerance_m).any().item()):
        raise RuntimeError("returned distances disagree with accepted same-state hit geometry")
    normal_error = (torch.linalg.vector_norm(raw_normals_w[accepted_beam_valid], dim=-1) - 1.0).abs()
    if bool((normal_error > config.unit_tolerance).any().item()):
        raise RuntimeError("returned valid normals must be unit length within tolerance")

    qualified_normals = torch.where(
        accepted_beam_valid.unsqueeze(-1), raw_normals_w, torch.zeros_like(raw_normals_w)
    )
    normals_handoff = _owned(qualified_normals.unsqueeze(0))
    validity_handoff = _owned(accepted_beam_valid.unsqueeze(0))
    observability = compute_exact_observability(
        normals_handoff,
        validity_handoff,
        ExactObservabilityConfig(unit_normal_tolerance=config.unit_tolerance),
    )
    result = RaycasterObservabilityResult(
        ray_hits_w=_owned(ray_hits_w),
        ray_distances_m=_owned(ray_distances_m),
        raw_surface_normals_w=_owned(raw_normals_w),
        surface_normals_w=normals_handoff,
        beam_valid=validity_handoff,
        hit_max_error_m=torch.zeros((), dtype=torch.float32, device=ray_starts_w.device),
        distance_max_error_m=_owned(distance_max_error),
        observability=observability,
    )
    direct = [getattr(result, field.name) for field in fields(result) if field.name != "observability"]
    pointers = [(value.device, value.untyped_storage().data_ptr()) for value in direct]
    input_pointers = {(value.device, value.untyped_storage().data_ptr()) for _, value in inputs}
    if len(set(pointers)) != len(pointers) or any(pointer in input_pointers for pointer in pointers):
        raise RuntimeError("adapter outputs must own mutually independent storage")
    return result


__all__ = [
    "RaycasterObservabilityConfig",
    "RaycasterObservabilityResult",
    "acquire_exact_raycaster_observability",
]
