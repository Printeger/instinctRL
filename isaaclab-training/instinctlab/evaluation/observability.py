"""Pure evaluation-only exact translational range-Jacobian observability."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch


@dataclass(frozen=True)
class ExactObservabilityConfig:
    """Immutable validation configuration for caller-supplied unit normals."""

    unit_normal_tolerance: float = 1.0e-5

    def __post_init__(self) -> None:
        value = self.unit_normal_tolerance
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("unit_normal_tolerance must be a real scalar")
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("unit_normal_tolerance must be finite and positive")
        object.__setattr__(self, "unit_normal_tolerance", value)


@dataclass(frozen=True)
class ExactObservabilityResult:
    """Detached, independently owned exact observability tensors."""

    numerical_rank: torch.Tensor
    singular_values: torch.Tensor
    sigma_min: torch.Tensor
    condition_number: torch.Tensor
    condition_valid: torch.Tensor
    weak_direction_w: torch.Tensor
    weak_direction_valid: torch.Tensor
    valid_ray_count: torch.Tensor
    total_ray_count: torch.Tensor
    valid_beam_fraction: torch.Tensor


def _owns_exact_storage(value: torch.Tensor) -> bool:
    expected_bytes = value.numel() * value.element_size()
    return (
        value._base is None
        and value.storage_offset() == 0
        and value.untyped_storage().nbytes() == expected_bytes
    )


def _validate_inputs(
    surface_normals_w: torch.Tensor,
    beam_valid: torch.Tensor,
    config: ExactObservabilityConfig,
) -> tuple[int, int, int]:
    if not isinstance(config, ExactObservabilityConfig):
        raise TypeError("config must be ExactObservabilityConfig")
    if not isinstance(surface_normals_w, torch.Tensor) or not isinstance(beam_valid, torch.Tensor):
        raise TypeError("surface_normals_w and beam_valid must be torch.Tensor values")
    if surface_normals_w.ndim != 4 or surface_normals_w.shape[-1] != 3:
        raise ValueError("surface_normals_w must have exact shape [T,N,R,3]")
    time_steps, environments, rays, _ = surface_normals_w.shape
    if time_steps <= 0 or environments <= 0 or rays <= 0:
        raise ValueError("surface_normals_w dimensions T,N,R must be positive")
    if beam_valid.shape != (time_steps, environments, rays):
        raise ValueError("beam_valid must have exact shape [T,N,R]")
    if surface_normals_w.dtype not in (torch.float32, torch.float64):
        raise TypeError("surface_normals_w must use float32 or float64")
    if beam_valid.dtype != torch.bool:
        raise TypeError("beam_valid must use bool")
    if surface_normals_w.device != beam_valid.device:
        raise ValueError("surface_normals_w and beam_valid must use the same device")
    if not _owns_exact_storage(surface_normals_w) or not _owns_exact_storage(beam_valid):
        raise ValueError("each input must own its exact storage and must not be a view")
    if surface_normals_w.untyped_storage().data_ptr() == beam_valid.untyped_storage().data_ptr():
        raise ValueError("surface_normals_w and beam_valid must not alias storage")
    if not bool(torch.isfinite(surface_normals_w).all().item()):
        raise ValueError("surface_normals_w must be finite")
    invalid = ~beam_valid
    if bool((surface_normals_w[invalid] != 0.0).any().item()):
        raise ValueError("invalid beams must use exact zero normal placeholders")
    valid_normals = surface_normals_w[beam_valid]
    if valid_normals.numel() > 0:
        norm_error = (torch.linalg.vector_norm(valid_normals, dim=-1) - 1.0).abs()
        if bool((norm_error > config.unit_normal_tolerance).any().item()):
            raise ValueError("valid surface normals must be unit length within tolerance")
    return time_steps, environments, rays


def _owned(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def compute_exact_observability(
    surface_normals_w: torch.Tensor,
    beam_valid: torch.Tensor,
    config: ExactObservabilityConfig = ExactObservabilityConfig(),
) -> ExactObservabilityResult:
    """Compute exact per-sample observability from world-frame surface normals.

    Every valid normal contributes one translational range-Jacobian row
    ``J_i = -n_i^T``. No simulator, runtime, actor, or control state is used.
    """

    time_steps, environments, rays = _validate_inputs(surface_normals_w, beam_valid, config)
    leading = (time_steps, environments)
    dtype = surface_normals_w.dtype
    device = surface_normals_w.device
    rank = torch.zeros(leading, dtype=torch.int64, device=device)
    spectrum = torch.zeros((*leading, 3), dtype=dtype, device=device)
    condition = torch.zeros(leading, dtype=dtype, device=device)
    condition_valid = torch.zeros(leading, dtype=torch.bool, device=device)
    weak = torch.zeros((*leading, 3), dtype=dtype, device=device)
    weak_valid = torch.zeros(leading, dtype=torch.bool, device=device)
    valid_count = beam_valid.sum(dim=-1, dtype=torch.int64)
    total_count = torch.full(leading, rays, dtype=torch.int64, device=device)

    epsilon = torch.finfo(dtype).eps
    tolerance_scale = max(rays, 3) * epsilon
    for time_index in range(time_steps):
        for environment_index in range(environments):
            selected = beam_valid[time_index, environment_index]
            count = int(valid_count[time_index, environment_index].item())
            if count == 0:
                continue
            jacobian = -surface_normals_w[time_index, environment_index, selected]
            _, singular_values_descending, vh = torch.linalg.svd(jacobian, full_matrices=True)
            padded = torch.zeros(3, dtype=dtype, device=device)
            padded[3 - singular_values_descending.numel():] = singular_values_descending.flip(0)
            spectrum[time_index, environment_index] = padded
            numerical_tolerance = tolerance_scale * padded[-1]
            sample_rank = torch.count_nonzero(padded > numerical_tolerance)
            rank[time_index, environment_index] = sample_rank
            if int(sample_rank.item()) == 3:
                condition[time_index, environment_index] = padded[-1] / padded[0]
                condition_valid[time_index, environment_index] = True

            smallest_multiplicity = torch.count_nonzero(
                (padded - padded[0]).abs() <= numerical_tolerance
            )
            if int(smallest_multiplicity.item()) == 1:
                direction = vh[-1].clone()
                absolute_direction = direction.abs()
                tied = absolute_direction >= absolute_direction.max() - numerical_tolerance
                sign_index = int(torch.nonzero(tied, as_tuple=False)[0, 0].item())
                if bool((direction[sign_index] < 0.0).item()):
                    direction.neg_()
                weak[time_index, environment_index] = direction
                weak_valid[time_index, environment_index] = True

    sigma_min = spectrum[..., 0].clone()
    fraction = valid_count.to(dtype=dtype) / total_count.to(dtype=dtype)
    result = ExactObservabilityResult(
        numerical_rank=_owned(rank),
        singular_values=_owned(spectrum),
        sigma_min=_owned(sigma_min),
        condition_number=_owned(condition),
        condition_valid=_owned(condition_valid),
        weak_direction_w=_owned(weak),
        weak_direction_valid=_owned(weak_valid),
        valid_ray_count=_owned(valid_count),
        total_ray_count=_owned(total_count),
        valid_beam_fraction=_owned(fraction),
    )
    output_values = [getattr(result, field.name) for field in fields(result)]
    storage_keys = {(value.device, value.untyped_storage().data_ptr()) for value in output_values}
    if len(storage_keys) != len(output_values):
        raise RuntimeError("observability outputs must own independent storage")
    return result


__all__ = ["ExactObservabilityConfig", "ExactObservabilityResult", "compute_exact_observability"]
