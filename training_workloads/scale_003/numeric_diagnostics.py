"""CPU-only diagnostics for the large-world float32 ray-distance contract."""

from __future__ import annotations

from typing import Any


def _owns_exact_storage(value: Any) -> bool:
    return (
        value._base is None
        and value.storage_offset() == 0
        and value.untyped_storage().nbytes() == value.numel() * value.element_size()
    )


def analyze_ray_distance_consistency(
    ray_starts_w: Any,
    ray_directions_w: Any,
    returned_ray_hits_w: Any,
    accepted_ray_hits_w: Any,
    ray_parameters: Any,
) -> dict[str, Any]:
    """Compare reverse world subtraction with the kernel's forward hit construction."""
    import torch

    tensors = {
        "ray_starts_w": ray_starts_w,
        "ray_directions_w": ray_directions_w,
        "returned_ray_hits_w": returned_ray_hits_w,
        "accepted_ray_hits_w": accepted_ray_hits_w,
        "ray_parameters": ray_parameters,
    }
    storage: set[int] = set()
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.device.type != "cpu" or value.dtype != torch.float32:
            raise TypeError(f"{name} must be a CPU float32 tensor")
        if value.requires_grad or not _owns_exact_storage(value):
            raise ValueError(f"{name} must be detached and own exact storage")
        pointer = value.untyped_storage().data_ptr()
        if pointer in storage:
            raise ValueError("diagnostic inputs must not alias storage")
        storage.add(pointer)
    if ray_starts_w.ndim != 3 or ray_starts_w.shape[-1] != 3:
        raise ValueError("ray starts must have shape [N,R,3]")
    if any(value.shape != ray_starts_w.shape for value in (ray_directions_w, returned_ray_hits_w, accepted_ray_hits_w)):
        raise ValueError("ray vector tensors must have identical shapes")
    if ray_parameters.shape != ray_starts_w.shape[:-1]:
        raise ValueError("ray parameters must have shape [N,R]")
    if not all(bool(torch.isfinite(value).all().item()) for value in tensors.values()):
        raise ValueError("diagnostic inputs must be finite")

    direction_norm = torch.linalg.vector_norm(ray_directions_w, dim=-1)
    if bool(((direction_norm - 1.0).abs() > 1.0e-5).any().item()):
        raise ValueError("ray directions must be unit length within the product tolerance")
    parameter_metric = ray_parameters * direction_norm
    reverse_metric = torch.linalg.vector_norm(returned_ray_hits_w - ray_starts_w, dim=-1)
    reconstructed = ray_starts_w + ray_parameters.unsqueeze(-1) * ray_directions_w
    forward_error = torch.linalg.vector_norm(reconstructed - returned_ray_hits_w, dim=-1)
    same_state_error = torch.linalg.vector_norm(returned_ray_hits_w - accepted_ray_hits_w, dim=-1)

    # These float64 calculations diagnose rounding of already-quantized float32 inputs. They do
    # not restore the world coordinates or ray parameters that were lost before this helper ran.
    start64 = ray_starts_w.to(torch.float64)
    direction64 = ray_directions_w.to(torch.float64)
    hit64 = returned_ray_hits_w.to(torch.float64)
    parameter64 = ray_parameters.to(torch.float64)
    reference_hit64 = start64 + parameter64.unsqueeze(-1) * direction64
    reverse64 = torch.linalg.vector_norm(hit64 - start64, dim=-1)
    parameter_metric64 = parameter64 * torch.linalg.vector_norm(direction64, dim=-1)
    coordinate_quantization = torch.linalg.vector_norm(hit64 - reference_hit64, dim=-1)
    subtraction_rounding = (reverse_metric.to(torch.float64) - reverse64).abs()
    parameter_rounding = (parameter_metric.to(torch.float64) - parameter_metric64).abs()

    def maximum(value: Any) -> float:
        return float(value.max().item())

    return {
        "schema_version": "scale-004-ray-distance-diagnostic.v1",
        "dtype": "torch.float32",
        "device": "cpu",
        "distance_tolerance_m": 1.0e-5,
        "direction_norm_max_error": maximum((direction_norm - 1.0).abs()),
        "reverse_world_subtraction_max_error_m": maximum((reverse_metric - parameter_metric).abs()),
        "forward_reconstruction_max_error_m": maximum(forward_error),
        "same_state_hit_max_error_m": maximum(same_state_error),
        "decomposition_against_float64_evaluation_of_float32_inputs": {
            "hit_coordinate_quantization_max_m": maximum(coordinate_quantization),
            "reverse_subtraction_rounding_max_m": maximum(subtraction_rounding),
            "parameter_metric_rounding_max_m": maximum(parameter_rounding),
            "information_recovery_claimed": False,
        },
        "recommended_check": "forward_reconstruction_plus_existing_same_state_hit_identity",
    }
