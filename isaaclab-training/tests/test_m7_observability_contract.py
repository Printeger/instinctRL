"""Synthetic, negative, ownership, and actor-isolation tests for M7 observability."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
import inspect
import math
from pathlib import Path

import pytest
import torch

from instinctlab.evaluation import (
    ExactObservabilityConfig,
    ExactObservabilityResult,
    compute_exact_observability,
)
from instinctlab.training.networks import HistoryActor


ROOT = Path(__file__).resolve().parents[1]
CONSUMER_ROOTS = ("control", "envs", "observations", "task", "training")


def _inputs(dtype: torch.dtype, geometry: str) -> tuple[torch.Tensor, torch.Tensor]:
    normals = torch.zeros(1, 1, 6, 3, dtype=dtype)
    valid = torch.zeros(1, 1, 6, dtype=torch.bool)
    axes = {
        "open": [],
        "wall": [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        "corridor": [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        "corner": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "anisotropic": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
    }[geometry]
    if axes:
        count = len(axes)
        normals[0, 0, :count] = torch.tensor(axes, dtype=dtype)
        valid[0, 0, :count] = True
    return normals, valid


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("geometry", "rank", "spectrum", "condition", "condition_valid", "weak", "weak_valid", "count"),
    [
        ("open", 0, [0.0, 0.0, 0.0], 0.0, False, [0.0, 0.0, 0.0], False, 0),
        ("wall", 1, [0.0, 0.0, math.sqrt(2.0)], 0.0, False, [0.0, 0.0, 0.0], False, 2),
        ("corridor", 2, [0.0, math.sqrt(2.0), math.sqrt(2.0)], 0.0, False, [0.0, 0.0, 1.0], True, 4),
        ("corner", 3, [1.0, 1.0, 1.0], 1.0, True, [0.0, 0.0, 0.0], False, 3),
        ("anisotropic", 3, [1.0, math.sqrt(2.0), math.sqrt(3.0)], math.sqrt(3.0), True, [1.0, 0.0, 0.0], True, 6),
    ],
)
def test_hand_computed_geometries_all_fields(
    dtype, geometry, rank, spectrum, condition, condition_valid, weak, weak_valid, count
):
    normals, valid = _inputs(dtype, geometry)
    result = compute_exact_observability(normals, valid)
    tolerance = 2.0e-6 if dtype == torch.float32 else 1.0e-12
    assert isinstance(result, ExactObservabilityResult)
    assert result.numerical_rank.shape == (1, 1) and result.numerical_rank.dtype == torch.int64
    assert result.singular_values.shape == (1, 1, 3) and result.singular_values.dtype == dtype
    assert result.sigma_min.shape == (1, 1) and result.sigma_min.dtype == dtype
    assert result.condition_number.shape == (1, 1) and result.condition_number.dtype == dtype
    assert result.condition_valid.shape == (1, 1) and result.condition_valid.dtype == torch.bool
    assert result.weak_direction_w.shape == (1, 1, 3) and result.weak_direction_w.dtype == dtype
    assert result.weak_direction_valid.shape == (1, 1) and result.weak_direction_valid.dtype == torch.bool
    assert result.valid_ray_count.shape == (1, 1) and result.valid_ray_count.dtype == torch.int64
    assert result.total_ray_count.shape == (1, 1) and result.total_ray_count.dtype == torch.int64
    assert result.valid_beam_fraction.shape == (1, 1) and result.valid_beam_fraction.dtype == dtype
    assert result.numerical_rank.item() == rank
    assert torch.allclose(result.singular_values[0, 0], torch.tensor(spectrum, dtype=dtype), atol=tolerance, rtol=tolerance)
    assert math.isclose(result.sigma_min.item(), spectrum[0], abs_tol=tolerance, rel_tol=tolerance)
    assert result.condition_valid.item() is condition_valid
    assert math.isclose(result.condition_number.item(), condition, abs_tol=tolerance, rel_tol=tolerance)
    assert result.weak_direction_valid.item() is weak_valid
    assert torch.allclose(result.weak_direction_w[0, 0], torch.tensor(weak, dtype=dtype), atol=tolerance, rtol=tolerance)
    assert result.valid_ray_count.item() == count and result.total_ray_count.item() == 6
    assert math.isclose(result.valid_beam_fraction.item(), count / 6.0, abs_tol=tolerance, rel_tol=tolerance)
    for field in fields(result):
        value = getattr(result, field.name)
        assert value.device == normals.device and not value.requires_grad and value.grad_fn is None
        assert torch.isfinite(value).all()


def test_batched_mixed_validity_preserves_leading_dimensions_and_counts():
    normals = torch.zeros(2, 2, 3, 3, dtype=torch.float64)
    valid = torch.zeros(2, 2, 3, dtype=torch.bool)
    fixtures = (
        ([[1.0, 0.0, 0.0]], 1),
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 2),
        ([], 0),
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], 3),
    )
    for index, (rows, count) in enumerate(fixtures):
        t, n = divmod(index, 2)
        if count:
            normals[t, n, :count] = torch.tensor(rows, dtype=torch.float64)
            valid[t, n, :count] = True
    result = compute_exact_observability(normals, valid)
    assert torch.equal(result.numerical_rank, torch.tensor([[1, 2], [0, 3]]))
    assert torch.equal(result.valid_ray_count, torch.tensor([[1, 2], [0, 3]]))
    assert torch.equal(result.total_ray_count, torch.full((2, 2), 3))
    assert torch.allclose(result.valid_beam_fraction, torch.tensor([[1 / 3, 2 / 3], [0, 1]], dtype=torch.float64))


@pytest.mark.parametrize("geometry", ["wall", "corridor", "corner", "anisotropic"])
def test_sign_flips_and_ray_permutations_preserve_invariants(geometry):
    normals, valid = _inputs(torch.float64, geometry)
    baseline = compute_exact_observability(normals, valid)
    permutation = torch.tensor([5, 2, 4, 0, 3, 1])
    changed_normals = (-normals[:, :, permutation]).clone()
    changed_valid = valid[:, :, permutation].clone()
    changed = compute_exact_observability(changed_normals, changed_valid)
    for name in (
        "numerical_rank", "singular_values", "sigma_min", "condition_number", "condition_valid",
        "weak_direction_w", "weak_direction_valid", "valid_ray_count", "total_ray_count",
        "valid_beam_fraction",
    ):
        assert torch.allclose(getattr(baseline, name), getattr(changed, name), atol=1e-12, rtol=1e-12)


def test_canonical_sign_uses_lowest_component_for_absolute_tie():
    direction = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64) / math.sqrt(2.0)
    orthogonal = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64) / math.sqrt(2.0)
    normals = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    normals[0, 0] = torch.stack((orthogonal, torch.tensor([0.0, 0.0, 1.0]), orthogonal))
    valid = torch.ones(1, 1, 3, dtype=torch.bool)
    result = compute_exact_observability(normals, valid)
    assert result.weak_direction_valid.item()
    assert torch.allclose(result.weak_direction_w[0, 0], direction, atol=1e-12)
    assert result.weak_direction_w[0, 0, 0] >= 0.0


def test_inputs_are_not_mutated_and_every_output_owns_distinct_storage():
    normals, valid = _inputs(torch.float64, "anisotropic")
    normals_snapshot, valid_snapshot = normals.clone(), valid.clone()
    result = compute_exact_observability(normals, valid)
    assert torch.equal(normals, normals_snapshot) and torch.equal(valid, valid_snapshot)
    values = [getattr(result, field.name) for field in fields(result)]
    input_pointers = {normals.untyped_storage().data_ptr(), valid.untyped_storage().data_ptr()}
    output_pointers = [value.untyped_storage().data_ptr() for value in values]
    assert len(set(output_pointers)) == len(output_pointers)
    assert input_pointers.isdisjoint(output_pointers)
    snapshot = [value.clone() for value in values]
    normals.zero_()
    valid.zero_()
    for value, expected in zip(values, snapshot, strict=True):
        assert torch.equal(value, expected)
    values[0].zero_()
    for value, expected in zip(values[1:], snapshot[1:], strict=True):
        assert torch.equal(value, expected)
    with pytest.raises(FrozenInstanceError):
        result.numerical_rank = torch.ones_like(result.numerical_rank)


def test_config_is_immutable_and_unit_tolerance_fails_closed_at_boundary():
    config = ExactObservabilityConfig(unit_normal_tolerance=1.0e-3)
    normals = torch.tensor([[[[1.0009, 0.0, 0.0]]]], dtype=torch.float64)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    compute_exact_observability(normals, valid, config)
    outside = torch.tensor([[[[1.0011, 0.0, 0.0]]]], dtype=torch.float64)
    with pytest.raises(ValueError, match="unit length"):
        compute_exact_observability(outside, valid.clone(), config)
    with pytest.raises(FrozenInstanceError):
        config.unit_normal_tolerance = 2.0e-3
    for invalid in (True, 0.0, -1.0, float("inf"), float("nan"), "small"):
        expected = TypeError if isinstance(invalid, (bool, str)) else ValueError
        with pytest.raises(expected):
            ExactObservabilityConfig(unit_normal_tolerance=invalid)


def test_numerical_rank_uses_declared_scale_aware_tolerance():
    for delta, expected_rank in ((1.0e-6, 2), (1.0e-8, 1)):
        normals = torch.tensor(
            [[[[1.0, 0.0, 0.0], [1.0, delta, 0.0]]]], dtype=torch.float32
        )
        normals /= torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
        valid = torch.ones(1, 1, 2, dtype=torch.bool)
        result = compute_exact_observability(normals, valid)
        tolerance = max(2, 3) * torch.finfo(torch.float32).eps * result.singular_values[0, 0, -1]
        assert result.numerical_rank.item() == expected_rank
        assert result.numerical_rank.item() == int(
            torch.count_nonzero(result.singular_values[0, 0] > tolerance).item()
        )


def test_shared_underlying_input_storage_fails_closed():
    backing = torch.zeros(144, dtype=torch.uint8)
    normals = backing.view(torch.float64).reshape(1, 1, 6, 3)
    valid = backing[:6].view(torch.bool).reshape(1, 1, 6)
    assert normals.untyped_storage().data_ptr() == valid.untyped_storage().data_ptr()
    with pytest.raises(ValueError, match="own its exact storage|alias storage"):
        compute_exact_observability(normals, valid)


@pytest.mark.parametrize(
    "mutation,exception",
    [
        ("normal_type", TypeError), ("valid_type", TypeError), ("swapped", (TypeError, ValueError)),
        ("normal_rank", ValueError), ("normal_tail", ValueError), ("valid_shape", ValueError),
        ("empty_time", ValueError), ("empty_environment", ValueError), ("empty_rays", ValueError),
        ("normal_dtype", TypeError), ("valid_dtype", TypeError), ("nonfinite", ValueError),
        ("invalid_nonzero", ValueError), ("nonunit", ValueError), ("normal_view", ValueError),
        ("valid_view", ValueError), ("wrong_config", TypeError),
    ],
)
def test_malformed_inputs_fail_closed(mutation, exception):
    normals, valid = _inputs(torch.float64, "corner")
    args = [normals, valid]
    config = ExactObservabilityConfig()
    if mutation == "normal_type": args[0] = normals.tolist()
    elif mutation == "valid_type": args[1] = valid.tolist()
    elif mutation == "swapped": args = [valid, normals]
    elif mutation == "normal_rank": args[0] = torch.zeros(1, 6, 3, dtype=torch.float64)
    elif mutation == "normal_tail": args[0] = torch.zeros(1, 1, 6, 2, dtype=torch.float64)
    elif mutation == "valid_shape": args[1] = torch.zeros(1, 1, 5, dtype=torch.bool)
    elif mutation == "empty_time": args = [torch.zeros(0, 1, 1, 3, dtype=torch.float64), torch.zeros(0, 1, 1, dtype=torch.bool)]
    elif mutation == "empty_environment": args = [torch.zeros(1, 0, 1, 3, dtype=torch.float64), torch.zeros(1, 0, 1, dtype=torch.bool)]
    elif mutation == "empty_rays": args = [torch.zeros(1, 1, 0, 3, dtype=torch.float64), torch.zeros(1, 1, 0, dtype=torch.bool)]
    elif mutation == "normal_dtype": args[0] = normals.to(torch.float16)
    elif mutation == "valid_dtype": args[1] = valid.to(torch.int64)
    elif mutation == "nonfinite": args[0][0, 0, 0, 0] = float("nan")
    elif mutation == "invalid_nonzero": args[0][0, 0, 5, 0] = 1.0
    elif mutation == "nonunit": args[0][0, 0, 0, 0] = 2.0
    elif mutation == "normal_view": args[0] = normals[:, :, :, :]
    elif mutation == "valid_view": args[1] = valid[:, :, :]
    elif mutation == "wrong_config": config = object()
    with pytest.raises(exception):
        compute_exact_observability(*args, config)


def test_missing_extra_and_device_mismatch_fail_closed():
    normals, valid = _inputs(torch.float64, "corner")
    with pytest.raises(TypeError):
        compute_exact_observability(normals)
    with pytest.raises(TypeError):
        compute_exact_observability(normals, valid, ExactObservabilityConfig(), "extra")
    meta_valid = torch.ones(valid.shape, dtype=torch.bool, device="meta")
    with pytest.raises(ValueError, match="same device"):
        compute_exact_observability(normals, meta_valid)


def test_public_signature_and_exports_are_exact_and_evaluation_only():
    signature = inspect.signature(compute_exact_observability)
    assert tuple(signature.parameters) == ("surface_normals_w", "beam_valid", "config")
    import instinctlab.evaluation as evaluation
    assert evaluation.compute_exact_observability is compute_exact_observability
    assert not hasattr(__import__("instinctlab"), "compute_exact_observability")


def test_recursive_consumers_do_not_import_core_or_name_surface_normals():
    for package in CONSUMER_ROOTS:
        for path in sorted((ROOT / "instinctlab" / package).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            assert "surface_normal" not in source.lower(), path
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all("evaluation.observability" not in alias.name for alias in node.names), path
                elif isinstance(node, ast.ImportFrom):
                    assert "evaluation.observability" not in (node.module or ""), path


def _policy() -> dict[str, torch.Tensor]:
    lidar = torch.zeros(2, 12, 59, 360, dtype=torch.float32)
    lidar[:, 0::3] = 20.0
    lidar[:, 1::3] = 1.0
    lidar[:, 2::3] = 1.0
    state = torch.zeros(2, 52, dtype=torch.float32)
    state[:, 5::13] = 1.0
    return {"lidar_grid": lidar, "state_vec": state}


def test_surface_normal_perturbations_cannot_change_real_history_actor():
    torch.manual_seed(703)
    actor = HistoryActor().eval()
    policy = _policy()
    first_normals, first_valid = _inputs(torch.float32, "corridor")
    second_normals, second_valid = _inputs(torch.float32, "anisotropic")
    first_observability = compute_exact_observability(first_normals, first_valid)
    second_observability = compute_exact_observability(second_normals, second_valid)
    assert not torch.equal(first_observability.numerical_rank, second_observability.numerical_rank)
    cpu_rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        _, first_output = actor.distribution(policy)
        first_action, first_log_probability, _ = actor.sample(policy)
    torch.random.set_rng_state(cpu_rng)
    with torch.no_grad():
        _, second_output = actor.distribution(policy)
        second_action, second_log_probability, _ = actor.sample(policy)
    assert torch.equal(first_output.alpha, second_output.alpha)
    assert torch.equal(first_output.beta, second_output.beta)
    assert torch.equal(first_action, second_action)
    assert torch.equal(first_log_probability, second_log_probability)
    changed_policy = _policy()
    changed_policy["state_vec"][:, 0] = 1.0
    with torch.no_grad():
        changed_output = actor(changed_policy)
    assert not torch.equal(first_output.alpha, changed_output.alpha) or not torch.equal(first_output.beta, changed_output.beta)


def test_m7_002_runtime_observability_remains_explicitly_unavailable():
    source = (ROOT / "instinctlab" / "evaluation" / "runtime.py").read_text(encoding="utf-8")
    assert 'int(ObservabilitySource.UNAVAILABLE)' in source
    assert '("observability_valid", torch.zeros(' in source
    assert '("observability_value", torch.zeros(' in source
