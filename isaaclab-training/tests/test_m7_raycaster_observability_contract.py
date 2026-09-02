"""Focused exact RayCaster normal acquisition and isolation contract tests."""

from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import FrozenInstanceError, fields
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

import instinctlab.evaluation.raycaster_observability as adapter
from instinctlab.evaluation.observability import ExactObservabilityResult
from instinctlab.evaluation.raycaster_observability import (
    RaycasterObservabilityConfig,
    RaycasterObservabilityResult,
    acquire_exact_raycaster_observability,
    acquire_exact_observability_from_source,
    resolve_frozen_raycaster_mesh,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixture(dtype=torch.float32):
    starts = torch.zeros(1, 4, 3, dtype=dtype)
    directions = torch.tensor(
        [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]]],
        dtype=dtype,
    ).reshape(1, 4, 3).clone()
    valid = torch.tensor([[True, True, True, False]])
    distance = torch.tensor([[2.0, 3.0, 4.0, float("inf")]], dtype=dtype)
    hits = torch.full_like(starts, float("inf"))
    hits[valid] = starts[valid] + distance[valid].unsqueeze(-1) * directions[valid]
    normals = torch.tensor(
        [[[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0],
            [float("inf"), float("inf"), float("inf")]]]], dtype=dtype
    ).reshape(1, 4, 3).clone()
    return starts, directions, hits, valid, distance, normals


def _install_cast(monkeypatch, mutation=None):
    starts, directions, hits, valid, distance, normals = _fixture()

    def fake(ray_starts_w, ray_directions_w, mesh, maximum_distance_m):
        del mesh, maximum_distance_m
        result = [hits.clone(), distance.clone(), normals.clone(), None]
        if mutation is not None:
            mutation(result, ray_starts_w, ray_directions_w, valid)
        return tuple(result)

    monkeypatch.setattr(adapter, "_raycast_mesh_with_normals", fake)
    return starts, directions, hits, valid


def _run(monkeypatch, mutation=None, config=RaycasterObservabilityConfig()):
    starts, directions, hits, valid = _install_cast(monkeypatch, mutation)
    mesh = SimpleNamespace(id=17, device="cuda:0")
    return acquire_exact_raycaster_observability(starts, directions, mesh, 40.0, hits, valid, config), (
        starts, directions, hits, valid
    )


def _run_parameter_case(monkeypatch, scale, parameter=2.0, returned_hit_offset=0.0, accepted_hit_offset=0.0):
    starts = torch.zeros(1, 1, 3, dtype=torch.float32)
    directions = torch.tensor([[[scale, 0.0, 0.0]]], dtype=torch.float32)
    valid = torch.ones(1, 1, dtype=torch.bool)
    accepted_hits = starts + float(parameter) * directions
    accepted_hits[0, 0, 0] += accepted_hit_offset
    returned_hits = starts + float(parameter) * directions
    returned_hits[0, 0, 0] += returned_hit_offset
    parameters = torch.tensor([[parameter]], dtype=torch.float32)
    normals = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)

    def fake(ray_starts_w, ray_directions_w, mesh, maximum_distance_m):
        assert ray_starts_w is starts and ray_directions_w is directions
        assert mesh.id == 23 and maximum_distance_m == 40.0
        return returned_hits.clone(), parameters.clone(), normals.clone(), None

    monkeypatch.setattr(adapter, "_raycast_mesh_with_normals", fake)
    result = acquire_exact_raycaster_observability(
        starts, directions, SimpleNamespace(id=23, device="cpu"), 40.0, accepted_hits, valid
    )
    return result, parameters


def test_admitted_nonunit_parameter_is_converted_to_metric_metres(monkeypatch):
    result, raw_parameters = _run_parameter_case(monkeypatch, 1.000008, parameter=2.0)
    independent_expected_metres = 2.000016
    assert result.ray_distances_m.item() == pytest.approx(independent_expected_metres, abs=3.0e-7)
    assert result.distance_max_error_m.item() <= RaycasterObservabilityConfig().distance_tolerance_m
    assert result.ray_distances_m.data_ptr() != raw_parameters.data_ptr()
    assert result.ray_distances_m.grad_fn is None and not result.ray_distances_m.requires_grad


def test_direction_error_float32_accepted_and_rejected_neighbors(monkeypatch):
    one = torch.tensor(1.0, dtype=torch.float32)
    rounded_boundary = torch.tensor(1.0 + RaycasterObservabilityConfig().unit_tolerance, dtype=torch.float32)
    accepted_scale = torch.nextafter(rounded_boundary, one).item()
    rejected_scale = rounded_boundary.item()
    assert accepted_scale - 1.0 <= RaycasterObservabilityConfig().unit_tolerance
    assert rejected_scale - 1.0 > RaycasterObservabilityConfig().unit_tolerance
    _run_parameter_case(monkeypatch, accepted_scale)
    with pytest.raises(ValueError, match="unit length"):
        _run_parameter_case(monkeypatch, rejected_scale)


def test_metric_distance_effective_float32_boundary(monkeypatch):
    base = torch.tensor(2.0, dtype=torch.float32)
    first_rejected = torch.tensor(2.0 + RaycasterObservabilityConfig().distance_tolerance_m, dtype=torch.float32)
    effective_boundary = torch.nextafter(first_rejected, base)
    below = torch.nextafter(effective_boundary, base)
    for parameter in (below.item(), effective_boundary.item()):
        starts, directions, hits, valid, _, normals = _fixture()
        returned = torch.tensor([[parameter, 3.0, 4.0, float("inf")]], dtype=torch.float32)
        monkeypatch.setattr(adapter, "_raycast_mesh_with_normals", lambda *_: (hits.clone(), returned.clone(), normals.clone(), None))
        result = acquire_exact_raycaster_observability(starts, directions, SimpleNamespace(id=1, device="cpu"), 40.0, hits, valid)
        assert result.distance_max_error_m.item() <= RaycasterObservabilityConfig().distance_tolerance_m
    starts, directions, hits, valid, _, normals = _fixture()
    rejected = torch.tensor([[first_rejected.item(), 3.0, 4.0, float("inf")]], dtype=torch.float32)
    monkeypatch.setattr(adapter, "_raycast_mesh_with_normals", lambda *_: (hits.clone(), rejected.clone(), normals.clone(), None))
    with pytest.raises(RuntimeError, match="distances disagree"):
        acquire_exact_raycaster_observability(starts, directions, SimpleNamespace(id=1, device="cpu"), 40.0, hits, valid)


@pytest.mark.parametrize("parameter,match", [
    (-1.0, "nonnegative"), (float("nan"), "finite"), (float("inf"), "finite"),
    (-float("inf"), "finite"),
])
def test_valid_raw_parameter_values_fail_closed(monkeypatch, parameter, match):
    starts, directions, hits, valid, distance, normals = _fixture()
    distance[0, 0] = parameter
    monkeypatch.setattr(adapter, "_raycast_mesh_with_normals", lambda *_: (hits.clone(), distance.clone(), normals.clone(), None))
    with pytest.raises(RuntimeError, match=match):
        acquire_exact_raycaster_observability(starts, directions, SimpleNamespace(id=1, device="cpu"), 40.0, hits, valid)


@pytest.mark.parametrize("parameter", [0.0, float("nan"), -float("inf")])
def test_invalid_beam_raw_parameter_requires_positive_infinity(monkeypatch, parameter):
    starts, directions, hits, valid, distance, normals = _fixture()
    distance[0, 3] = parameter
    monkeypatch.setattr(adapter, "_raycast_mesh_with_normals", lambda *_: (hits.clone(), distance.clone(), normals.clone(), None))
    with pytest.raises(RuntimeError, match="invalid ray parameters"):
        acquire_exact_raycaster_observability(starts, directions, SimpleNamespace(id=1, device="cpu"), 40.0, hits, valid)


@pytest.mark.parametrize("returned_offset,accepted_offset", [(2.0e-5, 0.0), (0.0, 2.0e-5)])
def test_hit_inconsistency_precedes_metric_acceptance(monkeypatch, returned_offset, accepted_offset):
    with pytest.raises(RuntimeError, match="sensor hits"):
        _run_parameter_case(monkeypatch, 1.0, returned_hit_offset=returned_offset, accepted_hit_offset=accepted_offset)


def test_positive_exact_handoff_all_fields_ownership_and_planar_degeneracy(monkeypatch):
    result, inputs = _run(monkeypatch)
    assert isinstance(result, RaycasterObservabilityResult)
    assert isinstance(result.observability, ExactObservabilityResult)
    assert result.ray_hits_w.shape == (1, 4, 3)
    assert result.ray_distances_m.shape == (1, 4)
    assert result.raw_surface_normals_w.shape == (1, 4, 3)
    assert result.surface_normals_w.shape == (1, 1, 4, 3)
    assert result.beam_valid.shape == (1, 1, 4)
    assert result.hit_max_error_m.shape == result.distance_max_error_m.shape == ()
    assert result.hit_max_error_m.item() == 0.0
    assert result.distance_max_error_m.item() == 0.0
    assert torch.isinf(result.raw_surface_normals_w[0, 3]).all()
    assert torch.equal(result.surface_normals_w[0, 0, 3], torch.zeros(3))
    exact = result.observability
    assert exact.numerical_rank.item() == 1
    assert torch.equal(exact.singular_values[0, 0, :2], torch.zeros(2))
    assert math.isclose(exact.singular_values[0, 0, 2].item(), math.sqrt(3.0), rel_tol=1e-6)
    assert exact.sigma_min.item() == 0.0
    assert not exact.condition_valid.item() and exact.condition_number.item() == 0.0
    assert not exact.weak_direction_valid.item() and torch.count_nonzero(exact.weak_direction_w) == 0
    assert exact.valid_ray_count.item() == 3 and exact.total_ray_count.item() == 4
    assert exact.valid_beam_fraction.item() == 0.75
    direct = [getattr(result, field.name) for field in fields(result) if field.name != "observability"]
    nested = [getattr(exact, field.name) for field in fields(exact)]
    pointers = [value.untyped_storage().data_ptr() for value in direct + nested]
    assert len(set(pointers)) == len(pointers)
    assert {value.untyped_storage().data_ptr() for value in inputs}.isdisjoint(pointers)
    snapshots = [value.clone() for value in direct + nested]
    for value in inputs:
        if value.dtype == torch.bool:
            value.logical_not_()
        else:
            value.zero_()
    for actual, expected in zip(direct + nested, snapshots, strict=True):
        assert torch.equal(actual, expected)
        assert actual.grad_fn is None and not actual.requires_grad
    direct[0].zero_()
    for actual, expected in zip((direct + nested)[1:], snapshots[1:], strict=True):
        assert torch.equal(actual, expected)
    with pytest.raises(FrozenInstanceError):
        result.beam_valid = torch.zeros_like(result.beam_valid)


def test_official_call_flags_and_argument_order_are_exact(monkeypatch):
    starts, directions, hits, valid, distance, normals = _fixture()
    captured = {}

    def official(ray_starts, ray_directions, *, mesh, max_dist, return_distance, return_normal):
        captured.update(locals())
        return hits.clone(), distance.clone(), normals.clone(), None

    fake_module = SimpleNamespace(raycast_mesh=official)
    monkeypatch.setitem(__import__("sys").modules, "isaaclab.utils.warp", fake_module)
    mesh = SimpleNamespace(id=9, device="cuda:0")
    result = acquire_exact_raycaster_observability(starts, directions, mesh, 40.0, hits, valid)
    assert result.observability.numerical_rank.item() == 1
    assert tuple(captured)[:6] == (
        "ray_starts", "ray_directions", "mesh", "max_dist", "return_distance", "return_normal"
    )
    assert captured["ray_starts"] is starts and captured["ray_directions"] is directions
    assert captured["mesh"] is mesh and captured["max_dist"] == 40.0
    assert captured["return_distance"] is True and captured["return_normal"] is True


@pytest.mark.parametrize(
    "name,mutation,match",
    [
        ("tuple_type", lambda result, *_: result.append(0), "four ordered"),
        ("face_id", lambda result, *_: result.__setitem__(3, torch.zeros(1, 4, dtype=torch.int32)), "face ids"),
        ("hit_shape", lambda result, *_: result.__setitem__(0, result[0][:, :3].clone()), "hits metadata"),
        ("distance_shape", lambda result, *_: result.__setitem__(1, result[1][:, :3].clone()), "ray parameters metadata"),
        ("normal_shape", lambda result, *_: result.__setitem__(2, result[2][:, :3].clone()), "normals metadata"),
        ("hit_dtype", lambda result, *_: result.__setitem__(0, result[0].double()), "hits dtype/device"),
        ("distance_dtype", lambda result, *_: result.__setitem__(1, result[1].double()), "ray parameters dtype/device"),
        ("normal_dtype", lambda result, *_: result.__setitem__(2, result[2].double()), "normals dtype/device"),
        ("hit_mismatch", lambda result, *_: result[0].__setitem__((0, 0, 0), 2.1), "sensor hits"),
        ("validity_mismatch", lambda result, *_: result[0].__setitem__((0, 3), torch.zeros(3)), "validity disagrees"),
        ("distance_mismatch", lambda result, *_: result[1].__setitem__((0, 0), 2.1), "distances disagree"),
        ("valid_normal_nonfinite", lambda result, *_: result[2].__setitem__((0, 0, 0), float("nan")), "valid normals"),
        ("valid_normal_nonunit", lambda result, *_: result[2].__setitem__((0, 0), torch.tensor([0.0, 0.0, 2.0])), "unit length"),
        ("miss_distance", lambda result, *_: result[1].__setitem__((0, 3), 40.0), "invalid ray parameters"),
        ("miss_normal", lambda result, *_: result[2].__setitem__((0, 3), torch.zeros(3)), "invalid normals"),
        ("mutate_start", lambda result, starts, *_: starts.__setitem__((0, 0, 0), 1.0), "mutated ray_starts"),
    ],
)
def test_return_metadata_fidelity_normal_and_mutation_fail_closed(monkeypatch, name, mutation, match):
    del name
    with pytest.raises((RuntimeError, ValueError), match=match):
        _run(monkeypatch, mutation)


@pytest.mark.parametrize(
    "case,exception",
    [
        ("starts_type", TypeError), ("directions_type", TypeError), ("hits_type", TypeError),
        ("valid_type", TypeError), ("starts_shape", ValueError), ("directions_shape", ValueError),
        ("hits_shape", ValueError), ("valid_shape", ValueError), ("empty_env", ValueError),
        ("empty_ray", ValueError), ("starts_dtype", TypeError), ("directions_dtype", TypeError),
        ("hits_dtype", TypeError), ("valid_dtype", TypeError), ("starts_nonfinite", ValueError),
        ("directions_nonfinite", ValueError), ("directions_nonunit", ValueError),
        ("valid_hit_nonfinite", ValueError), ("invalid_hit_finite", ValueError),
        ("distance_type", TypeError), ("distance_zero", ValueError), ("distance_nonfinite", ValueError),
        ("hit_beyond_max", ValueError), ("mesh", TypeError), ("config", TypeError),
        ("starts_view", ValueError), ("directions_view", ValueError), ("hits_view", ValueError),
        ("valid_view", ValueError),
    ],
)
def test_input_contract_negatives_fail_before_cast(monkeypatch, case, exception):
    starts, directions, hits, valid = _install_cast(monkeypatch)
    args = [starts, directions, SimpleNamespace(id=1, device="cuda:0"), 40.0, hits, valid]
    config = RaycasterObservabilityConfig()
    if case == "starts_type": args[0] = starts.tolist()
    elif case == "directions_type": args[1] = directions.tolist()
    elif case == "hits_type": args[4] = hits.tolist()
    elif case == "valid_type": args[5] = valid.tolist()
    elif case == "starts_shape": args[0] = torch.zeros(4, 3)
    elif case == "directions_shape": args[1] = torch.zeros(1, 3, 3)
    elif case == "hits_shape": args[4] = torch.zeros(1, 3, 3)
    elif case == "valid_shape": args[5] = torch.zeros(1, 3, dtype=torch.bool)
    elif case == "empty_env": args = [torch.zeros(0, 1, 3), torch.zeros(0, 1, 3), args[2], 40.0, torch.zeros(0, 1, 3), torch.zeros(0, 1, dtype=torch.bool)]
    elif case == "empty_ray": args = [torch.zeros(1, 0, 3), torch.zeros(1, 0, 3), args[2], 40.0, torch.zeros(1, 0, 3), torch.zeros(1, 0, dtype=torch.bool)]
    elif case == "starts_dtype": args[0] = starts.double()
    elif case == "directions_dtype": args[1] = directions.double()
    elif case == "hits_dtype": args[4] = hits.double()
    elif case == "valid_dtype": args[5] = valid.long()
    elif case == "starts_nonfinite": args[0][0, 0, 0] = float("nan")
    elif case == "directions_nonfinite": args[1][0, 0, 0] = float("inf")
    elif case == "directions_nonunit": args[1][0, 0, 0] = 2.0
    elif case == "valid_hit_nonfinite": args[4][0, 0, 0] = float("nan")
    elif case == "invalid_hit_finite": args[4][0, 3] = torch.zeros(3)
    elif case == "distance_type": args[3] = True
    elif case == "distance_zero": args[3] = 0.0
    elif case == "distance_nonfinite": args[3] = float("inf")
    elif case == "hit_beyond_max": args[3] = 1.0
    elif case == "mesh": args[2] = object()
    elif case == "config": config = object()
    elif case == "starts_view": args[0] = starts[:, :, :]
    elif case == "directions_view": args[1] = directions[:, :, :]
    elif case == "hits_view": args[4] = hits[:, :, :]
    elif case == "valid_view": args[5] = valid[:, :]
    with pytest.raises(exception):
        acquire_exact_raycaster_observability(*args, config)


def test_device_mismatch_and_shared_storage_fail_closed(monkeypatch):
    starts, directions, hits, valid = _install_cast(monkeypatch)
    meta_valid = torch.ones(valid.shape, dtype=torch.bool, device="meta")
    with pytest.raises(ValueError, match="same device"):
        acquire_exact_raycaster_observability(starts, directions, SimpleNamespace(id=1, device="cuda"), 40.0, hits, meta_valid)
    backing = torch.zeros(48, dtype=torch.uint8)
    aliased_starts = backing.view(torch.float32).reshape(1, 4, 3)
    aliased_valid = backing[:4].view(torch.bool).reshape(1, 4)
    with pytest.raises(ValueError, match="exact storage|alias storage"):
        acquire_exact_raycaster_observability(aliased_starts, directions, SimpleNamespace(id=1, device="cuda"), 40.0, hits, aliased_valid)


def test_config_boundaries_immutability_and_no_silent_normalization(monkeypatch):
    config = RaycasterObservabilityConfig(unit_tolerance=1.0e-3, distance_tolerance_m=1.0e-3)
    _run(monkeypatch, config=config)
    with pytest.raises(FrozenInstanceError):
        config.unit_tolerance = 1.0
    for kwargs in ({"unit_tolerance": True}, {"unit_tolerance": 0.0},
                   {"distance_tolerance_m": -1.0}, {"distance_tolerance_m": float("nan")}):
        with pytest.raises((TypeError, ValueError)):
            RaycasterObservabilityConfig(**kwargs)

    def slightly_nonunit(result, *_):
        result[2][0, 0] = torch.tensor([0.0, 0.0, 1.0009])

    qualified, _ = _run(monkeypatch, slightly_nonunit, config)
    assert qualified.surface_normals_w[0, 0, 0, 2].item() == pytest.approx(1.0009)


def _source() -> OrderedDict[str, object]:
    starts, directions, hits, valid, _, _ = _fixture()
    return OrderedDict((
        ('ray_starts_w', starts), ('ray_directions_w', directions),
        ('accepted_ray_hits_w', hits), ('accepted_beam_valid', valid),
        ('mesh_path', '/World/TestMesh'), ('maximum_distance_m', 40.0),
        ('source_step', torch.ones(1, dtype=torch.long)),
        ('reset_mask', torch.zeros(1, dtype=torch.bool)),
    ))


@pytest.mark.parametrize('mutation', [
    'missing', 'extra', 'reordered', 'step_shape', 'step_dtype', 'step_zero',
    'reset_shape', 'reset_dtype', 'step_view', 'reset_alias',
])
def test_ordered_environment_source_schema_fails_closed(monkeypatch, mutation):
    source = _source()
    monkeypatch.setattr(adapter, 'resolve_frozen_raycaster_mesh', lambda _: SimpleNamespace(id=1, device='cpu'))
    _install_cast(monkeypatch)
    if mutation == 'missing': source.pop('mesh_path')
    elif mutation == 'extra': source['extra'] = 1
    elif mutation == 'reordered': source.move_to_end('ray_starts_w')
    elif mutation == 'step_shape': source['source_step'] = torch.ones(1, 1, dtype=torch.long)
    elif mutation == 'step_dtype': source['source_step'] = torch.ones(1)
    elif mutation == 'step_zero': source['source_step'].zero_()
    elif mutation == 'reset_shape': source['reset_mask'] = torch.zeros(1, 1, dtype=torch.bool)
    elif mutation == 'reset_dtype': source['reset_mask'] = torch.zeros(1, dtype=torch.long)
    elif mutation == 'step_view': source['source_step'] = torch.ones(2, dtype=torch.long)[:1]
    elif mutation == 'reset_alias': source['reset_mask'] = source['accepted_beam_valid'][:, :1]
    with pytest.raises((TypeError, ValueError)):
        acquire_exact_observability_from_source(source)


def test_guarded_mesh_resolver_confines_cache_access(monkeypatch):
    mesh = SimpleNamespace(id=17, device='cuda:0')
    monkeypatch.setattr(adapter, 'frozen_raycaster_provenance', lambda: OrderedDict())
    monkeypatch.setitem(sys.modules, 'isaaclab.sensors', SimpleNamespace(
        RayCaster=SimpleNamespace(meshes={'/World/TestMesh': mesh})
    ))
    assert resolve_frozen_raycaster_mesh('/World/TestMesh') is mesh
    with pytest.raises(RuntimeError, match='unavailable'):
        resolve_frozen_raycaster_mesh('/World/Missing')


def test_module_is_not_reexported_and_consumers_remain_normal_free():
    evaluation = importlib.import_module("instinctlab.evaluation")
    assert not hasattr(evaluation, "acquire_exact_raycaster_observability")
    source = (ROOT / "instinctlab/evaluation/raycaster_observability.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert any(isinstance(node, ast.ImportFrom) and node.module == "isaaclab.utils.warp" for node in ast.walk(tree))
    for package in ("control", "envs", "observations", "task", "training"):
        for path in (ROOT / "instinctlab" / package).rglob("*.py"):
            consumer = path.read_text(encoding="utf-8").lower()
            assert "raycaster_observability" not in consumer, path
            assert "surface_normal" not in consumer, path


def test_m7_003_core_is_unmodified_and_m7_005_runtime_is_exact():
    core = ROOT / "instinctlab/evaluation/observability.py"
    runtime = ROOT / "instinctlab/evaluation/runtime.py"
    assert core.exists() and "compute_exact_observability" in core.read_text(encoding="utf-8")
    runtime_source = runtime.read_text(encoding="utf-8")
    assert "int(ObservabilitySource.EXACT)" in runtime_source
    assert '("observability_valid", torch.ones(' in runtime_source
    assert "distill_exact_runtime_observability" in runtime_source
