"""SCALE-004 forward reconstruction checks in the approved float32 domain."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

import instinctlab.evaluation.raycaster_observability as adapter
from instinctlab.evaluation.raycaster_observability import (
    RAYCASTER_GEOMETRY_COMPARISON_STANDARD,
    RaycasterObservabilityConfig,
    acquire_exact_observability_from_source,
)


_DIRECTION = (0.4942087233066559, 0.8693432807922363, 0.0)
_PARAMETER_M = 39.0


def _source(starts: torch.Tensor) -> OrderedDict[str, object]:
    environments = starts.shape[0]
    directions = torch.tensor(_DIRECTION, dtype=torch.float32).reshape(1, 1, 3).repeat(
        environments, 1, 1
    ).clone()
    accepted_hits = starts + _PARAMETER_M * directions
    return OrderedDict((
        ("ray_starts_w", starts),
        ("ray_directions_w", directions),
        ("accepted_ray_hits_w", accepted_hits),
        ("accepted_beam_valid", torch.ones(environments, 1, dtype=torch.bool)),
        ("mesh_path", "/World/MID360StaticScene/Scale003CombinedMesh"),
        ("maximum_distance_m", 40.0),
        ("source_step", torch.ones(environments, dtype=torch.long)),
        ("reset_mask", torch.zeros(environments, dtype=torch.bool)),
    ))


def _acquire(monkeypatch, source: OrderedDict[str, object], parameter_delta: float = 0.0,
             accepted_hit_delta: float = 0.0):
    accepted_hits = source["accepted_ray_hits_w"]
    assert isinstance(accepted_hits, torch.Tensor)
    if accepted_hit_delta:
        accepted_hits[0, 0, 0] += accepted_hit_delta
    returned_hits = source["ray_starts_w"] + _PARAMETER_M * source["ray_directions_w"]
    parameters = torch.full(
        source["accepted_beam_valid"].shape,
        _PARAMETER_M + parameter_delta,
        dtype=torch.float32,
    )
    normals = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).reshape(1, 1, 3).repeat(
        returned_hits.shape[0], 1, 1
    ).clone()
    monkeypatch.setattr(
        adapter, "resolve_frozen_raycaster_mesh", lambda _: SimpleNamespace(id=31, device="cpu")
    )
    monkeypatch.setattr(
        adapter,
        "_raycast_mesh_with_normals",
        lambda *_: (returned_hits.clone(), parameters.clone(), normals.clone(), None),
    )
    return acquire_exact_observability_from_source(source)


def test_local_and_500m_cases_record_old_and_new_residuals_separately(monkeypatch):
    tolerance = RaycasterObservabilityConfig().distance_tolerance_m
    assert RAYCASTER_GEOMETRY_COMPARISON_STANDARD == "forward-reconstruction-v1"

    local = _acquire(monkeypatch, _source(torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)))
    assert local.distance_max_error_m.item() <= tolerance
    assert local.reconstruction_max_error_m.item() == 0.0

    translated = _acquire(
        monkeypatch, _source(torch.tensor([[[500.0, 500.0, 1.0]]], dtype=torch.float32))
    )
    assert translated.distance_max_error_m.item() > tolerance
    assert translated.reconstruction_max_error_m.item() == 0.0
    assert translated.hit_max_error_m.item() == 0.0


def test_32_environment_100m_grid_uses_forward_reconstruction_gate(monkeypatch):
    indices = torch.arange(32, dtype=torch.float32)
    starts = torch.zeros(32, 1, 3, dtype=torch.float32)
    starts[:, 0, 0] = (indices.remainder(8) * 100.0)
    starts[:, 0, 1] = (torch.div(indices, 8, rounding_mode="floor") * 100.0)
    starts[:, 0, 2] = 1.0
    result = _acquire(monkeypatch, _source(starts))
    assert result.distance_max_error_m.item() > RaycasterObservabilityConfig().distance_tolerance_m
    assert result.reconstruction_max_error_m.item() == 0.0
    assert result.hit_max_error_m.item() == 0.0


def test_large_world_parameter_mismatch_still_fails(monkeypatch):
    source = _source(torch.tensor([[[500.0, 500.0, 1.0]]], dtype=torch.float32))
    with pytest.raises(RuntimeError, match="forward-reconstructed hits"):
        _acquire(monkeypatch, source, parameter_delta=1.0e-3)


def test_large_world_same_state_hit_mismatch_still_fails_first(monkeypatch):
    source = _source(torch.tensor([[[500.0, 500.0, 1.0]]], dtype=torch.float32))
    with pytest.raises(RuntimeError, match="same-state sensor hits"):
        _acquire(monkeypatch, source, accepted_hit_delta=1.0e-3)
