"""Discriminating semantic, negative, source, and atomic M5 tests."""

from __future__ import annotations

import ast
from dataclasses import fields
import inspect
import math
from pathlib import Path

import pytest
import torch

from instinctlab.control.anchor import (
    ANCHOR_CACHE_KEYS,
    ANCHOR_METRIC_KEYS,
    AnchorConfig,
    MeasurementAnchor,
    RESET_COMMAND_EXIT,
    RESET_EPISODE,
    RESET_EXPLICIT,
    RESET_MINIMUM_VALID,
    huber_loss,
)
from instinctlab.control.command_adapter import rotate_body_to_world
from instinctlab.control.governor import GovernorConfig, b0_governor, decode_governor, extract_latest_commands
from instinctlab.control.ics import (
    ICS_CACHE_KEYS,
    ICS_METRIC_KEYS,
    ICSConfig,
    RAY_COUNT,
    attenuate_command,
    extract_lidar_history,
    mounted_mid360_rays_body,
)
from instinctlab.control.pipeline import (
    INTERNAL_SNAPSHOT_KEYS,
    PUBLIC_RESULT_KEYS,
    ControlCoreConfig,
    ControlCorePipeline,
    ControlCoreResult,
)

ROOT = Path(__file__).resolve().parents[1]


def _actor(batch: int = 2, dtype: torch.dtype = torch.float32, device: str = "cpu"):
    lidar = torch.empty(batch, 12, 59, 360, dtype=dtype, device=device)
    lidar[:, 0::3] = 40.0
    lidar[:, 1::3] = 1.0
    lidar[:, 2::3] = 1.0
    state = torch.zeros(batch, 52, dtype=dtype, device=device)
    state[:, 3::13] = 0.0
    state[:, 4::13] = 0.0
    state[:, 5::13] = 1.0
    return lidar, state


def _anchor_inputs(batch: int = 2):
    ranges = torch.full((batch, 59, 360), 2.0)
    validity = torch.ones_like(ranges, dtype=torch.bool)
    reliability = torch.ones_like(ranges)
    command = torch.zeros(batch, 3)
    reset = torch.zeros(batch, dtype=torch.bool)
    return ranges, validity, reliability, command, reset


def _quat_yaw(degrees: float, batch: int = 1):
    half = math.radians(degrees) / 2.0
    return torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]]).repeat(batch, 1)


def _quat_rpy(roll: float, pitch: float, yaw: float):
    r, p, y = [math.radians(value) / 2.0 for value in (roll, pitch, yaw)]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return torch.tensor([[
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]])


def test_default_configs_and_signatures_are_exact():
    assert GovernorConfig() == GovernorConfig(0.5, 2.0, 0.0)
    assert AnchorConfig() == AnchorConfig(0.05, 0.15, 0.1, 0.1, 0.25)
    assert ICSConfig() == ICSConfig(0.8, 0.25, 2.0, 0.5, 0.1, 2.0, 0.0, 0.0, False, 1.0)
    assert tuple(inspect.signature(ControlCorePipeline.__call__).parameters) == (
        "self", "lidar_grid", "state_vec", "governor_action", "history_dt_s",
        "episode_reset", "explicit_reset", "controller_quaternion_wxyz",
    )
    with pytest.raises(ValueError):
        AnchorConfig(eps_enter=0.15, eps_exit=0.15)
    with pytest.raises(ValueError):
        ICSConfig(emergency_clearance_m=0.9)


def test_governor_endpoints_formula_smoothing_and_direction_clip():
    _, state = _actor()
    state[:, -7:-4] = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    state[:, -4:-1] = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    action = torch.tensor([[1.0, 1.0, 0.5, 0.0], [0.0, 0.5, 1.0, 0.5]])
    out = decode_governor(action, state, GovernorConfig(0.5, 1.0, 0.25))
    raw0 = 0.75 * torch.tensor([2.5, 0.0, -0.5]) + 0.25 * torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(out.alpha, action[:, :1])
    assert torch.allclose(out.v_corr_b, (2.0 * action[:, 1:] - 1.0) * 0.5)
    assert torch.allclose(out.v_gov_b[0], raw0 / raw0.norm())
    assert torch.all(torch.linalg.vector_norm(out.v_gov_b, dim=-1) <= 1.0)


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan")])
def test_governor_rejects_action_domain_instead_of_clamping(bad):
    _, state = _actor(1)
    action = torch.full((1, 4), 0.5)
    action[0, 0] = bad
    with pytest.raises(ValueError):
        decode_governor(action, state)
    with pytest.raises(ValueError):
        decode_governor(torch.zeros(1, 5), state)


def test_b0_and_exact_m4_latest_command_extraction():
    _, state = _actor(1)
    state[:, 6:9] = 9.0
    state[:, -7:-4] = torch.tensor([[3.0, 4.0, 0.0]])
    state[:, -4:-1] = torch.tensor([[0.1, 0.2, 0.3]])
    command, previous = extract_latest_commands(state)
    assert command.tolist() == [[3.0, 4.0, 0.0]]
    assert torch.allclose(previous, torch.tensor([[0.1, 0.2, 0.3]]))
    out = b0_governor(state, 2.0)
    assert out.alpha.tolist() == [[1.0]] and torch.count_nonzero(out.v_corr_b) == 0
    assert torch.allclose(out.v_gov_b, torch.tensor([[1.2, 1.6, 0.0]]))


def test_anchor_enter_hold_capture_immutability_and_huber_oracle():
    ranges, validity, reliability, command, reset = _anchor_inputs(1)
    anchor = MeasurementAnchor()
    first = anchor.update(ranges, validity, reliability, command, reset, reset)
    snapshot = anchor.cache_snapshot()
    assert first.metrics["active"].item() == 1.0
    changed = ranges + 0.5
    second = anchor.update(changed, validity, reliability, command, reset, reset)
    assert torch.equal(snapshot["range_star_m"], anchor.cache_snapshot()["range_star_m"])
    assert math.isclose(second.metrics["mean_absolute_error_m"].item(), 0.5, rel_tol=1e-6)
    assert math.isclose(second.metrics["huber_loss"].item(), 0.09375, rel_tol=1e-6)
    assert torch.allclose(huber_loss(torch.tensor([0.1, 0.5]), 0.25), torch.tensor([0.005, 0.09375]))


def test_anchor_reset_precedence_selectivity_and_minimum_valid():
    ranges, validity, reliability, command, reset = _anchor_inputs(2)
    anchor = MeasurementAnchor()
    anchor.update(ranges, validity, reliability, command, reset, reset)
    command[0, 0] = 0.15
    explicit = torch.tensor([True, False])
    episode = torch.tensor([True, False])
    out = anchor.update(ranges, validity, reliability, command, episode, explicit)
    assert out.metrics["reset_code"].reshape(-1).tolist() == [RESET_EPISODE, 0]
    assert out.metrics["active"].reshape(-1).tolist() == [0.0, 1.0]
    validity[1] = False
    ranges[1] = 40.0
    reliability[1] = 0.0
    out = anchor.update(ranges, validity, reliability, torch.zeros_like(command), reset, reset)
    assert out.metrics["reset_code"][1].item() == RESET_MINIMUM_VALID
    assert RESET_EXPLICIT < RESET_COMMAND_EXIT < RESET_MINIMUM_VALID


def test_anchor_mixed_reliability_keys_and_failed_call_atomicity():
    ranges, validity, reliability, command, reset = _anchor_inputs(1)
    reliability[:, :, 180:] = 0.05
    anchor = MeasurementAnchor()
    out = anchor.update(ranges, validity, reliability, command, reset, reset)
    before = anchor.cache_snapshot()
    assert set(out.metrics) == ANCHOR_METRIC_KEYS
    assert set(before) == ANCHOR_CACHE_KEYS
    assert math.isclose(out.metrics["usable_fraction"].item(), 0.5, rel_tol=1e-6)
    malformed = ranges.clone()
    malformed[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        anchor.update(malformed, validity, reliability, command, reset, reset)
    for key in before:
        assert torch.equal(before[key], anchor.cache_snapshot()[key])


def test_mounted_rays_count_order_norm_and_axis_sentinels():
    rays = mounted_mid360_rays_body(dtype=torch.float64)
    assert rays.shape == (RAY_COUNT, 3)
    assert torch.allclose(torch.linalg.vector_norm(rays, dim=-1), torch.ones(RAY_COUNT, dtype=torch.float64), atol=1e-12)
    elevation = math.radians(-7.0 + 7.0 * 59.0 / 58.0)
    expected = torch.tensor([
        math.cos(math.radians(30.0)) * math.cos(elevation) + math.sin(math.radians(30.0)) * math.sin(elevation),
        0.0,
        -math.sin(math.radians(30.0)) * math.cos(elevation) + math.cos(math.radians(30.0)) * math.sin(elevation),
    ], dtype=torch.float64)
    assert torch.allclose(rays[7 * 360 + 180], expected, atol=1e-12)
    assert rays[7 * 360 + 181, 1] > 0.0


def test_history_interleave_is_oldest_to_newest_and_zero_frames_are_ignored():
    lidar, _ = _actor(1)
    for index in range(4):
        lidar[:, index * 3] = float(index + 1)
    ranges, validity, reliability = extract_lidar_history(lidar)
    assert ranges[0, :, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    lidar[:, :3] = 0.0
    rays = mounted_mid360_rays_body()
    out = attenuate_command(lidar, rays, torch.zeros(1, 3), 0.1)
    assert torch.count_nonzero(out.cache["active_mask"]) == 0
    assert out.metrics["beta"].item() == 1.0


def _ics_case(clearance: float, speed: float, *, emergency: bool = False, config=ICSConfig()):
    lidar, _ = _actor(1)
    rays = mounted_mid360_rays_body()
    beam = int(torch.argmax(rays[:, 0]).item())
    row, column = divmod(beam, 360)
    lidar[:, 0::3, row, column] = clearance
    if emergency:
        lidar[:, 0::3, row, column] = 0.2
    return attenuate_command(lidar, rays, torch.tensor([[speed, 0.0, 0.0]]), 0.1, config)


def test_ics_empty_active_emergency_and_braking_oracle():
    empty = _ics_case(5.0, 1.0)
    assert empty.metrics["beta"].item() == 1.0
    emergency = _ics_case(0.2, 1.0, emergency=True)
    assert emergency.metrics["emergency"].item() == 1.0
    assert torch.count_nonzero(emergency.v_final_b) == 0
    active = _ics_case(1.0, 2.0)
    maximum_x = mounted_mid360_rays_body()[:, 0].max().item()
    expected = math.sqrt(2.0 * 2.0 * (1.0 - 0.8)) / (2.0 * maximum_x)
    assert math.isclose(active.metrics["beta"].item(), expected, rel_tol=2e-5)
    assert torch.allclose(active.v_final_b, active.metrics["beta"] * torch.tensor([[2.0, 0.0, 0.0]]))


def test_ics_monotonic_latency_margin_filter_and_mixed_invalid_40m():
    clear = _ics_case(1.2, 1.0).metrics["beta"].item()
    close = _ics_case(0.9, 1.0).metrics["beta"].item()
    fast = _ics_case(1.2, 2.0).metrics["beta"].item()
    margin = _ics_case(1.2, 1.0, config=ICSConfig(latency_s=0.1, clearance_margin_m=0.1)).metrics["beta"].item()
    assert close <= clear and fast <= clear and margin <= clear
    filter_lidar, _ = _actor(1)
    filter_rays = mounted_mid360_rays_body()
    filter_beam = int(torch.argmax(filter_rays[:, 0]).item())
    filter_row, filter_column = divmod(filter_beam, 360)
    filter_lidar[:, 0::3, filter_row, filter_column] = torch.tensor([1.3, 1.2, 1.1, 1.0]).reshape(1, 4)
    unfiltered = attenuate_command(filter_lidar, filter_rays, torch.zeros(1, 3), 0.1)
    filtered = attenuate_command(
        filter_lidar, filter_rays, torch.zeros(1, 3), 0.1, ICSConfig(range_rate_filter=True)
    )
    assert unfiltered.metrics["beta"].item() == 1.0 and filtered.metrics["beta"].item() < 1.0
    assert set(filtered.metrics) == ICS_METRIC_KEYS and set(filtered.cache) == ICS_CACHE_KEYS
    lidar, _ = _actor(1)
    lidar[:, 1::3, 0, 0] = 0.0
    lidar[:, 2::3, 0, 0] = 0.0
    out = attenuate_command(lidar, mounted_mid360_rays_body(), torch.tensor([[1.0, 0.0, 0.0]]), 0.1)
    assert out.cache["active_mask"][0, 0].item() is False
    lidar[:, 0::3, 7, 180] = 0.2
    lidar[:, 2::3, 7, 180] = 0.05
    stale = attenuate_command(lidar, mounted_mid360_rays_body(), torch.tensor([[1.0, 0.0, 0.0]]), 0.1)
    assert stale.metrics["emergency"].item() == 0.0


def test_ics_rejects_malformed_m4_pairs_and_nonfinite_governed_command():
    lidar, _ = _actor(1)
    rays = mounted_mid360_rays_body()
    lidar[:, 1, 0, 0] = 0.0
    lidar[:, 0, 0, 0] = 1.0
    lidar[:, 2, 0, 0] = 0.0
    with pytest.raises(ValueError):
        attenuate_command(lidar, rays, torch.zeros(1, 3), 0.1)
    lidar, _ = _actor(1)
    with pytest.raises(ValueError):
        attenuate_command(lidar, rays, torch.tensor([[float("nan"), 0.0, 0.0]]), 0.1)


def test_adapter_identity_yaws_roll_pitch_matrix_and_norm():
    body = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.allclose(rotate_body_to_world(body, _quat_yaw(0.0)), body)
    assert torch.allclose(rotate_body_to_world(body, _quat_yaw(90.0)), torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(rotate_body_to_world(body, _quat_yaw(-90.0)), torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6)
    quaternion = _quat_rpy(30.0, 20.0, 0.0)
    body_z = torch.tensor([[0.0, 0.0, 1.0]])
    world = rotate_body_to_world(body_z, quaternion)
    assert torch.allclose(world, torch.tensor([[0.29619813, -0.5, 0.8137977]]), atol=1e-6)
    assert torch.allclose(world.norm(dim=-1), body_z.norm(dim=-1))


def test_adapter_rejects_malformed_quaternion_and_tensor():
    with pytest.raises(ValueError):
        rotate_body_to_world(torch.zeros(1, 3), torch.tensor([[2.0, 0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        rotate_body_to_world(torch.zeros(2, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    with pytest.raises(TypeError):
        rotate_body_to_world(torch.zeros(1, 3), torch.ones(1, 4, dtype=torch.float64))


def test_pipeline_public_internal_keys_b0_fixed_action_and_bounds():
    lidar, state = _actor()
    state[:, -7:-4] = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    reset = torch.zeros(2, dtype=torch.bool)
    quaternion = _quat_yaw(0.0, 2)
    pipeline = ControlCorePipeline()
    out = pipeline(lidar, state, torch.full((2, 4), 0.5), 0.1, reset, reset, quaternion)
    assert {field.name for field in fields(ControlCoreResult)} == PUBLIC_RESULT_KEYS
    assert set(pipeline.internal_snapshot()) == INTERNAL_SNAPSHOT_KEYS
    assert torch.all(torch.linalg.vector_norm(out.v_final_b, dim=-1) <= 2.0)
    b0 = ControlCorePipeline(ControlCoreConfig(use_b0=True))
    b0_out = b0(lidar, state, torch.zeros(2, 4), 0.1, reset, reset, quaternion)
    assert torch.allclose(b0_out.v_gov_b, state[:, -7:-4])


def test_quaternion_noninterference_for_every_preadapter_result_cache_metric():
    lidar, state = _actor()
    state[:, -7:-4] = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    reset = torch.zeros(2, dtype=torch.bool)
    action = torch.tensor([[1.0, 0.5, 0.5, 0.5], [0.5, 0.6, 0.4, 0.5]])
    first, second = ControlCorePipeline(), ControlCorePipeline()
    a = first(lidar, state, action, 0.1, reset, reset, _quat_yaw(0.0, 2))
    b = second(lidar, state, action, 0.1, reset, reset, _quat_yaw(90.0, 2))
    for name in ("alpha", "v_corr_b", "v_gov_b", "v_final_b"):
        assert torch.equal(getattr(a, name), getattr(b, name))
    for metrics in ("anchor_metrics", "ics_metrics"):
        for key in getattr(a, metrics):
            assert torch.equal(getattr(a, metrics)[key], getattr(b, metrics)[key])
    for group in first.internal_snapshot():
        for key in first.internal_snapshot()[group]:
            assert torch.equal(first.internal_snapshot()[group][key], second.internal_snapshot()[group][key])
    assert not torch.equal(a.v_final_w[1], b.v_final_w[1])


def test_pipeline_failed_m4_action_quaternion_calls_are_atomic():
    lidar, state = _actor(1)
    reset = torch.zeros(1, dtype=torch.bool)
    pipeline = ControlCorePipeline()
    pipeline(lidar, state, torch.full((1, 4), 0.5), 0.1, reset, reset, _quat_yaw(0.0))
    before = pipeline.internal_snapshot()
    bad_inputs = [
        (lidar.clone(), state, torch.full((1, 4), 1.1), _quat_yaw(0.0)),
        (lidar, state, torch.full((1, 4), 0.5), torch.tensor([[2.0, 0.0, 0.0, 0.0]])),
    ]
    bad_inputs[0][0][0, 0, 0, 0] = 0.05
    for bad_lidar, bad_state, action, quaternion in bad_inputs:
        with pytest.raises(ValueError):
            pipeline(bad_lidar, bad_state, action, 0.1, reset, reset, quaternion)
        after = pipeline.internal_snapshot()
        for group in before:
            for key in before[group]:
                assert torch.equal(before[group][key], after[group][key])


def test_direct_m3_m4_ast_provenance_and_no_runtime_or_forbidden_dependency():
    m3_path = ROOT / "instinctlab" / "sensors" / "mid360.py"
    m4_path = ROOT / "instinctlab" / "observations" / "actor.py"
    m3_tree = ast.parse(m3_path.read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in m3_tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"AZIMUTH_SAMPLES", "ELEVATION_SAMPLES", "MOUNT_QUATERNION_WXYZ"}
    }
    ray_assignment = next(
        node for node in m3_tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "RAY_COUNT"
    )
    assert ast.unparse(ray_assignment.value) == "AZIMUTH_SAMPLES * ELEVATION_SAMPLES"
    assignments["RAY_COUNT"] = assignments["AZIMUTH_SAMPLES"] * assignments["ELEVATION_SAMPLES"]
    assert assignments == {
        "AZIMUTH_SAMPLES": 360, "ELEVATION_SAMPLES": 59, "RAY_COUNT": 21240,
        "MOUNT_QUATERNION_WXYZ": (0.9659258262890683, 0.0, 0.25881904510252074, 0.0),
    }
    m4_source = m4_path.read_text(encoding="utf-8")
    assert 'HISTORY_LENGTH = 4' in m4_source and 'LIDAR_FIELDS_PER_FRAME = 3' in m4_source
    control_source = "\n".join(path.read_text(encoding="utf-8").lower() for path in (ROOT / "instinctlab" / "control").glob("*.py"))
    for token in ("isaaclab", "omni.", "surface_normal", "odometry", "slam", "root_state", "privileged_height"):
        assert token not in control_source
    assert "controller_quaternion_wxyz" not in (ROOT / "instinctlab" / "control" / "governor.py").read_text()
    assert "controller_quaternion_wxyz" not in (ROOT / "instinctlab" / "control" / "anchor.py").read_text()
    assert "controller_quaternion_wxyz" not in (ROOT / "instinctlab" / "control" / "ics.py").read_text()
