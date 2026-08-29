"""Focused pure B0 direct-adapter numeric, boundary, and isolation tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import ast
import inspect
import math
from pathlib import Path

import pytest
import torch

import instinctlab.control as control
from instinctlab.control import B0DirectConfig, B0DirectResult, execute_b0_direct_velocity
from instinctlab.control.command_adapter import rotate_body_to_world
from instinctlab.control.governor import b0_governor


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)


def _tensor(values: object, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


def _identity(batch: int, dtype: torch.dtype) -> torch.Tensor:
    return _tensor([IDENTITY_QUATERNION] * batch, dtype)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_zero_below_exact_over_limit_and_batch_independence(dtype: torch.dtype) -> None:
    commands = _tensor(
        (
            (0.0, 0.0, 0.0),
            (1.0, -2.0, 0.5),
            (3.0, 4.0, 0.0),
            (6.0, 8.0, 0.0),
        ),
        dtype,
    )
    result = execute_b0_direct_velocity(
        commands, _identity(4, dtype), B0DirectConfig(5.0)
    )
    expected = _tensor(
        (
            (0.0, 0.0, 0.0),
            (1.0, -2.0, 0.5),
            (3.0, 4.0, 0.0),
            (3.0, 4.0, 0.0),
        ),
        dtype,
    )
    tolerance = 1.0e-6 if dtype is torch.float32 else 1.0e-12
    torch.testing.assert_close(result.issued_body_velocity_m_s, expected, atol=tolerance, rtol=0.0)
    torch.testing.assert_close(result.issued_world_velocity_m_s, expected, atol=tolerance, rtol=0.0)
    assert result.issued_body_velocity_m_s.dtype is dtype
    assert result.issued_world_velocity_m_s.dtype is dtype
    assert result.issued_body_velocity_m_s.device == commands.device
    assert torch.linalg.vector_norm(result.issued_body_velocity_m_s[3]).item() == pytest.approx(
        5.0, abs=tolerance
    )


def test_over_limit_direction_is_preserved() -> None:
    command = _tensor(((2.0, -3.0, 6.0),))
    result = execute_b0_direct_velocity(command, _identity(1, command.dtype), B0DirectConfig(2.5))
    original_direction = command / torch.linalg.vector_norm(command, dim=-1, keepdim=True)
    issued_direction = result.issued_body_velocity_m_s / torch.linalg.vector_norm(
        result.issued_body_velocity_m_s, dim=-1, keepdim=True
    )
    torch.testing.assert_close(issued_direction, original_direction, atol=1.0e-12, rtol=0.0)
    assert torch.linalg.vector_norm(result.issued_body_velocity_m_s).item() == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("quaternion", "expected"),
    (
        ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), (0.0, 1.0, 0.0)),
        ((math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)), (0.0, -1.0, 0.0)),
        ((0.5, 0.5, 0.5, 0.5), (0.0, 1.0, 0.0)),
    ),
)
def test_scalar_first_rotation_hand_oracles(
    quaternion: tuple[float, float, float, float],
    expected: tuple[float, float, float],
) -> None:
    result = execute_b0_direct_velocity(
        _tensor(((1.0, 0.0, 0.0),)),
        _tensor((quaternion,)),
        B0DirectConfig(2.0),
    )
    torch.testing.assert_close(
        result.issued_world_velocity_m_s,
        _tensor((expected,)),
        atol=1.0e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_exact_parity_with_accepted_b0_governor_and_rotation(dtype: torch.dtype) -> None:
    commands = _tensor(((0.5, -0.25, 0.75), (3.0, 4.0, 0.0)), dtype)
    quaternion = _tensor(
        (
            IDENTITY_QUATERNION,
            (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        ),
        dtype,
    )
    state = torch.zeros((2, 52), dtype=dtype)
    state[:, -7:-4] = commands
    state[:, -4:-1] = _tensor(((9.0, 8.0, 7.0), (6.0, 5.0, 4.0)), dtype)
    accepted = b0_governor(state, velocity_limit=2.0)
    result = execute_b0_direct_velocity(commands, quaternion, B0DirectConfig(2.0))
    assert torch.equal(accepted.alpha, torch.ones_like(accepted.alpha))
    assert torch.equal(accepted.v_corr_b, torch.zeros_like(accepted.v_corr_b))
    assert torch.equal(result.issued_body_velocity_m_s, accepted.v_gov_b)
    assert torch.equal(
        result.issued_world_velocity_m_s,
        rotate_body_to_world(accepted.v_gov_b, quaternion),
    )


@pytest.mark.parametrize(
    "value",
    (True, False, "2", None, 0.0, -1.0, float("nan"), float("inf"), -float("inf")),
)
def test_invalid_velocity_limit_rejects(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        B0DirectConfig(value)  # type: ignore[arg-type]


def test_config_is_frozen_normalized_and_required_by_exact_type() -> None:
    config = B0DirectConfig(2)
    assert config.velocity_limit_m_s == 2.0
    assert type(config.velocity_limit_m_s) is float
    with pytest.raises(FrozenInstanceError):
        config.velocity_limit_m_s = 3.0
    with pytest.raises(TypeError):
        execute_b0_direct_velocity(_tensor(((0.0, 0.0, 0.0),)), _identity(1, torch.float64), object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "command",
    (
        None,
        [[0.0, 0.0, 0.0]],
        torch.empty((0, 3), dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
        torch.zeros((1, 1, 3), dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((1, 4), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.int64),
        torch.zeros((1, 3), dtype=torch.float16),
    ),
)
def test_invalid_command_type_shape_or_dtype_rejects(command: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        execute_b0_direct_velocity(  # type: ignore[arg-type]
            command,
            torch.tensor((IDENTITY_QUATERNION,), dtype=torch.float32),
            B0DirectConfig(),
        )


def test_tensor_subclass_rejects_as_nonexact_tensor() -> None:
    parameter = torch.nn.Parameter(torch.zeros((1, 3)), requires_grad=False)
    with pytest.raises(TypeError, match="exact torch.Tensor"):
        execute_b0_direct_velocity(
            parameter, _identity(1, torch.float32), B0DirectConfig()
        )


@pytest.mark.parametrize(
    "quaternion",
    (
        None,
        [[1.0, 0.0, 0.0, 0.0]],
        torch.zeros((0, 4), dtype=torch.float32),
        torch.zeros(4, dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.zeros((2, 4), dtype=torch.float32),
        torch.tensor(((1, 0, 0, 0),), dtype=torch.int64),
        torch.tensor(((1.0, 0.0, 0.0, 0.0),), dtype=torch.float16),
        torch.tensor(((2.0, 0.0, 0.0, 0.0),), dtype=torch.float32),
    ),
)
def test_invalid_quaternion_type_shape_batch_dtype_or_norm_rejects(quaternion: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        execute_b0_direct_velocity(  # type: ignore[arg-type]
            torch.zeros((1, 3), dtype=torch.float32),
            quaternion,
            B0DirectConfig(),
        )


@pytest.mark.parametrize("field", ("command", "quaternion"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_inputs_reject(field: str, value: float) -> None:
    command = torch.zeros((1, 3), dtype=torch.float64)
    quaternion = _identity(1, torch.float64)
    if field == "command":
        command[0, 1] = value
    else:
        quaternion[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        execute_b0_direct_velocity(command, quaternion, B0DirectConfig())


def test_dtype_and_device_mismatch_reject_before_tensor_math() -> None:
    command = torch.zeros((1, 3), dtype=torch.float32)
    with pytest.raises(TypeError, match="share dtype"):
        execute_b0_direct_velocity(command, _identity(1, torch.float64), B0DirectConfig())
    meta_quaternion = torch.empty((1, 4), dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="share device"):
        execute_b0_direct_velocity(command, meta_quaternion, B0DirectConfig())


@pytest.mark.parametrize("field", ("command", "quaternion"))
def test_requires_grad_inputs_reject_without_mutation(field: str) -> None:
    command = torch.zeros((1, 3), dtype=torch.float64)
    quaternion = _identity(1, torch.float64)
    if field == "command":
        command.requires_grad_(True)
    else:
        quaternion.requires_grad_(True)
    command_before = command.detach().clone()
    quaternion_before = quaternion.detach().clone()
    with pytest.raises(ValueError, match="gradient-free"):
        execute_b0_direct_velocity(command, quaternion, B0DirectConfig())
    assert torch.equal(command.detach(), command_before)
    assert torch.equal(quaternion.detach(), quaternion_before)


def test_outputs_are_owned_detached_nonaliased_and_caller_mutation_isolated() -> None:
    command = _tensor(((0.25, -0.5, 0.75), (3.0, 4.0, 0.0)))
    quaternion = _identity(2, torch.float64)
    command_before = command.clone()
    quaternion_before = quaternion.clone()
    result = execute_b0_direct_velocity(command, quaternion, B0DirectConfig(2.0))
    expected_body = result.issued_body_velocity_m_s.clone()
    expected_world = result.issued_world_velocity_m_s.clone()
    assert torch.equal(command, command_before) and torch.equal(quaternion, quaternion_before)
    assert not result.issued_body_velocity_m_s.requires_grad
    assert not result.issued_world_velocity_m_s.requires_grad
    pointers = {
        command.untyped_storage().data_ptr(),
        quaternion.untyped_storage().data_ptr(),
        result.issued_body_velocity_m_s.untyped_storage().data_ptr(),
        result.issued_world_velocity_m_s.untyped_storage().data_ptr(),
    }
    assert len(pointers) == 4
    command.add_(100.0)
    quaternion.mul_(0.0)
    assert torch.equal(result.issued_body_velocity_m_s, expected_body)
    assert torch.equal(result.issued_world_velocity_m_s, expected_world)
    with pytest.raises(FrozenInstanceError):
        result.issued_body_velocity_m_s = expected_body


def test_repeated_calls_are_bitwise_deterministic_and_stateless_after_failure() -> None:
    command = _tensor(((0.1, 0.2, 0.3), (3.0, -4.0, 1.0)))
    quaternion = _tensor((IDENTITY_QUATERNION, (0.5, 0.5, 0.5, 0.5)))
    first = execute_b0_direct_velocity(command, quaternion, B0DirectConfig(1.5))
    invalid = quaternion.clone()
    invalid[0, 0] = 2.0
    command_before, invalid_before = command.clone(), invalid.clone()
    with pytest.raises(ValueError, match="unit length"):
        execute_b0_direct_velocity(command, invalid, B0DirectConfig(1.5))
    assert torch.equal(command, command_before) and torch.equal(invalid, invalid_before)
    second = execute_b0_direct_velocity(command, quaternion, B0DirectConfig(1.5))
    assert torch.equal(first.issued_body_velocity_m_s, second.issued_body_velocity_m_s)
    assert torch.equal(first.issued_world_velocity_m_s, second.issued_world_velocity_m_s)


def test_public_signature_imports_and_actor_controller_private_boundary() -> None:
    signature = inspect.signature(execute_b0_direct_velocity)
    assert tuple(signature.parameters) == (
        "commanded_body_velocity_m_s",
        "controller_private_quaternion_wxyz",
        "config",
    )
    assert all(parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values())
    source_path = ROOT / "instinctlab/control/baseline_b0.py"
    tree = ast.parse(source_path.read_text())
    imports: set[str] = set()
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
    assert imports == {
        "__future__",
        "dataclasses",
        "math",
        "torch",
        ".command_adapter",
        ".governor",
    }
    assert not identifiers.intersection(
        {
            "action", "state_vec", "lidar_grid", "mid360", "history", "imu",
            "anchor", "ics", "critic", "reward", "root_state", "odometry",
            "pose", "actual_velocity", "policy", "environment", "simulator",
        }
    )
    assert "commanded_body_velocity_m_s" in identifiers
    assert "controller_private_quaternion_wxyz" in identifiers


def test_no_actor_environment_training_or_evaluation_reverse_import() -> None:
    roots = (
        ROOT / "instinctlab/observations",
        ROOT / "instinctlab/envs",
        ROOT / "instinctlab/task",
        ROOT / "instinctlab/training",
        ROOT / "instinctlab/evaluation",
    )
    exact_exceptions = {
        ROOT / "instinctlab/evaluation/baseline_execution.py",
        ROOT / "instinctlab/evaluation/baseline_controller_execution.py",
    }
    for path in (source for root in roots for source in root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append((node.module or ""))
        exact_count = imported.count("instinctlab.control.baseline_b0")
        if path in exact_exceptions:
            assert exact_count == 1
            assert "instinctlab.control" not in imported
        else:
            assert exact_count == 0, path


def test_result_type_is_exact_frozen_public_record() -> None:
    result = execute_b0_direct_velocity(
        _tensor(((0.0, 0.0, 0.0),)), _identity(1, torch.float64), B0DirectConfig()
    )
    assert type(result) is B0DirectResult
    assert tuple(control.__all__) == (
        "AnchorConfig", "B0DirectConfig", "B0DirectResult", "ControlCoreConfig",
        "ControlCorePipeline", "ControlCoreResult", "GovernorConfig", "ICSConfig",
        "LeeControllerResult", "MeasurementAnchor", "RotorResult",
        "TaslabArticulationWrenchAdapter", "TaslabRotorModel", "WrenchResult",
        "attenuate_command", "b0_governor", "decode_governor",
        "execute_b0_direct_velocity", "mounted_mid360_rays_body",
        "rotate_body_to_world", "taslab_velocity_control",
    )
