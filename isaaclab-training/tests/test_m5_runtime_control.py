"""Analytic, negative, source, topology, and actor-boundary M5 runtime tests."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math
from pathlib import Path

import pytest
import torch

from instinctlab.assets.taslab_uav.config import TASLAB_UAV
from instinctlab.control.lee_velocity_controller import (
    GRAVITY_M_S2,
    quaternion_to_rotation_matrix_wxyz,
    taslab_mixer,
    taslab_velocity_control,
)
from instinctlab.control.rotor_actuation import ROTOR_NAMES, TAU_DOWN, TAU_UP, TaslabRotorModel
from instinctlab.control.runtime_adapter import EXPECTED_BODY_NAMES, TaslabArticulationWrenchAdapter
from instinctlab.observations.actor import APPROVED_INPUT_NAMES, OUTPUT_SLICE_PROVENANCE
from instinctlab.observations.audit import EXACT_ACTOR_KEYS

ROOT = Path(__file__).resolve().parents[1]
LEE = ROOT / "instinctlab/control/lee_velocity_controller.py"
ROTOR = ROOT / "instinctlab/control/rotor_actuation.py"
ADAPTER = ROOT / "instinctlab/control/runtime_adapter.py"
ARTICULATION = ROOT / "instinctlab/assets/taslab_uav/articulation.py"
SMOKE = ROOT / "tests/run_m5_taslab_control_smoke.py"
COMPARATOR = ROOT / "tests/compare_m5_taslab_control_smoke.py"


def _root(batch: int = 1, dtype: torch.dtype = torch.float64):
    root = torch.zeros(batch, 13, dtype=dtype)
    root[:, 2] = 2.0
    root[:, 3] = 1.0
    return root


def _maximum_thrust(dtype=torch.float64):
    rotor = TASLAB_UAV.rotors
    return torch.tensor(
        [kf * omega**2 for kf, omega in zip(rotor.force_constants, rotor.max_angular_velocities, strict=True)],
        dtype=dtype,
    )


class FakeArticulation:
    body_names = EXPECTED_BODY_NAMES
    num_bodies = 5

    def __init__(self, num_instances: int = 2, fail_apply: bool = False):
        self.num_instances = num_instances
        self.device = torch.device("cpu")
        self.fail_apply = fail_apply
        self.calls = []
        self.writes = 0

    def set_external_force_and_torque(self, forces, torques, **kwargs):
        if self.fail_apply:
            raise RuntimeError("fixture apply failure")
        self.calls.append((forces.clone(), torques.clone(), kwargs))

    def write_data_to_sim(self):
        self.writes += 1


def test_exact_taslab_controller_and_rotor_constants_mixer_order_signs():
    assert TASLAB_UAV.mass == 1.1583
    assert (TASLAB_UAV.inertia.xx, TASLAB_UAV.inertia.yy, TASLAB_UAV.inertia.zz) == (
        0.0051350687, 0.0056075235, 0.0052743931
    )
    assert TASLAB_UAV.rotors.angles == (5.49778714, 0.78539816, 2.35619449, 3.92699082)
    assert TASLAB_UAV.rotors.directions == (1.0, -1.0, 1.0, -1.0)
    assert ROTOR_NAMES == ("rotor_0", "rotor_1", "rotor_2", "rotor_3")
    assert TAU_UP == TAU_DOWN == 0.43
    assert TaslabRotorModel(1, "cpu", torch.float32).dt_s == 0.005
    mixer = taslab_mixer(_root())
    assert mixer.shape == (4, 4) and torch.linalg.matrix_rank(mixer) == 4
    allocation = torch.stack((
        torch.sin(torch.tensor(TASLAB_UAV.rotors.angles, dtype=torch.float64)) * torch.tensor(TASLAB_UAV.rotors.arm_lengths, dtype=torch.float64),
        -torch.cos(torch.tensor(TASLAB_UAV.rotors.angles, dtype=torch.float64)) * torch.tensor(TASLAB_UAV.rotors.arm_lengths, dtype=torch.float64),
        -torch.tensor(TASLAB_UAV.rotors.directions, dtype=torch.float64) * torch.tensor(TASLAB_UAV.rotors.moment_constants, dtype=torch.float64) / torch.tensor(TASLAB_UAV.rotors.force_constants, dtype=torch.float64),
        torch.ones(4, dtype=torch.float64),
    )).to(torch.float64)
    inertia4 = torch.diag(torch.tensor((TASLAB_UAV.inertia.xx, TASLAB_UAV.inertia.yy, TASLAB_UAV.inertia.zz, 1.0), dtype=torch.float64))
    assert torch.allclose(allocation @ mixer, inertia4, atol=1e-10, rtol=0.0)


def test_lee_identity_zero_velocity_hover_oracle_and_current_hold():
    root = _root()
    result = taslab_velocity_control(root, torch.zeros(1, 3, dtype=root.dtype))
    thrust_each = TASLAB_UAV.mass * GRAVITY_M_S2 / 4.0
    expected = thrust_each / _maximum_thrust() * 2.0 - 1.0
    assert torch.allclose(result.normalized_rotor_commands[0], expected, atol=2e-8, rtol=0.0)
    assert torch.equal(result.held_position_w_m, root[:, :3])
    assert torch.equal(result.held_yaw_rad, torch.zeros(1, 1, dtype=root.dtype))
    assert torch.allclose(result.desired_rotation_w, torch.eye(3, dtype=root.dtype).unsqueeze(0), atol=1e-12)
    assert math.isclose(result.collective_thrust_n.item(), TASLAB_UAV.mass * GRAVITY_M_S2, rel_tol=1e-12)


def test_lee_nontrivial_yaw_velocity_and_independent_rotation_oracle():
    root = _root()
    half = math.sqrt(0.5)
    root[:, 3:7] = torch.tensor([[half, 0.0, 0.0, half]], dtype=root.dtype)
    target = torch.tensor([[0.0, 0.4, 0.0]], dtype=root.dtype)
    result = taslab_velocity_control(root, target)
    rotation = quaternion_to_rotation_matrix_wxyz(root[:, 3:7])
    assert torch.allclose(rotation, torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=root.dtype), atol=1e-12)
    acceleration = torch.tensor([[0.0, -0.88, -GRAVITY_M_S2]], dtype=root.dtype)
    b3 = -acceleration / acceleration.norm(dim=-1, keepdim=True)
    b1 = torch.tensor([[0.0, 1.0, 0.0]], dtype=root.dtype)
    b2 = torch.linalg.cross(b3, b1, dim=-1)
    b2 = b2 / b2.norm(dim=-1, keepdim=True)
    expected_rotation = torch.stack((torch.linalg.cross(b2, b3, dim=-1), b2, b3), dim=-1)
    assert torch.allclose(result.desired_rotation_w, expected_rotation, atol=1e-12)
    assert math.isclose(result.held_yaw_rad.item(), math.pi / 2.0, rel_tol=1e-12)
    assert result.normalized_rotor_commands.shape == (1, 4)


def test_lee_output_is_affine_normalized_but_not_controller_clamped():
    root = _root()
    root[:, 10] = 100.0
    result = taslab_velocity_control(root, torch.zeros(1, 3, dtype=root.dtype))
    assert bool(((result.normalized_rotor_commands < -1.0) | (result.normalized_rotor_commands > 1.0)).any().item())
    source = LEE.read_text(encoding="utf-8")
    assert "nan_to_num" not in source and ".clamp(" not in source and "torch.clamp" not in source


@pytest.mark.parametrize("mutation", ["shape", "batch", "dtype", "nonfinite", "quaternion"])
def test_lee_rejects_malformed_inputs(mutation):
    root = _root()
    target = torch.zeros(1, 3, dtype=root.dtype)
    if mutation == "shape":
        root = root[:, :12]
    elif mutation == "batch":
        target = target.repeat(2, 1)
    elif mutation == "dtype":
        target = target.float()
    elif mutation == "nonfinite":
        root[0, 7] = float("nan")
    else:
        root[0, 3] = 2.0
    with pytest.raises((TypeError, ValueError)):
        taslab_velocity_control(root, target)


def test_lee_rejects_missing_non_tensor_and_cross_device_inputs():
    root = _root()
    with pytest.raises(TypeError):
        taslab_velocity_control(root)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        taslab_velocity_control(root.tolist(), torch.zeros(1, 3, dtype=root.dtype))  # type: ignore[arg-type]
    assert torch.cuda.is_available()
    with pytest.raises(ValueError):
        taslab_velocity_control(root, torch.zeros(1, 3, dtype=root.dtype, device="cuda:0"))


def test_rotor_boundary_mapping_recurrence_force_moment_and_out_of_range_clamp():
    model = TaslabRotorModel(1, "cpu", torch.float64)
    low = model.step(torch.full((1, 4), -1.0, dtype=torch.float64))
    assert torch.count_nonzero(low.throttle) == 0 and torch.count_nonzero(low.thrusts_n) == 0
    high = model.step(torch.ones(1, 4, dtype=torch.float64))
    assert torch.allclose(high.target_throttle, torch.ones_like(high.target_throttle))
    assert torch.allclose(high.throttle, torch.full_like(high.throttle, 0.43))
    assert torch.allclose(high.thrusts_n[0], 0.43**2 * _maximum_thrust())
    expected_moment = 0.43**2 * torch.tensor(TASLAB_UAV.rotors.moment_constants) * 2261.0**2 * -torch.tensor(TASLAB_UAV.rotors.directions)
    assert torch.allclose(high.reaction_moments_nm[0], expected_moment.to(torch.float64), atol=1e-10)
    second = model.step(torch.full((1, 4), 3.0, dtype=torch.float64))
    assert torch.allclose(second.throttle, torch.full_like(second.throttle, 0.43 + 0.43 * 0.57))
    below = model.step(torch.full((1, 4), -3.0, dtype=torch.float64))
    assert torch.allclose(below.target_throttle, torch.zeros_like(below.target_throttle))


def test_rotor_per_environment_isolation_reset_and_atomic_failure():
    model = TaslabRotorModel(2, "cpu", torch.float32)
    model.step(torch.tensor([[1.0] * 4, [-1.0] * 4]))
    assert torch.all(model.throttle[0] > 0.0) and torch.count_nonzero(model.throttle[1]) == 0
    before = model.throttle
    with pytest.raises(ValueError):
        model.step(torch.tensor([[float("nan")] * 4, [0.0] * 4]))
    assert torch.equal(before, model.throttle)
    model.reset(torch.tensor([0], dtype=torch.long))
    assert torch.count_nonzero(model.throttle[0]) == 0


def test_runtime_adapter_exact_local_wrench_order_and_write():
    robot = FakeArticulation()
    adapter = TaslabArticulationWrenchAdapter(2, "cpu", torch.float32)
    result = adapter.apply(robot, torch.ones(2, 4))
    forces, torques, kwargs = robot.calls[-1]
    assert kwargs == {"is_global": False}
    assert torch.equal(forces[:, 0], torch.zeros(2, 3))
    assert torch.equal(forces[:, 1:, :2], torch.zeros(2, 4, 2))
    assert torch.equal(forces[:, 1:, 2], result.rotor.thrusts_n)
    assert torch.equal(torques[:, 0, 2], result.rotor.reaction_moments_nm.sum(dim=-1))
    assert torch.count_nonzero(torques[:, 1:]) == 0
    adapter.write_data_to_sim(robot)
    assert robot.writes == 1


def test_runtime_adapter_topology_and_api_failure_are_atomic_then_reset_clears():
    adapter = TaslabArticulationWrenchAdapter(2, "cpu", torch.float32)
    wrong = FakeArticulation()
    wrong.body_names = ("base_link", "rotor_1", "rotor_0", "rotor_2", "rotor_3")
    before = adapter.rotors.throttle
    with pytest.raises(ValueError):
        adapter.apply(wrong, torch.ones(2, 4))
    assert torch.equal(before, adapter.rotors.throttle)
    wrong_device = FakeArticulation()
    wrong_device.device = torch.device("meta")
    with pytest.raises(ValueError):
        adapter.apply(wrong_device, torch.ones(2, 4))
    assert not wrong_device.calls and torch.equal(before, adapter.rotors.throttle)
    valid = FakeArticulation()
    with pytest.raises(ValueError):
        adapter.apply(valid, torch.ones(2, 3))
    assert not valid.calls and torch.equal(before, adapter.rotors.throttle)
    failing = FakeArticulation(fail_apply=True)
    with pytest.raises(RuntimeError):
        adapter.apply(failing, torch.ones(2, 4))
    assert torch.equal(before, adapter.rotors.throttle)
    robot = FakeArticulation()
    adapter.apply(robot, torch.ones(2, 4))
    adapter.reset(robot, torch.tensor([0], dtype=torch.long))
    assert torch.count_nonzero(adapter.rotors.throttle[0]) == 0
    forces, torques = adapter.pending_wrenches()
    assert torch.count_nonzero(forces[0]) == 0 and torch.count_nonzero(torques[0]) == 0
    reset_forces, reset_torques, kwargs = robot.calls[-1]
    assert kwargs["is_global"] is False and kwargs["env_ids"].tolist() == [0]
    assert torch.count_nonzero(reset_forces) == 0 and torch.count_nonzero(reset_torques) == 0


def test_mit_license_provenance_and_no_frozen_runtime_dependency():
    for path in (LEE, ROTOR):
        source = path.read_text(encoding="utf-8")
        assert source.startswith("# MIT License")
        assert "Copyright (c) 2023 Botian Xu, Tsinghua University" in source
        tree = ast.parse(source, filename=str(path))
        imports = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not any("omni_drones" in value or "isaac-training" in value for value in imports)
    combined = LEE.read_text() + ROTOR.read_text() + ADAPTER.read_text()
    for token in ("/home/mint/rl_dev/instinctRL", "rl_blackwell_test", "omni_drones"):
        assert token not in combined
    assert "rand" not in ROTOR.read_text().lower()


def test_root_state_remains_controller_private_and_accepted_surfaces_unchanged():
    actor = (ROOT / "instinctlab/observations/actor.py").read_text()
    audit = (ROOT / "instinctlab/observations/audit.py").read_text()
    governor = (ROOT / "instinctlab/control/governor.py").read_text()
    anchor = (ROOT / "instinctlab/control/anchor.py").read_text()
    ics = (ROOT / "instinctlab/control/ics.py").read_text()
    pipeline = (ROOT / "instinctlab/control/pipeline.py").read_text()
    command_adapter = (ROOT / "instinctlab/control/command_adapter.py").read_text()
    actor_clean_sources = (governor, anchor, ics, command_adapter, pipeline)
    forbidden = ("isaaclab", "omni.", "surface_normal", "odometry", "slam", "root_state", "privileged_height")
    for source in actor_clean_sources:
        for token in forbidden:
            assert token not in source.lower()
    assert "root_state_w" not in actor and "root_state_w" not in audit
    parameters = inspect.signature(taslab_velocity_control).parameters
    assert tuple(parameters) == ("root_state_w", "target_velocity_w", "measured_mass_kg", "measured_allocation")
    for name in ("measured_mass_kg", "measured_allocation"):
        assert parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None
    assert "root_state_w" in LEE.read_text()
    assert "target_velocity_w" in LEE.read_text()
    assert "root_state_w" not in ADAPTER.read_text()
    assert "v_final_w = rotate_body_to_world" in pipeline
    broadened = "\n".join(actor_clean_sources) + LEE.read_text()
    assert {token for token in forbidden if token in broadened.lower()} == {"root_state"}
    assert EXACT_ACTOR_KEYS == frozenset(("lidar_grid", "state_vec"))
    assert APPROVED_INPUT_NAMES == (
        "ranges_m", "validity", "sensor_frame_time_s", "policy_time_s",
        "body_angular_velocity_rad_s", "gravity_direction_body",
        "commanded_body_velocity_m_s", "previous_issued_body_velocity_m_s",
    )
    assert {item["output"] for item in OUTPUT_SLICE_PROVENANCE} == {"lidar_grid", "state_vec"}


def test_articulation_settings_remain_locked_and_only_topology_constants_added():
    source = ARTICULATION.read_text()
    for required in (
        "copy_from_source=False", "disable_gravity=False", "enable_gyroscopic_forces=True",
        "solver_position_iteration_count=4", "solver_velocity_iteration_count=0",
        "stiffness=0.0", "damping=0.0", 'TASLAB_BODY_NAMES = ("base_link", "rotor_0", "rotor_1", "rotor_2", "rotor_3")',
    ):
        assert required in source
    for forbidden in ("MassPropertiesCfg", "mass=", "density=", "inertia="):
        assert forbidden not in source


def test_runtime_program_contract_and_comparator_rejects_weak_surfaces():
    smoke = SMOKE.read_text()
    for required in (
        'DEVICE = "cuda:0"', "DT = 0.005", "HISTORY_DT_S = 0.1", "TOTAL_PHYSICS_STEPS = 256",
        "adapter.apply", "write_data_to_sim", "ControlCorePipeline",
        "ActorObservationHistory", "RayCaster", "taslab_velocity_control",
    ):
        assert required in smoke
    assert "AppLauncher(headless=True, enable_cameras=False)" in smoke
    for forbidden in ("reward", "ppo", "collector", "checkpoint", "domain_randomization"):
        assert forbidden not in smoke.lower()
    comparator = COMPARATOR.read_text()
    assert "TOLERANCE = 1.0e-5" in comparator
    assert "set(left) != set(right)" in comparator
    assert "not torch.isfinite" in comparator

    spec = importlib.util.spec_from_file_location("m5_runtime_comparator", COMPARATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = {"first": torch.tensor([1.0]), "second": torch.tensor([2.0])}
    negative_cases = (
        {"first": torch.tensor([1.0])},
        {"first": torch.tensor([1.0]), "second": torch.tensor([2.0]), "extra": torch.tensor([3.0])},
        {"second": torch.tensor([2.0]), "first": torch.tensor([1.0])},
        {"first": torch.tensor([float("nan")]), "second": torch.tensor([2.0])},
        {"first": torch.tensor([1.1]), "second": torch.tensor([2.0])},
    )
    for candidate in negative_cases:
        report = {"tensor_count": 0, "element_count": 0, "max_absolute_error": 0.0}
        with pytest.raises(AssertionError):
            module._compare(baseline, candidate, "root", report)
