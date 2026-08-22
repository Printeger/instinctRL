"""Discriminating analytic, negative, gradient, and isolation M6 tests."""

from __future__ import annotations

from dataclasses import fields
import importlib.util
import inspect
import math
from pathlib import Path

import pytest
import torch

from instinctlab.control.pipeline import ControlCoreConfig, ControlCorePipeline
from instinctlab.observations.audit import EXACT_ACTOR_KEYS, audit_actor_observation
from instinctlab.task.critic import CRITIC_DIMENSION, CRITIC_FIELD_ORDER, build_critic_state
from instinctlab.task.rewards import (
    REWARD_COMPONENT_KEYS,
    REWARD_DIAGNOSTIC_KEYS,
    RewardConfig,
    RewardResult,
    compute_reward,
)
from instinctlab.task.termination import (
    TERMINATION_ABOVE_HEIGHT,
    TERMINATION_BELOW_HEIGHT,
    TERMINATION_COLLISION,
    TERMINATION_MASK_KEYS,
    TERMINATION_NONE,
    TERMINATION_TIMEOUT,
    TerminationConfig,
    compute_termination,
)

ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = ROOT / "tests/compare_m6_task_contract_smoke.py"


def _reward_inputs(batch: int = 2, dtype: torch.dtype = torch.float64, device: str = "cpu"):
    zero3 = torch.zeros(batch, 3, dtype=dtype, device=device)
    zero1 = torch.zeros(batch, 1, dtype=dtype, device=device)
    false = torch.zeros(batch, 1, dtype=torch.bool, device=device)
    return {
        "v_cmd_b": zero3.clone(),
        "v_gov_b": zero3.clone(),
        "v_final_b": zero3.clone(),
        "previous_issued_v_final_b": zero3.clone(),
        "actual_velocity_b": zero3.clone(),
        "anchor_loss": zero1.clone(),
        "anchor_active": false.clone(),
        "ics_beta": torch.ones(batch, 1, dtype=dtype, device=device),
        "ics_emergency": false.clone(),
        "exact_clearance_m": torch.full((batch, 1), 2.0, dtype=dtype, device=device),
        "collision": false.clone(),
    }


def _root(batch: int = 2, dtype: torch.dtype = torch.float64, device: str = "cpu"):
    root = torch.zeros(batch, 13, dtype=dtype, device=device)
    root[:, 2] = 2.0
    root[:, 3] = 1.0
    return root


def _actor(batch: int = 2, dtype: torch.dtype = torch.float32, device: str = "cpu"):
    lidar = torch.empty(batch, 12, 59, 360, dtype=dtype, device=device)
    lidar[:, 0::3] = 40.0
    lidar[:, 1::3] = 1.0
    lidar[:, 2::3] = 1.0
    state = torch.zeros(batch, 52, dtype=dtype, device=device)
    for offset in range(0, 52, 13):
        state[:, offset + 5] = 1.0
    state[:, -7:-4] = torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=dtype, device=device)
    return {"lidar_grid": lidar, "state_vec": state}


def test_reward_defaults_keys_shapes_and_exact_unscaled_total():
    config = RewardConfig()
    assert config.actual_tracking_weight == 1.0 and config.clearance_safe_m == 0.8
    assert config.preservation_lower == 0.75 and config.preservation_upper == 1.05
    result = compute_reward(**_reward_inputs(), config=config)
    assert tuple(result.components) == REWARD_COMPONENT_KEYS
    assert tuple(result.diagnostics) == REWARD_DIAGNOSTIC_KEYS
    assert {field.name for field in fields(RewardResult)} == {"total", "components", "diagnostics"}
    assert all(value.shape == (2, 1) and value.dtype == torch.float64 for value in result.components.values())
    exposed_sum = sum(value for key, value in result.components.items() if key != "reward_total")
    assert torch.equal(result.total, exposed_sum)
    assert torch.equal(result.components["reward_total"], result.total)


def test_primary_tracking_uses_actual_velocity_not_final_proxy():
    values = _reward_inputs(1)
    values["v_cmd_b"][0, 0] = 1.0
    values["v_gov_b"][0, 0] = 1.0
    values["v_final_b"][0, 0] = 1.0
    first = compute_reward(**values)
    assert first.components["reward_actual_tracking"].item() == -1.0
    assert first.components["reward_final_proxy"].item() == 0.0
    values["actual_velocity_b"][0, 0] = 1.0
    second = compute_reward(**values)
    assert second.components["reward_actual_tracking"].item() == 0.0
    values["v_final_b"][0, 0] = 0.0
    third = compute_reward(**values)
    assert third.components["reward_actual_tracking"].item() == 0.0
    assert third.components["reward_final_proxy"].item() < 0.0


def test_ics_attenuation_and_emergency_gate_compliance_but_penalize_usage():
    values = _reward_inputs(2)
    values["v_cmd_b"][:, 0] = 1.0
    values["v_gov_b"][:, 0] = 1.0
    values["v_final_b"][:, 0] = torch.tensor([0.5, 0.0], dtype=torch.float64)
    values["ics_beta"][:, 0] = torch.tensor([0.5, 0.0], dtype=torch.float64)
    values["ics_emergency"][1, 0] = True
    result = compute_reward(**values)
    assert torch.count_nonzero(result.components["reward_actual_tracking"]) == 0
    assert torch.count_nonzero(result.components["reward_final_proxy"]) == 0
    assert torch.all(result.components["reward_intervention_usage"] < 0.0)
    assert torch.count_nonzero(result.diagnostics["ordinary_tracking_gate"]) == 0


def test_null_command_actual_speed_and_valid_active_anchor_are_distinct():
    values = _reward_inputs(2)
    values["actual_velocity_b"][:, 0] = 0.5
    values["anchor_loss"][:, 0] = 0.25
    values["anchor_active"][0, 0] = True
    result = compute_reward(**values)
    assert result.components["reward_null_actual_speed"].flatten().tolist() == [-1.0, -1.0]
    assert result.components["reward_anchor"].flatten().tolist() == [-1.0, 0.0]
    values["v_cmd_b"][0, 0] = 0.1
    values["v_gov_b"][0, 0] = 0.1
    values["v_final_b"][0, 0] = 0.1
    gated = compute_reward(**values)
    assert gated.components["reward_anchor"][0].item() == 0.0


def test_clearance_collision_smoothness_proxy_and_preservation_components():
    values = _reward_inputs(1)
    values["v_cmd_b"][0, 0] = 1.0
    values["v_gov_b"][0, 0] = 1.1
    values["v_final_b"][0, 0] = 1.2
    values["previous_issued_v_final_b"][0, 0] = 0.2
    values["actual_velocity_b"][0, 0] = 0.8
    values["exact_clearance_m"][0, 0] = 0.6
    values["collision"][0, 0] = True
    result = compute_reward(**values)
    assert math.isclose(result.components["reward_clearance"].item(), -0.2)
    assert result.components["reward_collision"].item() == -10.0
    assert result.components["reward_issued_smoothness"].item() == -0.05
    assert result.components["reward_governor_proxy"].item() < 0.0
    assert result.components["reward_final_proxy"].item() < 0.0
    assert result.components["reward_preservation"].item() < 0.0


def test_gradients_reach_actor_produced_reward_inputs_and_actual_velocity():
    values = _reward_inputs(1, dtype=torch.float64)
    for name, tensor in (
        ("v_gov_b", torch.tensor([[0.8, 0.1, 0.0]], dtype=torch.float64, requires_grad=True)),
        ("v_final_b", torch.tensor([[0.7, 0.1, 0.0]], dtype=torch.float64, requires_grad=True)),
        ("actual_velocity_b", torch.tensor([[0.6, 0.2, 0.0]], dtype=torch.float64, requires_grad=True)),
    ):
        values[name] = tensor
    values["v_cmd_b"][0, 0] = 1.0
    values["previous_issued_v_final_b"][0, 1] = 0.2
    result = compute_reward(**values)
    result.total.sum().backward()
    for name in ("v_gov_b", "v_final_b", "actual_velocity_b"):
        assert values[name].grad is not None and torch.isfinite(values[name].grad).all()
        assert torch.count_nonzero(values[name].grad) > 0


@pytest.mark.parametrize("field", [
    "v_cmd_b", "v_gov_b", "v_final_b", "previous_issued_v_final_b", "actual_velocity_b",
])
def test_reward_rejects_vector_shape_batch_dtype_and_nonfinite(field):
    values = _reward_inputs(2)
    values[field] = torch.zeros(2, 2, dtype=torch.float64)
    with pytest.raises((TypeError, ValueError)):
        compute_reward(**values)
    values = _reward_inputs(2)
    values[field][0, 0] = float("nan")
    with pytest.raises((TypeError, ValueError)):
        compute_reward(**values)
    values = _reward_inputs(2)
    values[field] = values[field].float()
    with pytest.raises(TypeError):
        compute_reward(**values)


def test_reward_missing_inputs_labels_domains_device_and_config_fail_closed():
    values = _reward_inputs(1)
    with pytest.raises(TypeError):
        compute_reward(**{key: value for key, value in values.items() if key != "actual_velocity_b"})
    for field in ("anchor_active", "ics_emergency", "collision"):
        malformed = _reward_inputs(1)
        malformed[field] = torch.zeros(1, 1, dtype=torch.float64)
        with pytest.raises(TypeError):
            compute_reward(**malformed)
    for field, bad in (("anchor_loss", -0.1), ("ics_beta", 1.1), ("exact_clearance_m", -0.1)):
        malformed = _reward_inputs(1)
        malformed[field][0, 0] = bad
        with pytest.raises(ValueError):
            compute_reward(**malformed)
    inconsistent = _reward_inputs(1)
    inconsistent["ics_emergency"][0, 0] = True
    with pytest.raises(ValueError):
        compute_reward(**inconsistent)
    if torch.cuda.is_available():
        mismatch = _reward_inputs(1)
        mismatch["exact_clearance_m"] = mismatch["exact_clearance_m"].cuda()
        with pytest.raises(TypeError):
            compute_reward(**mismatch)
    for config in (
        lambda: RewardConfig(actual_tracking_weight=0.0),
        lambda: RewardConfig(clearance_safe_m=0.0),
        lambda: RewardConfig(preservation_lower=2.0, preservation_upper=1.0),
        lambda: RewardConfig(full_compliance_beta=1.1),
    ):
        with pytest.raises((TypeError, ValueError)):
            config()


def test_termination_truth_table_boundaries_and_reason_precedence():
    height = torch.tensor([[2.0], [0.49], [4.01], [2.0], [0.4], [2.0]])
    collision = torch.tensor([[False], [False], [False], [False], [True], [True]])
    steps = torch.tensor([[1], [1], [1], [1000], [1000], [1]], dtype=torch.long)
    result = compute_termination(height_w_m=height, collision=collision, elapsed_steps=steps)
    assert result.terminated.flatten().tolist() == [False, True, True, False, True, True]
    assert result.truncated.flatten().tolist() == [False, False, False, True, True, False]
    assert result.reason_code.flatten().tolist() == [
        TERMINATION_NONE, TERMINATION_BELOW_HEIGHT, TERMINATION_ABOVE_HEIGHT,
        TERMINATION_TIMEOUT, TERMINATION_COLLISION, TERMINATION_COLLISION,
    ]
    assert tuple(result.masks) == TERMINATION_MASK_KEYS
    boundary = compute_termination(
        height_w_m=torch.tensor([[0.5], [4.0]]),
        collision=torch.zeros(2, 1, dtype=torch.bool),
        elapsed_steps=torch.tensor([[999], [999]], dtype=torch.long),
    )
    assert torch.count_nonzero(boundary.done) == 0


def test_termination_rejects_malformed_labels_steps_bounds_and_nonfinite():
    with pytest.raises(ValueError):
        TerminationConfig(minimum_height_w_m=1.0, maximum_height_w_m=1.0)
    with pytest.raises(ValueError):
        TerminationConfig(maximum_episode_steps=0)
    height = torch.ones(1, 1)
    collision = torch.zeros(1, 1, dtype=torch.bool)
    steps = torch.zeros(1, 1, dtype=torch.long)
    bad_cases = (
        {"height_w_m": height[:, :0], "collision": collision, "elapsed_steps": steps},
        {"height_w_m": height, "collision": collision.float(), "elapsed_steps": steps},
        {"height_w_m": height, "collision": collision, "elapsed_steps": steps.float()},
        {"height_w_m": torch.tensor([[float("nan")]]), "collision": collision, "elapsed_steps": steps},
        {"height_w_m": height, "collision": collision, "elapsed_steps": torch.tensor([[-1]])},
    )
    for case in bad_cases:
        with pytest.raises((TypeError, ValueError)):
            compute_termination(**case)


def test_critic_exact_field_order_values_shape_and_no_alias():
    root = _root(1)
    root[0, :3] = torch.tensor([1.0, 2.0, 3.0])
    root[0, 7:13] = torch.arange(6, dtype=torch.float64)
    actual = torch.tensor([[6.0, 7.0, 8.0]], dtype=torch.float64)
    clearance = torch.tensor([[0.9]], dtype=torch.float64)
    collision = torch.tensor([[True]])
    state = build_critic_state(
        root_state_w=root, actual_velocity_b=actual, exact_clearance_m=clearance, collision=collision
    )
    assert state.field_order == CRITIC_FIELD_ORDER and CRITIC_DIMENSION == 18
    assert state.privileged.shape == (1, 18)
    assert torch.equal(state.privileged[:, :13], root)
    assert torch.equal(state.privileged[:, 13:16], actual)
    assert state.privileged[0, 16:].tolist() == [0.9, 1.0]
    root[0, 0] = 99.0
    assert state.privileged[0, 0].item() == 1.0


def test_critic_privileged_perturbations_change_only_named_fields():
    root = _root(1)
    first = build_critic_state(
        root_state_w=root, actual_velocity_b=torch.zeros(1, 3, dtype=root.dtype),
        exact_clearance_m=torch.ones(1, 1, dtype=root.dtype), collision=torch.zeros(1, 1, dtype=torch.bool),
    ).privileged
    root2 = root.clone()
    root2[0, 0] = 2.0
    second = build_critic_state(
        root_state_w=root2, actual_velocity_b=torch.ones(1, 3, dtype=root.dtype),
        exact_clearance_m=torch.full((1, 1), 0.2, dtype=root.dtype), collision=torch.ones(1, 1, dtype=torch.bool),
    ).privileged
    assert not torch.equal(first, second)
    changed = torch.nonzero(first != second, as_tuple=False)[:, 1].unique().tolist()
    assert changed == [0, 13, 14, 15, 16, 17]


@pytest.mark.parametrize("mutation", ["root_shape", "velocity_shape", "dtype", "nonfinite", "quaternion", "clearance", "label"])
def test_critic_negative_contract(mutation):
    root = _root(1)
    actual = torch.zeros(1, 3, dtype=root.dtype)
    clearance = torch.ones(1, 1, dtype=root.dtype)
    collision = torch.zeros(1, 1, dtype=torch.bool)
    if mutation == "root_shape": root = root[:, :12]
    elif mutation == "velocity_shape": actual = actual[:, :2]
    elif mutation == "dtype": actual = actual.float()
    elif mutation == "nonfinite": actual[0, 0] = float("inf")
    elif mutation == "quaternion": root[0, 3] = 2.0
    elif mutation == "clearance": clearance[0, 0] = -1.0
    else: collision = collision.float()
    with pytest.raises((TypeError, ValueError)):
        build_critic_state(root_state_w=root, actual_velocity_b=actual, exact_clearance_m=clearance, collision=collision)


def test_privileged_changes_cannot_change_exact_actor_or_m5_control_outputs():
    actor = _actor()
    audit_actor_observation(actor)
    action = torch.full((2, 4), 0.5)
    reset = torch.zeros(2, dtype=torch.bool)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    first_pipeline, second_pipeline = ControlCorePipeline(ControlCoreConfig(use_b0=True)), ControlCorePipeline(ControlCoreConfig(use_b0=True))
    first = first_pipeline(actor["lidar_grid"], actor["state_vec"], action, 0.1, reset, reset, quaternion)
    actor_snapshot = {key: value.clone() for key, value in actor.items()}
    reward_a = _reward_inputs(2, dtype=torch.float32)
    reward_b = _reward_inputs(2, dtype=torch.float32)
    reward_b["actual_velocity_b"][:, 0] = 3.0
    reward_b["exact_clearance_m"][:] = 0.2
    reward_b["collision"][:] = True
    assert not torch.equal(compute_reward(**reward_a).total, compute_reward(**reward_b).total)
    build_critic_state(root_state_w=_root(2, torch.float32), actual_velocity_b=reward_b["actual_velocity_b"], exact_clearance_m=reward_b["exact_clearance_m"], collision=reward_b["collision"])
    second = second_pipeline(actor["lidar_grid"], actor["state_vec"], action, 0.1, reset, reset, quaternion)
    assert set(actor) == EXACT_ACTOR_KEYS
    for key in actor:
        assert torch.equal(actor[key], actor_snapshot[key])
    for name in ("alpha", "v_corr_b", "v_gov_b", "v_final_b", "v_final_w"):
        assert torch.equal(getattr(first, name), getattr(second, name))


def test_task_sources_are_pure_fail_closed_and_have_no_goal_or_training_semantics():
    task_root = ROOT / "instinctlab/task"
    source = "\n".join(path.read_text(encoding="utf-8").lower() for path in sorted(task_root.glob("*.py")))
    for forbidden in (
        "nan_to_num", "torch.no_grad", ".detach(", ".cpu(", "isaaclab", "omni.", "torchrl",
        "tensordict", "goal_distance", "reach_goal", "ppo", "optimizer", "collector", "rollout",
    ):
        assert forbidden not in source
    assert tuple(inspect.signature(compute_reward).parameters)[0] == "v_cmd_b"
    for path in (
        ROOT / "instinctlab/observations/actor.py", ROOT / "instinctlab/observations/audit.py",
        ROOT / "instinctlab/control/governor.py", ROOT / "instinctlab/control/anchor.py",
        ROOT / "instinctlab/control/ics.py", ROOT / "instinctlab/control/command_adapter.py",
        ROOT / "instinctlab/control/pipeline.py",
    ):
        assert "instinctlab.task" not in path.read_text(encoding="utf-8")


def test_termination_and_critic_reject_cross_device_inputs():
    assert torch.cuda.is_available()
    with pytest.raises(ValueError):
        compute_termination(
            height_w_m=torch.ones(1, 1),
            collision=torch.zeros(1, 1, dtype=torch.bool, device="cuda:0"),
            elapsed_steps=torch.zeros(1, 1, dtype=torch.long),
        )
    with pytest.raises(TypeError):
        build_critic_state(
            root_state_w=_root(1),
            actual_velocity_b=torch.zeros(1, 3, dtype=torch.float64, device="cuda:0"),
            exact_clearance_m=torch.ones(1, 1, dtype=torch.float64),
            collision=torch.zeros(1, 1, dtype=torch.bool),
        )


def test_complete_comparator_rejects_missing_extra_reordered_nonfinite_and_any_error():
    spec = importlib.util.spec_from_file_location("m6_comparator", COMPARATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = {"first": torch.tensor([1.0]), "second": torch.tensor([2], dtype=torch.long)}
    negative_cases = (
        {"first": torch.tensor([1.0])},
        {"first": torch.tensor([1.0]), "second": torch.tensor([2]), "extra": torch.tensor([3])},
        {"second": torch.tensor([2], dtype=torch.long), "first": torch.tensor([1.0])},
        {"first": torch.tensor([float("nan")]), "second": torch.tensor([2], dtype=torch.long)},
        {"first": torch.tensor([1.0 + torch.finfo(torch.float32).eps]), "second": torch.tensor([2], dtype=torch.long)},
    )
    for candidate in negative_cases:
        report = {"tensor_count": 0, "element_count": 0, "max_absolute_error": 0.0}
        with pytest.raises(AssertionError):
            module._compare(baseline, candidate, "root", report)
