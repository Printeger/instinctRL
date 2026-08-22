"""Static, schema, privilege, and fail-closed tests for M6-002."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from instinctlab.observations.actor import ActorObservationHistory, RAY_COUNT
from instinctlab.observations.audit import audit_actor_observation
from instinctlab.task.critic import CRITIC_DIMENSION, build_critic_state

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / "instinctlab/envs/taslab_direct_env.py"
RUNNER = ROOT / "tests/run_m6_environment_smoke.py"
COMPARATOR = ROOT / "tests/compare_m6_environment_smoke.py"
BASE = "8295f6622354988ad9e00734126ebfa4a9eeaf73"


def _source() -> str:
    return ENVIRONMENT.read_text(encoding="utf-8")


def _policy(batch: int = 2) -> dict[str, torch.Tensor]:
    history = ActorObservationHistory()
    zeros3 = torch.zeros(batch, 3)
    return history.update(
        torch.full((batch, RAY_COUNT), 40.0),
        torch.zeros(batch, RAY_COUNT, dtype=torch.bool),
        torch.zeros(batch, 1),
        torch.zeros(batch, 1),
        zeros3,
        torch.tensor([[0.0, 0.0, -1.0]]).repeat(batch, 1),
        zeros3,
        zeros3,
    )


def test_exact_config_spaces_and_non_b0_action_boundary_are_locked():
    source = _source()
    assert "PHYSICS_DT_S = 0.005" in source
    assert "DECIMATION = 20" in source and 'DEVICE = "cuda:0"' in source
    assert "shape=(ACTION_DIMENSION,)" in source and "low=0.0, high=1.0" in source
    assert 'ControlCoreConfig(use_b0=False)' in source
    assert "def step(self, action: torch.Tensor)" in source
    assert "self._validate_action(action)" in source
    assert "action must already share the environment device" in source
    assert "closed normalized [0,1]" in source


def test_policy_is_exact_m4_and_privilege_perturbation_is_discriminating():
    policy = _policy()
    audit_actor_observation(policy)
    assert tuple(policy) == ("lidar_grid", "state_vec")
    assert policy["lidar_grid"].shape == (2, 12, 59, 360)
    assert policy["state_vec"].shape == (2, 52)
    root = torch.zeros(2, 13)
    root[:, 2] = 2.0
    root[:, 3] = 1.0
    actual = torch.zeros(2, 3)
    clearance = torch.ones(2, 1)
    collision = torch.zeros(2, 1, dtype=torch.bool)
    first = build_critic_state(
        root_state_w=root, actual_velocity_b=actual,
        exact_clearance_m=clearance, collision=collision,
    ).privileged
    root[:, 0] = 7.0
    actual[:, 1] = 3.0
    collision[:, 0] = True
    second = build_critic_state(
        root_state_w=root, actual_velocity_b=actual,
        exact_clearance_m=clearance, collision=collision,
    ).privileged
    assert first.shape == second.shape == (2, CRITIC_DIMENSION)
    assert not torch.equal(first, second)
    assert torch.equal(policy["lidar_grid"], _policy()["lidar_grid"])
    assert torch.equal(policy["state_vec"], _policy()["state_vec"])


def test_recursive_actor_leakage_scan_rejects_every_privileged_name():
    forbidden = {
        "root_state_w", "pose", "actual_velocity_b", "exact_clearance_m", "collision",
        "critic", "reward", "termination", "controller", "forces", "rotors",
    }
    policy = {"policy": _policy()}
    names = {key for node in ast.walk(ast.parse(repr(policy))) if isinstance(node, ast.Constant)
             and isinstance(node.value, str) for key in [node.value]}
    assert not (names & forbidden)
    contaminated = {"policy": {**_policy(), "actual_velocity_b": torch.zeros(2, 3)}}
    with pytest.raises(ValueError):
        audit_actor_observation(contaminated["policy"])


def test_step_order_same_state_direct_metrics_and_atomic_reset_sources():
    source = _source()
    tree = ast.parse(source)
    methods = {node.name: ast.get_source_segment(source, node) for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert methods["step"].index("_validate_action") < methods["step"].index("super().step")
    assert "self._task = snapshot" in methods["_refresh_task_snapshot"]
    assert "self._refresh_task_snapshot()" in methods["_get_dones"]
    assert "self._task.actual_velocity_b" in methods["_get_rewards"]
    for metric in ("huber_loss", "active", "usable_fraction", "beta", "emergency"):
        assert f'["{metric}"]' in methods["_get_rewards"]
    reset = methods["_reset_idx"]
    for token in (
        "_wrench_adapter.reset", "write_root_pose_to_sim", "write_root_velocity_to_sim",
        "write_joint_state_to_sim", "_robot.reset", "super()._reset_idx", "_history.reset",
        "_control_pipelines[index] =", "_previous_issued_v_final_b[indices] = 0.0",
        "_episode_reset_pending[indices] = True",
    ):
        assert token in reset
    assert reset.index("_validate_reset_indices") < reset.index("_wrench_adapter.reset")


def test_public_producers_frames_and_actor_dependency_boundary():
    source = _source()
    assert ".data.root_state_w" in source
    assert ".data.root_quat_w" in source
    assert ".data.root_ang_vel_b" in source
    assert ".data.projected_gravity_b" in source
    assert ".data.ray_hits_w" in source and ".data.net_forces_w" in source
    assert "rotate_world_to_body_wxyz(root[:, 3:7], root[:, 7:10])" in source
    assert "exact_clearance_m = ranges_m.min" in source
    assert "nan_to_num" not in source
    imports = {node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)}
    assert not any(name and ("coordination" in name or "formal" in name or ".tests" in name) for name in imports)


def test_runner_and_comparator_are_persisted_fail_closed_programs():
    for path in (RUNNER, COMPARATOR):
        assert path.exists()
        ast.parse(path.read_text(encoding="utf-8"))
    compare = COMPARATOR.read_text(encoding="utf-8")
    assert "list(left) != list(right)" in compare
    assert "non-finite tensor" in compare
    assert "nonzero tensor error" in compare
    assert "max_absolute_error" in compare
    runner = RUNNER.read_text(encoding="utf-8")
    assert "POLICY_STEPS = 32" in runner and "SELECTIVE_RESET_STEP = 16" in runner
    for key in ("actions", "policy", "critic", "control", "reward", "done", "root_state_w",
                "forces_body_n", "rotor", "reset_marker", "episode_length"):
        assert f'"{key}"' in runner


def test_accepted_files_are_unmodified_from_order_base():
    # Formal evidence repeats this over the full accepted path manifest.
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--name-only", BASE, "--", "isaaclab-training/instinctlab/assets",
         "isaaclab-training/instinctlab/sensors", "isaaclab-training/instinctlab/observations",
         "isaaclab-training/instinctlab/control", "isaaclab-training/instinctlab/task"],
        cwd=ROOT.parent, check=True, capture_output=True, text=True,
    )
    assert result.stdout == ""
