#!/usr/bin/env python3
"""Standard-library static gates for the bounded M2 TASLAB runtime surface."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = ROOT / "isaaclab-training"
ASSET_ROOT = ISAACLAB_ROOT / "instinctlab/assets/taslab_uav"
ADAPTER = ASSET_ROOT / "articulation.py"
SMOKE = ISAACLAB_ROOT / "tests/run_m2_taslab_smoke.py"
COMPARATOR = ISAACLAB_ROOT / "tests/compare_m2_smoke.py"
EXPECTED_HASHES = {
    "taslab_uav.usd": "3e733ca1008b1a3620941a8c1cc2e82bd7006a6a4c441c921a292bf11d2aee99",
    "taslab_uav.yaml": "91b7319fd844303eb8e18f946434f3369988c8789e6ba0172b67bfd89177e3e4",
    "lee_controller_taslab_uav.yaml": "e5e0496666f3f245d604a949eedc5060370d71b95de0d2337d39617bacd1f8e6",
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _calls(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(_source(path), filename=str(path))):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                names.append(node.func.id)
    return names


class TestM2Static(unittest.TestCase):
    def test_frozen_asset_hashes(self) -> None:
        for name, expected in EXPECTED_HASHES.items():
            self.assertEqual(hashlib.sha256((ASSET_ROOT / name).read_bytes()).hexdigest(), expected)

    def test_adapter_uses_new_tree_relative_asset(self) -> None:
        text = _source(ADAPTER)
        self.assertIn('Path(__file__).resolve().parent / "taslab_uav.usd"', text)
        for forbidden in ("isaac-training", "/home/", "omniverse://", "http://", "https://"):
            self.assertNotIn(forbidden, text)

    def test_adapter_exact_spawn_settings(self) -> None:
        text = _source(ADAPTER)
        for required in ("copy_from_source=False", "disable_gravity=False", "enable_gyroscopic_forces=True",
                         "max_depenetration_velocity=10.0", "enabled_self_collisions=False",
                         "solver_position_iteration_count=4", "solver_velocity_iteration_count=0",
                         "sleep_threshold=0.005", "stabilization_threshold=0.001"):
            self.assertIn(required, text)

    def test_adapter_exact_initial_state(self) -> None:
        text = _source(ADAPTER)
        for required in ("pos=(0.0, 0.0, 2.0)", "rot=(1.0, 0.0, 0.0, 0.0)",
                         "lin_vel=(0.0, 0.0, 0.0)", "ang_vel=(0.0, 0.0, 0.0)",
                         'ROTOR_JOINT_EXPRESSION = r"rotor_[0-3]_joint"'):
            self.assertIn(required, text)

    def test_exactly_one_zero_gain_actuator(self) -> None:
        source = _source(ADAPTER)
        tree = ast.parse(source, filename=str(ADAPTER))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "ImplicitActuatorCfg"]
        self.assertEqual(len(calls), 1)
        segment = ast.get_source_segment(source, calls[0])
        self.assertIn("stiffness=0.0", segment)
        self.assertIn("damping=0.0", segment)
        self.assertIn("ROTOR_JOINT_EXPRESSION", segment)

    def test_no_physics_data_override(self) -> None:
        text = _source(ADAPTER)
        for forbidden in ("MassPropertiesCfg", "mass=", "density=", "inertia=", "center_of_mass=",
                          "CollisionPropertiesCfg", "joint_limits", "physics_material"):
            self.assertNotIn(forbidden, text)

    def test_app_launch_precedes_runtime_imports(self) -> None:
        text = _source(SMOKE)
        launch = text.index("AppLauncher(headless=True, enable_cameras=False)")
        for marker in ("import torch", "import isaaclab.sim", "from isaaclab.assets import Articulation"):
            self.assertGreater(text.index(marker), launch)

    def test_smoke_fixed_contract(self) -> None:
        text = _source(SMOKE)
        for required in ('DEVICE = "cuda:0"', "DT = 0.005", "STEPS_PER_TRAJECTORY = 32",
                         "TOTAL_PHYSICS_STEPS = 64", "robot.update(DT)"):
            self.assertIn(required, text)

    def test_no_control_or_later_phase_calls(self) -> None:
        calls = set(_calls(SMOKE))
        forbidden = {"set_external_force_and_torque", "set_joint_effort_target", "set_joint_position_target",
                     "set_joint_velocity_target", "set_forces_and_torques", "write_data_to_sim"}
        self.assertTrue(calls.isdisjoint(forbidden), calls & forbidden)
        text = _source(SMOKE).lower()
        for token in ("mid360", "raycaster", "ppo", "reward", "training", "actor_observation"):
            self.assertNotIn(token, text)

    def test_comparator_fixed_tolerance_and_standard_library(self) -> None:
        text = _source(COMPARATOR)
        self.assertIn("TOLERANCE = 1.0e-5", text)
        tree = ast.parse(text, filename=str(COMPARATOR))
        imported = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in tree.body
                     if isinstance(node, ast.ImportFrom) and node.module}
        self.assertTrue(imported <= {"__future__", "argparse", "json", "math", "pathlib", "typing"})


if __name__ == "__main__":
    unittest.main()
