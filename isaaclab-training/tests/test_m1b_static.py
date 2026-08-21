#!/usr/bin/env python3
"""Persisted standard-library/PyYAML static qualification for M1B-001."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import yaml


REPOSITORY = Path(__file__).resolve().parents[2]
NEW_ROOT = REPOSITORY / "isaaclab-training"
ASSET_DIR = NEW_ROOT / "instinctlab" / "assets" / "taslab_uav"
CONFIG_PATH = ASSET_DIR / "config.py"
BASE_COMMIT = "6eb521f2174a64030e30ea5e8ce8db1f89006619"
AUTHORIZATION_RECORD = "commander/authorizations/USER-20260821-TASLAB-REUSE.md"
AUTHORIZATION_SHA256 = "972e68c3677e9c07446106e3c69926753e97aa8b685bc10d2a88491fe6f48b7a"
ARTIFACTS = {
    "taslab_uav.usd": {
        "source_path": "isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.usd",
        "blob": "b16b1b16f21166d82579a9a5b190cb49191ddef5",
        "sha256": "3e733ca1008b1a3620941a8c1cc2e82bd7006a6a4c441c921a292bf11d2aee99",
        "size": 9563347,
        "file_type": "USD crate, version 0.8.0",
    },
    "taslab_uav.yaml": {
        "source_path": "isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.yaml",
        "blob": "babeb587f2db83764ca9812942e8e819d0813146",
        "sha256": "91b7319fd844303eb8e18f946434f3369988c8789e6ba0172b67bfd89177e3e4",
        "size": 3457,
        "file_type": "Unicode text, UTF-8 text",
    },
    "lee_controller_taslab_uav.yaml": {
        "source_path": "isaac-training/third_party/OmniDrones/omni_drones/controllers/cfg/lee_controller_taslab_uav.yaml",
        "blob": "d7aab1f4705898fd20759f14e0a7b0cbfe2d3a37",
        "sha256": "e5e0496666f3f245d604a949eedc5060370d71b95de0d2337d39617bacd1f8e6",
        "size": 1202,
        "file_type": "Unicode text, UTF-8 text",
    },
}


def run(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, cwd=REPOSITORY, check=True, capture_output=True)


def load_config():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("m1b_taslab_config", CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to construct config module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ArtifactIdentityTests(unittest.TestCase):
    def test_exact_source_git_objects_and_target_bytes(self):
        for name, expected in ARTIFACTS.items():
            target = ASSET_DIR / name
            source = run(["git", "cat-file", "blob", expected["blob"]]).stdout
            target_bytes = target.read_bytes()
            self.assertEqual(source, target_bytes, name)
            self.assertEqual(hashlib.sha256(target_bytes).hexdigest(), expected["sha256"])
            self.assertEqual(len(target_bytes), expected["size"])
            self.assertEqual(run(["git", "cat-file", "-t", expected["blob"]]).stdout.strip(), b"blob")
            self.assertEqual(run(["git", "hash-object", str(target)]).stdout.decode().strip(), expected["blob"])
            self.assertEqual(run(["file", "-b", str(target)]).stdout.decode().strip(), expected["file_type"])

    def test_provenance_and_notice_are_complete(self):
        provenance = json.loads((ASSET_DIR / "PROVENANCE.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["schema_version"], "instinctrl-taslab-provenance-1")
        self.assertEqual(provenance["base_commit"], BASE_COMMIT)
        self.assertEqual(provenance["authorization"]["record"], AUTHORIZATION_RECORD)
        self.assertEqual(provenance["authorization"]["sha256"], AUTHORIZATION_SHA256)
        records = {item["target"]: item for item in provenance["artifacts"]}
        self.assertEqual(set(records), set(ARTIFACTS))
        for name, expected in ARTIFACTS.items():
            record = records[name]
            self.assertEqual(record["source_path"], expected["source_path"])
            self.assertEqual(record["git_blob"], expected["blob"])
            self.assertEqual(record["sha256"], expected["sha256"])
            self.assertEqual(record["size_bytes"], expected["size"])
            self.assertEqual(record["git_object_type"], "blob")
            self.assertTrue(record["file_type"])
        notice = (ASSET_DIR / "NOTICE.md").read_text(encoding="utf-8")
        self.assertIn(AUTHORIZATION_RECORD, notice)
        self.assertIn(AUTHORIZATION_SHA256, notice)
        self.assertIn("does not claim or\ngrant broader third-party rights", notice)


class ActiveValueParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vehicle = yaml.safe_load((ASSET_DIR / "taslab_uav.yaml").read_text(encoding="utf-8"))
        cls.lee = yaml.safe_load((ASSET_DIR / "lee_controller_taslab_uav.yaml").read_text(encoding="utf-8"))
        cls.config_module = load_config()

    def test_vehicle_yaml_exact_active_values(self):
        expected_inertia = {
            "xx": 0.0051350687,
            "xy": 0.0000000049,
            "xz": 0.0002842395,
            "yy": 0.0056075235,
            "yz": 0.0000000307,
            "zz": 0.0052743931,
        }
        rotor = self.vehicle["rotor_configuration"]
        self.assertEqual(self.vehicle["name"], "taslab_uav")
        self.assertEqual(self.vehicle["mass"], 1.1583)
        self.assertEqual(self.vehicle["inertia"], expected_inertia)
        self.assertEqual(self.vehicle["l"], 0.24042)
        self.assertEqual(self.vehicle["drag_coef"], 0.2)
        self.assertEqual(rotor["num_rotors"], 4)
        self.assertEqual(rotor["arm_lengths"], [0.12021] * 4)
        self.assertEqual(rotor["rotor_angles"], [5.49778714, 0.78539816, 2.35619449, 3.92699082])
        self.assertEqual(rotor["directions"], [1.0, -1.0, 1.0, -1.0])
        self.assertEqual(rotor["force_constants"], [1.55e-06] * 4)
        self.assertEqual(rotor["moment_constants"], [2.39e-08] * 4)
        self.assertEqual(rotor["max_rotation_velocities"], [2261] * 4)

    def test_lee_yaml_exact_active_values(self):
        self.assertEqual(
            self.lee,
            {
                "position_gain": [4, 4, 4],
                "velocity_gain": [2.2, 2.2, 2.2],
                "attitude_gain": [0.7, 0.7, 0.035],
                "angular_rate_gain": [0.1, 0.1, 0.025],
            },
        )

    def test_declarative_config_matches_yaml_and_units(self):
        contract = self.config_module.TASLAB_UAV
        rotor = self.vehicle["rotor_configuration"]
        self.assertEqual(contract.name, self.vehicle["name"])
        self.assertEqual(contract.mass, self.vehicle["mass"])
        self.assertEqual(contract.length, self.vehicle["l"])
        self.assertEqual(contract.drag_coefficient, self.vehicle["drag_coef"])
        self.assertEqual(contract.rotor_count, rotor["num_rotors"])
        self.assertEqual(tuple(contract.inertia.__dict__[key] for key in ("xx", "xy", "xz", "yy", "yz", "zz")), tuple(self.vehicle["inertia"].values()))
        self.assertEqual(contract.rotors.arm_lengths, tuple(rotor["arm_lengths"]))
        self.assertEqual(contract.rotors.angles, tuple(rotor["rotor_angles"]))
        self.assertEqual(contract.rotors.directions, tuple(rotor["directions"]))
        self.assertEqual(contract.rotors.force_constants, tuple(rotor["force_constants"]))
        self.assertEqual(contract.rotors.moment_constants, tuple(rotor["moment_constants"]))
        self.assertEqual(contract.rotors.max_angular_velocities, tuple(rotor["max_rotation_velocities"]))
        self.assertEqual(contract.rotors.list_order, (0, 1, 2, 3))
        self.assertEqual(contract.rotors.usd_prim_paths, tuple(f"/taslab_uav/rotor_{index}" for index in range(4)))
        self.assertEqual(contract.lee_gains.position, tuple(self.lee["position_gain"]))
        self.assertEqual(contract.lee_gains.velocity, tuple(self.lee["velocity_gain"]))
        self.assertEqual(contract.lee_gains.attitude, tuple(self.lee["attitude_gain"]))
        self.assertEqual(contract.lee_gains.angular_rate, tuple(self.lee["angular_rate_gain"]))
        self.assertEqual(contract.mass_unit, "kg")
        self.assertEqual(contract.inertia.unit, "kg*m^2")
        self.assertEqual(contract.rotors.angle_unit, "rad")
        self.assertIn("unresolved", contract.drag_coefficient_unit)
        self.assertGreaterEqual(len(self.config_module.UNRESOLVED_RUNTIME_SEMANTICS), 5)


class IndependenceAndBoundaryTests(unittest.TestCase):
    def test_config_imports_standard_library_only(self):
        tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(imported, {"dataclasses", "pathlib"})

    def test_asset_paths_resolve_inside_new_tree(self):
        contract = load_config().TASLAB_UAV
        new_root = NEW_ROOT.resolve()
        for path in (contract.usd_path, contract.vehicle_yaml_path, contract.lee_yaml_path):
            resolved = path.resolve(strict=True)
            self.assertTrue(resolved.is_relative_to(new_root), resolved)

    def test_copied_content_has_no_external_absolute_dependency(self):
        forbidden = (
            b"/home/mint/rl_dev/instinctRL",
            b"/home/mint/rl_dev/instinctRL_coordination",
            b"/home/mint/rl_blackwell_test",
        )
        for name in ARTIFACTS:
            content = (ASSET_DIR / name).read_bytes()
            for marker in forbidden:
                self.assertNotIn(marker, content, f"{name}: {marker!r}")
        usd = (ASSET_DIR / "taslab_uav.usd").read_bytes()
        legacy_documentation = b"Generated from Composed Stage of root layer /home/mint/rl_dev/NavRL/isaac-training/third_party"
        self.assertEqual(usd.count(legacy_documentation), 1)
        for name in ("taslab_uav.yaml", "lee_controller_taslab_uav.yaml"):
            self.assertNotIn(b"isaac-training/third_party", (ASSET_DIR / name).read_bytes())

    def test_no_runtime_or_actor_surface_in_static_contract(self):
        tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
        names = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attributes = {node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        forbidden_runtime = {"articulationcfg", "simulationcontext", "rotorgroup", "ppo", "mid360"}
        forbidden_actor = {"actor_observation", "pose", "odometry", "slam_state", "obstacle_ground_truth", "privileged_state"}
        self.assertTrue(forbidden_runtime.isdisjoint(names | attributes))
        self.assertTrue(forbidden_actor.isdisjoint(names | attributes))
        functions = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.assertEqual(functions, [])

    def test_usd_inspector_has_only_read_only_stage_calls(self):
        inspector = Path(__file__).with_name("inspect_taslab_usd.py")
        tree = ast.parse(inspector.read_text(encoding="utf-8"))
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called.add(node.func.attr)
        self.assertIn("Open", called)
        self.assertIn("AppLauncher", called)
        self.assertTrue({"SimulationContext", "World", "play", "step", "spawn"}.isdisjoint(called))
        source = inspector.read_text(encoding="utf-8")
        self.assertIn("headless=True, enable_cameras=False", source)
        self.assertIn("os._exit", source)


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    unittest.main(verbosity=2)
