#!/usr/bin/env python3
"""Static and tensor-level gates for the bounded M3 MID360 contract."""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import sys
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = ROOT / "isaaclab-training"
MODULE = ISAACLAB_ROOT / "instinctlab/sensors/mid360.py"
SMOKE = ISAACLAB_ROOT / "tests/run_m3_mid360_smoke.py"
COMPARATOR = ISAACLAB_ROOT / "tests/compare_m3_smoke.py"
sys.path.insert(0, str(ISAACLAB_ROOT))


class _PatternBaseCfg:
    pass


class _OffsetCfg:
    def __init__(self, pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)) -> None:
        self.pos = pos
        self.rot = rot


class _RayCasterCfg:
    OffsetCfg = _OffsetCfg

    def __init__(self, **kwargs: object) -> None:
        self.attach_yaw_only = None
        for key, value in kwargs.items():
            setattr(self, key, value)


def _install_config_stubs() -> None:
    """Avoid importing Kit while exercising pure configuration and tensor logic."""
    modules = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.sensors": types.ModuleType("isaaclab.sensors"),
        "isaaclab.sensors.ray_caster": types.ModuleType("isaaclab.sensors.ray_caster"),
        "isaaclab.sensors.ray_caster.patterns": types.ModuleType("isaaclab.sensors.ray_caster.patterns"),
        "isaaclab.sensors.ray_caster.patterns.patterns_cfg": types.ModuleType(
            "isaaclab.sensors.ray_caster.patterns.patterns_cfg"
        ),
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
    }
    modules["isaaclab.sensors.ray_caster"].RayCasterCfg = _RayCasterCfg
    modules["isaaclab.sensors.ray_caster.patterns.patterns_cfg"].PatternBaseCfg = _PatternBaseCfg
    modules["isaaclab.utils"].configclass = lambda cls: cls
    sys.modules.update(modules)


_install_config_stubs()

from instinctlab.sensors.mid360 import (  # noqa: E402
    MID360_RAY_CASTER_CFG,
    MOUNT_QUATERNION_WXYZ,
    Mid360PatternCfg,
    mid360_pattern,
    ranges_and_validity,
)


def _calls(path: Path) -> set[str]:
    calls: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.add(node.func.id)
    return calls


class TestM3Static(unittest.TestCase):
    def test_pattern_shape_order_dtype_unit_and_contiguity(self) -> None:
        starts, directions = mid360_pattern(Mid360PatternCfg(), "cpu")
        self.assertEqual(starts.shape, (21240, 3))
        self.assertEqual(directions.shape, (21240, 3))
        self.assertEqual(starts.dtype, torch.float32)
        self.assertEqual(directions.dtype, torch.float32)
        self.assertTrue(starts.is_contiguous())
        self.assertTrue(directions.is_contiguous())
        self.assertTrue(torch.equal(starts, torch.zeros_like(starts)))
        self.assertLess(float((torch.linalg.vector_norm(directions, dim=-1) - 1.0).abs().max()), 2.0e-7)
        for elevation_index, azimuth_index in ((0, 0), (0, 359), (1, 0), (58, 359)):
            index = elevation_index * 360 + azimuth_index
            elevation = math.radians(-7.0 + elevation_index * (59.0 / 58.0))
            azimuth = math.radians(-180.0 + azimuth_index)
            expected = torch.tensor(
                [math.cos(elevation) * math.cos(azimuth),
                 math.cos(elevation) * math.sin(azimuth), math.sin(elevation)],
                dtype=torch.float32,
            )
            self.assertTrue(torch.allclose(directions[index], expected, atol=1.0e-6, rtol=0.0))

    def test_pattern_deterministic_hash(self) -> None:
        first_starts, first_directions = mid360_pattern(Mid360PatternCfg(), "cpu")
        second_starts, second_directions = mid360_pattern(Mid360PatternCfg(), "cpu")
        first_hash = hashlib.sha256(first_starts.numpy().tobytes() + first_directions.numpy().tobytes()).hexdigest()
        second_hash = hashlib.sha256(second_starts.numpy().tobytes() + second_directions.numpy().tobytes()).hexdigest()
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(len(first_hash), 64)
        print(f"pattern_sha256={first_hash}")

    def test_mount_and_ray_caster_configuration(self) -> None:
        cfg = MID360_RAY_CASTER_CFG
        self.assertEqual(cfg.prim_path, "/World/TASLAB/base_link")
        self.assertEqual(cfg.offset.pos, (0.0, 0.0, 0.05))
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in MOUNT_QUATERNION_WXYZ)), 1.0, places=14)
        w, x, y, z = MOUNT_QUATERNION_WXYZ
        rotated_forward_z = 2.0 * (x * z - w * y)
        self.assertAlmostEqual(rotated_forward_z, -0.5, places=12)
        self.assertEqual(cfg.ray_alignment, "base")
        self.assertIsNone(cfg.attach_yaw_only)
        self.assertEqual(cfg.update_period, 0.1)
        self.assertEqual(cfg.max_distance, 40.0)
        self.assertEqual(cfg.drift_range, (0.0, 0.0))
        self.assertEqual(cfg.ray_cast_drift_range, {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)})
        self.assertFalse(cfg.debug_vis)

    def test_closed_range_validity_and_fill(self) -> None:
        origins = torch.zeros((1, 3), dtype=torch.float32)
        hits = torch.tensor(
            [[[0.099, 0.0, 0.0], [0.1, 0.0, 0.0], [40.0, 0.0, 0.0],
              [40.001, 0.0, 0.0], [float("inf"), 0.0, 0.0], [float("nan"), 0.0, 0.0]]],
            dtype=torch.float32,
        )
        raw, serialized, valid = ranges_and_validity(hits, origins)
        self.assertEqual(valid.tolist(), [[False, True, True, False, False, False]])
        self.assertTrue(torch.allclose(
            serialized,
            torch.tensor([[40.0, 0.1, 40.0, 40.0, 40.0, 40.0]], dtype=torch.float32),
            atol=0.0,
            rtol=0.0,
        ))
        self.assertTrue(torch.isinf(raw[0, 4]))
        self.assertTrue(torch.isnan(raw[0, 5]))
        self.assertTrue(torch.isfinite(serialized).all())

    def test_runtime_surface_obeys_phase_boundary(self) -> None:
        smoke_text = SMOKE.read_text(encoding="utf-8")
        launch = smoke_text.index("AppLauncher(headless=True, enable_cameras=False)")
        for marker in ("import torch", "import isaaclab.sim", "from isaaclab.assets import Articulation"):
            self.assertGreater(smoke_text.index(marker), launch)
        self.assertNotIn("cfg.spawn.func", smoke_text)
        self.assertNotIn("_initialize_impl", smoke_text)
        forbidden_calls = {
            "set_external_force_and_torque", "set_joint_effort_target", "set_joint_position_target",
            "set_joint_velocity_target", "write_data_to_sim",
        }
        self.assertTrue(_calls(SMOKE).isdisjoint(forbidden_calls))
        lowered = (MODULE.read_text(encoding="utf-8") + smoke_text).lower()
        for token in ("actor_observation", "history_schema", "ppo", "reward", "training_loop", "controller"):
            self.assertNotIn(token, lowered)

    def test_comparator_is_standard_library_and_fixed_tolerance(self) -> None:
        text = COMPARATOR.read_text(encoding="utf-8")
        self.assertIn("TOLERANCE = 1.0e-5", text)
        tree = ast.parse(text, filename=str(COMPARATOR))
        imported = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in tree.body
                     if isinstance(node, ast.ImportFrom) and node.module}
        self.assertTrue(imported <= {"__future__", "argparse", "json", "math", "pathlib", "typing"})


if __name__ == "__main__":
    unittest.main()
