#!/usr/bin/env python3
"""Two-env, two-scan live exact RayCaster-normal acquisition qualification."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import fields
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from isaaclab.app import AppLauncher

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--output", type=Path, required=True)
PARSER.add_argument("--product-commit", required=True)
ARGS = PARSER.parse_args()
APP_LAUNCHER = AppLauncher(headless=True, enable_cameras=False)
SIMULATION_APP = APP_LAUNCHER.app

import torch
from isaaclab.sensors import RayCaster
from isaaclab.utils.warp import raycast_mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instinctlab.envs import TaslabDirectEnv, TaslabDirectEnvCfg
from instinctlab.evaluation.raycaster_observability import acquire_exact_raycaster_observability
from instinctlab.sensors.mid360 import RAY_COUNT, UPDATE_PERIOD_S, compose_mounted_sensor_pose, ranges_and_validity

NUM_ENVS = 2
SCANS = 2
PHYSICS_STEPS = 20
SEED = 7004
FIXED_ACTION = (0.5, 0.5, 0.5, 0.5)
ISAAC_REPO = Path("/home/mint/rl_blackwell_test/repos/IsaacLab")
ISAAC_HEAD = "37ddf626871758333d6ed89cf64ad702aef127d0"
SOURCE_HASHES = OrderedDict((
    ("source/isaaclab/isaaclab/utils/warp/ops.py", "6202538ae05e3f424a0dfdd4df7e00cc272e4213827cceafa5a0a72df8baea84"),
    ("source/isaaclab/isaaclab/utils/warp/kernels.py", "b38780baabb26b035595e6b6710e26381a18e99ed3bfbbc82887321aee7f82bf"),
    ("source/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py", "b6a84cb0e506f4a4724d3262e0f533ccb31d75128e1e46282220fcfa688e286c"),
))
EXPECTED_SIGNATURE = "(ray_starts: 'torch.Tensor', ray_directions: 'torch.Tensor', mesh: 'wp.Mesh', max_dist: 'float' = 1000000.0, return_distance: 'bool' = False, return_normal: 'bool' = False, return_face_id: 'bool' = False) -> 'tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]'"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ISAAC_REPO, check=True, capture_output=True, text=True).stdout


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _observability_tree(value: Any) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((field.name, _cpu(getattr(value, field.name))) for field in fields(value))


def _source_guard() -> OrderedDict[str, Any]:
    head = _git("rev-parse", "HEAD").strip()
    status = _git("status", "--porcelain=v1")
    if head != ISAAC_HEAD or status != " M apps/isaaclab.python.headless.kit\n":
        raise RuntimeError("frozen Isaac Lab head/status mismatch")
    actual = OrderedDict((relative, _hash(ISAAC_REPO / relative)) for relative in SOURCE_HASHES)
    if actual != SOURCE_HASHES:
        raise RuntimeError("frozen RayCaster/Warp source hash mismatch")
    version = (ISAAC_REPO / "VERSION").read_text(encoding="utf-8").strip()
    signature = str(inspect.signature(raycast_mesh))
    if version != "2.3.2" or signature != EXPECTED_SIGNATURE:
        raise RuntimeError(f"frozen API mismatch: version={version}, signature={signature}")
    return OrderedDict((
        ("isaac_lab_head", head), ("isaac_lab_version", version),
        ("isaac_lab_status", status.rstrip("\n")), ("raycast_mesh_signature", signature),
        ("source_sha256", actual),
    ))


def _policy_snapshot(env: TaslabDirectEnv) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value.detach().clone()) for name, value in env._policy.items())


def _capture(env: TaslabDirectEnv, label: str) -> OrderedDict[str, Any]:
    sensor = env._ray_caster
    for private in ("_ray_starts_w", "_ray_directions_w"):
        if not hasattr(sensor, private):
            raise RuntimeError(f"guarded private RayCaster buffer missing: {private}")
    mesh_path = env.cfg.ray_caster.mesh_prim_paths[0]
    if mesh_path not in RayCaster.meshes:
        raise RuntimeError("guarded frozen RayCaster mesh cache entry missing")
    mesh = RayCaster.meshes[mesh_path]
    mesh_id_before = int(mesh.id)
    sensor.update(UPDATE_PERIOD_S, force_recompute=True)
    ray_starts_w = sensor._ray_starts_w.detach().clone()
    ray_directions_w = sensor._ray_directions_w.detach().clone()
    accepted_hits_w = sensor.data.ray_hits_w.detach().clone()
    mount_position = torch.tensor(env.cfg.ray_caster.offset.pos, device=env.device, dtype=accepted_hits_w.dtype)
    mount_quaternion = torch.tensor(env.cfg.ray_caster.offset.rot, device=env.device, dtype=accepted_hits_w.dtype)
    sensor_origins_w, _ = compose_mounted_sensor_pose(
        sensor.data.pos_w.detach().clone(), sensor.data.quat_w.detach().clone(), mount_position, mount_quaternion
    )
    raw_ranges_m, _, accepted_valid = ranges_and_validity(accepted_hits_w, sensor_origins_w)
    accepted_valid = accepted_valid.detach().clone()
    policy_before = _policy_snapshot(env)
    source_snapshots = tuple(value.clone() for value in (ray_starts_w, ray_directions_w, accepted_hits_w, accepted_valid))
    result = acquire_exact_raycaster_observability(
        ray_starts_w,
        ray_directions_w,
        mesh,
        float(env.cfg.ray_caster.max_distance),
        accepted_hits_w,
        accepted_valid,
    )
    policy_after = _policy_snapshot(env)
    if any(not torch.equal(before, after) for before, after in zip(policy_before.values(), policy_after.values(), strict=True)):
        raise AssertionError("evaluation acquisition mutated actor policy inputs")
    if any(not torch.equal(actual, expected) for actual, expected in zip(
        (ray_starts_w, ray_directions_w, accepted_hits_w, accepted_valid), source_snapshots, strict=True
    )):
        raise AssertionError("evaluation acquisition mutated captured source tensors")
    if int(mesh.id) != mesh_id_before:
        raise AssertionError("evaluation acquisition mutated frozen mesh identity")
    valid_normals = result.raw_surface_normals_w[accepted_valid]
    invalid_normals = result.raw_surface_normals_w[~accepted_valid]
    normal_norm_error = (torch.linalg.vector_norm(valid_normals, dim=-1) - 1.0).abs()
    planar_xy_error = valid_normals[:, :2].abs()
    planar_z_error = (valid_normals[:, 2].abs() - 1.0).abs()
    exact = result.observability
    valid_count = int(accepted_valid.sum().item())
    miss_count = int((~accepted_valid).sum().item())
    gates = OrderedDict((
        ("shape_exact", tuple(ray_starts_w.shape) == (NUM_ENVS, RAY_COUNT, 3)),
        ("both_valid_and_miss", valid_count > 0 and miss_count > 0),
        ("accepted_miss_positive_infinity", bool((torch.isinf(accepted_hits_w[~accepted_valid]) & (accepted_hits_w[~accepted_valid] > 0)).all().item())),
        ("raw_normal_miss_positive_infinity", bool((torch.isinf(invalid_normals) & (invalid_normals > 0)).all().item())),
        ("qualified_miss_zero", bool((result.surface_normals_w[0][~accepted_valid] == 0).all().item())),
        ("valid_normals_unit", bool((normal_norm_error <= 1.0e-5).all().item())),
        ("valid_normals_world_plus_or_minus_z", bool((planar_xy_error <= 1.0e-6).all().item() and (planar_z_error <= 1.0e-6).all().item())),
        ("hits_exact", result.hit_max_error_m.item() == 0.0 and torch.equal(result.ray_hits_w, accepted_hits_w)),
        ("rank_one", bool((exact.numerical_rank == 1).all().item())),
        ("sigma_min_zero", bool((exact.sigma_min == 0).all().item())),
        ("condition_unavailable", bool((~exact.condition_valid).all().item() and (exact.condition_number == 0).all().item())),
        ("weak_direction_unavailable", bool((~exact.weak_direction_valid).all().item() and (exact.weak_direction_w == 0).all().item())),
        ("policy_inputs_unchanged", True), ("source_tensors_unchanged", True), ("mesh_identity_unchanged", True),
    ))
    if not all(gates.values()):
        raise AssertionError(f"live acquisition gate failure at {label}: {gates}")
    return OrderedDict((
        ("label", label),
        ("source", OrderedDict((
            ("ray_starts_w", _cpu(ray_starts_w)), ("ray_directions_w", _cpu(ray_directions_w)),
            ("accepted_ray_hits_w", _cpu(accepted_hits_w)), ("accepted_raw_ranges_m", _cpu(raw_ranges_m)),
            ("accepted_beam_valid", _cpu(accepted_valid)), ("maximum_distance_m", float(env.cfg.ray_caster.max_distance)),
            ("mesh_path", mesh_path), ("mesh_id", mesh_id_before),
        ))),
        ("acquisition", OrderedDict((
            ("ray_hits_w", _cpu(result.ray_hits_w)), ("ray_distances_m", _cpu(result.ray_distances_m)),
            ("raw_surface_normals_w", _cpu(result.raw_surface_normals_w)),
            ("surface_normals_w", _cpu(result.surface_normals_w)), ("beam_valid", _cpu(result.beam_valid)),
            ("hit_max_error_m", _cpu(result.hit_max_error_m)),
            ("distance_max_error_m", _cpu(result.distance_max_error_m)),
        ))),
        ("observability", _observability_tree(exact)),
        ("summary", OrderedDict((
            ("valid_count", valid_count), ("miss_count", miss_count),
            ("normal_norm_max_error", float(normal_norm_error.max().item())),
            ("planar_xy_max_abs", float(planar_xy_error.max().item())),
            ("planar_abs_z_max_error", float(planar_z_error.max().item())),
            ("distance_max_error_m", float(result.distance_max_error_m.item())),
        ))),
        ("gates", gates),
    ))


def main() -> None:
    if len(ARGS.product_commit) != 40:
        raise ValueError("product commit must be a full SHA")
    provenance = _source_guard()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    cfg = TaslabDirectEnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    env = TaslabDirectEnv(cfg)
    try:
        env.reset(seed=SEED)
        initial = _capture(env, "initial")
        action = torch.tensor(FIXED_ACTION, device=env.device, dtype=env._actions.dtype).repeat(NUM_ENVS, 1)
        env.step(action)
        stepped = _capture(env, "after_one_public_action")
        output = OrderedDict((
            ("schema_version", "m7.raycaster-observability-smoke.v1"),
            ("manifest", OrderedDict((
                ("num_envs", NUM_ENVS), ("scans", SCANS), ("rays_per_environment", RAY_COUNT),
                ("ray_samples", NUM_ENVS * SCANS * RAY_COUNT), ("public_actions", 1),
                ("physics_steps", PHYSICS_STEPS), ("decimation", cfg.decimation),
                ("physics_dt_s", cfg.sim.dt), ("device", str(env.device)), ("headless", True),
                ("seed", SEED), ("fixed_action", FIXED_ACTION), ("actor_inference_calls", 0),
                ("sampling_calls", 0), ("gradient_calls", 0), ("optimizer_calls", 0),
                ("training_performed", False), ("runtime_record_wiring", False),
            ))),
            ("provenance", OrderedDict((
                ("product_commit", ARGS.product_commit), ("fixture_id", "m7-raycaster-two-env-two-scan-static-plane-v1"),
                ("generator_id", "run_m7_raycaster_observability_smoke.py"), ("seed", SEED),
                ("frozen", provenance),
            ))),
            ("scans", (initial, stepped)),
            ("gates", OrderedDict((
                ("coverage_exact", NUM_ENVS * SCANS * RAY_COUNT == 84960),
                ("all_scan_gates", all(all(scan["gates"].values()) for scan in (initial, stepped))),
                ("actor_isolated", True), ("all_gates_pass", True),
            ))),
        ))
        if not all(output["gates"].values()):
            raise AssertionError(f"top-level gates failed: {output['gates']}")
        ARGS.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, ARGS.output)
        print({"status": "PASS", "ray_samples": 84960,
               "valid": sum(scan["summary"]["valid_count"] for scan in (initial, stepped)),
               "miss": sum(scan["summary"]["miss_count"] for scan in (initial, stepped))})
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
