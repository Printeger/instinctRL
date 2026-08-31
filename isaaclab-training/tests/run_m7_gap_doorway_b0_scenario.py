#!/usr/bin/env python3
"""Dedicated exact M7-020 gap-doorway B0 headless CUDA qualification."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
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
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply
from pxr import UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instinctlab.evaluation.baseline_environment import begin_ready_b0_environment
import instinctlab.evaluation.baseline_run as baseline_run_module
from instinctlab.evaluation.baseline_runtime import begin_ready_b0_runtime_evaluation
from instinctlab.evaluation.metrics import EvaluationMetricConfig, aggregate_metrics, compute_step_metrics
from instinctlab.evaluation.raycaster_observability import acquire_exact_observability_from_source
from instinctlab.evaluation.runtime import distill_exact_runtime_observability
from instinctlab.evaluation.scenario_gap_doorway import (
    CANONICAL_SCENARIO_PRIMS,
    COMMAND_SCHEDULE,
    FACE_INDEX_COUNT,
    FIXED_SEED,
    GROUND_PRIM_PATH,
    NUM_ENVS,
    GapDoorwayGeometryEvidence,
    GapDoorwayDirectionEvidence,
    GapDoorwayVisibilityEvidence,
    GEOMETRY_TOLERANCE_M,
    POINT_COUNT,
    POLICY_STEPS,
    RAYCASTER_MESH_PRIM_PATHS,
    RAY_MESH_PRIM_PATH,
    RAY_SCENE_PRIM_PATH,
    TRIANGLE_COUNT,
    FRAME_AUTHORED_NORMAL,
    FRAME_FACE_VERTEX_COUNTS,
    FRAME_FACE_VERTEX_INDICES,
    FRAME_MESH_PRIM_PATH,
    FRAME_POINTS_M,
    FRAME_SUBDIVISION_SCHEME,
    create_gap_doorway_fixture,
    finalize_gap_doorway_qualification,
)


actor_calls = [0]
training_calls = [0]
QUATERNION_NORM_TOLERANCE = 1.0e-5


def _expected_inventory() -> tuple[tuple[str, str, bool, str], ...]:
    rows = [
        (RAY_SCENE_PRIM_PATH, "Xform", False, "permitted_ray_scene_root"),
        (FRAME_MESH_PRIM_PATH, "Mesh", True, "permitted_gap_doorway_mesh"),
        (RAY_MESH_PRIM_PATH, "Mesh", False, "permitted_ray_plane_mesh"),
    ]
    for env_index in range(NUM_ENVS):
        for link in ("base_link", "rotor_0", "rotor_1", "rotor_2", "rotor_3"):
            for leaf, collision in (("collisions", True), ("visuals", False)):
                rows.append((
                    f"/World/envs/env_{env_index}/TASLAB/{link}/{leaf}",
                    "Mesh", collision, "permitted_taslab_articulation",
                ))
    rows.extend((
        (GROUND_PRIM_PATH, "Xform", False, "permitted_ground"),
        (GROUND_PRIM_PATH + "/Environment/Geometry", "Mesh", False, "permitted_ground"),
        (GROUND_PRIM_PATH + "/GroundPlane/CollisionPlane", "Plane", True, "permitted_ground"),
    ))
    return tuple(sorted(rows))


EXPECTED_FULL_STAGE_INVENTORY = _expected_inventory()


def _author_exact_frame() -> None:
    """Author the sole scenario mesh before environment/RayCaster construction."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim(RAY_SCENE_PRIM_PATH, "Xform")
    if stage.GetPrimAtPath(FRAME_MESH_PRIM_PATH).IsValid():
        raise AssertionError("gap-doorway mesh unexpectedly exists before dedicated authoring")
    mesh = UsdGeom.Mesh.Define(stage, FRAME_MESH_PRIM_PATH)
    mesh.CreatePointsAttr(list(FRAME_POINTS_M))
    mesh.CreateFaceVertexCountsAttr(list(FRAME_FACE_VERTEX_COUNTS))
    mesh.CreateFaceVertexIndicesAttr(list(FRAME_FACE_VERTEX_INDICES))
    mesh.CreateSubdivisionSchemeAttr(FRAME_SUBDIVISION_SCHEME)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return OrderedDict((field.name, _cpu(getattr(value, field.name))) for field in fields(value))
    if isinstance(value, dict):
        return OrderedDict((key, _cpu(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return value


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_mapping(metric: Any) -> OrderedDict[str, object]:
    return OrderedDict((field.name, getattr(metric, field.name)) for field in fields(metric))


def _reduce(values: torch.Tensor, mask: torch.Tensor, reduction: str) -> tuple[bool, float | None, float | None, int, int]:
    selected = values[mask]
    count = int(selected.numel())
    if count == 0:
        return False, None, None, 0, 0
    if reduction == "minimum":
        numerator = float(selected.min().item())
        return True, numerator, numerator, 1, count
    numerator = float(selected.sum().item())
    value = numerator / count
    if reduction == "root_mean_square":
        value = math.sqrt(value)
    return True, value, numerator, count, count


def _independent_metric_oracle(evaluation: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, OrderedDict[str, object]]:
    command = evaluation["commanded_velocity_b_m_s"]
    actual = evaluation["actual_velocity_b_m_s"]
    issued = evaluation["issued_velocity_b_m_s"]
    command_norm = torch.linalg.vector_norm(command, dim=-1)
    null_command = command_norm <= 1.0e-3
    intervention = (evaluation["ics_beta"] < 0.999) | evaluation["ics_emergency"]
    preservation_mask = (~null_command) & (~intervention)
    preservation = torch.zeros_like(command_norm)
    preservation[preservation_mask] = torch.linalg.vector_norm(issued[preservation_mask], dim=-1) / command_norm[preservation_mask]
    all_samples = torch.ones((POLICY_STEPS, NUM_ENVS), dtype=torch.bool, device=command.device)
    specs = (
        ("tracking_rmse_m_s", (actual - command).square().sum(dim=-1), all_samples, "root_mean_square", "m/s"),
        ("null_station_drift_mean_m", torch.linalg.vector_norm(evaluation["position_w_m"] - evaluation["station_reference_position_w_m"], dim=-1), null_command, "mean", "m"),
        ("null_station_speed_mean_m_s", torch.linalg.vector_norm(actual, dim=-1), null_command, "mean", "m/s"),
        ("anchor_error_mean_m", evaluation["anchor_error_m"], evaluation["anchor_valid"], "mean", "m"),
        ("minimum_clearance_m", evaluation["minimum_clearance_m"], evaluation["clearance_valid"], "minimum", "m"),
        ("collision_rate", evaluation["collision"].to(command.dtype), all_samples, "mean", "fraction"),
        ("ics_violation_rate", (evaluation["ics_worst_margin_m_s"] < 0.0).to(command.dtype), all_samples, "mean", "fraction"),
        ("command_preservation_ratio", preservation, preservation_mask, "mean", "ratio"),
        ("intervention_frequency", intervention.to(command.dtype), all_samples, "mean", "fraction"),
        ("observability_mean", evaluation["observability_value"], evaluation["observability_valid"], "mean", "source-defined"),
    )
    output = OrderedDict()
    for name, values, mask, reduction, unit in specs:
        available, value, numerator, denominator, valid_count = _reduce(values, mask, reduction)
        output[name] = OrderedDict((("name", name), ("reduction", reduction), ("unit", unit), ("available", available), ("value", value), ("numerator", numerator), ("denominator", denominator), ("valid_count", valid_count), ("total_count", 16)))
    return output


def _live_geometry(env: object) -> tuple[GapDoorwayGeometryEvidence, OrderedDict[str, object]]:
    stage = sim_utils.get_current_stage()
    selected_mesh_order = tuple(
        str(prim.GetPath()) for prim in stage.Traverse()
        if prim.GetTypeName() == "Mesh" and str(prim.GetPath()).startswith(RAY_SCENE_PRIM_PATH + "/")
    )
    geometry_types = {"Mesh", "Cube", "Sphere", "Capsule", "Cylinder"}
    full_inventory = []
    forbidden = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        prim_type = str(prim.GetTypeName())
        collision = bool(prim.HasAPI(UsdPhysics.CollisionAPI))
        if prim_type not in geometry_types and not collision and path not in (GROUND_PRIM_PATH, RAY_SCENE_PRIM_PATH):
            continue
        parts = path.split("/")
        taslab = len(parts) > 4 and parts[1] == "World" and parts[2] == "envs" and parts[3].startswith("env_") and parts[4] == "TASLAB"
        if path == GROUND_PRIM_PATH or path.startswith(GROUND_PRIM_PATH + "/"):
            classification = "permitted_ground"
        elif path == RAY_SCENE_PRIM_PATH:
            classification = "permitted_ray_scene_root"
        elif path == RAY_MESH_PRIM_PATH:
            classification = "permitted_ray_plane_mesh"
        elif path == FRAME_MESH_PRIM_PATH:
            classification = "permitted_gap_doorway_mesh"
        elif taslab:
            classification = "permitted_taslab_articulation"
        else:
            classification = "forbidden_scenario_geometry"
            forbidden.append(path)
        full_inventory.append((path, prim_type, collision, classification))
    full_inventory.sort()
    forbidden = sorted(forbidden)
    mesh_prim = stage.GetPrimAtPath(RAY_MESH_PRIM_PATH)
    if not mesh_prim.IsValid() or mesh_prim.GetTypeName() != "Mesh":
        raise AssertionError("live exact ray plane mesh is absent")
    mesh = UsdGeom.Mesh(mesh_prim)
    points = tuple(tuple(float(axis) for axis in point) for point in mesh.GetPointsAttr().Get())
    counts = tuple(int(value) for value in mesh.GetFaceVertexCountsAttr().Get())
    indices = tuple(int(value) for value in mesh.GetFaceVertexIndicesAttr().Get())
    unique_x = tuple(sorted({point[0] for point in points}))
    unique_y = tuple(sorted({point[1] for point in points}))
    unique_z = tuple(sorted({point[2] for point in points}))
    minimum = tuple(min(point[axis] for point in points) for axis in range(3))
    maximum = tuple(max(point[axis] for point in points) for axis in range(3))
    mesh_paths = tuple(env.cfg.ray_caster.mesh_prim_paths)
    frame_prim = stage.GetPrimAtPath(FRAME_MESH_PRIM_PATH)
    if not frame_prim.IsValid() or frame_prim.GetTypeName() != "Mesh":
        raise AssertionError("live exact gap-doorway mesh is absent")
    frame = UsdGeom.Mesh(frame_prim)
    frame_points = tuple(tuple(float(axis) for axis in point) for point in frame.GetPointsAttr().Get())
    frame_counts = tuple(int(value) for value in frame.GetFaceVertexCountsAttr().Get())
    frame_indices = tuple(int(value) for value in frame.GetFaceVertexIndicesAttr().Get())
    frame_subdivision = str(frame.GetSubdivisionSchemeAttr().Get())
    frame_collision = bool(frame_prim.HasAPI(UsdPhysics.CollisionAPI))
    evidence = GapDoorwayGeometryEvidence(
        CANONICAL_SCENARIO_PRIMS, tuple(forbidden),
        GROUND_PRIM_PATH, RAY_SCENE_PRIM_PATH, RAY_MESH_PRIM_PATH,
        mesh_paths, len(points), len(counts), len(indices), minimum, maximum,
        unique_x, unique_y, unique_z,
        FRAME_MESH_PRIM_PATH, str(frame_prim.GetTypeName()), frame_collision,
        frame_points, frame_counts, frame_indices, frame_subdivision,
        FRAME_AUTHORED_NORMAL,
    )
    oracle = OrderedDict((
        ("ground_exists", stage.GetPrimAtPath(GROUND_PRIM_PATH).IsValid()),
        ("ray_scene_type", stage.GetPrimAtPath(RAY_SCENE_PRIM_PATH).GetTypeName()),
        ("ray_mesh_type", mesh_prim.GetTypeName()),
        ("points_exact", len(points) == POINT_COUNT and unique_x == tuple(float(value) for value in range(-42, 43, 2)) and unique_y == tuple(float(value) for value in range(-42, 43, 2)) and unique_z == (0.0,)),
        ("faces_exact", len(counts) == TRIANGLE_COUNT and set(counts) == {3}),
        ("indices_exact", len(indices) == FACE_INDEX_COUNT and min(indices) == 0 and max(indices) == POINT_COUNT - 1),
        ("mesh_paths_exact", mesh_paths == RAYCASTER_MESH_PRIM_PATHS),
        ("selected_frame_first", bool(selected_mesh_order) and selected_mesh_order[0] == FRAME_MESH_PRIM_PATH),
        ("full_stage_geometry_inventory", len(full_inventory) > 3),
        ("forbidden_scenario_geometry_empty", forbidden == []),
        ("canonical_scenario_prims_exact", evidence.prim_paths_and_types == CANONICAL_SCENARIO_PRIMS),
        ("frame_points_exact", frame_points == FRAME_POINTS_M),
        ("frame_faces_exact", frame_counts == FRAME_FACE_VERTEX_COUNTS),
        ("frame_indices_exact", frame_indices == FRAME_FACE_VERTEX_INDICES),
        ("frame_bounds_exact", tuple(min(point[axis] for point in frame_points) for axis in range(3)) == (6.0, -42.0, 0.0) and tuple(max(point[axis] for point in frame_points) for axis in range(3)) == (6.0, 42.0, 8.0)),
        ("frame_collision_exact", frame_collision),
        ("frame_subdivision_exact", frame_subdivision == FRAME_SUBDIVISION_SCHEME),
        ("inventory_exact", tuple(full_inventory) == EXPECTED_FULL_STAGE_INVENTORY),
    ))
    if not all(oracle.values()):
        raise AssertionError(f"live geometry oracle failed: {oracle}")
    oracle["inventory"] = tuple(full_inventory)
    return evidence, oracle


def main() -> None:
    bundle_path = ARGS.output.with_suffix(".pt")
    if ARGS.output.exists() or bundle_path.exists():
        raise FileExistsError(ARGS.output)
    fixture = create_gap_doorway_fixture(product_commit=ARGS.product_commit)
    if not fixture.run_plan.to_mapping()["executable"]:
        raise AssertionError("fixture plan must be executable before environment construction")
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)
    torch.use_deterministic_algorithms(True)
    _author_exact_frame()
    accepted_create_prim = sim_utils.create_prim
    preserved_root_calls = [0]

    def preserve_authored_ray_root(prim_path, prim_type="Xform", *args, **kwargs):
        if prim_path == RAY_SCENE_PRIM_PATH and prim_type == "Xform":
            prim = sim_utils.get_current_stage().GetPrimAtPath(RAY_SCENE_PRIM_PATH)
            if prim.IsValid() and prim.GetTypeName() == "Xform":
                preserved_root_calls[0] += 1
                return prim
        return accepted_create_prim(prim_path, prim_type, *args, **kwargs)

    sim_utils.create_prim = preserve_authored_ray_root
    try:
        env = begin_ready_b0_environment(fixture.manifest, fixture.environment_config, None)
    finally:
        sim_utils.create_prim = accepted_create_prim
    if preserved_root_calls != [1]:
        raise AssertionError("dedicated runner must preserve the exact pre-authored RayCaster root once")
    geometry, geometry_oracle = _live_geometry(env)
    initial = env._get_observations()
    collector = begin_ready_b0_runtime_evaluation(fixture.manifest, initial["critic"].detach().clone(), fixture.evaluation_config)

    class ForbiddenPipeline:
        def __call__(self, *args, **kwargs):
            actor_calls[0] += 1
            raise AssertionError("learned/control pipeline must not execute")

    env._control_pipelines = [ForbiddenPipeline() for _ in range(NUM_ENVS)]
    reset_calls = [0]
    accepted_reset = env._reset_idx
    def traced_reset(env_ids):
        reset_calls[0] += 1
        return accepted_reset(env_ids)
    env._reset_idx = traced_reset
    source_steps = []
    negative_jamb_hit_counts = []
    positive_jamb_hit_counts = []
    lintel_hit_counts = []
    valid_frame_hit_counts = []
    frame_normals_exact = True
    world_commands = []
    intersection_parameters = []
    intersection_y = []
    intersection_z = []
    negative_y_margins = []
    positive_y_margins = []
    lower_z_margins = []
    upper_z_margins = []
    raw_quaternion_norms = []
    staged_identity = 0
    controller_root_snapshots = []
    accepted_controller_execution = baseline_run_module.execute_ready_b0_controller

    def capture_controller_root(*args, **kwargs):
        root_state = args[2] if len(args) > 2 else kwargs["controller_private_root_state_w"]
        controller_root_snapshots.append(root_state.detach().clone())
        return accepted_controller_execution(*args, **kwargs)

    baseline_run_module.execute_ready_b0_controller = capture_controller_root
    for step, values in enumerate(fixture.command_schedule, 1):
        original = torch.tensor(values, device=env.device, dtype=torch.float32)
        snapshot = original.detach().clone()
        staged = collector.prepare_command(original)
        if staged is original or staged.untyped_storage().data_ptr() == original.untyped_storage().data_ptr() or not torch.equal(staged, snapshot):
            raise AssertionError("collector command ownership oracle failed")
        controller_calls_before = len(controller_root_snapshots)
        observation, _, terminated, truncated, extras = env.step(staged)
        staged_identity += 1
        if bool(terminated.any().item()) or bool(truncated.any().item()):
            raise AssertionError("bounded scenario must not terminate/reset")
        pre_reset = extras["pre_reset"]
        root_state = pre_reset["root_state_w"]
        issued_body = pre_reset["control"]["v_final_b"]
        issued_world = pre_reset["control"]["v_final_w"]
        if len(controller_root_snapshots) - controller_calls_before != 20:
            raise AssertionError("each policy row must expose exactly twenty B0 controller quaternion snapshots")
        q_raw = controller_root_snapshots[-1][:, 3:7]
        q_raw_norm = torch.linalg.vector_norm(q_raw, dim=-1, keepdim=True)
        if not bool(torch.isfinite(q_raw_norm).all().item()) or bool(
            ((q_raw_norm - 1.0).abs() > QUATERNION_NORM_TOLERANCE).any().item()
        ):
            raise AssertionError("controller-private quaternion norm is non-finite or outside the accepted tolerance")
        q_norm = q_raw / q_raw_norm
        independently_rotated = quat_apply(q_norm, staged)
        if not torch.equal(issued_body, staged) or not torch.equal(issued_world, independently_rotated):
            raise AssertionError("environment/B0 issued command must equal the staged command and independent normalized quaternion rotation")
        approach = issued_world[:, 0]
        parameter = (6.0 - root_state[:, 0]) / approach
        projected_y = root_state[:, 1] + parameter * issued_world[:, 1]
        projected_z = root_state[:, 2] + parameter * issued_world[:, 2]
        negative_y_margin = projected_y + 2.0
        positive_y_margin = 2.0 - projected_y
        lower_z_margin = projected_z
        upper_z_margin = 3.0 - projected_z
        if bool((approach <= 0.0).any().item()) or bool((parameter <= 0.0).any().item()) or not bool(torch.isfinite(parameter).all().item()):
            raise AssertionError("every live world command must have a finite forward intersection with x=6 m")
        if bool((torch.stack((negative_y_margin, positive_y_margin, lower_z_margin, upper_z_margin)) <= 0.0).any().item()):
            raise AssertionError("every projected world command must intersect the strict doorway aperture")
        world_commands.append(tuple(tuple(float(item) for item in row) for row in issued_world.tolist()))
        parameter_row = tuple(float(item) for item in parameter.tolist())
        intersection_y_row = tuple(float(item) for item in projected_y.tolist())
        intersection_z_row = tuple(float(item) for item in projected_z.tolist())
        intersection_parameters.append(parameter_row)
        intersection_y.append(intersection_y_row)
        intersection_z.append(intersection_z_row)
        negative_y_margins.append(tuple(item + 2.0 for item in intersection_y_row))
        positive_y_margins.append(tuple(2.0 - item for item in intersection_y_row))
        lower_z_margins.append(intersection_z_row)
        upper_z_margins.append(tuple(3.0 - item for item in intersection_z_row))
        raw_quaternion_norms.append(tuple(float(item) for item in q_raw_norm.squeeze(-1).tolist()))
        source = extras["evaluation_ray_source"]
        acquisition = acquire_exact_observability_from_source(source)
        hits = acquisition.ray_hits_w
        normals = acquisition.raw_surface_normals_w
        valid = acquisition.beam_valid.squeeze(0)
        tolerance = GEOMETRY_TOLERANCE_M
        on_plane = (hits[..., 0] - 6.0).abs() <= tolerance
        negative_jamb = valid & on_plane & (hits[..., 1] <= -2.0)
        positive_jamb = valid & on_plane & (hits[..., 1] >= 2.0)
        lintel = valid & on_plane & (hits[..., 1] > -2.0) & (hits[..., 1] < 2.0) & (hits[..., 2] >= 3.0)
        membership = negative_jamb.to(torch.int8) + positive_jamb.to(torch.int8) + lintel.to(torch.int8)
        if not torch.equal(membership, valid.to(torch.int8)):
            raise AssertionError("every valid selected-frame hit must classify exactly once")
        expected_frame_normal = torch.tensor(FRAME_AUTHORED_NORMAL, device=normals.device, dtype=normals.dtype)
        frame_normals_exact = frame_normals_exact and torch.equal(normals[valid], expected_frame_normal.expand_as(normals[valid]))
        negative_row = tuple(int(value) for value in negative_jamb.sum(dim=1).tolist())
        positive_row = tuple(int(value) for value in positive_jamb.sum(dim=1).tolist())
        lintel_row = tuple(int(value) for value in lintel.sum(dim=1).tolist())
        valid_row = tuple(int(value) for value in valid.sum(dim=1).tolist())
        if any(count < 1 for row in (negative_row, positive_row, lintel_row) for count in row):
            raise AssertionError("each environment must observe all three frame surface families on every policy row")
        negative_jamb_hit_counts.append(negative_row)
        positive_jamb_hit_counts.append(positive_row)
        lintel_hit_counts.append(lintel_row)
        valid_frame_hit_counts.append(valid_row)
        exact = distill_exact_runtime_observability(source, acquisition)
        source_steps.append(exact.source_step.detach().cpu().clone())
        collector.commit_transition(
            staged_command=staged, pre_reset=extras["pre_reset"], exact_observability=exact,
            returned_critic=observation["critic"].detach().clone(),
            returned_episode_length=env.episode_length_buf.detach().clone(),
        )
    baseline_run_module.execute_ready_b0_controller = accepted_controller_execution
    environment_result = env.finalize()
    result = collector.finalize(environment_result)
    frame_visibility = GapDoorwayVisibilityEvidence(
        tuple(negative_jamb_hit_counts), tuple(positive_jamb_hit_counts),
        tuple(lintel_hit_counts), tuple(valid_frame_hit_counts), True,
        frame_normals_exact, GEOMETRY_TOLERANCE_M,
    )
    command_direction = GapDoorwayDirectionEvidence(
        fixture.command_schedule, tuple(world_commands), tuple(intersection_parameters),
        tuple(intersection_y), tuple(intersection_z), tuple(negative_y_margins),
        tuple(positive_y_margins), tuple(lower_z_margins), tuple(upper_z_margins),
    )
    qualification = finalize_gap_doorway_qualification(
        fixture=fixture, geometry=geometry, frame_visibility=frame_visibility,
        command_direction=command_direction,
        environment_result=environment_result,
        evaluation_result=result,
    )
    evaluation = OrderedDict(result.evaluation_inputs)
    details = OrderedDict(result.observability_details)
    accepted = OrderedDict((metric.name, _metric_mapping(metric)) for metric in aggregate_metrics(compute_step_metrics(evaluation, EvaluationMetricConfig())).metrics)
    independent = _independent_metric_oracle(evaluation)
    call_counts = OrderedDict((("actor", actor_calls[0]), ("training", training_calls[0]), ("environment_step", POLICY_STEPS), ("staged_identity", staged_identity), ("reset", reset_calls[0]), ("policy", result.contract.policy_steps), ("metric_samples", result.contract.metric_samples), ("physics", result.contract.physics_steps), ("lifecycle", result.contract.lifecycle_calls), ("wrench", result.contract.wrench_applications), ("environment_samples", result.contract.environment_samples), ("rays", result.contract.ray_samples), ("receipt", 1), ("evaluation_record", 1), ("qualification_record", 1)))
    expected_commands = torch.tensor(COMMAND_SCHEDULE, dtype=torch.float32)
    gates = OrderedDict((
        ("geometry_oracle", all(geometry_oracle.values())),
        ("preconstruction_root_preserved_once", preserved_root_calls == [1]),
        ("frame_visibility_oracle", all(count >= 1 for matrix in (negative_jamb_hit_counts, positive_jamb_hit_counts, lintel_hit_counts) for row in matrix for count in row) and frame_normals_exact),
        ("command_oracle", torch.equal(evaluation["commanded_velocity_b_m_s"].cpu(), expected_commands) and torch.equal(evaluation["issued_velocity_b_m_s"], evaluation["commanded_velocity_b_m_s"])),
        ("quaternion_norm_oracle", len(raw_quaternion_norms) == POLICY_STEPS and all(math.isfinite(value) and abs(value - 1.0) <= QUATERNION_NORM_TOLERANCE for row in raw_quaternion_norms for value in row)),
        ("aperture_direction_oracle", all(value > 0.0 for row in intersection_parameters for value in row) and all(value > 0.0 for matrix in (negative_y_margins, positive_y_margins, lower_z_margins, upper_z_margins) for row in matrix for value in row)),
        ("controller_snapshot_count_oracle", len(controller_root_snapshots) == POLICY_STEPS * 20),
        ("source_steps_exact", torch.equal(torch.stack(source_steps), torch.arange(1, 9).unsqueeze(1).expand(-1, 2))),
        ("metric_oracle", independent == accepted),
        ("coverage_exact", tuple(call_counts.values()) == (0, 0, 8, 8, 0, 8, 16, 160, 160, 160, 320, 339840, 1, 1, 1)),
        ("receipt_bound", result.ready_manifest_sha256 == environment_result.receipt.ready_manifest.manifest_sha256 and result.executed_manifest_sha256 == environment_result.receipt.executed_manifest.manifest_sha256),
        ("record_bound", qualification.to_mapping()["evaluation_record"]["mapping"] == result.record.to_mapping()),
        ("no_done_or_reset", not bool(evaluation["terminated"].any().item()) and not bool(evaluation["truncated"].any().item()) and reset_calls[0] == 0),
        ("privacy", actor_calls[0] == 0 and training_calls[0] == 0 and set(dict(result.evaluation_inputs)).isdisjoint({"ray_hits", "surface_normals", "mesh", "pose", "map"})),
        ("all_finite", all(bool(torch.isfinite(value).all().item()) for value in evaluation.values() if value.is_floating_point()) and all(bool(torch.isfinite(value).all().item()) for value in details.values() if value.is_floating_point())),
    ))
    gates["all_gates_pass"] = all(gates.values())
    payload = OrderedDict((
        ("fixture", OrderedDict((("manifest", fixture.manifest.canonical_json), ("plan", fixture.run_plan.canonical_json), ("schedule", fixture.command_schedule)))),
        ("geometry", _cpu(geometry)), ("geometry_oracle", geometry_oracle),
        ("frame_visibility", _cpu(frame_visibility)),
        ("command_direction", _cpu(command_direction)),
        ("raw_quaternion_norms", tuple(raw_quaternion_norms)),
        ("evaluation_inputs", _cpu(evaluation)), ("observability_details", _cpu(details)),
        ("command_identity_sha256", _cpu(result.command_identity_sha256)),
        ("source_identity_sha256", _cpu(result.source_identity_sha256)),
        ("result_identity_sha256", _cpu(result.result_identity_sha256)),
        ("metric_oracle", independent), ("evaluation_record", result.record.canonical_json),
        ("qualification_record", qualification.canonical_json), ("call_counts", call_counts),
        ("gates", gates),
    ))
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, bundle_path)
    summary = OrderedDict((
        ("status", "PASS" if gates["all_gates_pass"] else "FAIL"),
        ("product_commit", ARGS.product_commit), ("seed", FIXED_SEED),
        ("gpu", torch.cuda.get_device_name(0)), ("torch", str(torch.__version__)),
        ("catalog_sha256", fixture.run_plan.to_mapping()["catalog_sha256"]),
        ("manifest_sha256", fixture.manifest.manifest_sha256),
        ("run_plan_sha256", fixture.run_plan.run_plan_sha256),
        ("fixture_source_sha256", fixture.run_plan.to_mapping()["scenario_evidence_sha256"]),
        ("evaluation_record_sha256", hashlib.sha256(result.record.canonical_json.encode("ascii")).hexdigest()),
        ("qualification_sha256", qualification.qualification_sha256),
        ("bundle", bundle_path.name), ("bundle_sha256", _hash(bundle_path)),
        ("call_counts", call_counts), ("geometry_oracle", geometry_oracle), ("gates", gates),
        ("frame_visibility", _cpu(frame_visibility)),
        ("command_direction", _cpu(command_direction)),
        ("raw_quaternion_norms", tuple(raw_quaternion_norms)),
    ))
    ARGS.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "seed": FIXED_SEED, "policy_steps": POLICY_STEPS, "rays": result.contract.ray_samples}, sort_keys=True))
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if gates["all_gates_pass"] else 2)


if __name__ == "__main__":
    main()
