#!/usr/bin/env python3
"""Read-only TASLAB USD stage inventory using the frozen Isaac Lab app.

The application is launched only to make pxr available. This program opens
existing stages and performs schema/attribute queries; it creates no runtime
world, physics context, articulation, controller, or timeline activity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from isaaclab.app import AppLauncher


ROTOR_PATTERN = re.compile(r"(?:^|[_-])rotor[_-]?(\d+)$", re.IGNORECASE)
EXPECTED_ARM_LENGTHS = (0.12021, 0.12021, 0.12021, 0.12021)
EXPECTED_ANGLES = (5.49778714, 0.78539816, 2.35619449, 3.92699082)
MAPPING_TOLERANCE = 5.0e-6


def _round(value: float) -> float:
    return round(float(value), 10)


def _vector(value: object) -> list[float] | None:
    if value is None:
        return None
    try:
        return [_round(component) for component in value]
    except TypeError:
        return None


def _matrix(value: object) -> list[list[float]]:
    return [[_round(value[row][column]) for column in range(4)] for row in range(4)]


def _attribute_value(attribute: object) -> object:
    value = attribute.Get()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    vector = _vector(value)
    return vector if vector is not None else str(value)


def _inspect_stage(path: Path, Usd: object, UsdGeom: object, UsdPhysics: object, UsdUtils: object) -> dict[str, object]:
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open USD stage: {path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    prim_inventory: list[dict[str, object]] = []
    rigid_bodies: list[dict[str, object]] = []
    mass_apis: list[dict[str, object]] = []
    joints: list[dict[str, object]] = []
    rotor_candidates: list[dict[str, object]] = []

    for prim in stage.TraverseAll():
        prim_path = str(prim.GetPath())
        type_name = prim.GetTypeName()
        prim_inventory.append({"path": prim_path, "type": type_name})

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.RigidBodyAPI(prim)
            rigid_bodies.append(
                {
                    "path": prim_path,
                    "rigid_body_enabled": _attribute_value(api.GetRigidBodyEnabledAttr()),
                    "kinematic_enabled": _attribute_value(api.GetKinematicEnabledAttr()),
                }
            )
        if prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            mass_apis.append(
                {
                    "path": prim_path,
                    "mass": _attribute_value(api.GetMassAttr()),
                    "density": _attribute_value(api.GetDensityAttr()),
                    "center_of_mass": _attribute_value(api.GetCenterOfMassAttr()),
                    "diagonal_inertia": _attribute_value(api.GetDiagonalInertiaAttr()),
                    "principal_axes": _attribute_value(api.GetPrincipalAxesAttr()),
                }
            )
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            joints.append(
                {
                    "path": prim_path,
                    "type": type_name,
                    "body0_targets": sorted(str(item) for item in joint.GetBody0Rel().GetTargets()),
                    "body1_targets": sorted(str(item) for item in joint.GetBody1Rel().GetTargets()),
                    "local_pos0": _attribute_value(joint.GetLocalPos0Attr()),
                    "local_pos1": _attribute_value(joint.GetLocalPos1Attr()),
                    "local_rot0": _attribute_value(joint.GetLocalRot0Attr()),
                    "local_rot1": _attribute_value(joint.GetLocalRot1Attr()),
                    "axis": _attribute_value(prim.GetAttribute("physics:axis")),
                }
            )

        match = ROTOR_PATTERN.search(prim.GetName())
        if match:
            xformable = UsdGeom.Xformable(prim)
            world_matrix = xform_cache.GetLocalToWorldTransform(prim)
            local_matrix = xformable.GetLocalTransformation()
            rotor_candidates.append(
                {
                    "index_from_name": int(match.group(1)),
                    "path": prim_path,
                    "type": type_name,
                    "local_matrix": _matrix(local_matrix),
                    "world_matrix": _matrix(world_matrix),
                    "world_translation": _vector(world_matrix.ExtractTranslation()),
                    "world_axes": {
                        "x": [_round(world_matrix[row][0]) for row in range(3)],
                        "y": [_round(world_matrix[row][1]) for row in range(3)],
                        "z": [_round(world_matrix[row][2]) for row in range(3)],
                    },
                    "xform_ops": [
                        {"name": str(op.GetOpName()), "value": _attribute_value(op.GetAttr())}
                        for op in xformable.GetOrderedXformOps()
                    ],
                    "axis_attributes": {
                        attr.GetName(): _attribute_value(attr)
                        for attr in prim.GetAttributes()
                        if "axis" in attr.GetName().lower()
                    },
                }
            )

    rotor_candidates.sort(key=lambda item: (item["index_from_name"], item["path"]))
    expected_xy = [
        [_round(length * math.cos(angle)), _round(length * math.sin(angle))]
        for length, angle in zip(EXPECTED_ARM_LENGTHS, EXPECTED_ANGLES, strict=True)
    ]
    candidate_xy = [item["world_translation"][:2] for item in rotor_candidates]
    center_xy = (
        [_round(sum(point[axis] for point in candidate_xy) / len(candidate_xy)) for axis in range(2)]
        if candidate_xy
        else None
    )
    mapping_rows: list[dict[str, object]] = []
    mapping_proven = len(rotor_candidates) == 4
    seen_indices = {item["index_from_name"] for item in rotor_candidates}
    mapping_proven = mapping_proven and seen_indices == {0, 1, 2, 3}
    for candidate in rotor_candidates:
        index = int(candidate["index_from_name"])
        translation = candidate["world_translation"]
        actual_xy = translation[:2] if translation is not None else None
        centered_xy = (
            [_round(actual_xy[axis] - center_xy[axis]) for axis in range(2)]
            if actual_xy is not None and center_xy is not None
            else None
        )
        error = None
        nearest_index = None
        matches = False
        if centered_xy is not None and index < len(expected_xy):
            distances = [math.dist(centered_xy, position) for position in expected_xy]
            nearest_index = min(range(len(distances)), key=distances.__getitem__)
            error = distances[index]
            matches = nearest_index == index and error <= MAPPING_TOLERANCE
        mapping_proven = mapping_proven and matches
        mapping_rows.append(
            {
                "yaml_index": index,
                "prim_path": candidate["path"],
                "expected_xy_m": expected_xy[index] if index < len(expected_xy) else None,
                "actual_world_xy_stage_units": actual_xy,
                "rotor_centroid_xy_stage_units": center_xy,
                "centroid_relative_xy_stage_units": centered_xy,
                "nearest_yaml_index_by_centroid_relative_xy": nearest_index,
                "xy_error": None if error is None else _round(error),
                "matches_within_tolerance": matches,
            }
        )

    default_prim = stage.GetDefaultPrim()
    pseudo_root = stage.GetPseudoRoot()
    dependency_layers, dependency_assets, unresolved_dependencies = UsdUtils.ComputeAllDependencies(str(path))
    return {
        "path_label": path.name,
        "opened": True,
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "pseudo_root": str(pseudo_root.GetPath()),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": _round(UsdGeom.GetStageMetersPerUnit(stage)),
        "composition_dependencies": {
            "layer_count": len(dependency_layers),
            "external_asset_paths": sorted(str(item) for item in dependency_assets),
            "unresolved_paths": sorted(str(item) for item in unresolved_dependencies),
            "root_layer_external_references": sorted(str(item) for item in stage.GetRootLayer().GetExternalReferences()),
        },
        "prim_inventory": sorted(prim_inventory, key=lambda item: item["path"]),
        "rigid_bodies": sorted(rigid_bodies, key=lambda item: item["path"]),
        "mass_apis": sorted(mass_apis, key=lambda item: item["path"]),
        "joints": sorted(joints, key=lambda item: item["path"]),
        "rotor_candidates": rotor_candidates,
        "rotor_mapping": {
            "method": "unique nearest centroid-relative rotor world XY to YAML arm_length*cos/sin(angle), cross-checked against rotor_N index",
            "tolerance": MAPPING_TOLERANCE,
            "rows": mapping_rows,
            "proven": mapping_proven,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--target-usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    _application = app_launcher.app
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    source = _inspect_stage(args.source_usd.resolve(), Usd, UsdGeom, UsdPhysics, UsdUtils)
    target = _inspect_stage(args.target_usd.resolve(), Usd, UsdGeom, UsdPhysics, UsdUtils)
    comparable_source = {**source, "path_label": "taslab_uav.usd"}
    comparable_target = {**target, "path_label": "taslab_uav.usd"}
    semantic_match = comparable_source == comparable_target
    result = {
            "schema_version": "m1b-usd-inspection-1",
            "operation_audit": {
                "app_launcher": "headless=True, enable_cameras=False",
                "stage_operation": "Usd.Stage.Open(load=Usd.Stage.LoadAll) and read-only schema queries",
                "physics_initialized": False,
                "simulation_context_created": False,
                "timeline_played": False,
                "world_created": False,
                "articulation_spawned": False,
                "step_called": False,
                "shutdown": "os._exit after result write/print/flush; baseline-known headless close handling",
            },
            "source": source,
            "target": target,
            "source_target_semantic_match": semantic_match,
            "rotor_mapping_proven": bool(source["rotor_mapping"]["proven"] and target["rotor_mapping"]["proven"]),
    }
    normalized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(normalized, encoding="utf-8")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    print(json.dumps({"output": str(args.output), "sha256": digest, "semantic_match": semantic_match, "rotor_mapping_proven": result["rotor_mapping_proven"]}, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if semantic_match and result["rotor_mapping_proven"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
