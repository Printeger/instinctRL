"""Reconstruct a small no-disturbance command-replay window for CPU shadow checks."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.r6_direct.evidence import validate_binding
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import save
from tools.r6_evidence import checked
from training_workloads.unified_h1.command_replay import (
    CLEAR_PASS_EXECUTIONS, EARLY_RETREAT_EXECUTION, RECOVERY_RETREAT_EXECUTION, SIDE_HOVER_RETREAT_EXECUTION,
    TUBE_CONFIRM_EXECUTION, arm_execution, route_layouts,
)
from training_workloads.unified_h1.command_replay_audit import body_to_world
from training_workloads.unified_h1.escape_holdout_scene import scene_geometry


def mesh_ranges(origin, directions, triangles):
    distances = np.full(len(directions), np.inf)
    for a, b, c in triangles:
        edge1, edge2 = b-a, c-a
        normal = np.cross(directions, edge2)
        determinant = (normal*edge1).sum(axis=1)
        nonparallel = np.abs(determinant) > 1e-10
        inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=nonparallel)
        offset = origin-a
        u = inverse*(normal*offset).sum(axis=1)
        crossed = np.cross(offset, edge1)
        v = inverse*(directions*crossed).sum(axis=1)
        depth = inverse*(edge2@crossed)
        hits = nonparallel & (u >= 0.) & (v >= 0.) & (u+v <= 1.) & (depth >= .1) & (depth <= 40.)
        distances[hits] = np.minimum(distances[hits], depth[hits])
    return distances


def reconstruct(initial, row, previous, lane, rays, triangles):
    orientation = row['attitude_wxyz'][lane]
    root = initial['root_state_w'][lane] if previous is None else previous['root_state_after_w'][lane]
    origin = np.asarray(root[:3])+body_to_world(orientation, (0., 0., .05))
    directions = rays@np.asarray([body_to_world(orientation, axis) for axis in np.eye(3)])
    depths = mesh_ranges(origin, directions, triangles)
    centre = np.asarray(row['mover_position_w'][lane])-origin
    radial = directions@centre
    discriminant = radial**2-centre@centre+.34**2
    root = np.sqrt(np.maximum(discriminant, 0.))
    first = radial-root
    hit = np.where(first > 0., first, radial+root)
    if row['mover_winning_beams'][lane]:
        closer = (discriminant >= 0.) & (hit >= .1) & (hit <= 40.) & (hit < depths)
        depths[closer] = hit[closer]
        beam = row['mover_beam'][lane]
        error = abs(depths[beam]-row['mover_witness_range_m'][lane])
        if error > .002:
            raise ValueError(f'actor mover witness reconstruction mismatch {row["step"]}: {error}')
    else:
        error = None
    valid = np.isfinite(depths)
    return np.asarray([np.where(valid, depths, 40.), valid, valid], dtype=np.float32), error


def audit(binding, arm, lane, start, end):
    import torch
    from instinctlab.control.governor import decode_governor
    from instinctlab.control.ics import ICSConfig, attenuate_command
    from instinctlab.training.hover_reflex import HoverReflex

    _, ctx, report = validate_binding(binding)
    arm_data = report['summary']['models'][arm]['layout_routes']
    params = ctx['contract']['parameters']
    geometry_pin = pin(ctx['output']/'initial_preflight/geometry.json')
    geometry = json.loads(checked(geometry_pin).read_bytes())
    origins = torch.tensor(geometry['env_origins'], dtype=torch.float32)
    vertices, faces, _, _ = scene_geometry(origins, params['seed'], command_routes=route_layouts(params['experiment_id']))
    surface = np.asarray(vertices)[np.asarray(faces).reshape(-1,3)]
    origin = np.asarray(geometry['env_origins'][lane])
    near = ((surface.max(axis=1) >= origin-45.).all(axis=1)
            & (surface.min(axis=1) <= origin+45.).all(axis=1))
    surface = surface[near]
    calibration = pin(ctx['output']/f'{arm}-lidar-rays.f32')
    rays = np.fromfile(checked(calibration), dtype='<f4').reshape(-1,3)
    trace = checked(arm_data['trace'])
    with trace.open() as handle:
        initial = json.loads(next(handle))
        rows = [json.loads(row) for row in handle]
    if not 3 <= start < end <= len(rows) or end-start > 100:
        raise ValueError('use a bounded scope of at most 100 policy steps')
    if any(row['reset'][lane] for row in rows[:end]):
        raise ValueError('shadow scope must stay in the first flight')
    if initial['layouts'][lane] != 'street_gap':
        raise ValueError('this reconstruction requires the authored street mover')
    frames, errors = {}, []
    for i in range(start-3, end):
        frames[i], error = reconstruct(initial, rows[i], rows[i-1] if i else None, lane, rays, surface)
        if error is not None:
            errors.append(float(error))
    ray_tensor = torch.tensor(rays)
    reflex = HoverReflex(device='cpu', dtype=torch.float32)
    reflex._full_rays.copy_(ray_tensor)
    reflex._rays.copy_(ray_tensor.reshape(59, 360, 3)[::3].reshape(-1,3))
    execution = arm_execution(params, arm)
    if execution not in (TUBE_CONFIRM_EXECUTION, RECOVERY_RETREAT_EXECUTION,
                         EARLY_RETREAT_EXECUTION, SIDE_HOVER_RETREAT_EXECUTION, *CLEAR_PASS_EXECUTIONS):
        raise ValueError('shadow audit is registered only for the v2 sweep settings')
    config = ICSConfig(clearance_margin_m=.15, motion_tube_radius_m=.60, motion_tube_confirm_closure=True)
    results = []
    for i in range(start, end):
        row = rows[i]
        lidar = torch.tensor(np.concatenate([frames[k] for k in range(i-3, i+1)])).reshape(1,12,59,360)
        state = torch.tensor(row['actor_state'][lane]).reshape(1,52)
        if state.reshape(4,13)[:, -1].abs().max() > .001:
            raise ValueError('shadow audit requires fresh undisturbed histories')
        retreat = reflex.retreat_direction(lidar, state, hover_only=False)[0]
        raw = state[0,45:48]
        heading = raw/raw.norm().clamp_min(.001)
        tangent = torch.stack((-retreat[1],retreat[0],retreat[2]))
        recorded = torch.tensor(row['v_gov_b'][lane])
        selected = decode_governor(torch.tensor([row['selected_action'][lane]]), state).v_gov_b[0]
        if not torch.allclose(selected, recorded, atol=2.e-6, rtol=0.):
            raise ValueError('recorded selected action does not reproduce the governor')
        original = decode_governor(torch.tensor([row['action'][lane]]), state).v_gov_b[0]
        proposals = torch.stack([recorded, original, raw, -.30*heading, .30*retreat,
                                 .30*tangent, -.30*tangent, torch.zeros_like(raw)])
        result = attenuate_command(lidar.expand(len(proposals),-1,-1,-1), ray_tensor, proposals, .1, config)
        outcomes = {}
        for j, name in enumerate(('recorded_governor', 'actor_without_escape', 'latest_command', 'reverse',
                                  'away_from_return', 'tangent_left', 'tangent_right', 'hold')):
            outcomes[name] = dict(beta=float(result.metrics['beta'][j,0]),
                                issued_w_m_s=body_to_world(row['attitude_wxyz'][lane], result.v_final_b[j].tolist()))
        results.append(dict(step=i, facing_closure=float(retreat@heading),
                          retreat_b=retreat.tolist(),
                          retreat_w=body_to_world(row['attitude_wxyz'][lane], retreat.tolist()),
                          actor_witness_beams=row['mover_winning_beams'][lane],
                          measured_ics_beta=row['ics_beta'][lane], candidates=outcomes))
    return dict(schema_version='instinctrl-command-ray-shadow/1', binding=binding, geometry=geometry_pin,
                calibration=calibration, trace=arm_data['trace'], arm=arm, lane=lane,
                max_reconstructed_witness_error_m=max(errors, default=None), steps=results,
                scientific_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--lane', type=int, choices=range(32), required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if stop_paths():
        raise RuntimeError('STOP')
    result = audit(pin(args.binding), args.arm, args.lane, args.start, args.end)
    if stop_paths():
        raise RuntimeError('STOP')
    print(save(args.output, result))


if __name__ == '__main__':
    main()
