"""Recompute sampled physical clearance from closed command-replay originals."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import save
from tools.r6_evidence import checked
from training_workloads.unified_h1.command_replay import LANES, conditions, route_layouts
from training_workloads.unified_h1.escape_holdout_scene import scene_geometry


def segment_distance_squared(points, start, end):
    direction = end-start
    fraction = np.clip(((points-start)*direction).sum(axis=1) / (direction@direction), 0., 1.)
    offset = points-start-fraction[:, None]*direction
    return (offset*offset).sum(axis=1)


def triangle_distance_squared(points, triangle):
    a, b, c = triangle
    edge1, edge2 = b-a, c-a
    delta = points-a
    d00, d01, d11 = edge1@edge1, edge1@edge2, edge2@edge2
    d20 = (delta*edge1).sum(axis=1)
    d21 = (delta*edge2).sum(axis=1)
    determinant = d00*d11-d01*d01
    if determinant <= 0.:
        raise ValueError('degenerate collision triangle')
    v = (d11*d20-d01*d21)/determinant
    w = (d00*d21-d01*d20)/determinant
    on_face = (v >= 0.) & (w >= 0.) & (v+w <= 1.)
    projected = delta - v[:, None]*edge1 - w[:, None]*edge2
    face = (projected*projected).sum(axis=1)
    boundary = np.minimum.reduce([segment_distance_squared(points, x, y) for x,y in ((a,b),(b,c),(c,a))])
    return np.where(on_face, face, boundary)


def static_distance(positions, triangles):
    # Ground plane and collision triangles validated by the R6 preflight.
    squared = positions[:, 2]**2
    padding = squared.max()**.5
    near = ((triangles.max(axis=1) >= positions.min(axis=0)-padding).all(axis=1)
            & (triangles.min(axis=1) <= positions.max(axis=0)+padding).all(axis=1))
    for triangle in triangles[near]:
        squared = np.minimum(squared, triangle_distance_squared(positions, triangle))
    return np.sqrt(squared)


def summarize(binding, envelope_pin):
    import torch

    bound, ctx, report = validate_binding(binding)
    params = ctx['contract']['parameters']
    trials = conditions(params)
    if not report['summary']['witnesses_replayed']:
        raise ValueError('a completely audited command replay is required')
    envelope = read(envelope_pin)
    checked(envelope['asset'])
    if envelope['status'] != 'USD_COLLIDER_BOUND_REQUIRES_LIVE_JOINT_RADIUS_VERIFICATION':
        raise ValueError('unrecognized collider envelope')
    radius = envelope['maximum_rest_pose_rotation_envelope_m']

    geometry_pin = pin(ctx['output']/'initial_preflight/geometry.json')
    geometry = read(geometry_pin)
    if geometry['status'] != 'PASS' or geometry['envs'] != LANES:
        raise ValueError('preflight collision geometry unavailable')
    origins = torch.tensor(geometry['env_origins'], dtype=torch.float32)
    _, _, vertices, indices = scene_geometry(origins, params['seed'], command_routes=route_layouts(params['experiment_id']))
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(indices).reshape(-1,3)]
    if len(triangles) != geometry['collision_wall_triangles']:
        raise ValueError('collision fixture differs from preflight')
    models = {}
    for name, arms in report['summary']['models'].items():
        models[name] = {}
        for trial, item in arms.items():
            trace = checked(item['trace'])
            with trace.open() as source:
                next(source)  # Initial state is bound in the audit and used for continuity checks.
                rows = [json.loads(row) for row in source]
            contact_or_end = [len(rows) if lane['first_reset_step'] is None else lane['first_reset_step']+1
                              for lane in item['lanes']]
            roots = np.asarray([row['root_state_after_w'] for row in rows], dtype=np.float64)[:, :, :3]
            movers = np.asarray([row['mover_position_after_w'] for row in rows], dtype=np.float64)
            values = []
            for lane, (end, score) in enumerate(zip(contact_or_end, item['lanes'], strict=True)):
                static = static_distance(roots[:end, lane], triangles)
                mover = None
                if trials[trial] is not None and score['layout'] == 'street_gap':
                    mover = np.linalg.norm(roots[:end, lane]-movers[:end, lane], axis=-1)-.34
                nearest = static if mover is None else np.minimum(static, mover)
                step = int(nearest.argmin())
                values.append(dict(lane=lane, layout=score['layout'], trace=item['trace'], samples=end,
                                   minimum_clearance_lower_bound_m=float(nearest[step]-radius), step=step,
                                   closest_surface='mover' if mover is not None and mover[step] < static[step] else 'static'))
            models[name][trial] = values
    return dict(schema_version='instinctrl-sampled-physical-clearance/1', binding=binding, geometry=geometry_pin,
                envelope=envelope_pin, sample_period_ms=100, models=models, scientific_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--envelope', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if stop_paths():
        raise RuntimeError('STOP')
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(pin(args.binding), pin(args.envelope))
    if stop_paths():
        raise RuntimeError('STOP')
    print(save(args.output, result))


if __name__ == '__main__':
    main()
