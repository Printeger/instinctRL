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


def feasible_tracking(before, measured, commands, mover, triangles, body_radius):
    """Diagnostic straight-line swept clearance from the measured start state."""
    # Estimate whether the requested motion fits over the next six policy
    # boundaries. The extra half-interval of possible travel is conservative
    # for the sampled straight-line segment, with no missing future samples.
    sample_steps = np.arange(0, 7, 2)
    steps = np.arange(len(before))[:, None]+sample_steps[None, :]
    horizon_complete = steps[:, -1] < len(before)
    points = before[:, None, :]+commands[:, None, :]*(sample_steps[None, :, None]*.1)
    speed = np.linalg.norm(commands, axis=1)
    static = static_distance(points.reshape(-1, 3), triangles).reshape(points.shape[:2]).min(axis=1)-.1*speed
    if mover is None:
        clearance = static
    else:
        samples = mover[steps.clip(max=len(before)-1)]
        speed_bound = np.max(np.linalg.norm(np.diff(mover, axis=0), axis=1), initial=0.)/.1
        moving = np.linalg.norm(points-samples, axis=2).min(axis=1)-.34-.1*(speed+speed_bound)
        clearance = np.minimum(static, moving)
    moving_intent = speed > .03
    feasible = moving_intent & horizon_complete & (clearance >= body_radius+.15)
    projection = (commands*measured).sum(axis=1)/np.maximum(speed, .001)
    error = np.linalg.norm(commands-measured, axis=1)
    slow = feasible & (projection < .6*speed)
    changes = np.diff(np.r_[False, slow, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    worst = int(np.argmax(ends-starts)) if starts.size else None
    return dict(
        feasible_moving_steps=int(feasible.sum()), requested_moving_steps=int(moving_intent.sum()),
        rmse_m_s=float(np.sqrt(np.mean(error[feasible]**2))) if feasible.any() else None,
        mean_along_intent_m_s=float(projection[feasible].mean()) if feasible.any() else None,
        fraction_at_least_60_percent_of_intent=float((~slow[feasible]).mean()) if feasible.any() else None,
        longest_slow_feasible_window_s=float((ends[worst]-starts[worst])*.1) if worst is not None else 0.,
        longest_slow_feasible_start_step=int(starts[worst]) if worst is not None else None,
    )


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
                initial = json.loads(next(source))
                rows = [json.loads(row) for row in source]
            contact_or_end = [len(rows) if lane['first_reset_step'] is None else lane['first_reset_step']+1
                              for lane in item['lanes']]
            states = np.asarray([row['root_state_after_w'] for row in rows], dtype=np.float64)
            roots = states[:, :, :3]
            before = np.concatenate([np.asarray(initial['root_state_w'], dtype=np.float64)[None, :, :3],
                                     roots[:-1]], axis=0)
            velocities = states[:, :, 7:10]
            commands = np.asarray([[v['velocity_w_m_s'] for v in row['raw_event']]
                                   if isinstance(row['raw_event'], list)
                                   else [row['raw_event']['velocity_w_m_s']]*LANES for row in rows])
            movers = np.asarray([row['mover_position_after_w'] for row in rows], dtype=np.float64)
            movers_before = np.asarray([row['mover_position_w'] for row in rows], dtype=np.float64)
            values = []
            for lane, (end, score) in enumerate(zip(contact_or_end, item['lanes'], strict=True)):
                static = static_distance(roots[:end, lane], triangles)
                mover = None
                if trials[trial] is not None and score['layout'] == 'street_gap':
                    mover = np.linalg.norm(roots[:end, lane]-movers[:end, lane], axis=-1)-.34
                nearest = static if mover is None else np.minimum(static, mover)
                step = int(nearest.argmin())
                tracking = feasible_tracking(before[:end, lane], velocities[:end, lane], commands[:end, lane],
                                             None if mover is None else movers_before[:end, lane], triangles, radius)
                values.append(dict(lane=lane, layout=score['layout'], trace=item['trace'], samples=end,
                                   minimum_clearance_lower_bound_m=float(nearest[step]-radius), step=step,
                                   closest_surface='mover' if mover is not None and mover[step] < static[step] else 'static',
                                   tracking_in_feasible_moving_windows=tracking))
            models[name][trial] = values
    return dict(schema_version='instinctrl-sampled-physical-clearance/1', binding=binding, geometry=geometry_pin,
                envelope=envelope_pin, sample_period_ms=100, diagnostic_clearance_m=.15, sweep_horizon_s=.6,
                models=models, scientific_acceptance=False)


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
