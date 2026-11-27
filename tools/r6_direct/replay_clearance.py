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
from training_workloads.unified_h1.command_replay_audit import dense_clearance
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


def frozen_opportunity(velocities, commands, first_flight_steps, start, end):
    """Count each externally registered nonzero command, including after a failure."""
    intents = commands[start:end]
    speed = np.linalg.norm(intents, axis=1)
    expected = speed > .03
    if not expected.any():
        raise ValueError('registered opportunity contains no movement command')
    observed = np.arange(start, end) < first_flight_steps
    measured = velocities[start:end]
    projected = (intents*measured).sum(axis=1)/np.maximum(speed, .001)
    meets = observed & expected & (projected >= .6*speed)
    used = observed & expected
    total = int(expected.sum())
    error = np.linalg.norm(intents[used]-measured[used], axis=1)
    return dict(scheduled_moving_steps=total, first_flight_moving_steps=int(used.sum()),
                met_steps=int(meets.sum()), met_fraction=float(meets.sum()/total),
                measured_progress_m=float(projected[used].sum()*.1),
                requested_progress_m=float(speed[expected].sum()*.1),
                observed_only_rmse_m_s=float(np.sqrt(np.mean(error**2))) if len(error) else None)


def frozen_stop(before, after, velocities, commands, first_flight_steps, start, end):
    """Require a stable stop across the rest of a predeclared zero-input window."""
    if start <= 0 or end-start < 20 or end > len(commands):
        raise ValueError('scheduled stop must cover two seconds of recorded zero input')
    if np.linalg.norm(commands[start-1]) <= .03 or np.any(commands[start:end]):
        raise ValueError('scheduled stop does not begin with a move-to-zero change')
    observed_end = min(first_flight_steps, end)
    if observed_end <= start:
        return dict(first_flight_completed_window=False, stopped_after_s=None, max_drift_m=None)
    speed = np.linalg.norm(velocities[start:observed_end], axis=1)
    stable = np.flatnonzero(np.logical_and.accumulate((speed <= .07)[::-1])[::-1])
    stopped = (int(stable[0])+1)*.1 if len(stable) >= 5 and observed_end == end else None
    drift = np.linalg.norm(after[start:observed_end]-before[start], axis=1)
    return dict(first_flight_completed_window=observed_end == end, stopped_after_s=stopped,
                max_drift_m=float(drift.max()))


def projected_clearance(before, measured_before, targets, mover, triangles, body_radius):
    """Score a held request over 2.4 seconds with a fixed velocity response model."""
    offsets = np.arange(0, 25, 2)
    time = offsets*.1
    ticks = np.arange(len(before))[:, None]+offsets
    complete = ticks[:, -1] < len(before)
    lag = .35*(1.-np.exp(-time/.35))
    path = before[:, None]+time[None, :, None]*targets[:, None]
    path += lag[None, :, None]*(measured_before-targets)[:, None]
    speed = np.maximum(np.linalg.norm(measured_before, axis=1), np.linalg.norm(targets, axis=1))
    nearest = static_distance(path.reshape(-1, 3), triangles).reshape(ticks.shape).min(axis=1)-.1*speed
    if mover is not None:
        centers = mover[ticks.clip(max=len(before)-1)]
        mover_speed = np.linalg.norm(np.diff(mover, axis=0), axis=1).max(initial=0.)/.1
        moving = np.linalg.norm(path-centers, axis=2).min(axis=1)-.34-.1*(speed+mover_speed)
        nearest = np.minimum(nearest, moving)
    return np.where(complete, nearest-body_radius, np.nan)


def hazard_responses(nominal, protected, commands, measured, *, floor=.15):
    """Audit intervention and recovery against the current request at each boundary."""
    risky = np.isfinite(nominal) & (nominal < floor)
    edges = np.diff(np.r_[False, risky, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    events = []
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        improved = np.flatnonzero(protected[start:end]-nominal[start:end] >= .04)
        next_start = int(starts[i+1]) if i+1 < len(starts) else len(nominal)
        intent = commands[end:next_start]
        speed = np.linalg.norm(intent, axis=1)
        feasible = (speed > .03) & (nominal[end:next_start] >= floor)
        eligible = np.flatnonzero(feasible)
        projected = (intent*measured[end:next_start]).sum(axis=1)/np.maximum(speed,.001)
        tracking = feasible & (projected >= .6*speed)
        stable = np.convolve(tracking.astype(np.int8), np.ones(5, dtype=np.int8), mode='valid') if len(tracking) >= 5 else []
        recovered = np.flatnonzero(np.asarray(stable) == 5)
        events.append(dict(start_step=int(start), first_clear_step=int(end),
                           intervention_after_s=float(improved[0]*.1) if len(improved) else None,
                           first_feasible_request_step=int(end+eligible[0]) if len(eligible) else None,
                           recovery_after_s=float((recovered[0]+1-eligible[0])*.1) if len(recovered) else None))
    return dict(risk_windows=len(events), events=events)


def measured_clearance(samples, lane, end, triangles, *, crossing):
    # Neighboring intervals share endpoints throughout the first flight.
    starts = samples[:end, :-1, lane].reshape(-1, samples.shape[-1])
    points = np.concatenate((starts, samples[end-1, -1:, lane]), axis=0)
    best = (float('inf'), 0, 'static')
    for start in range(0, len(points), 4096):
        batch = points[start:start+4096].astype(np.float64)
        static = static_distance(batch[:, :3], triangles)
        moving = np.linalg.norm(batch[:, :3]-batch[:, 3:6], axis=-1)-.34 if crossing else None
        clearance = (np.minimum(static, moving) if crossing else static)-batch[:, 6]
        offset = int(clearance.argmin())
        if clearance[offset] < best[0]:
            best = (float(clearance[offset]), start+offset,
                    'mover' if crossing and moving[offset] < static[offset] else 'static')
    lower, tick, surface = best
    return dict(samples=len(points), minimum_clearance_lower_bound_m=lower,
                elapsed_physics_tick=tick, closest_surface=surface,
                maximum_measured_link_envelope_m=float(points[:, 6].max()))


def summarize(binding, envelope_pin, opportunity_pin=None, *, hazard_preview=False):
    import torch

    bound, ctx, report = validate_binding(binding)
    params = ctx['contract']['parameters']
    trials = conditions(params)
    if not report['summary']['witnesses_replayed']:
        raise ValueError('a completely audited command replay is required')
    envelope = read(envelope_pin)
    checked(envelope['asset'])
    if 'clearance_envelope' in params and params['clearance_envelope'] != envelope_pin:
        raise ValueError('clearance envelope differs from the launch pin')
    if envelope['status'] != 'USD_COLLIDER_BOUND_REQUIRES_LIVE_JOINT_RADIUS_VERIFICATION':
        raise ValueError('unrecognized collider envelope')
    radius = envelope['maximum_rest_pose_rotation_envelope_m']
    opportunities = []
    stop_windows = []
    if opportunity_pin is not None:
        frozen = read(opportunity_pin)
        if (set(frozen) not in ({'schema_version', 'run_id', 'windows'},
                              {'schema_version', 'run_id', 'windows', 'stop_windows'})
                or frozen['schema_version'] != 'instinctrl-frozen-progress-opportunities/1'
                or frozen['run_id'] != ctx['plan']['run_id'] or not frozen['windows']):
            raise ValueError('invalid predeclared progress opportunity identity')
        opportunities = frozen['windows']
        stop_windows = frozen.get('stop_windows', [])
        for row in opportunities:
            if (set(row) != {'id', 'layout', 'start_step', 'end_step'}
                    or type(row['start_step']) is not int or type(row['end_step']) is not int
                    or not 0 <= row['start_step'] < row['end_step']):
                raise ValueError('invalid predeclared progress window')
        if len({row['id'] for row in opportunities}) != len(opportunities):
            raise ValueError('duplicate predeclared progress window')
        seen = set()
        for row in stop_windows:
            if (set(row) != {'id', 'lane', 'start_step', 'end_step'} or row['id'] in seen
                    or type(row['lane']) is not int or not 0 <= row['lane'] < LANES
                    or type(row['start_step']) is not int or type(row['end_step']) is not int
                    or not 0 < row['start_step'] < row['end_step']):
                raise ValueError('invalid scheduled stop')
            seen.add(row['id'])

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
            dense_pin = item.get('clearance')
            dense = None if dense_pin is None else dense_clearance(checked(dense_pin), len(rows))
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
            if hazard_preview:
                velocity_before = np.concatenate((np.asarray(initial['root_state_w'])[None, :, 7:10], velocities[:-1]), axis=0)
                issued = np.asarray([row['v_final_w'] for row in rows], dtype=np.float64)
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
                fixed = {}
                for row in opportunities:
                    if row['layout'] == score['layout']:
                        finish = row['end_step']
                        if finish > len(rows):
                            raise ValueError('predeclared window exceeds workload horizon')
                        fixed[row['id']] = frozen_opportunity(velocities[:, lane], commands[:, lane], end,
                                                               row['start_step'], finish)
                values.append(dict(lane=lane, layout=score['layout'], trace=item['trace'], samples=end,
                                   minimum_clearance_lower_bound_m=float(nearest[step]-radius), step=step,
                                   closest_surface='mover' if mover is not None and mover[step] < static[step] else 'static',
                                   tracking_in_feasible_moving_windows=tracking,
                                   externally_scheduled_progress=fixed))
                if stop_windows:
                    values[-1]['externally_scheduled_stops'] = {
                        row['id']: frozen_stop(before[:, lane], roots[:, lane], velocities[:, lane], commands[:, lane],
                                               end, row['start_step'], row['end_step'])
                        for row in stop_windows if row['lane'] == lane}
                if dense is not None:
                    values[-1]['physics_tick_clearance'] = dict(
                        source=dense_pin, sample_period_ms=5,
                        **measured_clearance(dense, lane, end, triangles, crossing=mover is not None))
                if hazard_preview:
                    collider = values[-1].get('physics_tick_clearance', {}).get('maximum_measured_link_envelope_m', radius)
                    path_mover = None if mover is None else movers_before[:end, lane]
                    nominal = projected_clearance(before[:end, lane], velocity_before[:end, lane],
                                                  commands[:end, lane], path_mover, triangles, collider)
                    physical = projected_clearance(before[:end, lane], velocity_before[:end, lane],
                                                   issued[:end, lane], path_mover, triangles, collider)
                    values[-1]['hazard_response_preview'] = hazard_responses(nominal, physical, commands[:end, lane], velocities[:end, lane])
            models[name][trial] = values
    return dict(schema_version='instinctrl-sampled-physical-clearance/1', binding=binding, geometry=geometry_pin,
                envelope=envelope_pin, sample_period_ms=100, diagnostic_clearance_m=.15, sweep_horizon_s=.6,
                opportunities=opportunity_pin, models=models, scientific_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--envelope', type=Path, required=True)
    parser.add_argument('--opportunities', type=Path)
    parser.add_argument('--hazard-preview', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if stop_paths():
        raise RuntimeError('STOP')
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(pin(args.binding), pin(args.envelope), pin(args.opportunities) if args.opportunities else None,
                       hazard_preview=args.hazard_preview)
    if stop_paths():
        raise RuntimeError('STOP')
    print(save(args.output, result))


if __name__ == '__main__':
    main()
