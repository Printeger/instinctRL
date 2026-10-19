"""CPU motion/geometry scoring for R6's separately registered test schema.

No actor imports, simulator state mutation, or old-suite seed remapping.
Passing motion metrics is necessary but not sufficient: the runner must also
independently validate reset, measurement, force, RNG and artifact provenance.
"""
import math

DIRECTIONS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
              (2**-.5, 2**-.5, 0), (-2**-.5, 2**-.5, 0),
              (2**-.5, -2**-.5, 0), (-2**-.5, -2**-.5, 0))
MODES = ('toward', 'away', 'parallel', 'zero')


def norm(v): return math.sqrt(sum(x*x for x in v))
def sub(a, b): return tuple(x-y for x, y in zip(a, b))
def dot(a, b): return sum(x*y for x, y in zip(a, b))
def rms(v): return math.sqrt(sum(x*x for x in v)/len(v))


def vector(v):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError('VECTOR_SHAPE')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in v):
        raise ValueError('NONFINITE_VECTOR')
    return tuple(v)


def surface_distance(position, origin):
    x, y, z = sub(vector(position), vector(origin))
    outside = lambda value, lo, hi: max(lo-value, value-hi, 0)
    return min(norm((z, outside(x, -42, 42), outside(y, -42, 42))),
               norm((x-6, outside(y, -6, 6), outside(z, 0, 12))),
               norm((y-6, outside(x, -6, 6), outside(z, 0, 12))))


def lane_role(block, lane):
    if type(lane) is not int or not 0 <= lane < 32:
        raise ValueError('LANE_ID')
    if block['task'] != 'wall_modes':
        return 'd'+str(lane % 8), DIRECTIONS[lane % 8]
    axis = 0 if lane < 16 else 1
    mode = MODES[(lane % 16)//4]
    direction = [0., 0., 0.]
    if mode in ('toward', 'away'): direction[axis] = 1 if mode == 'toward' else -1
    elif mode == 'parallel': direction[1-axis] = 1 if lane % 2 == 0 else -1
    return ('x_wall' if axis == 0 else 'y_wall')+':'+mode, tuple(direction)


def command(block, lane, step):
    _, direction = lane_role(block, lane)
    task = block['task']
    if task == 'wall_modes':
        mode = MODES[(lane % 16)//4]
        active = mode == 'toward' or (mode in ('away', 'parallel') and step < 100)
        factor = 1 if active else 0
    elif task == 'stop': factor = 1 if step < 100 else 0
    elif task == 'reverse': factor = 0 if step < 50 else (1 if step < 125 else -1)
    elif task == 'pulse_hold': factor = 0
    else: raise ValueError('UNKNOWN_TASK')
    return tuple(factor*block['speed_m_s']*x for x in direction)


def consecutive(values, predicate, start=0, end=None, count=11):
    run = 0
    for i in range(start, len(values) if end is None else end):
        run = run+1 if predicate(values[i]) else 0
        if run >= count: return i
    return None


def confirmation(speeds, start, *, boundary='PRE_STEP', limit=3.):
    i = consecutive(speeds, lambda v: v <= .08, start)
    if i is None: return None
    elapsed = (i+1-start if boundary == 'PRE_STEP' else i-start)*.1
    return i if elapsed <= limit+1e-12 else None


def hold(positions, velocities, index, count=201):
    if index is None or len(positions) < index+count: return False
    p = positions[index:index+count]; v = velocities[index:index+count]
    return rms([norm(x) for x in v]) <= .08 and max(norm(sub(x, p[0])[:2]) for x in p) <= .30


def z_gate(positions, velocities, initial):
    return rms([norm(v) for v in velocities]) <= .08 and max(
        norm(sub(p, initial)[:2]) for p in positions) <= .30


def _validate(block, lane, trace, run_id):
    if trace['schema_version'] != 'r6-motion-trace/1' or trace['run_id'] != run_id:
        raise ValueError('TRACE_RUN_IDENTITY')
    if trace['block_id'] != block['id'] or type(trace['lane']) is not int or trace['lane'] != lane:
        raise ValueError('TRACE_LANE_IDENTITY')
    if block['lanes'] != 32 or block['steps'] != (200 if block['task'] == 'reverse' else 330):
        raise ValueError('BLOCK_CONTRACT')
    vector(trace['env_origin_w_m'])
    initial = trace['initial']; rows = trace['rows']
    if not isinstance(rows, list) or len(rows) != block['steps']:
        raise ValueError('MISSING_REQUIRED_SAMPLE')
    episode = initial['episode_id']
    if type(episode) is not int: raise ValueError('EPISODE_ID')
    for i, row in enumerate([initial]+rows):
        vector(row['sensor_position_w_m']); vector(row['velocity_w_m_s'])
        if type(row['episode_id']) is not int or row['episode_id'] != episode:
            raise ValueError('RESET_INTERRUPTED_NO_SPLICE')
        for flag in ('collision', 'height_violation', 'missing'):
            if type(row[flag]) is not bool: raise ValueError('INVALID_FLAG')
            if row[flag]: raise ValueError(flag.upper())
        if type(row['time_s']) not in (int, float) or not math.isfinite(row['time_s']) or abs(row['time_s']-i*.1)>1e-6:
            raise ValueError('OBSERVATION_TIMING')
        if i:
            if type(row['policy_step']) is not int or row['policy_step'] != i-1:
                raise ValueError('POLICY_STEP_SEQUENCE')
            actual = vector(row['command_w_m_s'])
            if norm(sub(actual, command(block, lane, i-1))) > 1e-6:
                raise ValueError('COMMAND_MISMATCH')
    return initial, rows


def score_lane(block, lane, trace, *, run_id):
    """Missing or malformed traces fail in the original registered denominator."""
    group, direction = lane_role(block, lane)
    result = dict(block_id=block['id'], lane=lane, subgroup=group, motion_pass=False,
                  scientific_acceptance='REQUIRES_INDEPENDENT_INPUT_CHAIN_AUDIT', metrics={})
    try:
        initial, rows = _validate(block, lane, trace, run_id)
        positions = [r['sensor_position_w_m'] for r in rows]
        velocities = [r['velocity_w_m_s'] for r in rows]
        speeds = [norm(v) for v in velocities]
        all_positions = [initial['sensor_position_w_m']]+positions
        geometry = [surface_distance(p, trace['env_origin_w_m']) for p in all_positions]
        height = max(abs(p[2]-all_positions[0][2]) for p in all_positions)
        metrics = result['metrics']
        metrics.update(min_distance_m=min(geometry), max_height_deviation_m=height)
        if min(geometry) < .8 or height > .20:
            result['reason'] = 'ALL_TIME_GEOMETRY_OR_HEIGHT'; return result
        task = block['task']
        mode = group.split(':')[-1]
        if task == 'wall_modes' and mode in ('away', 'parallel'): task = 'stop'
        if task == 'stop':
            errors = [norm(sub(velocities[i], command(block, lane, i))) for i in range(100)]
            moving = consecutive(velocities, lambda v: dot(v, direction) >= block['speed_m_s']/2, end=100) is not None
            index = confirmation(speeds, 100)
            metrics.update(incoming_rmse=rms(errors), incoming_stable_rmse=rms(errors[20:]),
                           incoming_vz_rms=rms([v[2] for v in velocities[:100]]),
                           movement_prerequisite=moving, confirmation_index=index)
            passed = moving and metrics['incoming_rmse'] <= .15 and metrics['incoming_stable_rmse'] <= .10 and metrics['incoming_vz_rms'] <= .08 and hold(positions, velocities, index)
        elif task == 'reverse':
            errors = [norm(sub(v, command(block, lane, i))) for i, v in enumerate(velocities)]
            confirmations = [consecutive(errors, lambda e: e <= .10, start, start+30) for start in (50,125)]
            movement = [consecutive(velocities, lambda v, sign=sign: sign*dot(v, direction) >= block['speed_m_s']/2, start, end) is not None for start,end,sign in ((50,125,1),(125,200,-1))]
            projections = [dot(velocities[i], command(block, lane, i))/block['speed_m_s'] for i in range(50,200)]
            metrics.update(nonzero_rmse=rms(errors[50:]), stable_rmse=rms(errors[69:125]+errors[144:]),
                           confirmation_indices=confirmations, movement_prerequisites=movement,
                           peak_projection=max(projections))
            passed = all(i is not None for i in confirmations) and all(movement) and metrics['nonzero_rmse']<=.15 and metrics['stable_rmse']<=.10 and max(projections)<=1.5*block['speed_m_s'] and z_gate(positions[:50], velocities[:50], all_positions[0])
        elif task == 'pulse_hold':
            index = confirmation(speeds, 52)
            metrics['confirmation_index'] = index
            passed = z_gate(positions[:50], velocities[:50], all_positions[0]) and hold(positions, velocities, index)
        elif task == 'wall_modes' and mode == 'zero':
            passed = z_gate(positions, velocities, all_positions[0])
        elif task == 'wall_modes' and mode == 'toward':
            first = next((i for i, d in enumerate(geometry[1:]) if d<=1.1), None)
            index = confirmation(speeds, first if first is not None else 0,
                                 boundary='OBSERVATION' if first is not None else 'PRE_STEP',
                                 limit=3 if first is not None else 20)
            moving = index is not None and consecutive(
                velocities, lambda v: dot(v, direction)>=block['speed_m_s']/2,
                end=max(0, index-10)) is not None
            enough_hold = index is not None and len(rows)-index >= 101
            passed = moving and enough_hold and rms(speeds[index:])<=.08
            metrics.update(trigger_index=first, confirmation_index=index, movement_prerequisite=moving)
        else: raise ValueError('UNKNOWN_TASK')
        result.update(motion_pass=bool(passed), reason='PASS' if passed else 'MOTION_GATE_FAILED')
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        result.update(reason='INVALID_EVIDENCE', detail=str(exc))
    return result


def wilson(successes, total):
    if total <= 0 or not 0 <= successes <= total: raise ValueError('DENOMINATOR')
    z = 1.959963984540054; p = successes/total; d = 1+z*z/total
    center = (p+z*z/(2*total))/d
    delta = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0., center-delta), min(1., center+delta)]


def score_block(block, traces, *, run_id):
    """Reject duplicates, preserve missing lanes, never shrink denominators."""
    indexed = {}
    for trace in traces:
        lane = trace.get('lane')
        if type(lane) is not int or not 0<=lane<32 or lane in indexed:
            raise ValueError('DUPLICATE_OR_UNKNOWN_LANE')
        indexed[lane] = trace
    rows = [score_lane(block, lane, indexed.get(lane, {}), run_id=run_id) for lane in range(32)]
    successes = sum(r['motion_pass'] for r in rows)
    groups = {}
    for r in rows: groups.setdefault(r['subgroup'], []).append(r)
    return dict(block_id=block['id'], domain=block['domain'], motion_pass=successes==32,
                successes=successes, total=32, wilson_95=wilson(successes,32), lanes=rows,
                subgroups={k: {'successes':sum(r['motion_pass'] for r in v), 'total':len(v),
                                'wilson_95':wilson(sum(r['motion_pass'] for r in v),len(v))}
                           for k,v in groups.items()},
                scientific_acceptance='REQUIRES_INDEPENDENT_INPUT_CHAIN_AUDIT')
