"""Offline motion replay. This is neither job settlement nor scientific acceptance.

The caller must separately bind the closed job, inventory, authorized protocol and
model. No simulator imports, model invocation, seed substitution or raw writes.
"""
import hashlib
import json
from pathlib import Path

from . import robustness

MAX_JSON_BYTES = 32 * 1024**2
MAX_RAW_BYTES = 256 * 1024**2
MAX_SUMMARY_BYTES = 48 * 1024


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def pin(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


class Inputs:
    """Keep the exact bytes consumed and reject changed or aliased originals."""
    def __init__(self):
        self.pins = {}

    def read(self, path, maximum=MAX_JSON_BYTES, lines=False):
        path = Path(path).absolute()
        if path.resolve() != path or not path.is_file():
            raise ValueError('MISSING_OR_ALIASED_INPUT')
        with path.open('rb') as stream:
            data = stream.read(maximum+1)
        if len(data) > maximum:
            raise ValueError('INPUT_SIZE_BOUND')
        identity = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        if str(path) in self.pins and self.pins[str(path)] != identity:
            raise ValueError('INPUT_CHANGED')
        self.pins[str(path)] = identity
        def invalid(value):
            raise ValueError('NONFINITE_JSON')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('DUPLICATE_JSON_KEY')
                result[key] = value
            return result
        def decode(raw):
            return json.loads(raw, parse_constant=invalid, object_pairs_hook=unique)
        return [decode(row) for row in data.splitlines()] if lines else decode(data)

    def verify(self):
        for expected in self.pins.values():
            if pin(expected['path']) != expected or Path(expected['path']).resolve() != Path(expected['path']):
                raise ValueError('INPUT_CHANGED')


def replay_blocks(output, blocks, *, run_id, seed, split, inputs=None):
    """Every registered block and all 32 lanes remain in the result on failure."""
    inputs = inputs if inputs is not None else Inputs()
    ids = [b['id'] for b in blocks]
    if not blocks or len(set(ids)) != len(ids) or any(
            not isinstance(i, str) or not i or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in i)
            for i in ids):
        raise ValueError('BLOCK_IDENTITIES')
    if not run_id or type(seed) is not int or split not in ('development', 'final'):
        raise ValueError('RUN_IDENTITY')
    results = []
    for block in blocks:
        if block['lanes'] != 32 or block['domain'] not in ('target', 'stress'):
            raise ValueError('BLOCK_DENOMINATOR_OR_DOMAIN')
        folder = Path(output)/'r6'/block['id']
        row = dict(block_id=block['id'], domain=block['domain'], total=32,
                   replay_complete=False, saved_result_equal=False, scoring=None, error=None)
        try:
            traces = inputs.read(folder/'motion_traces.json')
            score = robustness.score_block(block, traces, run_id=run_id)
            row['scoring'] = score
            saved = inputs.read(folder/'result.json')
            if any(saved.get(k) != v for k,v in dict(run_id=run_id, block_id=block['id'], seed=seed, split=split,
                    schema_version='r6-block-result/1').items()):
                raise ValueError('SAVED_RESULT_IDENTITY')
            row['replay_complete'] = True
            row['saved_result_equal'] = encoded(score) == encoded(saved['scoring'])
            if not row['saved_result_equal']:
                row['error'] = 'SAVED_SCORE_MISMATCH'
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            row['error'] = type(error).__name__+':'+str(error)[:240]
        results.append(row)
    inputs.verify()
    return results


def replay_original(output, *, inputs=None):
    """Use the unchanged original evaluator; no canonicalized seed view."""
    from . import evaluation as ev
    inputs = inputs if inputs is not None else Inputs()
    output = Path(output)
    raw = output/'evaluation_raw'/'r6_baseline_u640_original'
    traces = []
    for case in ev.CASE_ORDER:
        path = raw/(case+'.jsonl')
        rows = inputs.read(path, MAX_RAW_BYTES, lines=True)
        if type(rows[0]['seed']) is not int or rows[0]['seed'] != 20260909:
            raise ValueError('ORIGINAL_SEED_MISMATCH')
        for key in ('env_origins_w_m', 'initial_sensor_position_w_m', 'initial_root_state_w'):
            if len(rows[0][key]) != 32:
                raise ValueError('ORIGINAL_LANE_DENOMINATOR')
        for row in rows[1:]:
            for key in ('command_b', 'actual_velocity_b', 'sensor_positions_w_m', 'episode_ids',
                        'collision', 'below_height', 'above_height', 'ray_clearance_m', 'q', 'ics_beta', 'root_state_w'):
                if len(row[key]) != 32:
                    raise ValueError('ORIGINAL_LANE_DENOMINATOR')
        traces.extend(ev.replay_raw_case(str(path), case=case))
    score = ev.evaluate_checkpoint(traces, update=640)
    wrows = inputs.read(raw/'W.jsonl', MAX_RAW_BYTES, lines=True)[1:]
    motion = [ev.wall_motion_prerequisite([r['root_state_w'][lane][7+(lane>=16)] for r in wrows]) for lane in range(32)]
    wall = dict(per_lane=motion, all32=all(motion))
    saved = inputs.read(output/'r6_original_suite.json')
    saved_wall = inputs.read(output/'r6_original_wall_motion.json')
    inputs.verify()
    return dict(scoring=score, wall_motion=wall, saved_result_equal=encoded(score)==encoded(saved),
                saved_wall_equal=encoded(wall)==encoded(saved_wall), evaluator=pin(ev.__file__))


def summarize(blocks, original):
    """Bounded decision input with evidence failures separate from motion failures."""
    rows = []
    for row in blocks:
        score = row['scoring']
        rows.append(dict(block_id=row['block_id'], domain=row['domain'], total=32,
            replay_complete=row['replay_complete'], saved_result_equal=row['saved_result_equal'], error=row['error'],
            successes=None if score is None else score['successes'],
            subgroups=None if score is None else score['subgroups'],
            failed_lanes=None if score is None else [r['lane'] for r in score['lanes'] if not r['motion_pass']]))
    result = dict(schema_version='r6-motion-replay-summary/1', blocks=rows,
        original=original, scientific_acceptance=False,
        pending=['EXACT_JOB_INVENTORY_AND_MODEL_BINDING', 'MEASUREMENT_CHAIN_REPLAY',
                 'RESET_FORCE_AND_RANDOM_STREAM_AUDIT', 'INDEPENDENT_SCIENTIFIC_REVIEW'])
    if len(encoded(result)) > MAX_SUMMARY_BYTES:
        raise ValueError('SUMMARY_SIZE_BOUND')
    return result


def analyze(output, protocol_path, *, run_id, seed, split):
    inputs = Inputs()
    protocol = inputs.read(protocol_path)
    blocks = replay_blocks(output, protocol['blocks'], run_id=run_id, seed=seed, split=split, inputs=inputs)
    original = None
    original_summary = dict(replay_complete=False, error=None)
    try:
        original = replay_original(output, inputs=inputs)
        if original['evaluator']['sha256'] != protocol['old_suite']['source']['sha256']:
            raise ValueError('ORIGINAL_EVALUATOR_HASH')
        original_summary.update(replay_complete=True, saved_result_equal=original['saved_result_equal'],
            saved_wall_equal=original['saved_wall_equal'], cases=original['scoring']['cases'],
            all_gates_pass=original['scoring']['all_gates_pass'], wall_motion=original['wall_motion'])
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        original_summary['error'] = type(error).__name__+':'+str(error)[:240]
    inputs.verify()
    summary = summarize(blocks, original_summary)
    report = dict(schema_version='r6-motion-replay/1', run_id=run_id, seed=seed, split=split,
        protocol_frozen=protocol.get('frozen') is True, blocks=blocks, original=original,
        inputs=list(inputs.pins.values()), scorer=pin(robustness.__file__), analyzer=pin(__file__),
        scientific_acceptance=False, summary=summary)
    return report, summary
