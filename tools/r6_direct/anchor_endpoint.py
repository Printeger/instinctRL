"""Read the third-training stop endpoint from existing direct evaluation audits."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from tools.r6_evidence import checked, file_pin
from training_workloads.unified_h1.robustness_audit import read_audit


BLOCKS = ('S-delay-drop', 'S-clean-low')
STOP_STEP = 100
FIRST_STEP = 110
LAST_STEP = 209
LANES = 32
MIN_USABLE = 0.1


def endpoint(records):
    errors = [[] for _ in range(LANES)]
    coverage = [[] for _ in errors]
    invalid = [[] for _ in errors]
    expected_step = STOP_STEP - 1
    for record in records:
        if record['kind'] != 'step_complete':
            continue
        row = record['payload']
        step = row['policy_step']
        if not STOP_STEP - 1 <= step <= LAST_STEP:
            continue
        if step != expected_step:
            raise ValueError('missing, duplicate or out-of-order endpoint step')
        expected_step += 1
        anchor = row['control']['anchor_metrics']
        fields = ('active', 'reset_code', 'usable_fraction', 'mean_absolute_error_m')
        if not all(len(anchor[key]) == LANES for key in fields) or any(
            len(row[key]) != LANES for key in ('reset_mask', 'lanes')
        ):
            raise ValueError('unexpected endpoint lane count')
        for lane in range(LANES):
            active, reset, usable, error = (anchor[key][lane][0] for key in fields)
            sample = row['lanes'][lane]
            expected_zero = step >= STOP_STEP
            if (sample['command_w_m_s'] == [0, 0, 0]) != expected_zero:
                raise ValueError('stop does not match the registered step')
            if not (math.isfinite(error) and error >= 0 and 0 <= usable <= 1
                    and active in (0, 1)):
                raise ValueError('invalid recorded anchor metric')
            if (row['reset_mask'][lane] or sample['episode_id'] != 0 or sample['missing']
                    or sample['collision'] or sample['height_violation'] or reset != 0
                    or active != int(expected_zero) or (expected_zero and usable < MIN_USABLE)):
                invalid[lane].append(step)
            if step >= FIRST_STEP:
                errors[lane].append(1000 * error)
                coverage[lane].append(usable)
    if expected_step != LAST_STEP + 1:
        raise ValueError('incomplete endpoint window')
    return [
        dict(lane=lane, anchor_mae_mm=None if bad else mean(values),
             mean_usable_fraction=mean(usable), min_usable_fraction=min(usable),
             invalid_steps=bad)
        for lane, (values, usable, bad) in enumerate(zip(errors, coverage, invalid))
    ]


def from_report(report_path, block_id):
    report_pin = file_pin(report_path)
    report = json.loads(checked(report_pin).read_text())
    if block_id not in BLOCKS:
        raise ValueError('unregistered stop block')
    block, = (row for row in report['blocks'] if row['block_id'] == block_id)
    if (not block['replay_complete'] or not block['saved_result_equal']
            or block['total'] != LANES or not block['measurement_replay']['bit_exact_delivery']):
        raise ValueError('incomplete audited evaluation block')
    archive = Path(block['measurement_replay']['audit']['path'])
    pins = {}
    for name in ('result.json', 'audit.bin'):
        path = archive.with_name(name)
        item, = (item for item in report['inputs'] if Path(item['path']) == path)
        pins[name] = {key: item[key] for key in ('path', 'sha256')}
        if checked(pins[name]).stat().st_size != item['bytes']:
            raise ValueError('audited file length changed')
    result = json.loads(checked(pins['result.json']).read_text())
    run_id, seed, split = (report[key] for key in ('run_id', 'seed', 'split'))
    if (result['run_id'], result['block_id'], result['seed'], result['split']) != (
        run_id, block_id, seed, split
    ):
        raise ValueError('block result identity mismatch')
    records = read_audit(archive, run_id=run_id, block_id=block_id,
                         expected_records=result['audit_records'])
    identity = next(records)
    if (identity['kind'] != 'identity' or identity['payload']['seed'] != seed
            or identity['payload']['split'] != split
            or identity['payload']['block']['id'] != block_id
            or identity['payload']['block']['task'] != 'stop'):
        raise ValueError('stop audit identity mismatch')
    lanes = endpoint(records)
    scores = [lane['anchor_mae_mm'] for lane in lanes]
    complete = all(score is not None for score in scores)
    return dict(run_id=run_id, seed=seed, split=split, block=identity['payload']['block'],
                report=report_pin, inputs=pins,
                lane_scores=lanes, anchor_mae_mm=mean(scores) if complete else None,
                between_lane_sd_mm=stdev(scores) if complete else None,
                registered_successes=block['scoring']['successes'])


def compare(reference_path, trial_path):
    blocks = {}
    for block_id in BLOCKS:
        reference = from_report(reference_path, block_id)
        trial = from_report(trial_path, block_id)
        if any(reference[key] != trial[key] for key in ('seed', 'split', 'block')):
            raise ValueError('endpoint requires the same evaluation block, seed and split')
        diffs = [
            b['anchor_mae_mm'] - a['anchor_mae_mm']
            if a['anchor_mae_mm'] is not None and b['anchor_mae_mm'] is not None else None
            for a, b in zip(reference['lane_scores'], trial['lane_scores'])
        ]
        blocks[block_id] = dict(
            reference=reference, trial=trial, trial_minus_reference_mm=diffs,
            mean_paired_delta_mm=mean(diffs) if all(d is not None for d in diffs) else None,
        )
    return dict(schema_version='instinctrl-stop-anchor-endpoint/1', primary_block=BLOCKS[0],
                window_policy_steps=[FIRST_STEP, LAST_STEP], blocks=blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference_report', type=Path)
    parser.add_argument('trial_report', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference_report, args.trial_report), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
