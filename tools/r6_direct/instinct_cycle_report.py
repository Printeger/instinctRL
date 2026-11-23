"""Summarize exact-cycle evaluation against the two frozen, matched references."""

from math import hypot
from pathlib import Path
from statistics import mean

from tools.r6_direct.anchor_endpoint import BLOCKS, FIRST_STEP, LAST_STEP, STOP_STEP
from tools.r6_direct.evidence import read
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import FORMAL, HISTORICAL, EVAL, save
from tools.r6_direct.study_report import compare, score_model


def displacement(report_pin, block_id):
    report = read(report_pin)
    block, = (r for r in report['blocks'] if r['block_id'] == block_id)
    path = Path(block['measurement_replay']['audit']['path']).with_name('motion_traces.json')
    pinned, = (r for r in report['inputs'] if Path(r['path']) == path)
    traces = read({k: pinned[k] for k in ('path', 'sha256')})
    if [r['lane'] for r in traces] != list(range(32)):
        raise ValueError(f'Incomplete or unordered motion cohort: {path}')
    drifts = []
    for trace in traces:
        window = trace['rows'][STOP_STEP - 1:LAST_STEP + 1]
        if ([row['policy_step'] for row in window] != list(range(STOP_STEP - 1, LAST_STEP + 1))
                or any(row['missing'] or row['episode_id'] != window[0]['episode_id'] for row in window)):
            drifts.append(None)
            continue
        reference = window[0]['sensor_position_w_m']
        drifts.append(max(hypot(row['sensor_position_w_m'][0] - reference[0],
                                 row['sensor_position_w_m'][1] - reference[1])
                          for row in trace['rows'][FIRST_STEP:LAST_STEP + 1]))
    valid = [v for v in drifts if v is not None]
    return dict(lane_max_horizontal_m=drifts, valid_lanes=len(valid), total_lanes=32,
                mean_m=mean(valid) if valid else None, worst_m=max(valid) if valid else None)


def main():
    if paths := stop_paths():
        raise RuntimeError(f'STOP {paths}')
    evidence_pin = pin(FORMAL / 'FORMAL_EVIDENCE.json')
    evidence = read(evidence_pin)
    historical = read(evidence['references'])['models']
    jobs = [m for m in historical if (m['arm'], m['pair']) in (('baseline', 0), ('control', 1))]
    jobs.append(dict(arm='candidate', pair=1, run_id=EVAL, report=evidence['evaluation_report']))
    baseline, control, candidate = map(score_model, jobs)
    if len({r['seed'] for r in (baseline, control, candidate)}) != 1:
        raise ValueError('Reports do not share the historical evaluation seed')
    for model in (baseline, control, candidate):
        model['stop_displacement'] = {block: displacement(model['report'], block) for block in BLOCKS}
    settings_pin = pin(HISTORICAL / 'FREEZE.json')
    settings = read(settings_pin)
    comparisons = {name: compare(settings, [candidate], [reference])
                   for name, reference in (('against_baseline', baseline), ('against_equal_training', control))}
    training = read(evidence['training_report'])['summary']
    output = dict(schema_version='instinctrl-first-cycle-formal-comparison/1', evidence=evidence_pin,
                  paired_seed=candidate['seed'], settings=settings_pin,
                  training_counters=training['observed_counters'],
                  models=[baseline, control, candidate], comparisons=comparisons)
    if paths := stop_paths():
        raise RuntimeError(f'STOP {paths}')
    print(save(FORMAL / 'RESULTS.json', output))


if __name__ == '__main__':
    main()
