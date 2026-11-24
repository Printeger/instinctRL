"""Recompute matched persistence outcomes from closed physical traces."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.import_legacy import pin
from tools.r6_direct.instinct_cycle import save
from tools.r6_direct.instinct_cycle_finish import guard
from tools.r6_evidence import checked


def lane_outcome(rows, lane, score):
    rows = rows[:next((i + 1 for i, row in enumerate(rows) if row['terminal'][lane]), len(rows))]
    axis = rows[0]['axis_w'][lane]
    tangent = (-axis[1], axis[0])
    origin = rows[0]['initial_position_w'][lane]
    visible = [row for row in rows if row['pre']['winning_beams'][lane]]
    cues = [row for row in rows if math.hypot(*row['reflex_direction_b'][lane][:2]) > .1]
    held = [row for row in rows if row['held_bias_b'] is not None
            and math.hypot(*row['held_bias_b'][lane][:2]) > .001]
    first = cues[0]['step'] if cues else None
    after_cue = rows[first:] if first is not None else []
    first_side = (sum(x * v for x, v in zip(tangent, cues[0]['v_final_w'][lane])) if cues else 0.)
    signed = math.copysign(1., first_side)
    issued = .1 * sum(signed * sum(x * v for x, v in zip(tangent, row['v_final_w'][lane]))
                       for row in after_cue)
    displacement = [sum(x * (v - p) for x, v, p in zip(tangent, row['vehicle_position_w'][lane], origin))
                    for row in rows]
    gap = [math.dist(row['vehicle_position_w'][lane], row['post']['sphere_position_w'][lane])
           for row in rows if row['post']['active'][lane]]
    return dict(score, first_visible=visible[0]['step'] if visible else None, first_cue=first,
                steps_with_cue=len(cues), steps_held=len(held),
                visible_after_cue_but_gyro_blocked=sum(row['gyro_peak'][lane] >= .22
                    and row['pre']['winning_beams'][lane] > 0 for row in after_cue),
                ics_zero_during_hold=sum(row['ics_beta'][lane] == 0. for row in held),
                issued_initial_side_integral_m=issued,
                maximum_lateral_displacement_m=max(map(abs, displacement)),
                minimum_root_to_sphere_center_m=min(gap, default=None))


def summarize(result):
    bound, _, report = validate_binding(result['analysis'])
    if bound['report'] is None or not report['summary']['witnesses_replayed']:
        raise ValueError('Missing independent physical trace audit')
    outcomes = {}
    for arm, model in report['summary']['models'].items():
        by_mode = defaultdict(list)
        for group in model['groups']:
            trace = checked(group['trace'])
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            for score in group['lanes']:
                by_mode[score['mode']].append(lane_outcome(rows, score['lane'], score))
        outcomes[arm] = by_mode
    reference, candidate = outcomes['reference'], outcomes['candidate']
    paired = {}
    for mode, baseline in reference.items():
        trial = candidate[mode]
        fields = ('tracking_rmse_mps', 'progress_m', 'horizontal_displacement_m',
                  'maximum_lateral_displacement_m', 'issued_initial_side_integral_m',
                  'steps_with_cue', 'steps_held', 'visible_after_cue_but_gyro_blocked', 'ics_zero_during_hold')
        paired[mode] = dict(episodes=len(trial),
                            reference_collisions=sum(row['collision'] for row in baseline),
                            candidate_collisions=sum(row['collision'] for row in trial),
                            collision_to_pass=sum(a['collision'] and not b['collision'] for a, b in zip(baseline, trial, strict=True)),
                            pass_to_collision=sum(not a['collision'] and b['collision'] for a, b in zip(baseline, trial, strict=True)),
                            means={arm: {field: fmean(v[field] for v in rows) for field in fields}
                                   for arm, rows in (('reference', baseline), ('candidate', trial))})
    return dict(analysis=result['analysis'], outcomes=outcomes, paired=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    guard()
    result_pin = pin(args.result)
    output = summarize(read(result_pin))
    guard()
    save(args.destination, dict(result=result_pin, **output))


if __name__ == '__main__':
    main()
