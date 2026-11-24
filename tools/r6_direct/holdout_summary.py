"""Compare the prespecified held-out physical endpoints on paired episodes."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean, median

from tools.r6_direct.evidence import validate_binding
from tools.r6_direct.import_legacy import pin
from tools.r6_direct.instinct_cycle import save
from tools.r6_direct.instinct_cycle_finish import guard
from tools.r6_evidence import checked


def compare_holdout(binding):
    _, _, report = validate_binding(binding)
    models = {}
    for arm, model in report['summary']['models'].items():
        by_case = defaultdict(list)
        for group in model['groups']:
            for row in group['lanes']:
                by_case[(row['layout'], row['mode'])].append(row)
        models[arm] = by_case
    cases = []
    fields = ('progress_m', 'tracking_rmse_mps', 'recovery_rmse_mps', 'maximum_horizontal_excursion_m')
    for (layout, mode), candidate in sorted(models['candidate'].items()):
        key = layout, mode
        case = dict(layout=layout, mode=mode, episodes=len(candidate), means={}, collisions={}, paired_effects={})
        for arm, outcomes in models.items():
            rows = outcomes[key]
            if len(rows) != len(candidate) or [r['lane'] for r in rows] != [r['lane'] for r in candidate]:
                raise ValueError('Holdout pairing key differs across models')
            case['collisions'][arm] = sum(r['collision'] for r in rows)
            case['means'][arm] = {field: fmean(r[field] for r in rows) for field in fields}
            if arm != 'candidate':
                case['paired_effects'][arm] = dict(
                    collision_to_pass=sum(a['collision'] and not b['collision'] for a, b in zip(rows, candidate, strict=True)),
                    pass_to_collision=sum(not a['collision'] and b['collision'] for a, b in zip(rows, candidate, strict=True)))
        cases.append(case)
    ordinary = {r['layout']: r for r in cases if r['mode'] == 'static_travel'}
    for case in cases:
        if case['mode'] == 'approach':
            progress = ordinary[case['layout']]['means']
            case['progress_fraction'] = {arm: m['progress_m'] / progress[arm]['progress_m']
                                        if progress[arm]['progress_m'] > 0. else None
                                        for arm, m in case['means'].items()}
    forwards = [r for r in cases if r['mode'] in ('open_front', 'approach')]
    collisions = {arm: sum(r['collisions'][arm] for r in forwards) for arm in models}
    motion_pass = all(r['means']['candidate']['progress_m'] >= max(1.5, .95*r['means']['reference']['progress_m'])
                      and r['means']['candidate']['tracking_rmse_mps'] <= r['means']['reference']['tracking_rmse_mps']+.010
                      for r in ordinary.values())
    hover_pass = all(r['means']['candidate']['maximum_horizontal_excursion_m']
                     <= r['means']['reference']['maximum_horizontal_excursion_m']+.005
                     for r in cases if r['mode'] == 'static_hover')
    gates = dict(front_collisions_halved=2*collisions['candidate'] <= collisions['reference'],
                 no_new_collisions=all(r['paired_effects']['reference']['pass_to_collision'] == 0 for r in cases),
                 ordinary_travel=motion_pass, still_hover=hover_pass)
    return dict(cases=cases, front_episode_collisions=collisions, engineering_gates=gates,
                receipts_verified=True, gates_pass=all(gates.values()))


def mechanisms(binding):
    """Describe the observed sensing-to-motion chain on the initial episodes."""
    _, _, report = validate_binding(binding)
    cases = defaultdict(list)
    for arm, model in report['summary']['models'].items():
        for group in model['groups']:
            raw = [json.loads(line) for line in checked(group['trace']).read_text().splitlines()]
            for lane_info in group['lanes']:
                if lane_info['mode'].startswith('static_'):
                    continue
                lane = lane_info['lane']
                end = next((i+1 for i,row in enumerate(raw) if row['terminal'][lane]), len(raw))
                rows = raw[:end]
                first_visible = next((i for i,row in enumerate(rows) if row['pre']['winning_beams'][lane]), None)
                active = [i for i,row in enumerate(rows) if row['held_bias_b'] is not None
                          and math.hypot(*row['held_bias_b'][lane][:2]) > .1]
                axis = rows[0]['axis_w'][lane]
                side = (-axis[1], axis[0])
                issued = [sum(x*y for x,y in zip(side, r['v_final_w'][lane])) for r in rows]
                location = rows[0]['initial_position_w'][lane]
                lateral = [abs(sum(x*(y-z) for x,y,z in zip(side, r['vehicle_position_w'][lane], location)))
                           for r in rows]
                signed_issued = .1 * sum(issued[active[0]:]) if active else None
                cases[(arm, lane_info['layout'], lane_info['mode'])].append(dict(
                    collision=lane_info['collision'], visible_s=None if first_visible is None else first_visible*.1,
                    first_option_s=None if not active else active[0]*.1, option_active_s=len(active)*.1,
                    option_ics_rejected_s=.1*sum(rows[i]['ics_beta'][lane] == 0 for i in active),
                    absolute_net_lateral_command_m=None if signed_issued is None else abs(signed_issued),
                    maximum_lateral_displacement_m=max(lateral),
                    minimum_root_to_obstacle_center_m=min((math.dist(r['vehicle_position_w'][lane],
                        r['post']['sphere_position_w'][lane]) for r in rows if r['post']['active'][lane]), default=None)))
    result = []
    for (arm, layout, mode), episodes in sorted(cases.items()):
        for collision in (False, True):
            rows = [row for row in episodes if row['collision'] == collision]
            if not rows:
                continue
            medians = {}
            for field in rows[0].keys() - {'collision'}:
                values = [row[field] for row in rows if row[field] is not None]
                medians[field] = median(values) if values else None
            result.append(dict(arm=arm, layout=layout, mode=mode, collision=collision, episodes=len(rows),
                               saw_mover=sum(row['visible_s'] is not None for row in rows),
                               applied_option=sum(row['first_option_s'] is not None for row in rows),
                               medians=medians))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binding', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    guard()
    binding = pin(args.binding)
    rows = mechanisms(binding)
    guard()
    save(args.destination, dict(binding=binding, analyzer=pin(Path(__file__)), episodes_by_outcome=rows))


if __name__ == '__main__':
    main()
