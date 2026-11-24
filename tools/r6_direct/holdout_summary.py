"""Compare the prespecified held-out physical endpoints on paired episodes."""

from collections import defaultdict
from statistics import fmean

from tools.r6_direct.evidence import validate_binding


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
