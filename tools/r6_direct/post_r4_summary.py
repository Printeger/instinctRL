"""Recompute matched physical outcomes from the committed direct assay."""

from collections import defaultdict
import json
import math
from statistics import fmean, median

from tools.r6_direct.evidence import validate_binding
from tools.r6_evidence import checked


def rear(binding):
    _, _, report = validate_binding(binding)
    result = {}
    for arm, item in report['summary']['models'].items():
        group, = (group for group in item['groups'] if all(lane['mode'] == 'open_rear' for lane in group['lanes']))
        trace = checked(group['trace'])
        raw = [json.loads(line) for line in trace.read_text().splitlines()]
        rows = defaultdict(list)
        for episode in group['lanes']:
            lane = episode['lane']
            terminal = next((r['step'] for r in raw if r['terminal'][lane]), len(raw) - 1)
            steps = raw[:terminal+1]
            visible = next((r['step'] for r in steps if r['pre']['winning_beams'][lane]), None)
            delivered = next((r['step'] for r in steps if r['mover_witness_delivered'][lane]), None)
            trigger = next((r['step'] for r in steps if r['escape_recovery']['active'][lane]), None)
            emergency = next((r['step'] for r in steps if r['ics_emergency'][lane]), None)
            contact = next((r['step'] for r in steps if r['collision'][lane]), None)
            horizon = terminal + 1 if emergency is None else emergency
            axis = steps[0]['axis_w'][lane]
            side = axis[1], -axis[0]
            issued = [r['v_final_w'][lane][0]*side[0] + r['v_final_w'][lane][1]*side[1] for r in steps]
            away = [-sum(speed*direction for speed,direction in zip(r['v_final_w'][lane],axis)) for r in steps]
            origin = steps[0]['initial_position_w'][lane]
            position = [sum(a*(b-c) for a,b,c in zip(side, r['vehicle_position_w'][lane], origin)) for r in steps]
            start = horizon if trigger is None else trigger
            first_lateral = next((r['step'] for r in steps[start:] if abs(issued[r['step']]) > .03), None)
            baseline = 0. if start == 0 else position[start-1]
            rows[episode['layout']].append(dict(lane=lane, visible_step=visible, delivered_step=delivered,
                trigger_step=trigger, first_issued_lateral_step=first_lateral, emergency_step=emergency, contact_step=contact,
                visible_to_emergency_steps=None if visible is None or emergency is None else emergency-visible,
                visible_to_contact_steps=None if visible is None or contact is None else contact-visible,
                triggered_lateral_mps=None if trigger is None else abs(issued[trigger]),
                triggered_axial_away_mps=None if trigger is None else away[trigger],
                pre_emergency_axial_away_command_m=sum(away[start:horizon])*.1,
                pre_emergency_lateral_command_m=abs(sum(issued[start:horizon]))*.1,
                pre_emergency_lateral_motion_m=max((abs(p-baseline) for p in position[start:horizon]), default=0.),
                maximum_physical_lateral_displacement_m=max(map(abs, position))))
        result[arm] = {layout: dict(trajectories=lanes, contacts=sum(r['contact_step'] is not None for r in lanes),
            denominator=len(lanes), medians={key: median(values) if values else None for key in lanes[0] if key != 'lane'
                   for values in [[r[key] for r in lanes if r[key] is not None]]}) for layout,lanes in rows.items()}
    return result


def physical(binding):
    _, _, report = validate_binding(binding)
    arms = {}
    per_case = {}
    for arm, data in report['summary']['models'].items():
        by_case = defaultdict(list)
        for group in data['groups']:
            for lane in group['lanes']:
                by_case[lane['layout'], lane['mode']].append(lane)
        per_case[arm] = by_case
        recorded = defaultdict(dict)
        for (layout, mode), rows in sorted(by_case.items()):
            recorded[layout][mode] = dict(denominator=len(rows), contacts=sum(row['collision'] for row in rows),
                tracking_rmse_mps=fmean(row['tracking_rmse_mps'] for row in rows),
                recovery_rmse_mps=fmean(row['recovery_rmse_mps'] for row in rows),
                recovery_progress_m=fmean(row['recovery_progress_m'] for row in rows),
                no_contact_and_recovered=sum(not row['collision'] and row['recovery_progress_m'] >= .5 for row in rows),
                progress_m=fmean(row['progress_m'] for row in rows),
                mean_maximum_horizontal_excursion_m=fmean(row['maximum_horizontal_excursion_m'] for row in rows),
                worst_horizontal_excursion_m=max(row['maximum_horizontal_excursion_m'] for row in rows))
        modes = dict(forward={'open_front', 'approach'}, rear={'open_rear'},
                     nonzero={'approach','cross_left','cross_right','static_travel'},
                     still_hover={'static_hover'}, total={mode for _, mode in by_case})
        categories = {}
        for name, selected in modes.items():
            rows = [r for (_, mode), lanes in by_case.items() if mode in selected for r in lanes]
            categories[name] = dict(denominator=len(rows), contacts=sum(row['collision'] for row in rows),
                                    tracking_rmse_mps=math.sqrt(fmean(r['tracking_rmse_mps']**2 for r in rows)))
        arms[arm] = dict(by_layout=dict(recorded), categories=categories)
    candidate = per_case['candidate']
    comparisons = {}
    for name, other in per_case.items():
        if name == 'candidate':
            continue
        if other.keys() != candidate.keys():
            raise ValueError('Physical case mismatch')
        paired = defaultdict(dict)
        for (layout, mode), trials in sorted(candidate.items()):
            reference = other[layout, mode]
            if [r['lane'] for r in trials] != [r['lane'] for r in reference]:
                raise ValueError('Physical lane mismatch')
            pairs = list(zip(trials, reference, strict=True))
            paired[layout][mode] = dict(saved=sum(ref['collision'] and not trial['collision'] for trial,ref in pairs),
                new=sum(trial['collision'] and not ref['collision'] for trial,ref in pairs),
                recovery_progress_delta_m=fmean(trial['recovery_progress_m'] - ref['recovery_progress_m'] for trial,ref in pairs))
        comparisons[name] = dict(paired)
    return dict(arms=arms, pairs=comparisons)
