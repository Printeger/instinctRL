"""Recompute the next continuation's frozen training and independent evidence."""

import argparse
from collections import defaultdict
import json
from math import sqrt
from pathlib import Path
from statistics import fmean

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.holdout_summary import compare_holdout
from tools.r6_direct.import_legacy import pin, process_identity
from tools.r6_direct.instinct_cycle import HISTORICAL, save
from tools.r6_direct.instinct_cycle_finish import guard
from tools.r6_direct.post_r4_audit import paired_r6, verify_training
from tools.r6_direct.study_report import score_model
from training_workloads.unified_h1.recovery_coverage import audit as audit_coverage


CATEGORIES = dict(forward=('open_front', 'approach'), rear=('open_rear',),
                  nonzero=('approach', 'cross_left', 'cross_right', 'static_travel'),
                  still_hover=('static_hover',))


def physical_metrics(summary):
    arms = {}
    for arm, model in summary['models'].items():
        layouts = defaultdict(list)
        for group in model['groups']:
            for lane in group['lanes']:
                layouts[lane['layout']].append(lane)
        arms[arm] = {}
        for layout, lanes in layouts.items():
            values = {}
            for category, modes in CATEGORIES.items():
                selected = [lane for lane in lanes if lane['mode'] in modes]
                approaches = [lane for lane in selected if lane['mode'] == 'approach']
                metrics = dict(denominator=len(selected), contacts=sum(lane['collision'] for lane in selected),
                               tracking_rmse_mps=sqrt(fmean(lane['tracking_rmse_mps']**2 for lane in selected)))
                if approaches:
                    metrics.update(recovery_denominator=len(approaches),
                                   mean_fixed_window_progress_m=fmean(r['recovery_progress_m'] for r in approaches),
                                   fixed_window_tracking_rmse_mps=sqrt(fmean(r['recovery_rmse_mps']**2 for r in approaches)),
                                   joint_recovery=sum(not r['collision'] and r['recovery_progress_m']>=.5
                                                      for r in approaches))
                if category == 'still_hover':
                    excursions = [r['maximum_horizontal_excursion_m'] for r in selected]
                    metrics.update(mean_horizontal_excursion_m=fmean(excursions), worst_horizontal_excursion_m=max(excursions))
                values[category] = metrics
            values['all'] = dict(denominator=len(lanes), contacts=sum(lane['collision'] for lane in lanes))
            arms[arm][layout] = values
    return arms


def coverage_metrics(result):
    totals = defaultdict(int)
    segments = []
    for segment, saved in zip(result['segments'], result['pipe_coverage'], strict=True):
        recomputed = audit_coverage(segment['analysis'])
        if recomputed != read(saved):
            raise ValueError('Saved physical training coverage differs from the originals')
        segments.append(dict(evidence=saved, phases=recomputed['phases']))
        for phase in recomputed['phases']:
            totals['pipe_lane_steps'] += phase['pipe_lane_steps']
            for key, n in phase['totals'].items():
                totals[key] += n
    return dict(totals=dict(totals), segments=segments)


def audit(evaluations):
    assembly = read(evaluations)
    assignments = read(assembly['assignments'])
    protocol = read(assignments['protocol'])
    formal = read(assignments['formal'])
    training = verify_training(formal)
    coverage = coverage_metrics(formal)
    decision = read(read(formal['design'])['decision'])
    if (decision['independent_protocol'] != assignments['protocol']
            or training['counts']['updates'] != protocol['added_updates']
            or training['counts']['transitions'] < 524288
            or formal['segments'][-1]['plan']['target_update'] != protocol['fixed_endpoint']):
        raise ValueError('Formal execution differs from the frozen endpoint')
    minimums = decision['coverage_minimums']
    coverage['minimums'] = minimums
    coverage['passed'] = all(coverage['totals'].get(k, 0)>=n for k, n in minimums.items())

    physical_bound = read(assembly['physical'])
    physical_pin = physical_bound['binding']
    _, physical_ctx, physical_result = validate_binding(physical_pin)
    physical_params = physical_ctx['contract']['parameters']
    roles = assignments['roles']
    mapping = dict(reference='qualified', control='origin', candidate='end')
    if 'prior' in roles:
        mapping['prior'] = 'prior'
    expected = protocol['physical']
    if (physical_params['seed'] != expected['seed']
            or physical_params['models'] != {a: roles[r]['checkpoint'] for a, r in mapping.items()}
            or physical_params['arm_options'] != {a: roles[r]['execution'] for a,r in mapping.items()}
            or physical_ctx['state']['counters']['nontraining_transitions'] != expected['transitions_per_arm']*len(mapping)):
        raise ValueError('Physical roles, seed, execution or counts differ from the frozen assignment')
    metrics = physical_metrics(physical_result['summary'])
    for layouts in metrics.values():
        if set(layouts) != set(expected['layouts']):
            raise ValueError('Physical layouts differ from the frozen assignment')
        for category, denominator in expected['denominators'].items():
            if sum(results[category]['denominator'] for results in layouts.values()) != denominator:
                raise ValueError('Physical denominators differ from the frozen assignment')

    r6 = {}
    for arm in assignments['r6_arm_order']:
        evaluation = read(assembly['r6'][arm])
        bound, ctx, report = validate_binding(evaluation['binding'])
        params = ctx['contract']['parameters']
        counters = ctx['state']['counters']
        if (params['evaluation_seed'] != protocol['r6']['original_seed']
                or params['r6']['seed'] != protocol['r6']['perturbation_seed']
                or params['r6']['split'] != protocol['r6']['perturbation_split']
                or counters['training_transitions'] != 0 or counters['real_optimizer_updates'] != 0
                or counters['nontraining_transitions'] != protocol['r6']['transitions_per_arm']):
            raise ValueError(f'R6 protocol or counts drifted for {arm}')
        row = score_model(dict(run_id=report['run_id'], report=bound['report'], arm=arm, pair=1))
        requested = params['evaluation_selection']
        if row['checkpoint'] != roles[arm]['checkpoint'] or requested.get('execution_override') != roles[arm].get('execution_override'):
            raise ValueError(f'R6 role assignment drifted for {arm}')
        r6[arm] = dict(row, binding=evaluation['binding'], receipt=bound['receipt'],
                       transitions=counters['nontraining_transitions'])
    prior_gates_pin = pin(HISTORICAL/'FREEZE.json')
    tolerance = protocol['r6']['max_paired_tracking_increase_mps']
    r6_pairs = {role: paired_r6(r6['end'], arm, tolerance, read(prior_gates_pin))
                for role, arm in r6.items() if role != 'end'}
    if 'prior' in r6:
        r6_pairs['origin_vs_prior_same_weights'] = paired_r6(r6['origin'], r6['prior'], tolerance, read(prior_gates_pin))

    receipts = []
    for reservation in read(assembly['ledger'])['new_reservations']:
        if reservation['id'] == 'ROUND-SYSTEM':
            continue
        receipt_pin = reservation['receipt']
        receipt = read(receipt_pin)
        owners = [receipt['parent'], *receipt['owned_processes']]
        if receipt['status'] != 'CLOSED' or receipt['remaining'] or any(process_identity(p['pid']) == p for p in owners):
            raise ValueError('An owned process remains live')
        receipts.append(dict(run_id=reservation['id'], receipt=receipt_pin,
                             native_exit=receipt['exit_code'], native_error=receipt['error'],
                             retained_bytes=reservation.get('closed_evidence_bytes'), wall_seconds=receipt['wall_seconds']))
    return dict(evaluations=evaluations, protocol=assignments['protocol'], training=training, coverage=coverage,
                physical=dict(binding=physical_pin, by_layout=metrics, pairs=physical_bound['metrics']['pairs'],
                              guards=compare_holdout(physical_pin), transitions=physical_ctx['state']['counters']['nontraining_transitions']),
                r6=r6, r6_pairs=r6_pairs, prior_r6_gates=prior_gates_pin, receipts=receipts, all_processes_closed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluations', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    guard()
    result = audit(pin(args.evaluations))
    guard()
    print(json.dumps(save(args.output, result)))


if __name__ == '__main__':
    main()
