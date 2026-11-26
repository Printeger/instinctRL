"""Recompute the next continuation's frozen training and independent evidence."""

import argparse
from collections import defaultdict
import json
from math import dist, sqrt
from pathlib import Path
from statistics import fmean

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.holdout_summary import compare_holdout
from tools.r6_direct.import_legacy import pin, process_identity
from tools.r6_direct.instinct_cycle import HISTORICAL, save
from tools.r6_direct.instinct_cycle_finish import guard
from tools.r6_direct.post_r4_audit import paired_r6, verify_training
from tools.r6_direct.study_report import score_model
from tools.r6_evidence import checked
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


def resource_journal(ctx):
    gate = Path(ctx['plan']['attempt_dir'])/'prelaunch'
    journal = pin(gate/'runtime_resource_observations.jsonl')
    summary = pin(gate/'runtime_resources.json')
    with checked(journal).open() as stream:
        observations = [json.loads(line) for line in stream]
    if (not observations or observations != read(summary)['observations']
            or any(r['free_memory_bytes'] < r['minimum_free_memory_bytes'] for r in observations)):
        raise ValueError('Runtime memory decisions differ from the closed resource journal')
    return dict(journal=journal, summary=summary, observations=len(observations),
                minimum_free_memory_bytes=min(r['free_memory_bytes'] for r in observations))


def origin_guards(physical, protocol):
    limits = protocol['physical']['gates']
    ordinary, hover = {}, {}
    for case in physical['cases']:
        trial, start = (case['means'][arm] for arm in ('candidate', 'control'))
        if case['mode'] == 'static_travel':
            tracking = trial['tracking_rmse_mps']-start['tracking_rmse_mps']
            progress = trial['progress_m']/start['progress_m'] if start['progress_m'] > 0 else None
            ordinary[case['layout']] = dict(tracking_delta_mps=tracking, progress_ratio=progress,
                passed=tracking <= limits['maximum_ordinary_rmse_delta_mps']
                and progress is not None and progress >= limits['minimum_ordinary_progress_ratio'])
        if case['mode'] == 'static_hover':
            delta = trial['maximum_horizontal_excursion_m']-start['maximum_horizontal_excursion_m']
            hover[case['layout']] = dict(drift_delta_m=delta, passed=delta <= limits['maximum_hover_increment_m'])
    pipe, = (case for case in physical['cases'] if (case['layout'], case['mode']) == ('open_tube', 'approach'))
    progress = pipe['means']['candidate']['recovery_progress_m']-pipe['means']['control']['recovery_progress_m']
    saved = sum(case['paired_effects']['control']['collision_to_pass'] for case in physical['cases'])
    new = sum(case['paired_effects']['control']['pass_to_collision'] for case in physical['cases'])
    gates = dict(no_new_contacts=new == 0, ordinary=all(case['passed'] for case in ordinary.values()),
                 hover=all(case['passed'] for case in hover.values()),
                 pipe_progress=progress >= -limits['maximum_pipe_recovery_progress_loss_m'])
    return dict(new_contacts=new, saved_contacts=saved, ordinary=ordinary, hover=hover,
                pipe_progress_delta_m=progress, gates=gates, passed=all(gates.values()))


def development_dose(curve_pin, formal, decision, protocol, coverage):
    middle, end = (row['checkpoint'] for row in formal['segments'])
    curve = read(curve_pin)
    _, ctx, report = validate_binding(curve['binding'])
    parameters = ctx['contract']['parameters']
    execution = formal['execution']
    if (parameters['seed'] != protocol['development_curve']['seed']
            or parameters['models'] != dict(reference=middle, candidate=end)
            or parameters['arm_options'] != dict(reference=execution, candidate=execution)):
        raise ValueError('Development dose differs from the frozen nodes or seed')
    admission = next(p for p in decision['development_evidence'] if Path(p['path']).name == 'FORMAL-ADMISSION.json')
    baseline = read(read(admission)['physical'])
    _, start_ctx, start_report = validate_binding(baseline['binding'])
    start_params = start_ctx['contract']['parameters']
    if (start_params['seed'] != parameters['seed'] or start_params['models']['reference'] != formal['start']
            or start_params['arm_options']['reference'] != execution):
        raise ValueError('Development baseline differs from the frozen origin')
    start_groups = start_report['summary']['models']['reference']['groups']
    for model in report['summary']['models'].values():
        for start, trial in zip(start_groups, model['groups'], strict=True):
            with checked(start['trace']).open() as stream:
                left = json.loads(stream.readline())
            with checked(trial['trace']).open() as stream:
                right = json.loads(stream.readline())
            if (any(left[k] != right[k] for k in ('modes', 'layout', 'axis_w', 'command_w'))
                    or any(dist(a, b) > .0002 for first, second in (
                        (left['initial_position_w'], right['initial_position_w']),
                        (left['pre']['sphere_position_w'], right['pre']['sphere_position_w']))
                        for a, b in zip(first, second, strict=True))):
                raise ValueError('Development dose initial pairing differs from the prior origin')
    snapshots = [baseline['metrics']['arms']['reference'], curve['metrics']['arms']['reference'],
                 curve['metrics']['arms']['candidate']]
    rows = {}
    for update, arm in zip(protocol['development_curve']['absolute_checkpoints'], snapshots, strict=True):
        pipe = arm['by_layout']['open_tube']
        rows[str(update)] = dict(categories=arm['categories'], ordinary=pipe['static_travel'], recovery=pipe['approach'],
                                 hover=pipe['static_hover'])
    start, _, final = rows.values()
    thresholds = protocol['dose_effect_outcomes']
    differences = dict(ordinary_progress_m=final['ordinary']['progress_m']-start['ordinary']['progress_m'],
                       recovery_progress_m=final['recovery']['recovery_progress_m']-start['recovery']['recovery_progress_m'])
    return dict(curve=curve_pin, nodes=rows, terminal_deltas=differences, matched_initial_geometry=True,
                coverage_passed=coverage['passed'],
                preserves_both_progress=coverage['passed']
                and differences['ordinary_progress_m'] >= -thresholds['ordinary_pipe_progress_margin_m']
                and differences['recovery_progress_m'] >= -thresholds['recovery_pipe_progress_margin_m'])


def audit(evaluations, curve):
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
    journals = {}
    if 'recovery' in decision:
        for segment in formal['segments']:
            _, segment_ctx, _ = validate_binding(segment['analysis'])
            journals[segment_ctx['plan']['run_id']] = resource_journal(segment_ctx)

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
    physical_guards = compare_holdout(physical_pin)
    start_guards = origin_guards(physical_guards, protocol)
    execution_contrast = None
    if 'prior' in roles:
        execution_contrast = defaultdict(dict)
        for group, other in zip(*(physical_result['summary']['models'][a]['groups']
                                 for a in ('control', 'prior')), strict=True):
            for trial, prior in zip(group['lanes'], other['lanes'], strict=True):
                counts = execution_contrast[trial['layout']].setdefault(trial['mode'], dict(saved=0, new=0))
                counts['saved'] += prior['collision'] and not trial['collision']
                counts['new'] += trial['collision'] and not prior['collision']

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
        if ctx['plan']['result_payload_cap_bytes'] != assignments['r6_result_payload_cap_bytes']:
            raise ValueError(f'R6 launch payload differs from the bound resource envelope for {arm}')
        if 'recovery' in decision:
            journals[ctx['plan']['run_id']] = resource_journal(ctx)
        r6[arm] = dict(row, binding=evaluation['binding'], receipt=bound['receipt'],
                       transitions=counters['nontraining_transitions'])
    prior_gates_pin = pin(HISTORICAL/'FREEZE.json')
    tolerance = protocol['r6']['max_paired_tracking_increase_mps']
    r6_frozen = dict(read(prior_gates_pin), clean_increase_mm=protocol['r6']['max_clean_stop_anchor_increase_mm'],
                     coverage_drop=protocol['r6']['max_coverage_drop_fraction'], tracking_rmse_increase_mps=tolerance)
    r6_pairs = {role: paired_r6(r6['end'], arm, tolerance, r6_frozen)
                for role, arm in r6.items() if role != 'end'}
    if 'prior' in r6:
        r6_pairs['origin_vs_prior_same_weights'] = paired_r6(r6['origin'], r6['prior'], tolerance, r6_frozen)

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
    dose = development_dose(curve, formal, decision, protocol, coverage)
    promotion = dict(coverage=coverage['passed'], physical_qualified=physical_guards['gates_pass'],
                     physical_origin=start_guards['passed'], r6_qualified=r6_pairs['qualified']['passed'],
                     r6_origin=r6_pairs['origin']['passed'])
    return dict(evaluations=evaluations, protocol=assignments['protocol'], training=training, coverage=coverage,
                development_dose=dose, promotion=promotion, promotion_passed=all(promotion.values()),
                physical=dict(binding=physical_pin, by_layout=metrics, pairs=physical_bound['metrics']['pairs'],
                              guards=physical_guards, same_execution_guards=start_guards,
                              origin_vs_prior_same_weights=execution_contrast,
                              transitions=physical_ctx['state']['counters']['nontraining_transitions']),
                r6=r6, r6_pairs=r6_pairs, prior_r6_gates=prior_gates_pin, receipts=receipts,
                runtime_resource_journals=journals, all_processes_closed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluations', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--development-curve', type=Path, required=True)
    args = parser.parse_args()
    guard()
    result = audit(pin(args.evaluations), pin(args.development_curve))
    guard()
    print(json.dumps(save(args.output, result)))


if __name__ == '__main__':
    main()
