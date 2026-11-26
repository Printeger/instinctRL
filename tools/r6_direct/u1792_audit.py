"""Compare the matched short dose and audit each locked u1792 continuation."""

import argparse
import json
from pathlib import Path

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.holdout_summary import compare_holdout
from tools.r6_direct.import_legacy import pin, process_identity
from tools.r6_direct.instinct_cycle import HISTORICAL, save
from tools.r6_direct.instinct_cycle_finish import guard
from tools.r6_direct.next_long_audit import coverage_metrics, origin_guards, physical_metrics, resource_journal
from tools.r6_direct.post_r4_audit import paired_r6, verify_training
from tools.r6_direct.study_report import score_model
from tools.r6_evidence import checked as evidence_file


PHYSICAL_GATES = dict(maximum_ordinary_rmse_delta_mps=.01, minimum_ordinary_progress_ratio=.95,
                      maximum_hover_increment_m=.005, maximum_pipe_recovery_progress_loss_m=.03)


def short_comparison(spec_pin):
    spec = read(spec_pin)
    binding = spec['binding']
    _, ctx, report = validate_binding(binding)
    parameters = ctx['contract']['parameters']
    protocol = read(spec['protocol'])
    if parameters['seed'] != protocol['physical_seed']:
        raise ValueError('The short comparison requires the frozen development stream')
    checked = {}
    for role, result_pin in spec['training'].items():
        result = read(result_pin)
        audit = verify_training(result)
        if (audit['counts']['updates'] != protocol['added_updates'] or
                result['segments'][0]['plan']['training_seed'] != protocol['training_seed'] or
                parameters['models'][role] != result['end'] or parameters['models']['control'] != result['start'] or
                set(parameters['arm_options'].values()) != {result['execution']}):
            raise ValueError('The short comparison requires matched start, dose, seed and execution')
        checked[role] = coverage_metrics(result)
    results = compare_holdout(binding)
    comparisons = {}
    for reference in ('control', 'reference'):
        cases = [dict(row, means=dict(candidate=row['means']['candidate'], control=row['means'][reference]),
                      paired_effects=dict(control=row['paired_effects'][reference])) for row in results['cases']]
        checks = origin_guards(dict(cases=cases), dict(physical=dict(gates=PHYSICAL_GATES)))
        pipe = [case for case in results['cases'] if case['layout'] == 'open_tube']
        approach, = (case for case in pipe if case['mode'] == 'approach')
        recovered = approach['recovered_without_contact']
        saved = sum(case['paired_effects'][reference]['collision_to_pass'] for case in pipe)
        checks.update(pipe_saved=saved, pipe_recovery_count_delta=recovered['candidate']-recovered[reference],
                      improved=saved > 0 or checks['pipe_progress_delta_m'] >= .03)
        checks['qualifies'] = checks['passed'] and checks['improved'] and checks['pipe_recovery_count_delta'] >= 0
        comparisons[reference] = checks
    coverage = {role: value['totals'] for role,value in checked.items()}
    keys = ('visible_closing_episodes', 'eligible_recovery_steps', 'issued_recovery_steps', 'quiet_hover_episodes')
    coverage_pass = all(coverage['candidate'][key] >= coverage['reference'][key] for key in keys)
    return dict(input=spec_pin, metrics=physical_metrics(report['summary']), comparisons=comparisons,
                coverage=checked, coverage_pass=coverage_pass,
                passed=coverage_pass and all(checks['qualifies'] for checks in comparisons.values()))


def formal_evidence(evaluations_pin):
    assembly = read(evaluations_pin)
    assignments = read(assembly['assignments'])
    protocol = read(assignments['protocol'])
    roles = assignments['roles']
    r6 = {}
    for role in assignments['r6_arm_order']:
        binding = read(assembly['r6'][role])['binding']
        bound, ctx, result = validate_binding(binding)
        params = ctx['contract']['parameters']
        counters = ctx['state']['counters']
        expected = protocol['r6']
        if (params['evaluation_seed'] != expected['original_seed'] or params['r6']['seed'] != expected['perturbation_seed']
                or params['r6']['split'] != expected['perturbation_split']
                or counters['training_transitions'] != 0 or counters['real_optimizer_updates'] != 0
                or counters['nontraining_transitions'] != expected['transitions_per_arm']):
            raise ValueError(f'R6 stream differs for role {role}')
        model = score_model(dict(run_id=result['run_id'], report=bound['report'], arm=role, pair=1))
        if (model['checkpoint'] != roles[role]['checkpoint']
                or ctx['plan']['result_payload_cap_bytes'] != assignments['r6_result_payload_cap_bytes']):
            raise ValueError(f'R6 assignment differs for role {role}')
        r6[role] = dict(model, binding=binding, receipt=bound['receipt'], transitions=counters['nontraining_transitions'])
    tolerance = protocol['r6']['max_paired_tracking_increase_mps']
    prior_gates = dict(read(pin(HISTORICAL/'FREEZE.json')),
                       clean_increase_mm=protocol['r6']['max_clean_stop_anchor_increase_mm'],
                       coverage_drop=protocol['r6']['max_coverage_drop_fraction'], tracking_rmse_increase_mps=tolerance)
    shared_cases = {}
    shared_control_contacts = {}
    rounds = {}
    for arm, formal_pin in assignments['formal'].items():
        result = read(formal_pin)
        verification = verify_training(result)
        decision = read(verification['decision'])
        coverage = coverage_metrics(result)
        minimums = decision['coverage_minimums']
        received = {key: coverage['totals'].get(key, 0) for key in minimums}
        training_journals = {}
        for row in result['segments']:
            _, ctx, _ = validate_binding(row['analysis'])
            training_journals[ctx['plan']['run_id']] = resource_journal(ctx)
        physical = read(assembly['physical'][arm])
        _, ctx, report = validate_binding(physical['binding'])
        params = ctx['contract']['parameters']
        mapping = dict(candidate=arm, control='origin', reference='qualified')
        expected = protocol['physical']
        if (params['seed'] != expected['seed'] or params['models'] != {a: roles[r]['checkpoint'] for a,r in mapping.items()}
                or params['arm_options'] != {a: roles[r]['execution'] for a,r in mapping.items()}
                or ctx['state']['counters']['nontraining_transitions'] != 3*expected['transitions_per_arm']):
            raise ValueError(f'Physical assignment differs for {arm}')
        metrics = physical_metrics(report['summary'])
        for layouts in metrics.values():
            if set(layouts) != set(expected['layouts']) or any(sum(rows[key]['denominator'] for rows in layouts.values()) != value
                                                             for key,value in expected['denominators'].items()):
                raise ValueError('Physical holdout has an incomplete denominator')
        for key in ('control', 'reference'):
            groups = report['summary']['models'][key]['groups']
            signature = []
            for group in groups:
                with evidence_file(group['trace']).open() as stream:
                    first = json.loads(next(stream))
                signature.append({k:first[k] for k in ('layout', 'modes', 'initial_position_w', 'axis_w', 'command_w')})
            if shared_cases.setdefault(key, signature) != signature:
                raise ValueError(f'The repeated common physical {key} initial assignments did not match')
            shared_control_contacts.setdefault(key, {})[arm] = [sum(lane['collision'] for lane in group['lanes'])
                                                                for group in groups]
        paired = compare_holdout(physical['binding'])
        same_execution = origin_guards(paired, protocol)
        r6_pairs = {ref: paired_r6(r6[arm], r6[ref], tolerance, prior_gates) for ref in ('origin','qualified')}
        gates = dict(coverage=all(received[key]>=value for key,value in minimums.items()),
                     physical_qualified=paired['gates_pass'], physical_origin=same_execution['passed'],
                     r6_qualified=r6_pairs['qualified']['passed'], r6_origin=r6_pairs['origin']['passed'])
        rounds[arm] = dict(training=verification, coverage=coverage, minimums=minimums,
                           physical=metrics, physical_pairing=paired, same_execution=same_execution,
                           r6_pairs=r6_pairs, training_runtime_journals=training_journals, gates=gates,
                           passed=all(gates.values()))
    receipts = []
    for row in read(assembly['ledger'])['new_reservations']:
        if row['id'] == 'ROUND-SYSTEM':
            continue
        receipt = read(row['receipt'])
        if receipt['status'] != 'CLOSED' or receipt['remaining'] or any(process_identity(i['pid']) == i
                for i in [receipt['parent'], *receipt['owned_processes']]):
            raise ValueError('There is an unclosed supervised process')
        receipts.append(dict(run_id=row['id'], receipt=row['receipt'], native_exit=receipt['exit_code'],
                             native_error=receipt['error'], retained_bytes=row.get('closed_evidence_bytes')))
    return dict(evaluations=evaluations_pin, protocol=assignments['protocol'], rounds=rounds,
                valid_formal_count=sum(result['passed'] for result in rounds.values()), r6=r6, receipts=receipts,
                shared_control_contacts=shared_control_contacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('short', 'formal'))
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    guard()
    result = (short_comparison if args.mode=='short' else formal_evidence)(pin(args.input))
    guard()
    print(json.dumps(save(args.output, result)))


if __name__ == '__main__':
    main()
