"""Check the three formal closure chains and the independent paired evaluations."""

import argparse
import json
from pathlib import Path

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.holdout_summary import compare_holdout
from tools.r6_direct.import_legacy import pin, process_identity
from tools.r6_direct.instinct_cycle import HISTORICAL, save
from tools.r6_direct.instinct_cycle_finish import guard
from tools.r6_direct.post_r4_summary import physical
from tools.r6_direct.study_report import compare, score_model


def verify_training(result):
    design = read(result['design'])
    recipe = read(design['recipe'])
    checkpoints = []
    rng = []
    counts = dict(updates=0, transitions=0, optimizer_steps_actor=0, optimizer_steps_critic=0)
    predecessor = result['start']
    rows = []
    for row in result['segments']:
        bound, ctx, report = validate_binding(row['analysis'])
        summary = report['summary']
        initialization = row['plan']['initialization']
        retained_rng = summary['rng_initialization_audit']
        if (row['receipt'] != bound['receipt'] or initialization['checkpoint'] != predecessor
                or row['plan'] != ctx['contract']['parameters']['r6_training_plan']
                or ctx['state']['counters'] != row['counters'] or report['issues']
                or not summary['probability_gate_recomputed'] or retained_rng['status'] != 'PASS'
                or retained_rng['physical_continuity'] is not False):
            raise ValueError('Training source, counters or checkpoint inheritance drifted')
        if rng and retained_rng['resume_rng_equal'] is not True:
            raise ValueError('The segment RNG did not resume bit-exactly')
        rng.append(retained_rng)
        for checkpoint in summary['checkpoints']:
            audit = checkpoint['state_audit']
            if (checkpoint['pin'] != pin(checkpoint['pin']['path']) or not checkpoint['metadata_verified']
                    or not audit['actor_critic_state_loaded'] or not audit['adam_state_loaded']):
                raise ValueError('Saved model/Adam audit is incomplete')
            checkpoints.append(checkpoint)
        if checkpoints[-1]['pin'] != row['checkpoint']:
            raise ValueError('Segment checkpoint does not match its audited terminal node')
        counters = row['counters']
        counts['updates'] += counters['completed_updates']
        counts['transitions'] += counters['training_transitions']
        counts['optimizer_steps_actor'] += counters['actor_optimizer_steps']
        counts['optimizer_steps_critic'] += counters['critic_optimizer_steps']
        predecessor = row['checkpoint']
        rows.append(dict(receipt=row['receipt'], probability=summary['probability_replay'],
                         native_exit=bound['native_exit_code'], native_error=bound['native_error'],
                         measurement_frames=sum(stream.get('rows',0) or 0 for stream in report['streams']
                             if stream['logical_name'].endswith('/measurement.bin'))))
    if (predecessor != result['end'] or result['start'] != design['initial']['checkpoint']
            or counts['updates'] != recipe['branch_updates'] or counts['transitions'] != counts['updates']*512
            or counts['optimizer_steps_actor'] != counts['updates']*8
            or counts['optimizer_steps_critic'] != counts['updates']*8):
        raise ValueError('Frozen training volume, optimizer clock or endpoint differs')
    return dict(start=result['start'], end=result['end'], counts=counts, rng=rng, checkpoints=checkpoints,
                recipe=design['recipe'], source=design['source'], decision=design['decision'], segments=rows,
                supervised_seconds=result['training_supervised_seconds'])


def paired_r6(trial, reference, tolerance, prior_gates):
    a, b = trial['tracking_rmse_mps'], reference['tracking_rmse_mps']
    differences = {k: a[k]-b[k] for k in sorted(a.keys() & b.keys())}
    missing = sorted(a.keys() ^ b.keys())
    worst = max(differences, key=differences.get, default=None)
    worst_delta = differences.get(worst)
    legacy = compare(prior_gates,
        [dict(trial, tracking_rmse_mps={k: a[k] for k in differences})],
        [dict(reference, tracking_rmse_mps={k: b[k] for k in differences})])
    return dict(deltas_mps=differences, missing=missing, worst_block=worst, worst_delta_mps=worst_delta,
                legacy_r6_gates=legacy, passed=legacy['guardrails_pass'] and not missing
                and worst_delta is not None and worst_delta<=tolerance)


def audit(evaluations):
    assembly = read(evaluations)
    assignments = read(assembly['assignments'])
    protocol = read(assignments['protocol'])
    arms, training, results = {}, [], []
    for arm in assignments['arm_order']:
        evaluation = read(assembly['r6'][arm])
        bound, ctx, report = validate_binding(evaluation['binding'])
        params = ctx['contract']['parameters']
        counts = ctx['state']['counters']
        if (params['r6']['seed'] != protocol['r6']['perturbation_seed']
                or params['r6']['split'] != protocol['r6']['perturbation_split']
                or params['evaluation_seed'] != protocol['r6']['original_seed']
                or counts['nontraining_transitions'] != protocol['r6']['transitions_per_arm']):
            raise ValueError('R6 evaluation does not match the frozen protocol')
        model = score_model(dict(run_id=ctx['plan']['run_id'], arm=arm, pair=1, report=bound['report']))
        arms[arm] = dict(model, binding=evaluation['binding'], receipt=bound['receipt'],
                         wall_seconds=evaluation['wall_seconds'], transitions=counts['nontraining_transitions'])
    if arms['qualified']['checkpoint'] != assignments['qualified']:
        raise ValueError('Qualified policy differs')
    tolerance = protocol['r6']['criteria']['max_paired_tracking_rmse_increase_mps']
    prior_gates_pin = pin(HISTORICAL/'FREEZE.json')
    prior_gates = read(prior_gates_pin)
    for index, (round_pin, physical_pin) in enumerate(zip(assignments['formal'], assembly['physical'], strict=True), 1):
        closed = read(round_pin)
        turn = verify_training(closed)
        training.append(turn)
        evaluation = read(physical_pin)
        binding = evaluation['binding']
        bound, ctx, report = validate_binding(binding)
        params = ctx['contract']['parameters']
        models = dict(candidate=turn['end'], control=turn['start'], reference=assignments['qualified'])
        if (params['models'] != models or params['arm_options'] != dict(candidate=closed['execution'],
                control=closed['execution'], reference=None) or params['seed'] != protocol['physical']['seed']):
            raise ValueError('Physical pairing or protocol drifted')
        candidate, origin = arms[f'r{index}-end'], arms[f'r{index}-origin']
        if candidate['checkpoint'] != turn['end'] or origin['checkpoint'] != turn['start']:
            raise ValueError('R6 endpoint/origin pairing drifted')
        comparison = physical(binding)
        for name, metrics in comparison['arms'].items():
            if any(metrics['categories'][k]['denominator'] != protocol['physical']['denominators'][k]
                   for k in metrics['categories']):
                raise ValueError(f'{name} initial denominator drifted')
        gates = compare_holdout(binding)
        results.append(dict(round=index, evaluation=physical_pin, evidence=binding, receipt=bound['receipt'],
            wall_seconds=evaluation['wall_seconds'], transitions=ctx['state']['counters']['nontraining_transitions'],
            physical=comparison, physical_gates=gates['engineering_gates'],
            r6_qualified=paired_r6(candidate, arms['qualified'], tolerance, prior_gates),
            r6_origin=paired_r6(candidate, origin, tolerance, prior_gates)))
    ledger = read(assembly['ledger'])
    receipts = []
    for reservation in ledger['new_reservations']:
        if reservation['id'] == 'ROUND-SYSTEM':
            continue
        item = read(reservation['receipt'])
        owners = [item['parent'], *item['owned_processes']]
        if item['status'] != 'CLOSED' or item['remaining'] or any(process_identity(p['pid']) == p for p in owners):
            raise ValueError('Owned runtime process is not closed')
        receipts.append(dict(run=reservation['id'], receipt=reservation['receipt'], native_exit=item['exit_code'],
                             error=item['error'], bytes=reservation.get('closed_evidence_bytes'), wall_seconds=item['wall_seconds']))
    return dict(evaluations=evaluations, protocol=assignments['protocol'], prior_r6_gates=prior_gates_pin,
                training=training, r6=arms,
                physical=results, receipts=receipts, all_processes_closed=True)


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
