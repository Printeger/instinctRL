"""Freeze the inputs and handoffs for the first bounded instinctRL cycle."""

import argparse
import json
from pathlib import Path

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.import_legacy import ROOT, pin, stop_paths
from tools.r6_direct.prepare import build
from tools.r6_direct.resource_policy import CYCLE_PREFIX, CYCLE_SCHEMA, THIRD_CAPS, OUTPUT_CAP
from tools.r6_direct.settle import settle_time
from tools.r6_direct.storage import SYSTEM_ROOT, SYSTEM_STORAGE_SCHEMA
from tools.r6_direct.io import atomic
from tools.r6_direct.lifecycle import writer
from tools.r6_evidence import canonical, persist_exact


FORMAL = ROOT / 'docs/experiments/instinct-first-cycle/formal'
HISTORICAL = ROOT / 'docs/experiments/third-training/formal'
TRAIN = CYCLE_PREFIX + 'TRAIN-01'
TRAIN_RECOVERY = TRAIN + '-R1'
EVAL = CYCLE_PREFIX + 'EVAL-01'
DIAG = CYCLE_PREFIX + 'PROBABILITY-DIAGNOSTIC'
LEDGER = FORMAL / 'RESOURCE_LEDGER.json'
TRAIN_TEMPLATE = HISTORICAL / 'R6-DIRECT-THIRD-FORMAL-CONTROL-01-contract.json'
EVAL_TEMPLATE = HISTORICAL / 'R6-DIRECT-THIRD-EVAL-CONTROL-01-contract.json'
DIAG_TEMPLATE = HISTORICAL / 'R6-DIRECT-THIRD-FORMAL-CONTROL-01-DIAGNOSTIC-R1-inputs.json'


def load(path):
    return json.loads(path.read_bytes())


def save(path, value):
    if path.exists():
        if load(path) != value:
            raise ValueError(f'Existing frozen input differs: {path}')
    else:
        persist_exact(path, canonical(value) + b'\n')
    return pin(path)


def destination(run_id):
    root = SYSTEM_ROOT / run_id
    return dict(run_id=run_id, attempt_dir=str(root), output_root=str(root / 'control'),
                native_dir=str(root / 'native'), system_results=True, system_native=True)


def run_inputs(run_id, snapshot, contract, predecessor, *, source=None):
    initial = load(HISTORICAL / 'R6-DIRECT-THIRD-FORMAL-CONTROL-01-inputs.json')
    return save(FORMAL / (run_id + '-inputs.json'), dict(destination(run_id),
        schema_version='r6-direct-run-inputs/1', snapshot=str(snapshot),
        source=source or pin(FORMAL / 'source.json'), contract=contract,
        extension_inputs=initial['extension_inputs'], ipc_wrapper=initial['ipc_wrapper'],
        resource_policy=pin(FORMAL / 'RESOURCE_POLICY.json'), storage_reservation_id='CYCLE1-SYSTEM',
        result_payload_cap_bytes=read(contract)['parameters']['output_cap_bytes'],
        predecessor_receipt=predecessor))


def prepare_training():
    # Must run inside the clean detached execution tree.
    from tools.r6_direct.source import ROOT as snapshot, manifest, training_identity
    if snapshot == ROOT:
        raise ValueError('Prepare from the detached cycle snapshot')
    source = manifest()
    save(FORMAL / 'source.json', source)
    storage = save(FORMAL / 'STORAGE.json', dict(schema_version=SYSTEM_STORAGE_SCHEMA,
                   root=str(SYSTEM_ROOT), new_volume_allocation_bytes=0))
    caps = dict(allocated_gpu_seconds=10000, live_wall_seconds=10000, queue_seconds=540,
                live_starts=3, simulator_initializations=2, training_transitions=65536,
                nontraining_transitions=193600, training_checkpoint_writes=5,
                major_scientific_candidates=1, **{OUTPUT_CAP: 32 * 1024**3})
    authority = save(FORMAL / 'RESOURCE_POLICY.json', dict(schema_version=CYCLE_SCHEMA,
                     user_authority=pin(FORMAL / 'AUTHORITY.md'), caps=caps, elapsed_seconds=18000))
    zero = dict.fromkeys(THIRD_CAPS, 0)
    save(LEDGER, dict(resource_policy=authority, caps=caps, historical_charged=zero,
                     new_used=zero, protected_confirmation_delivery_repair=zero, status='FROZEN',
                     new_reservations=[dict(id='CYCLE1-SYSTEM', status='UNRELEASED',
                                            resources={OUTPUT_CAP: 0}, storage_plan=storage)]))
    params = load(TRAIN_TEMPLATE)['parameters']
    plan = params['r6_training_plan']
    recipe = read(plan['recipe'])
    curriculum = read(recipe['curriculum'])
    curriculum['phases'][1].update(delay_cycles=1, drop_every_steps=24)
    identity = training_identity(source)
    recipe.update(candidate_id='R6-CANDIDATE-INSTINCT-CYCLE1', source_identity=identity,
                  diagnosis=[pin(FORMAL.parent / 'REPORT.md')],
                  hypothesis='Delay and drop exposure in the robust phase improves condition-specific recovery.',
                  comparison='One complete candidate against same-seed, same-update historical control and u640.',
                  curriculum=save(FORMAL / 'curriculum.json', curriculum))
    plan.update(experiment_id=TRAIN, recipe=save(FORMAL / 'recipe.json', recipe))
    params.update(experiment_id=TRAIN, source_identity=identity)
    contract = save(FORMAL / 'TRAIN-contract.json', build('contract', dict(operation='train', parameters=params)))
    predecessor = pin(SYSTEM_ROOT / 'R6-DIRECT-THIRD-EVAL-CANDIDATE-03/control/receipt.json')
    return run_inputs(TRAIN, snapshot, contract, predecessor)


def recover_training():
    from tools.r6_direct.source import ROOT as snapshot, manifest, training_identity
    if snapshot == ROOT:
        raise ValueError('Recover from the clean detached execution tree')
    original = load(FORMAL / (TRAIN + '-inputs.json'))
    attempt = Path(original['attempt_dir'])
    out = Path(original['output_root'])
    stderr = (out / 'stderr.log').read_text()
    if ('ValueError: R6 overlapping output roots' not in stderr
            or (attempt / 'state.json').exists() or (attempt / 'kit_settings.json').exists()
            or any((attempt / 'prelaunch').iterdir())
            or any((attempt / 'results/research-r6-training').iterdir())):
        raise ValueError('Recovery requires exact before-gate failure and no workload state')
    with writer():
        ledger = load(LEDGER)
        row, = (r for r in ledger['new_reservations'] if r['id'] == TRAIN)
        predecessor = row['receipt']
        closed = read(predecessor)
        if closed['exit_code'] != 1 or closed['error'] is not None:
            raise ValueError('Unexpected before-gate exit')
        if 'pre_app_abort' not in row:
            settle_time(ledger, TRAIN)
            # All actual supervised time and retained bytes stay charged. This
            # child never entered admission or recorded an AppLauncher start.
            row['pre_app_abort'] = dict(stderr=pin(out / 'stderr.log'),
                                        receipt=predecessor, reserved_before_recovery=dict(row['resources']),
                                        scientific_launches=0, training_transitions=0)
            for counter in ('live_starts', 'queue_seconds', 'simulator_initializations',
                            'training_transitions', 'training_checkpoint_writes', 'major_scientific_candidates'):
                row['resources'][counter] = 0
            atomic(LEDGER, ledger)
    source = manifest()
    source_pin = save(FORMAL / 'source-r1.json', source)
    identity = training_identity(source)
    params = read(original['contract'])['parameters']
    plan = params['r6_training_plan']
    recipe = read(plan['recipe'])
    recipe['source_identity'] = identity
    plan.update(experiment_id=TRAIN_RECOVERY, recipe=save(FORMAL / 'recipe-r1.json', recipe))
    params.update(experiment_id=TRAIN_RECOVERY, source_identity=identity)
    contract = save(FORMAL / 'TRAIN-r1-contract.json', build('contract', dict(operation='train', parameters=params)))
    return run_inputs(TRAIN_RECOVERY, snapshot, contract, predecessor, source=source_pin)


def active_training_inputs():
    recovery = FORMAL / (TRAIN_RECOVERY + '-inputs.json')
    return load(recovery if recovery.exists() else FORMAL / (TRAIN + '-inputs.json'))


def settle(run_id):
    with writer():
        ledger = load(LEDGER)
        row, = (r for r in ledger['new_reservations'] if r['id'] == run_id)
        if 'settlement' not in row:
            settle_time(ledger, run_id)
            ledger['status'] = 'WAITING_FOR_REVIEW' if run_id == EVAL else 'CONTROLLED_HANDOFF'
            atomic(LEDGER, ledger)
        return row['receipt']


def prepare_diagnostic():
    inputs = active_training_inputs()
    report = load(Path(inputs['attempt_dir']) / 'analysis-cpu/report.json')
    retained, = (r['pin'] for r in report['streams']
                 if r['logical_name'] == 'pilot_diagnostic/update_641_full_input.pt')
    spec = load(DIAG_TEMPLATE)
    del spec['native_volume']
    spec.update(destination(DIAG), resource_policy=inputs['resource_policy'],
                launcher_script=pin(Path(inputs['snapshot']) / 'tools/r6_direct/cuda_diagnostic.py'),
                predecessor_receipt=settle(inputs['run_id']), probability_replay=dict(
                    backend='isaaclab_seeded_reset', contract=inputs['contract'], retained=retained))
    return save(FORMAL / (DIAG + '-inputs.json'), spec)


def seal_diagnostic():
    path = FORMAL / (DIAG + '-inputs.json')
    spec = load(path)
    receipt = settle(DIAG)
    out = Path(spec['output_root'])
    result = pin(Path(spec['native_dir']) / 'probability_result.json')
    if read(receipt)['exit_code'] != 0 or read(receipt)['error'] is not None:
        raise ValueError('Device probability verification did not complete')
    if pin(Path(spec['attempt_dir']) / 'probability_result.json')['sha256'] != result['sha256']:
        raise ValueError('Device probability result mirror differs')
    origin = read(active_training_inputs()['contract'])['parameters']['r6_training_plan']['initialization']['checkpoint']
    return save(Path(spec['attempt_dir']) / 'sealed-evidence.json', dict(
        schema_version='r6-probability-sealed-evidence/1', plan=pin(path),
        run_inputs=pin(out / 'RUN_INPUTS.json'), intent=pin(out / 'intent.json'),
        receipt=receipt, result=result, source_trace=spec['source_trace'], origin=origin))


def prepare_evaluation():
    train_inputs = active_training_inputs()
    attempt = Path(train_inputs['attempt_dir'])
    analysis = attempt / 'analysis/binding.json'
    if not analysis.exists():
        analysis = attempt / 'analysis-cpu/binding.json'
    analysis = pin(analysis)
    _, ctx, report = validate_binding(analysis)
    terminal, = (r['pin'] for r in report['summary']['checkpoints'] if r['update'] == 768)
    plan = ctx['contract']['parameters']['r6_training_plan']
    params = load(EVAL_TEMPLATE)['parameters']
    selection = params['evaluation_selection']
    comparison = read(selection['comparison'])
    comparison.update(id='INSTINCT-CYCLE1-PAIRED-REGRESSION', candidate_id='R6-CANDIDATE-INSTINCT-CYCLE1',
                      hypothesis=read(plan['recipe'])['hypothesis'])
    selection.update(checkpoint=terminal, training_plan=plan, training_analysis=analysis,
                     comparison=save(FORMAL / 'EVAL-comparison.json', comparison))
    params['experiment_id'] = EVAL
    params['r6']['run_id'] = EVAL
    contract = save(FORMAL / 'EVAL-contract.json', build('contract', dict(operation='evaluate', parameters=params)))
    predecessor = settle(DIAG if (FORMAL / (DIAG + '-inputs.json')).exists() else train_inputs['run_id'])
    return run_inputs(EVAL, Path(train_inputs['snapshot']), contract, predecessor, source=train_inputs['source'])


def main():
    actions = dict(train=prepare_training, recover=recover_training, diagnostic=prepare_diagnostic, seal=seal_diagnostic,
                   evaluate=prepare_evaluation, close=lambda: settle(EVAL))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=actions)
    if paths := stop_paths():
        raise RuntimeError(f'STOP {paths}')
    print(json.dumps(actions[parser.parse_args().step](), indent=2))


if __name__ == '__main__':
    main()
