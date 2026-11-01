"""Bind direct-run evidence without Store, jobs, or invented authorizations."""
import json
from pathlib import Path

from tools.r6_evidence import checked, file_pin
from tools.r6_direct.import_legacy import process_identity


def require(ok, reason):
    if not ok:
        raise ValueError('DIRECT_EVIDENCE_' + reason)


def read(pin):
    return json.loads(checked(pin).read_text())


def context(input_pin):
    from training_workloads.unified_h1.direct_contract import validate_runtime_contract
    plan = read(input_pin)
    contract = validate_runtime_contract(read(plan['contract']))
    require(plan['run_id'] == contract['parameters']['experiment_id'], 'RUN_ID')
    attempt = Path(plan['attempt_dir'])
    root = Path(plan['output_root'])
    require(all(p.is_absolute() and p.resolve() == p and not p.is_symlink()
                for p in (attempt, root)), 'PATH_ALIAS')
    pins = {name: file_pin(root / filename) for name, filename in
            (('receipt', 'receipt.json'), ('launch', 'launch.json'), ('run_record', 'RUN_INPUTS.json'))}
    receipt, launch, record = (read(pins[k]) for k in ('receipt', 'launch', 'run_record'))
    require(record['plan'] == plan and record['input'] == input_pin
            and record['execution_mode'] == 'local-direct'
            and launch['input'] == input_pin and launch['parent'] == receipt['parent'], 'RUN_BINDING')
    require(receipt['status'] == 'CLOSED' and receipt['remaining'] == [], 'NOT_CLOSED')
    for owner in [receipt['parent'], *receipt['owned_processes']]:
        require(process_identity(owner['pid']) != owner, 'PROCESS_STILL_LIVE')
    pins.update(workload_state=file_pin(attempt / 'state.json'),
                modules=file_pin(attempt / 'CPU_MODULES.json'))
    state, modules = read(pins['workload_state']), read(pins['modules'])
    require(state['identity'] in receipt['owned_processes'], 'WORKLOAD_IDENTITY')
    require(modules['execution_mode'] == 'local-direct' and modules['forbidden_modules'] == []
            and modules['forbidden_pythonpath'] == []
            and not any(n == 'rlflow' or n.startswith('rlflow.') for n in modules['loaded_modules']), 'DEPENDENCIES')
    return dict(plan=plan, contract=contract, output=attempt / 'results' / contract['mode'],
                pins=pins, receipt=receipt, state=state)


def verify_inputs(report):
    seen = set()
    for item in report['inputs']:
        pin = {k: item[k] for k in ('path', 'sha256')}
        path = checked(pin)
        require(str(path) not in seen and path.stat().st_size == item['bytes'], 'REPORT_INPUT')
        seen.add(str(path))
    return seen


def bind(input_pin, report_pin):
    ctx = context(input_pin)
    report = read(report_pin)
    require(report.get('run_inputs') == input_pin and 'authorization' not in report
            and report['scientific_acceptance'] is False, 'REPORT_BINDING')
    seen = verify_inputs(report)
    require({input_pin['path'], ctx['plan']['contract']['path'],
             *(p['path'] for p in ctx['pins'].values())} <= seen, 'LIFECYCLE_INPUTS')
    return dict(schema_version='r6-direct-bound-analysis/1', run_inputs=input_pin,
                report=report_pin, **ctx['pins'], native_exit_code=ctx['receipt']['exit_code'],
                native_error=ctx['receipt']['error'], scientific_acceptance=False)


def validate_binding(pin):
    bound = read(pin)
    require(bound == bind(bound['run_inputs'], bound['report']), 'BINDING_DRIFT')
    ctx = context(bound['run_inputs'])
    # Cleanup failure is retained, never silently converted to a clean exit.
    receipt = ctx['receipt']
    require((receipt['exit_code'] == 0 and receipt['error'] is None)
            or (receipt['error'] == "TimeoutError('DIRECT_CLEANUP_TIMEOUT')"
                and receipt['exit_code'] in (-15, -9)), 'NATIVE_FAILURE')
    require(ctx['state']['status'] in ('COMPLETE', 'PASS_PENDING_CLOSE')
            and ctx['state'].get('error') is None, 'WORKLOAD_FAILED')
    return bound, ctx, read(bound['report'])


def analysis_context(output, input_pin):
    ctx = context(input_pin)
    require(Path(output) == ctx['output'], 'EXACT_OUTPUT')
    return ctx


def add_inputs(report, input_pin, ctx):
    pins = [input_pin, ctx['plan']['contract'], *ctx['pins'].values()]
    indexed = {p['path']: p for p in report['inputs']}
    for pin in pins:
        path = checked(pin)
        indexed[pin['path']] = dict(pin, bytes=path.stat().st_size)
    report['inputs'] = sorted(indexed.values(), key=lambda p: p['path'])
    report['run_inputs'] = input_pin
    report.pop('authorization', None)
    verify_inputs(report)
