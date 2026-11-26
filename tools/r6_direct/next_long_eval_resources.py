"""Size the next long-run R6 replays from closed, comparable local evidence."""

import shutil
from pathlib import Path

from tools.r6_direct.evidence import read
from tools.r6_direct.import_legacy import ROOT, process_identity
from tools.r6_direct.storage import CONTROL_ALLOWANCE, SYSTEM_FLOOR, SYSTEM_ROOT, evidence_bytes
from tools.r6_evidence import file_pin
from training_workloads.unified_h1.direct_contract import PROTOCOL_SHA


MIB = 2**20
R6_OUTPUT_CAP = 8 * 2**30
QUOTA_ROUNDING = 8 * MIB
ARCHIVES = {'range_bytes': 'measurements.bin', 'audit_bytes': 'audit.bin'}
HISTORICAL = (
    ('instinct-r4-recovery', 'evaluations', ('candidate', 'origin', 'qualified')),
    ('instinct-post-r4-three', 'r6', ('qualified', 'r1-origin', 'r1-end',
                                     'r2-origin', 'r2-end', 'r3-origin', 'r3-end')),
)


def _measure_arm(arm, reservations):
    binding = read(arm['binding'])
    receipt = read(binding['receipt'])
    state = read(binding['workload_state'])
    plan = read(binding['run_inputs'])
    contract = read(plan['contract'])
    report = read(binding['report'])
    run_id = plan['run_id']
    root = SYSTEM_ROOT / run_id
    reservation = reservations[run_id]
    params = contract['parameters']
    r6 = params['r6']
    block_ids = r6['block_ids']
    owners = [receipt['parent'], *receipt['owned_processes']]
    if (binding['receipt'] != reservation['receipt'] or receipt['status'] != 'CLOSED'
            or receipt['remaining'] or any(process_identity(owner['pid']) == owner for owner in owners)
            or state['identity'] not in receipt['owned_processes']
            or state['status'] not in ('COMPLETE', 'PASS_PENDING_CLOSE') or state['error'] is not None
            or state['counters']['nontraining_transitions'] != 193600
            or arm['transitions'] != 193600 or Path(plan['attempt_dir']) != root
            or root.resolve() != root or params['experiment_id'] != run_id
            or contract['operation'] != 'evaluate' or r6['protocol']['sha256'] != PROTOCOL_SHA
            or report['run_id'] != run_id or report['run_inputs'] != binding['run_inputs']
            or not report['protocol_frozen'] or len(block_ids) != len(set(block_ids)) or len(block_ids) != 16
            or [row['block_id'] for row in report['blocks']] != block_ids
            or not all(row['replay_complete'] and row['saved_result_equal'] for row in report['blocks'])
            or not report['original']['saved_result_equal'] or not report['original']['saved_wall_equal']):
        raise ValueError(f'Historical R6 sizing run is incomplete or incomparable: {run_id}')

    by_path = {entry['path']: entry for entry in report['inputs']}
    files = {}
    for block_id in block_ids:
        folder = root / 'results' / contract['mode'] / 'r6' / block_id
        files[block_id] = {}
        for quota_field, filename in ARCHIVES.items():
            path = folder / filename
            entry = by_path[str(path)]
            if (path.resolve() != path or not path.is_file()
                    or not 0 < path.stat().st_size == entry['bytes'] <= r6['output_quotas'][block_id][quota_field]):
                raise ValueError(f'Historical R6 sizing file differs from its bound inventory: {path}')
            # Use the already bound digest and check bytes without rereading payloads.
            files[block_id][quota_field] = entry
    return block_ids, dict(run_id=run_id, root=str(root), binding=arm['binding'],
                           receipt=binding['receipt'], report=binding['report'], contract=plan['contract'],
                           closed_evidence_bytes=reservation['closed_evidence_bytes'],
                           retained_bytes=evidence_bytes(root), files=files)


def r6_output_plan():
    """Return an auditable per-block quota proposal with no changes to R6 scoring."""
    sources = []
    runs = []
    block_ids = None
    for experiment, field, roles in HISTORICAL:
        docs = ROOT / 'docs/experiments' / experiment
        audit_pin = file_pin(docs / 'COMPLETION_AUDIT.json')
        ledger_pin = file_pin(docs / 'RESOURCE_LEDGER.json')
        audit = read(audit_pin)
        ledger = read(ledger_pin)
        sources.append(dict(completion_audit=audit_pin, ledger=ledger_pin))
        reservations = {row['id']: row for row in ledger['new_reservations']}
        for role in roles:
            ids, run = _measure_arm(audit[field][role], reservations)
            if block_ids is None:
                block_ids = ids
            if ids != block_ids or any(prior['run_id'] == run['run_id'] for prior in runs):
                raise ValueError('Historical R6 sizing arms differ in suite order or repeat a run')
            runs.append(run)

    maxima = {block_id: {field: max(run['files'][block_id][field]['bytes'] for run in runs)
                         for field in ARCHIVES} for block_id in block_ids}
    denominator = 2 * QUOTA_ROUNDING
    quotas = {block_id: {field: ((3 * size + denominator - 1) // denominator) * QUOTA_ROUNDING
                         for field, size in measurements.items()} for block_id, measurements in maxima.items()}
    output_quota_bytes = sum(sum(block.values()) for block in quotas.values())
    # Match direct_contract.validate_baseline's TOTAL_OUTPUT check.
    original_suite_allowance = 64 * MIB
    per_block_allowance = 16 * MIB
    contract_minimum = original_suite_allowance + len(block_ids) * per_block_allowance + output_quota_bytes
    if contract_minimum > R6_OUTPUT_CAP or any(value > 2 * 2**30 for block in quotas.values() for value in block.values()):
        raise ValueError('Historical R6 sizing envelope exceeds the registered per-arm cap')
    return dict(schema_version='instinctrl-next-long-r6-output-plan/1', historical_sources=sources,
                historical_runs=runs, historical_r6_retained_bytes_max=max(run['retained_bytes'] for run in runs),
                protocol_sha256=PROTOCOL_SHA, block_ids=block_ids, output_quotas=quotas,
                observed_file_maxima_bytes=maxima, quota_multiple_numerator=3, quota_multiple_denominator=2,
                quota_rounding_bytes=QUOTA_ROUNDING, output_quota_bytes=output_quota_bytes,
                original_suite_allowance_bytes=original_suite_allowance, per_block_allowance_bytes=per_block_allowance,
                contract_minimum_bytes=contract_minimum, output_cap_bytes=R6_OUTPUT_CAP,
                control_allowance_bytes=CONTROL_ALLOWANCE, launch_cap_bytes=R6_OUTPUT_CAP + CONTROL_ALLOWANCE)


def forecast(output_plan, *, remaining_phases, remaining_r6_arms,
             expected_r6_retained_bytes=None, other_bytes=0):
    """Measure free space now, then forecast the largest remaining serial admission.

    Supply non-R6 phases in their remaining launch order. The phase launch_cap_bytes
    is its result payload cap, before the supervisor's control allowance. The
    retained estimate includes its whole directory after closure. Existing outputs
    are already reflected in the current free-space measurement.
    """
    if (type(remaining_r6_arms) is not int or remaining_r6_arms < 0
            or type(other_bytes) is not int or other_bytes < 0
            or (expected_r6_retained_bytes is not None
                and (type(expected_r6_retained_bytes) is not int or expected_r6_retained_bytes < 0))):
        raise ValueError('Invalid remaining R6 forecast')
    r6_retained = max(output_plan['historical_r6_retained_bytes_max'], expected_r6_retained_bytes or 0)
    phases = [*remaining_phases, *(dict(name=f'R6-{ordinal}', launch_cap_bytes=output_plan['output_cap_bytes'],
                                      retained_bytes=r6_retained) for ordinal in range(1, remaining_r6_arms + 1))]
    retained = other_bytes
    peak = retained
    forecast_rows = []
    for phase in phases:
        cap, estimate = phase['launch_cap_bytes'], phase['retained_bytes']
        if type(cap) is not int or cap <= 0 or type(estimate) is not int or not 0 <= estimate <= cap + CONTROL_ALLOWANCE:
            raise ValueError(f"Invalid launch cap or retained estimate for {phase['name']}")
        admission = retained + cap + CONTROL_ALLOWANCE
        forecast_rows.append(dict(name=phase['name'], payload_cap_bytes=cap, retained_bytes=estimate,
                                  cumulative_before_launch_bytes=retained,
                                  additional_free_required_at_launch_bytes=admission))
        peak = max(peak, admission)
        retained += estimate

    free = shutil.disk_usage(SYSTEM_ROOT).free
    required = SYSTEM_FLOOR + max(peak, retained)
    return dict(schema_version='instinctrl-next-long-storage-forecast/1', measured_directory=str(SYSTEM_ROOT),
                measured_free_bytes=free, system_minimum_free_bytes=SYSTEM_FLOOR, other_bytes=other_bytes,
                control_allowance_bytes=CONTROL_ALLOWANCE, required_free_bytes=required,
                expected_new_retained_bytes=retained, headroom_bytes=free - required,
                fits=free >= required, phases=forecast_rows)
