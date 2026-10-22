"""One-time time accounting for closed NEW runs; historical charges stay intact."""
import math
from tools.r6_direct.import_legacy import pin, process_identity
import json
from pathlib import Path


def settle_time(ledger, run_id):
    rows=[r for r in ledger['new_reservations'] if r['id']==run_id]
    if len(rows)!=1:raise ValueError('DIRECT_SETTLEMENT_ID')
    row=rows[0]
    if row['status']!='UNRELEASED' or 'settlement' in row:raise ValueError('DIRECT_ALREADY_SETTLED')
    evidence=row['receipt']
    if pin(evidence['path'])!=evidence:raise ValueError('DIRECT_RECEIPT_DRIFT')
    receipt=json.loads(Path(evidence['path']).read_text())
    if receipt['status']!='CLOSED' or receipt['remaining']:raise ValueError('DIRECT_SETTLEMENT_CLEANUP')
    for owner in [receipt['parent'],*receipt['owned_processes']]:
        if process_identity(owner['pid'])==owner:raise ValueError('DIRECT_SETTLEMENT_LIVE')
    seconds=receipt['wall_seconds']
    if type(seconds) not in (float,int) or not math.isfinite(seconds) or seconds<=0:raise ValueError('DIRECT_SETTLEMENT_TIME')
    # Baseline supervisor starts its clock before creating its only live child
    # and ends after complete cleanup. Diagnostics have a separate earlier gate
    # and are deliberately excluded from this narrower settlement.
    if not run_id.startswith('R6-DIRECT-BASELINE-'):raise ValueError('DIRECT_SETTLEMENT_SCOPE')
    original=dict(row['resources']); charged=math.ceil(seconds)
    if any(charged>original[k] for k in ('allocated_gpu_seconds','live_wall_seconds')):
        raise ValueError('DIRECT_SETTLEMENT_OVER_RESERVATION')
    row['settlement']=dict(original_resources=original,receipt=evidence,
        policy='Ceiling of entire supervised lifetime including gate and cleanup; all other resources remain reserved. No move to new_used or historical refund.',
        charged_seconds=charged)
    for key in ('allocated_gpu_seconds','live_wall_seconds'):row['resources'][key]=charged
    row['status']='TIME_SETTLED_OTHER_RESOURCES_UNRELEASED'
    return ledger
