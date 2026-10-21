"""Read-only pre-baseline capacity reconciliation, never an admission receipt."""
import json
from pathlib import Path
from tools.r6_budget import charged, totals
from tools.r6_recovery import file_pin
from rlflow.store import sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'docs/experiments/r6-robustness-auto'
DISK = 'output_actual_plus_unreleased_reservations_bytes'
MIB = 1024**2
# Exact second snapshot consumes the already planned allowance; no refund.
NEW_BASELINE_SNAPSHOT_EVENT = '4cd9a0dfd5957c2b6521621fe928ef48bc640374b63018e86796c4f1cb96a246'


def reconcile(state, draft, allocation_key):
    if any(e['kind'] == 'job' for e in state['events'].values()):
        raise ValueError('PRE_BASELINE_ONLY')
    planned = dict(draft['development_components_bytes'])
    planned['three_result_filesystem_allowances'] = 3*96*MIB
    # Bundle the exact already-reserved baseline image/control/FS charge.
    planned['initial_full_baseline'] += 160*MIB
    planned['three_result_volume_control_allowances'] -= 64*MIB
    planned['three_result_filesystem_allowances'] -= 96*MIB
    role_buckets = {'source-snapshot':'historical_snapshot_reservation',
        'native-volume-provision':'consumed_cpu_volume_and_jit_reservations',
        'cpu-native-jit':'consumed_cpu_volume_and_jit_reservations',
        'result-volume-provision':'result_volume_cpu02_reservation'}
    actual = {k:0 for k in planned}; rows=[]
    actual['unclassified_used_disk'] = state['budget']['used'][DISK]
    for reservation in state['budget']['reservations']:
        key=sha(reservation['identity']);event=state['events'].get(key)
        if event is None or event['identity'] != reservation['identity']:
            raise ValueError('RESERVATION_EVENT_MISSING')
        size=charged(reservation,state['budget']['caps']).get(DISK,0)
        if key==NEW_BASELINE_SNAPSHOT_EVENT and event['identity']['role']=='source-snapshot':
            bucket='new_baseline_snapshot'
        elif event['kind']=='model':bucket='controller_and_reviews'
        elif key==allocation_key and event['identity']['role']=='job-storage-allocation':
            bucket='initial_full_baseline'
        else:bucket=role_buckets.get(event['identity']['role'],'unclassified_reservations')
        actual[bucket]=actual.get(bucket,0)+size
        rows.append(dict(event_key=key,bucket=bucket,charged_bytes=size))
    accounted=sum(actual.values())
    if accounted != totals(state['budget'])[DISK]:raise ValueError('DISK_RECONCILIATION')
    components={k:dict(planned_bytes=planned.get(k,0),charged_bytes=actual.get(k,0),
        remaining_planned_bytes=max(0,planned.get(k,0)-actual.get(k,0)),
        projected_bytes=max(planned.get(k,0),actual.get(k,0))) for k in sorted(set(planned)|set(actual))}
    total=sum(v['projected_bytes'] for v in components.values())
    final=dict(draft['final_components_bytes']);final['seven_result_filesystem_allowances']=7*96*MIB
    final_total=sum(final.values());final_protection=((final_total+1024**3-1)//1024**3)*1024**3
    dev_limit=state['budget']['caps'][DISK]-final_protection
    return dict(status='ARITHMETIC_ONLY_NOT_ACCEPTED',capacity_pass=False,components=components,
        reservation_mapping=rows,current_charged_bytes=accounted,
        future_planned_bytes=sum(v['remaining_planned_bytes'] for v in components.values()),
        development_projected_bytes=total,development_limit_bytes=dev_limit,
        development_headroom_bytes=dev_limit-total,final_components=final,
        final_projected_bytes=final_total,proposed_final_protection_bytes=final_protection,
        existing_ledger_final_protection_bytes=state['budget']['protected_confirmation_delivery_repair'][DISK],
        arithmetic_fits=total<=dev_limit,ledger_changed=False,
        limitations=['Fixed-layout capacity remains unmeasured; no production admission.',
            'Training dual-volume wiring and accepted time allocation remain incomplete.',
            'All existing model charges, including repair, conservatively remain in development controller pool.',
            'No implicit refund: each bucket charges max(plan, actual plus unreleased reservation).'])


def run():
    paths=[BASE/'batch/state.json',BASE/'CAPACITY_DUAL_VOLUME_DRAFT.json',
           BASE/'production-storage-baseline-01/REGISTRATION.json']
    state,draft,registration=[json.loads(p.read_text()) for p in paths]
    report=reconcile(state,draft,registration['event_key'])
    report['inputs']=[file_pin(p) for p in paths]
    report['tool']=file_pin(Path(__file__).resolve())
    return report


if __name__=='__main__':print(json.dumps(run(),indent=2))
