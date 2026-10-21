import copy
import json
from tools.reconcile_r6_capacity import BASE, DISK, reconcile, run


def inputs():
    return [json.loads((BASE/p).read_text()) for p in ['batch/state.json','CAPACITY_DUAL_VOLUME_DRAFT.json','production-storage-baseline-01/REGISTRATION.json']]


def test_baseline_prepaid_charge_replaces_future_allowance():
    r=run();b=r['components']['initial_full_baseline']
    assert b['charged_bytes']==b['planned_bytes']==20250099712
    assert b['remaining_planned_bytes']==0
    assert r['current_charged_bytes']+r['future_planned_bytes']==r['development_projected_bytes']
    assert r['development_headroom_bytes']==598409216
    assert r['proposed_final_protection_bytes']==93*1024**3
    assert not r['capacity_pass'] and not r['ledger_changed']


def test_overrun_consumes_headroom_instead_of_refunding_or_disappearing():
    s,d,reg=inputs();before=copy.deepcopy(s)
    row=next(r for r in s['budget']['reservations'] if r['identity']==s['events'][reg['event_key']]['identity'])
    row['observations']=[dict(resources={DISK:row['resources'][DISK]+1024**3})]
    r=reconcile(s,d,reg['event_key'])
    assert r['development_headroom_bytes']==598409216-1024**3
    assert not r['arithmetic_fits']
    assert before['events']==s['events']


def test_read_only_reconciliation_preserves_ledger():
    p=BASE/'batch/state.json';before=p.read_bytes();run();assert p.read_bytes()==before
