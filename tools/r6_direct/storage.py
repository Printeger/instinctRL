"""Bind newly reserved volumes without inheriting old storage acceptance."""
import json
from pathlib import Path
from tools.r6_evidence import checked


def fresh_binding(plan, ledger, contract, *, third=False):
    identity=plan['storage_reservation_id']
    rows=[r for r in ledger['new_reservations'] if r['id']==identity]
    if len(rows)!=1:raise ValueError('DIRECT_STORAGE_RESERVATION')
    row=rows[0]
    if row['status']!='UNRELEASED':raise ValueError('DIRECT_STORAGE_RESERVATION_STATUS')
    spec=json.loads(checked(row['storage_plan']).read_text())
    reuse=spec.get('schema_version')=='instinctrl-third-storage-reuse/1'
    if reuse:
        from tools.r6_direct.resource_policy import OUTPUT_CAP, THIRD_PREFIX
        if (not third or not plan['run_id'].startswith(THIRD_PREFIX)
                or spec.get('new_volume_allocation_bytes')!=0
                or row['resources']!={OUTPUT_CAP:0}
                or any(type(plan.get(key)) is not int or plan[key]<=0
                       for key in ('result_payload_cap_bytes','native_minimum_free_bytes'))):
            raise ValueError('DIRECT_STORAGE_REUSE_SCOPE')
    total=0
    for kind,destination in (('native','native_dir'),('result','attempt_dir')):
        actual=json.loads(checked(plan[kind+'_volume']).read_text())
        expected=spec[kind]
        if reuse and expected['manifest']!=plan[kind+'_volume']:
            raise ValueError('DIRECT_STORAGE_REUSE_MANIFEST')
        for key in ('volume_id','target','image','maximum_backing_bytes','owner_uid','owner_gid'):
            if actual[key]!=expected[key]:raise ValueError('DIRECT_STORAGE_MANIFEST:'+key)
        if Path(plan[destination])!=Path(expected['target'])/plan['run_id']:
            raise ValueError('DIRECT_STORAGE_DESTINATION')
        total+=expected['maximum_backing_bytes']
    if not reuse and row['resources']['output_actual_plus_unreleased_reservations_bytes']<total:
        raise ValueError('DIRECT_STORAGE_UNACCOUNTED')
    # Control metadata allowance is distinct from the two backing images.
    if contract['parameters']['output_cap_bytes']!=total+64*1024**2:
        raise ValueError('DIRECT_STORAGE_CAP')
    return identity
