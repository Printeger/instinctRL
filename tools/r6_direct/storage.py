"""Bind newly reserved volumes without inheriting old storage acceptance."""
import json
from pathlib import Path
from tools.r6_evidence import checked


def fresh_binding(plan, ledger, contract):
    identity=plan['storage_reservation_id']
    rows=[r for r in ledger['new_reservations'] if r['id']==identity]
    if len(rows)!=1:raise ValueError('DIRECT_STORAGE_RESERVATION')
    row=rows[0]
    if row['status']!='UNRELEASED':raise ValueError('DIRECT_STORAGE_RESERVATION_STATUS')
    spec=json.loads(checked(row['storage_plan']).read_text())
    total=0
    for kind,destination in (('native','native_dir'),('result','attempt_dir')):
        actual=json.loads(checked(plan[kind+'_volume']).read_text())
        expected=spec[kind]
        for key in ('volume_id','target','image','maximum_backing_bytes','owner_uid','owner_gid'):
            if actual[key]!=expected[key]:raise ValueError('DIRECT_STORAGE_MANIFEST:'+key)
        if Path(plan[destination])!=Path(expected['target'])/plan['run_id']:
            raise ValueError('DIRECT_STORAGE_DESTINATION')
        total+=expected['maximum_backing_bytes']
    if row['resources']['output_actual_plus_unreleased_reservations_bytes']<total:
        raise ValueError('DIRECT_STORAGE_UNACCOUNTED')
    # Control metadata allowance is distinct from the two backing images.
    if contract['parameters']['output_cap_bytes']!=total+64*1024**2:
        raise ValueError('DIRECT_STORAGE_CAP')
    return identity
