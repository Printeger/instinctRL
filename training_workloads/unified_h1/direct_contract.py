"""Local-direct scientific run contract. No Store or legacy approval objects.

Baseline reuses the exact frozen development protocol, its denominators and
original suite. Source/asset and process/resource admission are separate checks
performed by the direct entry before constructing the simulator.
"""
import copy
import re
from .robustness_contract import baseline, pinned_json, budget_from_parameters
from .robustness_prelaunch import POLICY

PROTOCOL_SHA = '9a26583bb53930f78237244350e9cc9bd33b8657ad631ab983d7ca60243a3822'
PARAMETERS = {'experiment_id','group','training_seed','evaluation_seed',
              'baseline_bindings','r6','wall_seconds','output_cap_bytes'}
R6_FIELDS = {'protocol','block_ids','run_id','seed','split','expected_mass_kg','output_quotas'}


def require(ok, reason):
    if not ok: raise ValueError('R6_DIRECT_' + reason)


def validate_baseline(parameters):
    require(isinstance(parameters,dict) and set(parameters)==PARAMETERS,'BASELINE_FIELDS')
    p=copy.deepcopy(parameters)
    require(isinstance(p['experiment_id'],str) and re.fullmatch(r'R6-DIRECT-[A-Z0-9-]{1,80}',p['experiment_id']), 'IDENTITY')
    require(p['group']=='R1' and type(p['training_seed']) is int and p['training_seed']==20260908
            and type(p['evaluation_seed']) is int and p['evaluation_seed']==20260909,'ORIGINAL_SEEDS')
    require(type(p['wall_seconds']) is int and 1<=p['wall_seconds']<=3600,'WALL')
    require(type(p['output_cap_bytes']) is int and 0<p['output_cap_bytes']<=20250099712,'OUTPUT_CAP')
    baseline(p)  # Rehashes actual frozen u640 and checks ICS .15 and baseline identity.
    r=p['r6']; require(isinstance(r,dict) and set(r)==R6_FIELDS,'R6_FIELDS')
    require(r['protocol']['sha256']==PROTOCOL_SHA,'PROTOCOL_IDENTITY')
    protocol=pinned_json(r['protocol'])
    require(protocol['frozen'] is True and protocol['resource_admission']==POLICY,'PROTOCOL_RESOURCE_POLICY')
    require(protocol['old_suite']['seed']==20260909 and protocol['old_suite']['transitions']==32960,'OLD_SUITE')
    require(r['split']=='development' and type(r['seed']) is int and r['seed']==2026101901,'DEVELOPMENT_STREAM')
    require(r['run_id']==p['experiment_id'],'RUN_ID')
    require(type(r['expected_mass_kg']) in (int,float) and abs(r['expected_mass_kg']-1.181517923)<1e-8,'MASS')
    blocks=protocol['blocks']; ids=[b['id'] for b in blocks]
    require(len(ids)==16 and len(set(ids))==16 and r['block_ids']==ids,'ALL_BLOCKS_AND_ORDER')
    require(isinstance(r['output_quotas'],dict) and set(r['output_quotas'])==set(ids),'QUOTA_BLOCKS')
    needed=64*1024**2
    for block in blocks:
        require(block['lanes']==32 and block['steps']==(200 if block['task']=='reverse' else 330),'DENOMINATOR')
        quota=r['output_quotas'][block['id']]
        require(isinstance(quota,dict) and set(quota)=={'range_bytes','audit_bytes'},'QUOTA_FIELDS')
        require(all(type(v) is int and 0<v<=2*1024**3 for v in quota.values()),'QUOTA_LIMIT')
        needed+=sum(quota.values())+16*1024**2
    require(needed<=p['output_cap_bytes'],'TOTAL_OUTPUT')
    return p


def baseline_runtime_contract(parameters):
    p=validate_baseline(parameters)
    return dict(schema_version='r6-local-runtime-contract/1',execution_mode='local-direct',
                operation='baseline',mode='research-r6-evaluation',parameters=p,
                budgets=budget_from_parameters(p))


def validate_runtime_contract(value):
    require(isinstance(value,dict) and set(value)=={'schema_version','execution_mode','operation','mode','parameters','budgets'},'CONTRACT_FIELDS')
    require(value['schema_version']=='r6-local-runtime-contract/1' and value['execution_mode']=='local-direct','MODE')
    require(value['operation']=='baseline','OPERATION_NOT_IMPLEMENTED')
    expected=baseline_runtime_contract(value['parameters'])
    require(value==expected,'EXACT_CONTRACT_BUDGET')
    return expected
