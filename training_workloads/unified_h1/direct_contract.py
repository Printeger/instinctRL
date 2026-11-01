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


def validate_baseline(parameters, *, selected=None, final=False):
    require(isinstance(parameters,dict) and set(parameters)==PARAMETERS,'BASELINE_FIELDS')
    p=copy.deepcopy(parameters)
    require(isinstance(p['experiment_id'],str) and re.fullmatch(r'R6-DIRECT-[A-Z0-9-]{1,80}',p['experiment_id']), 'IDENTITY')
    expected_seed=20260908 if selected is None else selected['model']['training_seed']
    require(p['group']=='R1' and type(p['training_seed']) is int and p['training_seed']==expected_seed
            and type(p['evaluation_seed']) is int and p['evaluation_seed']==20260909,'ORIGINAL_SEEDS')
    require(type(p['wall_seconds']) is int and 1<=p['wall_seconds']<=3600,'WALL')
    require(type(p['output_cap_bytes']) is int and 0<p['output_cap_bytes'],'OUTPUT_CAP')
    baseline(p)  # Rehashes actual frozen u640 and checks ICS .15 and baseline identity.
    r=p['r6']; require(isinstance(r,dict) and set(r)==R6_FIELDS,'R6_FIELDS')
    require(r['protocol']['sha256']==PROTOCOL_SHA,'PROTOCOL_IDENTITY')
    protocol=pinned_json(r['protocol'])
    require(protocol['frozen'] is True and protocol['resource_admission']==POLICY,'PROTOCOL_RESOURCE_POLICY')
    require(protocol['old_suite']['seed']==20260909 and protocol['old_suite']['transitions']==32960,'OLD_SUITE')
    if selected is None:
        require(r['split']=='development' and type(r['seed']) is int and r['seed']==2026101901,'DEVELOPMENT_STREAM')
    else:
        require(p['baseline_bindings']==selected['selection']['baseline_bindings']
            and r['split']==('final' if final else 'development')
            and type(r['seed']) is int and 0<=r['seed']<2**31
            and all(r[k]==selected['comparison'][k] for k in ('protocol','seed','split','block_ids')),'EXACT_COMPARISON')
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


def training_runtime_contract(parameters):
    from . import robustness_training_plan as training
    from .checkpoint_compatibility import validate_source_identity
    fields={'experiment_id','group','training_seed','evaluation_seed','baseline_bindings',
            'r6_training_plan','r6_expected_mass_kg','output_cap_bytes','source_identity'}
    require(isinstance(parameters,dict) and set(parameters)==fields,'TRAINING_FIELDS')
    p=copy.deepcopy(parameters)
    plan=training.validate(p['r6_training_plan'])
    require(plan['schema_version']=='r6-local-training-plan/1','LOCAL_PLAN_REQUIRED')
    require(p['experiment_id']==plan['experiment_id'] and p['group']=='R1','TRAINING_IDENTITY')
    require(type(p['training_seed']) is int and p['training_seed']==plan['training_seed']
            and type(p['evaluation_seed']) is int and p['evaluation_seed']==20260909,'TRAINING_SEEDS')
    baseline(p)
    require(type(p['r6_expected_mass_kg']) in (int,float)
            and abs(p['r6_expected_mass_kg']-1.181517923)<1e-8,'MASS')
    require(type(p['output_cap_bytes']) is int
            and plan['storage']['payload_bytes']<=p['output_cap_bytes'],'OUTPUT_CAP')
    validate_source_identity(p['source_identity'])
    require(p['source_identity']==pinned_json(plan['recipe'])['source_identity'],'TRAINING_SOURCE')
    return dict(schema_version='r6-local-runtime-contract/1',execution_mode='local-direct',
                operation='train',mode='research-r6-training',parameters=p,
                budgets=training.required_budget(plan))


def training_inputs(backend, plan):
    """Select an explicit runtime contract without converting it to approval."""
    if getattr(backend,'execution_mode',None)=='local-direct':
        inputs=validate_runtime_contract(backend.runtime_contract)
        require(getattr(backend,'authorization',None) is None,'NO_LEGACY_AUTHORIZATION')
        require(plan['schema_version']=='r6-local-training-plan/1','LOCAL_PLAN_REQUIRED')
    else:
        inputs=backend.authorization
        require(plan['schema_version']=='r6-training-plan/1','LEGACY_PLAN_REQUIRED')
    from .robustness_training_plan import required_budget
    if not (inputs['mode']=='research-r6-training'
            and inputs['parameters'].get('r6_training_plan')==plan
            and inputs['budgets']==required_budget(plan)):
        raise PermissionError('R6_EXECUTION_EXACT_PLAN_BUDGET')
    return inputs


def evaluation_runtime_contract(parameters, *, final=False):
    from . import robustness_final_selection, robustness_evaluation_selection
    field='final_selection' if final else 'evaluation_selection'
    require(isinstance(parameters,dict) and set(parameters)==PARAMETERS|{field},'EVALUATION_FIELDS')
    value=parameters[field]
    require(value['schema_version']==('r6-local-final-selection/1' if final else 'r6-local-evaluation-selection/1'),
            'LOCAL_SELECTION_REQUIRED')
    module=robustness_final_selection if final else robustness_evaluation_selection
    selected=module.validate(value)
    p=validate_baseline({k:v for k,v in parameters.items() if k!=field},selected=selected,final=final)
    p[field]=copy.deepcopy(value)
    return dict(schema_version='r6-local-runtime-contract/1',execution_mode='local-direct',
                operation='final' if final else 'evaluate',
                mode='research-r6-final-evaluation' if final else 'research-r6-candidate-evaluation',
                parameters=p,budgets=budget_from_parameters(p))


def evaluation_inputs(backend, *, final=False):
    if getattr(backend,'execution_mode',None)=='local-direct':
        require(backend.authorization is None,'NO_LEGACY_AUTHORIZATION')
        contract=validate_runtime_contract(backend.runtime_contract)
        require(contract['operation']==('final' if final else 'evaluate'),'EVALUATION_OPERATION')
        return contract
    from . import robustness_final_contract, robustness_candidate_contract
    module=robustness_final_contract if final else robustness_candidate_contract
    auth=backend.authorization
    p=module.validate_parameters(auth['mode'],backend.research)
    require(auth['parameters']==p and auth['budgets']==module.required_budget(module.MODE,p),'EXACT_STAGE_BUDGET')
    return auth


def render_runtime_contract(parameters):
    from tools.r6_video_presentation import load
    fields={'experiment_id','extraction','wall_seconds','output_cap_bytes'}
    require(isinstance(parameters,dict) and set(parameters)==fields,'RENDER_FIELDS')
    p=copy.deepcopy(parameters)
    require(isinstance(p['experiment_id'],str) and re.fullmatch(r'R6-DIRECT-[A-Z0-9-]{1,80}',p['experiment_id']),'IDENTITY')
    require(type(p['wall_seconds']) is int and 1<=p['wall_seconds']<=3600,'WALL')
    require(type(p['output_cap_bytes']) is int and 1024**3<=p['output_cap_bytes'],'OUTPUT_CAP')
    load(p['extraction'])  # CPU rechecks every extracted trajectory and original pin.
    budgets=dict(live_process_starts=1,simulator_initializations=1,wall_seconds=p['wall_seconds'],
                 **{k:0 for k in ('training_transitions','nontraining_transitions','optimizer_updates',
                                  'checkpoint_writes','initialization_ticks','explicit_resets','automatic_resets')})
    return dict(schema_version='r6-local-runtime-contract/1',execution_mode='local-direct',
                operation='render',mode='research-r6-video-replay',parameters=p,budgets=budgets)


def validate_runtime_contract(value):
    require(isinstance(value,dict) and set(value)=={'schema_version','execution_mode','operation','mode','parameters','budgets'},'CONTRACT_FIELDS')
    require(value['schema_version']=='r6-local-runtime-contract/1' and value['execution_mode']=='local-direct','MODE')
    builders={'baseline':baseline_runtime_contract,'train':training_runtime_contract,
              'evaluate':evaluation_runtime_contract,'final':lambda p:evaluation_runtime_contract(p,final=True),
              'render':render_runtime_contract}
    require(value['operation'] in builders,'OPERATION_NOT_IMPLEMENTED')
    expected=builders[value['operation']](value['parameters'])
    require(value==expected,'EXACT_CONTRACT_BUDGET')
    return expected
