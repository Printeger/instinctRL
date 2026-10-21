"""Exact prospective R6 training stage; no approval or job submission.

This is distinct from the fixed-policy baseline evaluation contract. Training
capacity and source/method compatibility need their own exact independent
reviews. Formal workload dispatch requires this contract; control-plane job issuance
still independently binds the exact attempt and resource reservations.
"""
from pathlib import Path
from .robustness_contract import baseline,pinned_json,accepted_review,object_sha,USER_SHA,REQUIREMENTS_SHA
from .research_contract import validate_pin
from .checkpoint_compatibility import validate_source_identity
from .robustness_training_plan import validate as validate_plan,required_budget as plan_budget

MODE='research-r6-training'
FIELDS={'experiment_id','group','training_seed','evaluation_seed','output_cap_bytes','master_authority',
    'experiment_card','baseline_bindings','checkpoint_compatibility','capacity_review',
    'r6_training_plan','r6_expected_mass_kg','native_volume','result_volume','extension_inputs'}


def require(condition,reason):
    if not condition:raise PermissionError('R6_TRAINING_STAGE_'+reason)


MANAGED={'experiment_card','native_volume','result_volume','output_cap_bytes'}


def validate_parameters(mode,value):
    return _validate(mode,value,materialized=True)


def validate_draft(mode,value):
    return _validate(mode,value,materialized=False)


def _validate(mode,value,*,materialized):
    require(mode==MODE and isinstance(value,dict)
        and set(value)==(FIELDS if materialized else FIELDS-MANAGED),'FIELDS')
    p=dict(value);plan=validate_plan(p['r6_training_plan'])
    require(p['experiment_id']==plan['experiment_id'] and p['group']=='R1','IDENTITY')
    require(type(p['training_seed'])is int and p['training_seed']==plan['training_seed']
        and type(p['evaluation_seed'])is int and p['evaluation_seed']==20260909,'SEED')
    require(type(p['r6_expected_mass_kg'])in (int,float)
        and abs(p['r6_expected_mass_kg']-1.181517923)<=1e-8,'PINNED_MASS')
    master=pinned_json(p['master_authority'])
    require(master.get('id')=='R6-ROBUSTNESS-AUTO-20261019'
        and master.get('status')=='REGISTERED_USER_DELEGATION','MASTER')
    for key,digest in (('user_authority',USER_SHA),('requirements',REQUIREMENTS_SHA)):
        validate_pin(master[key]);require(master[key]['sha256']==digest,'USER_AUTHORITY')
    b=baseline(p);initial=plan['initialization']
    if initial['kind']=='COMMON_U640_NEW_STREAM':
        require(initial['checkpoint']==b['selected_model'] and initial['source_identity']==b['source_identity']
            and initial['method_hashes']==b['method_hashes'],'COMMON_ORIGIN')
    from .native_volume import MAXIMUM,validate as validate_native
    from .result_volume import CONTROL_ALLOWANCE,workload_write_cap
    from .extension_inputs import validate as validate_extensions
    payload=plan['storage']['payload_bytes']
    aggregate=payload+96*1024**2+MAXIMUM+CONTROL_ALLOWANCE
    if materialized:
        require(type(p['output_cap_bytes'])is int and p['output_cap_bytes']==aggregate,'AGGREGATE_OUTPUT')
    capacity=pinned_json(p['capacity_review'])
    require(set(capacity)=={'schema_version','plan_sha256','payload_bytes','job_output_cap_bytes','budgets','independent_review'}
        and capacity['schema_version']=='r6-training-capacity/1'
        and capacity['plan_sha256']==object_sha(plan) and capacity['payload_bytes']==payload
        and capacity['job_output_cap_bytes']==aggregate and capacity['budgets']==plan_budget(plan),'CAPACITY')
    accepted_review(capacity['independent_review'],'PASS_R6_TRAINING_CAPACITY',
        payload_sha256=object_sha({k:v for k,v in capacity.items() if k!='independent_review'}))
    validate_extensions(p['extension_inputs'])
    if materialized:
        validate_native(p['native_volume'])
        require(workload_write_cap(p,minimum_free_bytes=payload)==payload,'PAYLOAD_VOLUME')
        card=pinned_json(p['experiment_card'])
        require(card.get('experiment_id')==p['experiment_id']
            and card.get('profiles',{}).get('R1')=={k:v for k,v in p.items() if k!='experiment_card'},'CARD')
    pinned_json(p['checkpoint_compatibility'])
    return p


def required_budget(mode,value):
    p=validate_parameters(mode,value)
    return plan_budget(p['r6_training_plan'])


def source_changes(old,new):
    return {k:dict(origin=old.get(k),execution=new.get(k)) for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}


def validate_compatibility(pin,plan,config):
    plan=validate_plan(plan);recipe=pinned_json(plan['recipe']);initial=plan['initialization']
    origin=validate_source_identity(initial['source_identity']);execution=validate_source_identity(config['resolved_identity'])
    require(execution==recipe['source_identity'],'RECIPE_EXECUTION')
    record=pinned_json(pin)
    fields={'schema_version','origin_source_identity','execution_source_identity','origin_config','execution_config',
        'recipe','checkpoint','method','changed_product_sources','changed_workload_sources','independent_review'}
    require(set(record)==fields and record['schema_version']=='r6-training-compatibility/1','COMPATIBILITY_FIELDS')
    require(record['origin_source_identity']==origin and record['execution_source_identity']==execution
        and record['recipe']==plan['recipe'] and record['checkpoint']==initial['checkpoint']
        and record['method']==recipe['method'],'COMPATIBILITY_BINDINGS')
    old=pinned_json(record['origin_config']);new=pinned_json(record['execution_config'])
    require(record['origin_config']['sha256']==origin['config_sha256']
        and record['execution_config']['sha256']==execution['config_sha256']
        and new=={k:v for k,v in config.items() if k!='resolved_identity'},'CONFIG_BINDINGS')
    for key in ('actor','training','execution','runtime','versions'):
        require(old[key]==new[key],'INVARIANT_'+key)
    # The underlying H1 method/config stays fixed. Explicit reviewed R6 method
    # settings are carried in the recipe, not silently embedded in a new base.
    for key in ('version','proposal_sha256','hashes'):
        require(old['method'][key]==new['method'][key],'BASE_METHOD')
    metric='training_workloads/unified_h1/evaluation.py'
    require(old['workload']['source_sha256'].get(metric) is not None
        and old['workload']['source_sha256'][metric]==new['workload']['source_sha256'].get(metric),'ORIGINAL_METRIC')
    changes={section:source_changes(old[section]['source_sha256'],new[section]['source_sha256'])
             for section in ('product','workload')}
    allowed={'isaaclab-training/instinctlab/envs/taslab_direct_env.py',
        'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py',
        'isaaclab-training/instinctlab/task/rewards_h1.py',
        'isaaclab-training/instinctlab/control/ics.py',
        *('isaaclab-training/instinctlab/observations/'+n+'.py' for n in ('robustness','robustness_adapter','robustness_physics'))}
    from .robustness_contract import SERIALIZER_PATH,SERIALIZER_DELTA
    if changes['product'].get(SERIALIZER_PATH)==SERIALIZER_DELTA:allowed.add(SERIALIZER_PATH)
    # Source-manifest additions are eligible only when the file already existed
    # unchanged in the origin checkout, except the explicit R6 adapter files.
    for name,delta in changes['product'].items():
        if name not in allowed and delta['origin'] is None and delta['execution'] is not None:
            validate_pin(dict(path=str(Path(old['product']['path'])/name),sha256=delta['execution']))
            allowed.add(name)
    require(set(changes['product'])<=allowed,'PRODUCT_BOUNDARY')
    for section in ('product','workload'):
        require(changes[section]==record['changed_'+section+'_sources'],'EXACT_'+section+'_DELTA')
        for value in (old,new):
            for relative,digest in value[section]['source_sha256'].items():
                require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'SOURCE_PATH')
                validate_pin(dict(path=str(Path(value[section]['path'])/relative),sha256=digest))
    accepted_review(record['independent_review'],'PASS_R6_TRAINING_COMPATIBILITY',
        payload_sha256=object_sha({k:v for k,v in record.items() if k!='independent_review'}))
    return dict(origin)
