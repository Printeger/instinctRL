"""Exact candidate development evaluation stage and formal workload wrapper.

Shares frozen denominator/resource checks with baseline evaluation, while exact
selection and method bindings are mandatory and baseline authority is rejected.
"""
from . import robustness_contract as base
from . import robustness_evaluation_selection as selection

MODE='research-r6-candidate-evaluation'


def validate_parameters(mode,value):
    selection.require(mode==MODE and isinstance(value,dict) and 'evaluation_selection' in value,'CANDIDATE_MODE')
    selected=selection.validate(value['evaluation_selection'])
    return base._validate_parameters(mode,value,selected=selected)


def required_budget(mode,value):
    return base.budget_from_parameters(validate_parameters(mode,value))


def evaluate_registered(backend):
    from training_workloads.scale_003.persistence import atomic_json_replace
    from .robustness_candidate_restore import restore
    p=validate_parameters(backend.authorization['mode'],backend.research)
    selection.require(backend.authorization['parameters']==p and backend.authorization['budgets']==required_budget(MODE,p),'EXACT_STAGE_BUDGET')
    result=restore(backend,p['evaluation_selection']);selected=result['selection'];model=selected['model'];method=selected['method']
    label=f"r6_candidate_u{model['update']}_original"
    atomic_json_replace(backend.output,'r6_model_binding.json',dict(checkpoint=model['checkpoint'],metadata=result['metadata'],
        checkpoint_source_identity=model['source_identity'],method_hashes=model['method_hashes'],
        evaluation_selection_sha256=base.object_sha(p['evaluation_selection']),comparison=p['evaluation_selection']['comparison'],
        method=p['evaluation_selection']['method'],actual_method_binding=result['method_binding'],
        ics_clearance_margin_m=method['ics_clearance_margin_m'],original_evaluation_label=label,
        physical_history_continuity_claimed=False,actor_input_extension=False))
    original=base.evaluate_original_suite(backend,update=model['update'],label=label)
    new=backend.evaluate_r6_blocks()
    complete=dict(original=original,new_blocks=new,model=model['checkpoint'],
        selection_sha256=base.object_sha(p['evaluation_selection']),comparison=p['evaluation_selection']['comparison'],
        original_seed=20260909,new_seed=p['r6']['seed'],scientific_acceptance=False,
        evidence_status='REQUIRES_INDEPENDENT_RAW_REPLAY_AND_PAIRED_COMPARISON')
    atomic_json_replace(backend.output,'r6_candidate_evaluation_complete.json',complete)
    return complete


def validate_compatibility(pin,value,config):
    record=base.pinned_json(pin)
    payload={k:v for k,v in record.items() if k!='independent_review'}
    origin=validate_compatibility_payload(payload,value,config)
    base.accepted_review(record['independent_review'],'PASS_R6_CANDIDATE_COMPATIBILITY',payload_sha256=base.object_sha(payload))
    return origin


def validate_compatibility_payload(record,value,config):
    from pathlib import Path
    from .research_contract import validate_pin
    from .checkpoint_compatibility import validate_source_identity
    from .robustness_training_contract import source_changes
    selected=selection.validate(value);model=selected['model']
    origin=validate_source_identity(model['source_identity']);execution=validate_source_identity(config['resolved_identity'])
    fields={'schema_version','selection_sha256','origin_source_identity','execution_source_identity','origin_config','execution_config',
        'checkpoint','method','changed_product_sources','changed_workload_sources'}
    selection.require(set(record)==fields and record['schema_version']=='r6-candidate-compatibility/1','COMPATIBILITY_FIELDS')
    selection.require(record['selection_sha256']==base.object_sha(value) and record['origin_source_identity']==origin
        and record['execution_source_identity']==execution and record['checkpoint']==model['checkpoint']
        and record['method']==value['method'],'COMPATIBILITY_BINDING')
    old=base.pinned_json(record['origin_config']);new=base.pinned_json(record['execution_config'])
    selection.require(record['origin_config']['sha256']==origin['config_sha256']
        and record['execution_config']['sha256']==execution['config_sha256']
        and new=={k:v for k,v in config.items() if k!='resolved_identity'},'CONFIG_BINDING')
    for key in ('actor','training','execution','runtime','versions'):
        selection.require(old[key]==new[key],'INVARIANT_'+key)
    for key in ('version','proposal_sha256','hashes'):
        selection.require(old['method'][key]==new['method'][key],'BASE_METHOD')
    metric='training_workloads/unified_h1/evaluation.py'
    selection.require(old['workload']['source_sha256'].get(metric) is not None
        and old['workload']['source_sha256'][metric]==new['workload']['source_sha256'].get(metric),'ORIGINAL_METRIC')
    changes={section:source_changes(old[section]['source_sha256'],new[section]['source_sha256']) for section in ('product','workload')}
    allowed={'isaaclab-training/instinctlab/envs/taslab_direct_env.py','isaaclab-training/instinctlab/envs/unified_h1_direct_env.py',
        'isaaclab-training/instinctlab/task/rewards_h1.py','isaaclab-training/instinctlab/control/ics.py',
        *('isaaclab-training/instinctlab/observations/'+n+'.py' for n in ('robustness','robustness_adapter','robustness_physics'))}
    if changes['product'].get(base.SERIALIZER_PATH)==base.SERIALIZER_DELTA:allowed.add(base.SERIALIZER_PATH)
    for name,delta in changes['product'].items():
        if name not in allowed and delta['origin'] is None and delta['execution'] is not None:
            validate_pin(dict(path=str(Path(old['product']['path'])/name),sha256=delta['execution']));allowed.add(name)
    selection.require(set(changes['product'])<=allowed,'PRODUCT_BOUNDARY')
    for section in ('product','workload'):
        selection.require(record['changed_'+section+'_sources']==changes[section],'EXACT_'+section+'_DELTA')
        for source in (old,new):
            for relative,digest in source[section]['source_sha256'].items():
                selection.require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'SOURCE_PATH')
                validate_pin(dict(path=str(Path(source[section]['path'])/relative),sha256=digest))
    return dict(origin)
