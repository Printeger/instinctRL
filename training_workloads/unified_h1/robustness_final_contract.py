"""Frozen final evaluation stage; no optimizer, training or checkpoint writes."""
from . import robustness_contract as base
from . import robustness_final_selection as selection
from .robustness_candidate_contract import _validate_compatibility_payload

MODE=selection.MODE


def validate_parameters(mode,value):
    selection.require(mode==MODE and isinstance(value,dict) and 'final_selection' in value,'FINAL_MODE')
    selected=selection.validate(value['final_selection'])
    return base._validate_parameters(mode,value,selected=selected,final=True)


def validate_draft(mode,value,payload_bytes):
    selection.require(mode==MODE and isinstance(value,dict) and 'final_selection' in value,'FINAL_MODE')
    selected=selection.validate(value['final_selection'])
    return base._validate_parameters(mode,value,selected=selected,final=True,materialized=False,payload_bytes=payload_bytes)


def required_budget(mode,value):
    return base.budget_from_parameters(validate_parameters(mode,value))


def validate_compatibility_payload(record,value,config):
    selected=selection.validate(value)
    return _validate_compatibility_payload(record,value,config,selected,
        schema='r6-final-compatibility/1',method=selected['method_pin'])


def validate_compatibility(pin,value,config):
    record=base.pinned_json(pin);payload={k:v for k,v in record.items() if k!='independent_review'}
    origin=validate_compatibility_payload(payload,value,config)
    base.accepted_review(record['independent_review'],'PASS_R6_FINAL_COMPATIBILITY',payload_sha256=base.object_sha(payload))
    return origin


def compatibility_payload(origin_config,execution_config,value,config):
    from .robustness_training_contract import source_changes
    selected=selection.validate(value)
    old=base.pinned_json(origin_config);new=base.pinned_json(execution_config)
    payload=dict(schema_version='r6-final-compatibility/1',selection_sha256=base.object_sha(value),
        origin_source_identity=selected['model']['source_identity'],execution_source_identity=config['resolved_identity'],
        origin_config=origin_config,execution_config=execution_config,checkpoint=selected['model']['checkpoint'],
        method=selected['method_pin'],changed_product_sources=source_changes(old['product']['source_sha256'],new['product']['source_sha256']),
        changed_workload_sources=source_changes(old['workload']['source_sha256'],new['workload']['source_sha256']))
    validate_compatibility_payload(payload,value,config)
    return payload


def evaluate_registered(backend):
    from training_workloads.scale_003.persistence import atomic_json_replace
    from .robustness_final_restore import restore
    p=validate_parameters(backend.authorization['mode'],backend.research)
    selection.require(backend.authorization['parameters']==p and backend.authorization['budgets']==required_budget(MODE,p),'EXACT_STAGE_BUDGET')
    loaded=restore(backend,p['final_selection']);selected=loaded['selection'];model=selected['model']
    label=f"r6_final_u{model['update']}_original"
    binding=dict(checkpoint=model['checkpoint'],metadata=loaded['metadata'],checkpoint_source_identity=model['source_identity'],
        method_hashes=model['method_hashes'],final_selection=p['final_selection'],
        final_selection_sha256=base.object_sha(p['final_selection']),assignment=selected['assignment'],
        comparison=selected['comparison'],method=selected['method_pin'],actual_method_binding=loaded['method_binding'],
        ics_clearance_margin_m=.15 if selected['method'] is None else selected['method']['ics_clearance_margin_m'],
        original_evaluation_label=label,physical_history_continuity_claimed=False,actor_input_extension=False)
    atomic_json_replace(backend.output,'r6_model_binding.json',binding)
    original=base.evaluate_original_suite(backend,update=model['update'],label=label)
    new=backend.evaluate_r6_blocks()
    complete=dict(original=original,new_blocks=new,model=model['checkpoint'],final_selection=p['final_selection'],
        selection_sha256=base.object_sha(p['final_selection']),assignment=selected['assignment'],comparison=selected['comparison'],
        original_seed=20260909,new_seed=p['r6']['seed'],scientific_acceptance=False,
        evidence_status='REQUIRES_INDEPENDENT_RAW_REPLAY_AND_COMPLETE_FINAL_BATCH')
    atomic_json_replace(backend.output,'r6_final_evaluation_complete.json',complete)
    return complete
