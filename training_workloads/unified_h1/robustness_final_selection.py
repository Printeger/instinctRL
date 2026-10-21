"""Exact final-batch selection for frozen execution; never a development selector.

Journal, registered-review receipt and Git-publication provenance are checked by
the controller before authority issuance. This runtime side rechecks the pinned
freeze, review, commitment/reveal, entire assignment set and checkpoint evidence.
"""
import re
from . import robustness_contract as base
from . import robustness_evaluation_selection as development
from . import robustness_training_plan as training
from .robustness_training_method import validate as validate_method
from .robustness_holdout import reveal
from .research_contract import validate_pin

MODE='research-r6-final-evaluation'
FIELDS={'schema_version','batch','assignment_index'}


def require(value,reason):
    if not value:raise PermissionError('R6_FINAL_SELECTION_'+reason)


def validate(value):
    require(isinstance(value,dict) and set(value)==FIELDS and value['schema_version']=='r6-final-selection/1','FIELDS')
    batch=base.pinned_json(value['batch'])
    require(set(batch)=={'schema_version','freeze','freeze_review','holdout_event_key','publication','publication_commit',
        'realization','assignments','protocol','scientific_acceptance','dispatch_authorized'}
        and batch['schema_version']=='r6-final-batch-inputs/1'
        and batch['scientific_acceptance']is False and batch['dispatch_authorized']is False,'BATCH')
    require(isinstance(batch['publication_commit'],str) and re.fullmatch('[a-f0-9]{40}',batch['publication_commit']),
        'PUBLICATION_COMMIT')
    freeze=base.pinned_json(batch['freeze'])
    require(set(freeze)=={'schema_version','branch','protocol','recipe','selection_rule','models','evidence'}
        and freeze['schema_version']=='r6-final-freeze/2' and batch['protocol']==freeze['protocol'],'FREEZE')
    base.accepted_review(batch['freeze_review'],'PASS_R6_FINAL_FREEZE',freeze_sha256=batch['freeze']['sha256'])
    public=base.pinned_json(batch['publication']);realization=base.pinned_json(batch['realization'])
    require(public['freeze_sha256']==batch['freeze']['sha256'] and public['protocol_sha256']==freeze['protocol']['sha256']
        and realization==reveal(public,realization['nonce']),'REVEALED_COMMITMENT')
    baseline_only=freeze['branch']=='BASELINE_SUFFICIENT'
    if baseline_only:
        require(set(freeze['models'])=={'baseline'} and freeze['selection_rule']==dict(kind='unchanged_baseline',update=640,
            training_transitions=0,final_realization_indices=[0,1,2],all_original_and_registered_blocks_each_stream=True,
            select_using_holdout=False),'BASELINE_RULE')
        pairs=[('baseline',i) for i in range(3)]
    else:
        require(freeze['branch']=='THREE_CONTINUATIONS' and set(freeze['models'])=={'baseline','seed1','seed2','seed3'},'MODEL_SET')
        pairs=[(role,0) for role in ('baseline','seed1','seed2','seed3')]
    for pin in freeze['models'].values():validate_pin(pin)
    expected=[dict(model_role=role,checkpoint=freeze['models'][role],realization_index=i,
        evaluation_seed=realization['realization_seeds'][i],split='final') for role,i in pairs]
    require(batch['assignments']==expected,'EXACT_ASSIGNMENTS')
    index=value['assignment_index'];require(type(index)is int and 0<=index<len(expected),'INDEX')
    assignment=expected[index];baseline_pin=freeze['evidence']['baseline_bindings']
    baseline=base.baseline({'baseline_bindings':baseline_pin})
    require(freeze['models']['baseline']==baseline['selected_model'],'BASELINE_IDENTITY')
    protocol=base.pinned_json(freeze['protocol'])
    require(protocol.get('frozen')is True and len(protocol['blocks'])==16
        and len({b['id'] for b in protocol['blocks']})==16,'FULL_PROTOCOL')
    plan=None;method=None;method_pin=None
    if assignment['model_role']=='baseline':
        if baseline_only:require(freeze['recipe']==baseline_pin,'UNCHANGED_BASELINE')
        model=dict(checkpoint=baseline['selected_model'],update=640,training_seed=20260908,
            source_identity=baseline['source_identity'],method_hashes=baseline['method_hashes'],ppo_config=None)
    else:
        ordinal=int(assignment['model_role'][-1])-1
        streams=freeze['evidence']['streams'];require(len(streams)==3,'THREE_STREAMS')
        stream=streams[ordinal];context=base.pinned_json(stream['context'])
        profile=base.pinned_json(context['baseline_profile']);plan=training.validate(profile['r6_training_plan'])
        recipe=base.pinned_json(plan['recipe']);method_pin=recipe['method'];method=validate_method(base.pinned_json(method_pin))
        require(plan['phase']=='confirmation' and plan['recipe']==freeze['recipe']
            and plan['selection_freeze']==freeze['evidence']['recipe_freeze']
            and plan['training_seed']==recipe['confirmation_seeds'][ordinal]
            and plan['start_update']==640 and plan['target_update']==640+recipe['branch_updates']
            and profile['baseline_bindings']==baseline_pin,'CONFIRMATION_IDENTITY')
        require(freeze['selection_rule']==dict(kind='terminal_checkpoint',common_start_update=640,
            target_update=plan['target_update'],select_using_holdout=False)
            and stream['checkpoint']==assignment['checkpoint'] and stream['plan_sha256']==base.object_sha(plan)
            and stream['training_seed']==plan['training_seed'] and stream['target_update']==plan['target_update'],'TERMINAL_SELECTION')
        model=development._training_model(plan,stream['analysis'],assignment['checkpoint'],method)
        require(model['update']==plan['target_update'],'TERMINAL_CHECKPOINT')
    # Metadata verification shares the existing strict serializer clock rules.
    normalized=dict(role='baseline_control' if plan is None else 'trained_candidate',training_plan=plan,
        baseline_bindings=baseline_pin)
    return dict(final_selection=value,selection=normalized,model=model,method=method,method_pin=method_pin,
        batch=batch,freeze=freeze,assignment=assignment,
        comparison=dict(protocol=freeze['protocol'],block_ids=[b['id'] for b in protocol['blocks']],
            seed=assignment['evaluation_seed'],split='final'),scientific_acceptance=False,dispatch_authorized=False)


def verify_loaded(selected,metadata):
    return development.verify_loaded(selected,metadata)
