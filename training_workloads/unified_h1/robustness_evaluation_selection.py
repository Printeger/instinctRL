"""Reviewed development comparison and exact policy selection, not live authority.

Final holdout requires a separate freeze/release contract; this module rejects it.
"""
from pathlib import Path
from . import robustness_contract as base
from . import robustness_training_plan as training
from .robustness_training_method import validate as validate_method
from .research_contract import validate_pin

FIELDS={'schema_version','phase','role','baseline_bindings','checkpoint','training_plan',
        'training_analysis','method','comparison','independent_review'}


def require(ok,reason):
    if not ok:raise PermissionError('R6_EVALUATION_SELECTION_'+reason)


def validate(value):
    return _validate(value,reviewed=True)


def validate_draft(value):
    return _validate(value,reviewed=False)


def _validate(value,*,reviewed):
    require(isinstance(value,dict) and set(value)==(FIELDS if reviewed else FIELDS-{'independent_review'}) and value['schema_version']=='r6-evaluation-selection/1','FIELDS')
    require(value['phase']=='development' and value['role'] in ('baseline_control','trained_candidate'),'DEVELOPMENT_ROLE')
    baseline=base.baseline(value);validate_pin(value['checkpoint'])
    method=validate_method(base.pinned_json(value['method']))
    comparison=base.pinned_json(value['comparison'])
    require(set(comparison)=={'schema_version','id','candidate_id','hypothesis','protocol','split','seed','block_ids','method','baseline_checkpoint_sha256'}
        and comparison['schema_version']=='r6-paired-comparison/1','COMPARISON_FIELDS')
    require(isinstance(comparison['id'],str) and 1<=len(comparison['id'])<=100
        and isinstance(comparison['hypothesis'],str) and 1<=len(comparison['hypothesis'])<=4096,'COMPARISON_IDENTITY')
    require(comparison['method']==value['method'] and comparison['split']=='development'
        and type(comparison['seed'])is int and 0<=comparison['seed']<2**31
        and comparison['baseline_checkpoint_sha256']==base.BASELINE_SHA,'PAIRED_METHOD_STREAM')
    protocol=base.pinned_json(comparison['protocol'])
    require(protocol.get('frozen') is True and len(protocol['blocks'])==16
        and comparison['block_ids']==[b['id'] for b in protocol['blocks']]
        and len(set(comparison['block_ids']))==16,'FULL_FROZEN_PROTOCOL')
    # The same comparison pin is consumed by both policy roles; method changes
    # cannot be mistaken for learning gains from a different controller baseline.
    if value['role']=='baseline_control':
        require(value['checkpoint']==baseline['selected_model'] and value['training_plan'] is None
            and value['training_analysis'] is None,'FIXED_BASELINE_CONTROL')
        model=dict(checkpoint=value['checkpoint'],update=640,training_seed=20260908,
            source_identity=baseline['source_identity'],method_hashes=baseline['method_hashes'],ppo_config=None)
    else:
        plan=training.validate(value['training_plan']);recipe=base.pinned_json(plan['recipe'])
        require(plan['phase']=='development' and recipe['candidate_id']==comparison['candidate_id']
            and value['method']==recipe['method'],'TRAINED_METHOD')
        model=_training_model(plan,value['training_analysis'],value['checkpoint'],method)
    # Review explicitly adjudicates the pinned training evidence/native status,
    # comparison design and the selected node; no analysis alone grants selection.
    if reviewed:
        base.accepted_review(value['independent_review'],'PASS_R6_EVALUATION_SELECTION',
            payload_sha256=base.object_sha({k:v for k,v in value.items() if k!='independent_review'}))
    return dict(selection=value,model=model,method=method,comparison=comparison,
        scientific_acceptance=False,dispatch_authorized=False)


def _training_model(plan,analysis_pin,checkpoint,method):
    """Closed checkpoint evidence for a validated development or confirmation plan."""
    recipe=base.pinned_json(plan['recipe'])
    binding=base.pinned_json(analysis_pin)
    require(binding['schema_version']=='r6-bound-training-analysis/1' and binding['scientific_acceptance'] is False
        and binding['training_plan_sha256']==base.object_sha(plan),'TRAINING_ANALYSIS')
    for name in ('spec','authorization','closure','inventory','postrun','report'):validate_pin(binding[name])
    auth=base.pinned_json(binding['authorization']);report=base.pinned_json(binding['report']);spec=base.pinned_json(binding['spec'])
    require(spec['pins']['authorization']==binding['authorization'] and spec['h1']['mode']=='research-r6-training'
        and spec['h1']['attempt']==auth['attempt'],'TRAINING_SPEC')
    require(auth['mode']=='research-r6-training' and auth['parameters']['r6_training_plan']==plan
        and auth['budgets']==training.required_budget(plan),'TRAINING_AUTHORIZATION')
    require(report['schema_version']=='r6-training-analysis/1' and report['authorization']==binding['authorization']
        and report['plan_sha256']==base.object_sha(plan) and report['scientific_acceptance'] is False
        and report['issues']==[],'TRAINING_REPORT')
    summary=report['summary'];require(summary['issue_count']==0 and summary['probability_gate_recomputed'] is True
        and summary['rng_initialization_audit']['status']=='PASS','TRAINING_AUDITS')
    matched=[row for row in summary['checkpoints'] if row.get('pin')==checkpoint]
    require(len(matched)==1,'EXACT_CHECKPOINT_NODE');row=matched[0];update=row['update']
    require(type(update)is int and update in plan['checkpoint_updates'] and row['metadata_verified'] is True
        and row['state_audit']['actor_critic_state_loaded'] is True and row['state_audit']['adam_state_loaded'] is True,'CHECKPOINT_AUDIT')
    require(Path(checkpoint['path'])==Path(auth['attempt'])/'results/research-r6-training/checkpoints'/f'u{update}.pt','CHECKPOINT_ATTEMPT')
    inventory=base.pinned_json(binding['inventory']);closure=base.pinned_json(binding['closure']);postrun=base.pinned_json(binding['postrun'])
    require(all(v['job_id']==binding['job_id'] for v in (inventory,closure,postrun))
        and inventory['spec']==binding['spec'] and inventory['closure']==binding['closure']
        and postrun['closure']==binding['closure'] and postrun['inventory']==binding['inventory']
        and postrun['postrun_complete'] is True and inventory['output_inventory_complete'] is True,'CLOSED_JOB_CHAIN')
    require(closure['spec_sha256']==base.object_sha(spec),'CLOSURE_SPEC')
    indexed={r['path']:r for r in inventory['files']};seen=set()
    require(len(indexed)==len(inventory['files']),'DUPLICATE_INVENTORY')
    for item in report['inputs']:
        pin={k:item[k] for k in ('path','sha256')};path=validate_pin(pin)
        require(pin['path'] not in seen and path.stat().st_size==item['bytes'],'REPORT_INPUT_IDENTITY');seen.add(pin['path'])
        if pin!=plan['initialization']['checkpoint']:
            require(pin['path'] in indexed and all(item[k]==indexed[pin['path']][k] for k in ('sha256','bytes')),'REPORT_INPUT_INVENTORY')
    require({checkpoint['path'],binding['authorization']['path'],plan['initialization']['checkpoint']['path']}<=seen,'REPORT_INPUT_MISSING')
    require(binding['native_status']==closure['native_status'] and binding['native_exit_code']==closure['native_exit_code'],'NATIVE_STATUS')
    originals=[r for r in inventory['files'] if {k:r[k] for k in ('path','sha256')}==checkpoint]
    require(len(originals)==1 and originals[0]['bytes']==validate_pin(checkpoint).stat().st_size,'CHECKPOINT_INVENTORY')
    return dict(checkpoint=checkpoint,update=update,training_seed=plan['training_seed'],source_identity=recipe['source_identity'],
        method_hashes=dict(plan['initialization']['method_hashes'],r6_recipe=plan['recipe']['sha256'],r6_method=recipe['method']['sha256'],curriculum='curriculum-h1-r6-v1'),
        ppo_config=dict(method['ppo'],permutation_seed=plan['training_seed']))


def verify_loaded(selection,metadata):
    model=selection['model'];update=model['update']
    expected=dict(update=update,transition=update*512,global_policy_step=update*16,actor_optimizer=update*8,critic_optimizer=update*8)
    require(metadata['counters']==expected and all(type(v)is int for v in metadata['counters'].values())
        and type(metadata['training_seed'])is int and metadata['training_seed']==model['training_seed'],'LOADED_COUNTERS_SEED')
    if selection['selection']['role']=='baseline_control':
        expected_curriculum=dict(version='curriculum-h1-v1',completed_policy_steps=update*16)
    else:
        plan=selection['selection']['training_plan']
        expected_curriculum=dict(version='curriculum-h1-r6-v1',completed_policy_steps=update*16,
            recipe_sha256=plan['recipe']['sha256'],origin_checkpoint_sha256=base.BASELINE_SHA,branch_seed=model['training_seed'])
    require(metadata['curriculum']==expected_curriculum,'LOADED_CURRICULUM')
    return model
