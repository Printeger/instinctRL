"""Pre-confirmation recipe selection evidence; no holdout release or success claim."""
from pathlib import Path
from . import robustness_contract as base
from .research_contract import validate_pin


def require(ok,reason):
    if not ok:raise PermissionError('R6_CONFIRMATION_'+reason)


def evaluation_evidence(pin,protocol_pin):
    """Authenticate complete development replay inputs, preserving native failures.

    This verifies saved evidence relationships, not physical execution or independent
    scientific acceptance. The recipe-freeze reviewer must inspect the originals.
    """
    from . import robustness_evaluation_selection as selection
    bound=base.pinned_json(pin)
    if bound.get('schema_version')=='r6-direct-bound-analysis/1':
        from .direct_selection import evaluation_evidence as direct_evidence
        return direct_evidence(pin,protocol_pin)
    require(bound['schema_version']=='r6-bound-motion-analysis/1' and bound['scientific_acceptance']is False
        and bound['issues']==[] and bound['protocol']==protocol_pin,'BOUND_EVALUATION')
    objects={k:base.pinned_json(bound[k]) for k in ('spec','authorization','closure','inventory','postrun','report','model_binding')}
    spec=objects['spec'];auth=objects['authorization'];close=objects['closure'];inv=objects['inventory'];post=objects['postrun'];report=objects['report']
    require(spec['pins']['authorization']==bound['authorization'] and spec['h1']['mode']==auth['mode']=='research-r6-candidate-evaluation'
        and spec['h1']['attempt']==auth['attempt'],'EVALUATION_AUTHORIZATION')
    output=Path(auth['attempt'])/'results'/auth['mode'];request=auth['parameters']['r6']
    require(request['protocol']==protocol_pin and request['split']=='development','DEVELOPMENT_ONLY')
    require(close['event_key']==inv['event_key']==post['job_event_key']==bound['event_key']
        and close['job_id']==inv['job_id']==post['job_id']==bound['job_id'] and close['spec_sha256']==base.object_sha(spec)
        and inv['spec']==bound['spec'] and inv['closure']==post['closure']==bound['closure']
        and post['inventory']==bound['inventory'] and inv['output_inventory_complete']is True
        and post['postrun_complete']is True and post['postrun_queries_used']==2
        and close['native_status']==bound['native_status'] and close['native_exit_code']==bound['native_exit_code'],'CLOSED_JOB_CHAIN')
    require(len(post['receipts'])==2 and len({p['path'] for p in post['receipts']})==2,'POSTRUN_RECEIPTS')
    for receipt in post['receipts']:validate_pin(receipt)
    require(report['schema_version']=='r6-motion-replay/1' and report['scientific_acceptance']is False
        and report['authorization']==bound['authorization'] and report['candidate_binding_issues']==[]
        and all(report[k]==request[k] for k in ('run_id','seed','split')),'REPORT_IDENTITY')
    selected=selection.validate(auth['parameters']['evaluation_selection']);model=selected['model'];binding=objects['model_binding']
    require(selected['comparison']['protocol']==protocol_pin and selected['comparison']['seed']==request['seed']
        and binding['checkpoint']==model['checkpoint'] and binding['method']==selected['selection']['method']
        and binding['evaluation_selection_sha256']==base.object_sha(selected['selection'])
        and report['candidate']['selection_sha256']==binding['evaluation_selection_sha256']
        and report['candidate']['checkpoint']==model['checkpoint'] and report['candidate']['update']==model['update'],'MODEL_IDENTITY')
    protocol=base.pinned_json(protocol_pin);blocks=protocol['blocks']
    require(len(blocks)==16 and [r['block_id'] for r in report['blocks']]==[b['id'] for b in blocks]
        and request['block_ids']==[b['id'] for b in blocks],'FULL_BLOCKS')
    required={bound['authorization']['path'],bound['model_binding']['path']}
    for block,row in zip(blocks,report['blocks']):
        require(row['replay_complete']is True and row['saved_result_equal']is True and row['total']==32
            and row['domain']==block['domain'] and row['measurement_replay']['bit_exact_delivery']is True
            and row['physics_replay']['staging_reconstructed']is True,'COMPLETE_REPLAY')
        # A negative new-domain result remains admissible evidence for review.
        # No new success threshold is introduced here.
        required.update(str(output/'r6'/block['id']/name) for name in ('motion_traces.json','result.json','measurements.bin','audit.bin'))
    original=report['summary']['original']
    require(original['replay_complete']is True and original['saved_result_equal']is True and original['saved_wall_equal']is True
        and original['all_gates_pass']is True and original['wall_motion']['all32']is True
        and original['wall_motion']['per_lane']==[True]*32,'ORIGINAL_REGRESSION')
    label=f"r6_candidate_u{model['update']}_original"
    require(binding['original_evaluation_label']==report['candidate']['original_label']==label,'ORIGINAL_LABEL')
    required.update(str(output/'evaluation_raw'/label/(name+'.jsonl')) for name in ('Z','T','S','X','W'))
    required.update(str(output/name) for name in ('r6_original_suite.json','r6_original_wall_motion.json'))
    indexed={r['path']:r for r in inv['files']};require(len(indexed)==len(inv['files']),'DUPLICATE_INVENTORY')
    seen=set();external=[protocol_pin]
    if protocol.get('physics_replay',{}).get('mount_source'):external.append(protocol['physics_replay']['mount_source'])
    for item in report['inputs']:
        p={k:item[k] for k in ('path','sha256')};path=validate_pin(p)
        require(p['path'] not in seen and path.stat().st_size==item['bytes'],'INPUT_IDENTITY');seen.add(p['path'])
        if p in external:continue
        row=indexed.get(p['path'])
        require(row is not None and all(row[k]==item[k] for k in ('sha256','bytes'))
            and (p==bound['authorization'] or output in path.parents),'INPUT_INVENTORY')
    require(required<=seen and protocol_pin['path'] in seen,'COMPLETE_INPUT_SET')
    return selected


def validate_registered_prefix(lineage, recipe, trained):
    """Bind a shorter confirmation recipe to an actually evaluated registered node.

    Historical recipe/checkpoint identities remain unchanged. Only a complete
    curriculum-phase prefix may be selected; methods and stream lists stay exact.
    """
    from . import robustness_training_plan as training
    rebound='source_rebinding' in lineage
    require(set(lineage)==({'kind','development_recipe','source_rebinding'} if rebound else {'kind','development_recipe'})
        and lineage['kind']=='REGISTERED_DEVELOPMENT_PREFIX','PREFIX_FIELDS')
    plan=trained['selection']['training_plan']
    require(plan['schema_version']=='r6-local-training-plan/1'
        and plan['recipe']==lineage['development_recipe'],'PREFIX_PROVENANCE')
    old=training.validate_recipe(base.pinned_json(lineage['development_recipe']))
    changed={'branch_updates','checkpoint_offsets','curriculum'}
    if rebound:
        validate_source_rebinding(lineage['source_rebinding'],old['source_identity'],recipe['source_identity'])
        changed.add('source_identity')
    require(all(recipe[k]==old[k] for k in old.keys()-changed),'PREFIX_RECIPE_DRIFT')
    end=recipe['branch_updates']
    require(type(end)is int and end<old['branch_updates'] and end in old['checkpoint_offsets']
        and recipe['checkpoint_offsets']==[n for n in old['checkpoint_offsets'] if n<=end],
        'PREFIX_REGISTERED_NODE')
    source=base.pinned_json(old['curriculum']);actual=base.pinned_json(recipe['curriculum'])
    expected=dict(source,phases=[p for p in source['phases'] if p['end_offset']<=end])
    require(expected['phases'] and expected['phases'][-1]['end_offset']==end
        and actual==expected,'PREFIX_CURRICULUM')
    require(plan['phase']=='development' and plan['start_update']==640
        and plan['initialization']['kind']=='COMMON_U640_NEW_STREAM'
        and plan['target_update']==trained['model']['update']==640+end
        and plan['checkpoint_updates']==[640+n for n in recipe['checkpoint_offsets']],
        'PREFIX_ACTUAL_ENDPOINT')


def validate_source_rebinding(pin,old_identity,new_identity):
    """Review an exact engineering source migration without relabeling old runs."""
    from tools.r6_direct.source import training_identity
    value=base.pinned_json(pin)
    require(set(value)=={'schema_version','original_source','replacement_source','changes','independent_review'}
        and value['schema_version']=='r6-confirmation-source-rebinding/1','SOURCE_REBIND_FIELDS')
    old=base.pinned_json(value['original_source']);new=base.pinned_json(value['replacement_source'])
    require(old['schema_version']==new['schema_version']=='r6-direct-source/1'
        and training_identity(old)==old_identity and training_identity(new)==new_identity
        and old_identity!=new_identity,'SOURCE_REBIND_IDENTITY')
    require(all(old[k]==new[k] for k in ('scientific_config','execution','actor','training','method','product_deltas')),
        'SOURCE_REBIND_METHOD')
    changes=[dict(path=k,before=old['sources'].get(k),after=new['sources'].get(k))
        for k in sorted(old['sources'].keys()|new['sources'].keys())
        if old['sources'].get(k)!=new['sources'].get(k)]
    require(changes==value['changes'] and bool(changes),'SOURCE_REBIND_DIFF')
    payload={k:v for k,v in value.items() if k!='independent_review'}
    base.accepted_review(value['independent_review'],'PASS_R6_CONFIRMATION_SOURCE_REBIND',
        payload_sha256=base.object_sha(payload))


def validate_selection_evidence(pin,recipe_pin,recipe):
    value=base.pinned_json(pin)
    prefix=value.get('schema_version')=='r6-recipe-selection-evidence/2'
    fields={'schema_version','protocol','split_manifest','protocol_review','paired_evaluations','checkpoint_selection'}
    amended='split_amendment' in value
    if amended:fields.add('split_amendment')
    require((not amended or prefix) and set(value)==(fields|{'recipe_lineage'} if prefix else fields)
        and value['schema_version'] in ('r6-recipe-selection-evidence/1','r6-recipe-selection-evidence/2'),'SELECTION_EVIDENCE')
    protocol=base.pinned_json(value['protocol']);split=base.pinned_json(value['split_manifest'])
    require(protocol.get('frozen')is True and split['schema_version']=='r6-split-manifest/1'
        and split['final']['seed_realization_generated']is False,'UNREVEALED_SPLIT')
    review=base.accepted_review(value['protocol_review'],'PASS_R6_PROTOCOL',protocol_sha256=value['protocol']['sha256'])
    reviewed_split=value['split_manifest']
    if amended:
        reviewed_split=validate_seed_amendment(value['split_amendment'],value['protocol'],protocol,value['split_manifest'])
    else:
        require(protocol.get('split_manifest',reviewed_split)==reviewed_split,'PROTOCOL_SPLIT_BINDING')
    require(all(p in review['evidence_pins'] for p in (value['protocol'],reviewed_split)),'REVIEWED_SPLIT')
    require(recipe['confirmation_seeds']==split['confirmation_training_seeds']
        and len(set(recipe['confirmation_seeds']))==3 and not set(recipe['confirmation_seeds'])&set(recipe['development_seeds']),'THREE_REGISTERED_STREAMS')
    require(value['checkpoint_selection']==dict(kind='terminal_checkpoint',common_start_update=640,
        target_update=640+recipe['branch_updates'],select_using_holdout=False),'FIXED_SELECTION_RULE')
    paired=value['paired_evaluations'];require(set(paired)=={'baseline_control','trained_candidate'},'PAIRED_EVIDENCE')
    selected={role:evaluation_evidence(p,value['protocol']) for role,p in paired.items()}
    for role,item in selected.items():
        require(item['selection']['role']==role and item['selection']['method']==recipe['method'],'PAIRED_ROLE_METHOD')
    control=selected['baseline_control']['selection'];trained=selected['trained_candidate']['selection']
    require(control['comparison']==trained['comparison'] and control['baseline_bindings']==trained['baseline_bindings']
        and trained['training_plan']['phase']=='development','PAIRED_DESIGN')
    require(selected['baseline_control']['model']['update']==640
        and selected['trained_candidate']['model']['update']==640+recipe['branch_updates'],
        'EVALUATED_ENDPOINT')
    if prefix:
        validate_registered_prefix(value['recipe_lineage'],recipe,selected['trained_candidate'])
    else:
        require(trained['training_plan']['recipe']==recipe_pin,'PAIRED_DESIGN')
    return dict(evidence=value,scientific_acceptance=False,holdout_released=False)


def validate_seed_amendment(pin,protocol_pin,protocol,split_pin):
    """Explicit reviewed override of confirmation seeds, never test distribution."""
    amendment=base.pinned_json(pin)
    fields={'schema_version','protocol','original_split','replacement_split','justification',
        'evidence_pins','independent_review'}
    require(set(amendment)==fields and amendment['schema_version']=='r6-confirmation-seed-amendment/1'
        and amendment['protocol']==protocol_pin
        and amendment['original_split']==protocol['split_manifest']
        and amendment['replacement_split']==split_pin,'SEED_AMENDMENT_BINDING')
    old=base.pinned_json(amendment['original_split']);new=base.pinned_json(split_pin)
    require(set(old)==set(new) and all(old[k]==new[k] for k in old if k!='confirmation_training_seeds')
        and old['confirmation_training_seeds']!=new['confirmation_training_seeds']
        and new['final']['seed_realization_generated']is False,'SEED_ONLY_AMENDMENT')
    from .robustness_training_plan import seeds
    seeds(new['confirmation_training_seeds'],3)
    require(isinstance(amendment['justification'],str) and bool(amendment['justification'].strip())
        and isinstance(amendment['evidence_pins'],list) and bool(amendment['evidence_pins']),
        'SEED_AMENDMENT_EVIDENCE')
    for evidence in amendment['evidence_pins']:validate_pin(evidence)
    payload={k:v for k,v in amendment.items() if k!='independent_review'}
    base.accepted_review(amendment['independent_review'],'PASS_R6_CONFIRMATION_SEED_AMENDMENT',
        payload_sha256=base.object_sha(payload))
    return amendment['original_split']


def freeze_payload(recipe_pin,recipe_review,selection_evidence, *, local=False):
    """Exact input for the independent recipe freeze review; never acceptance."""
    from . import robustness_training_plan as training
    recipe=training.validate_recipe(base.pinned_json(recipe_pin)) if local else training.recipe(recipe_pin,recipe_review)
    validate_selection_evidence(selection_evidence,recipe_pin,recipe)
    return dict(schema_version='r6-recipe-freeze/1',recipe=recipe_pin,
        confirmation_seeds=recipe['confirmation_seeds'],selection_evidence=selection_evidence)


def build_plans(freeze_pin,recipe_review,baseline_bindings,experiment_ids,wall_seconds,storage, *, local=False):
    """Three full common-origin plans, ordered by frozen seeds. No reservation."""
    from . import robustness_training_plan as training
    from .native_volume import MAXIMUM
    from .result_volume import CONTROL_ALLOWANCE
    freeze=base.pinned_json(freeze_pin)
    expected=freeze_payload(freeze['recipe'],recipe_review,freeze['selection_evidence'],local=local)
    require(set(freeze)==set(expected)|{'independent_review'} and all(freeze[k]==v for k,v in expected.items()),'EXACT_FREEZE')
    base.accepted_review(freeze['independent_review'],'PASS_R6_RECIPE_FREEZE',payload_sha256=base.object_sha(expected))
    recipe=training.validate_recipe(base.pinned_json(freeze['recipe'])) if local else training.recipe(freeze['recipe'],recipe_review)
    baseline=base.baseline(dict(baseline_bindings=baseline_bindings))
    require(isinstance(experiment_ids,list) and len(experiment_ids)==3 and len(set(experiment_ids))==3,'THREE_EXPERIMENTS')
    initial=dict(kind='COMMON_U640_NEW_STREAM',checkpoint=baseline['selected_model'],
        source_identity=baseline['source_identity'],method_hashes=baseline['method_hashes'])
    plans=[];expansions=[]
    for experiment,seed in zip(experiment_ids,freeze['confirmation_seeds']):
        plan=dict(schema_version='r6-training-plan/1',experiment_id=experiment,phase='confirmation',
            recipe=freeze['recipe'],recipe_review=recipe_review,selection_freeze=freeze_pin,training_seed=seed,
            start_update=640,target_update=640+recipe['branch_updates'],
            checkpoint_updates=[640+n for n in recipe['checkpoint_offsets']],initialization=dict(initial),
            wall_seconds=wall_seconds,storage=dict(storage))
        if local:
            plan['schema_version']='r6-local-training-plan/1'
            plan.pop('recipe_review')
        expanded=training.expanded_budget(plan,physics_decimation=20)
        plans.append(plan);expansions.append(expanded)
    totals={k:sum(e['budgets'][k] for e in expansions) for k in expansions[0]['budgets']}
    aggregate=storage['payload_bytes']+96*1024**2+MAXIMUM+CONTROL_ALLOWANCE
    charged=dict(live_starts=3,simulator_initializations=3,
        training_transitions=totals['training_transitions'],training_checkpoint_writes=totals['checkpoint_writes'],
        nontraining_transitions=0,live_wall_seconds=totals['wall_seconds']+3*190,
        allocated_gpu_seconds=totals['wall_seconds']+3*190,
        output_actual_plus_unreleased_reservations_bytes=3*aggregate)
    return dict(schema_version='r6-three-confirmation-plans/1',freeze=freeze_pin,baseline_bindings=baseline_bindings,
        plans=plans,expanded_budgets=expansions,total_workload_budgets=totals,training_job_reservations=charged,
        scope='THREE_TRAINING_JOBS_ONLY_FINAL_EVALUATION_DELIVERY_AND_REPAIR_REQUIRE_SEPARATE_BUDGET',
        scientific_acceptance=False,dispatch_authorized=False,holdout_released=False)
