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


def validate_selection_evidence(pin,recipe_pin,recipe):
    value=base.pinned_json(pin)
    require(set(value)=={'schema_version','protocol','split_manifest','protocol_review','paired_evaluations','checkpoint_selection'}
        and value['schema_version']=='r6-recipe-selection-evidence/1','SELECTION_EVIDENCE')
    protocol=base.pinned_json(value['protocol']);split=base.pinned_json(value['split_manifest'])
    require(protocol.get('frozen')is True and split['schema_version']=='r6-split-manifest/1'
        and split['final']['seed_realization_generated']is False,'UNREVEALED_SPLIT')
    review=base.accepted_review(value['protocol_review'],'PASS_R6_PROTOCOL',protocol_sha256=value['protocol']['sha256'])
    require(all(p in review['evidence_pins'] for p in (value['protocol'],value['split_manifest'])),'REVIEWED_SPLIT')
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
        and trained['training_plan']['recipe']==recipe_pin and trained['training_plan']['phase']=='development','PAIRED_DESIGN')
    return dict(evidence=value,scientific_acceptance=False,holdout_released=False)


def freeze_payload(recipe_pin,recipe_review,selection_evidence):
    """Exact input for the independent recipe freeze review; never acceptance."""
    from . import robustness_training_plan as training
    recipe=training.recipe(recipe_pin,recipe_review)
    validate_selection_evidence(selection_evidence,recipe_pin,recipe)
    return dict(schema_version='r6-recipe-freeze/1',recipe=recipe_pin,
        confirmation_seeds=recipe['confirmation_seeds'],selection_evidence=selection_evidence)


def build_plans(freeze_pin,recipe_review,baseline_bindings,experiment_ids,wall_seconds,storage):
    """Three full common-origin plans, ordered by frozen seeds. No reservation."""
    from . import robustness_training_plan as training
    from .native_volume import MAXIMUM
    from .result_volume import CONTROL_ALLOWANCE
    freeze=base.pinned_json(freeze_pin)
    expected=freeze_payload(freeze['recipe'],recipe_review,freeze['selection_evidence'])
    require(set(freeze)==set(expected)|{'independent_review'} and all(freeze[k]==v for k,v in expected.items()),'EXACT_FREEZE')
    base.accepted_review(freeze['independent_review'],'PASS_R6_RECIPE_FREEZE',payload_sha256=base.object_sha(expected))
    recipe=training.recipe(freeze['recipe'],recipe_review)
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
