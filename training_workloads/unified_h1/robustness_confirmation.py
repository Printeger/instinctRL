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
