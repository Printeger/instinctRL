"""Freeze unchanged u640 for three final environment streams, without training.

Development success permits this prospective branch; it is not final acceptance.
The independent freeze reviewer must still audit physical and random execution.
"""
import json
from rlflow.store import sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_analysis_binding import bind_analysis
from tools.r6_decision import record as parent_decision
from tools.r6_prepare_authorization import verify_frozen
from tools.r6_budget import totals
from training_workloads.unified_h1 import robustness_contract as base
from training_workloads.unified_h1.robustness import lane_role


def require(value,reason):
    if not value:raise RecoveryError('BASELINE_VALIDATION_'+reason)


def development_pass(report,protocol):
    """Check all frozen target lanes/subgroups; keep stress failures reportable."""
    require(report['split']=='development' and report['scientific_acceptance']is False,'DEVELOPMENT_REPORT')
    blocks=protocol['blocks'];rows=report['blocks']
    require(len(blocks)==len(rows)==16 and [r['block_id'] for r in rows]==[b['id'] for b in blocks]
        and len({b['id'] for b in blocks})==16,'FULL_BLOCK_SET')
    targets=0
    for block,row in zip(blocks,rows):
        require(block['domain'] in ('target','stress') and row['domain']==block['domain']
            and block['lanes']==row['total']==32,'DOMAIN_DENOMINATOR')
        require(row['replay_complete']is True and row['saved_result_equal']is True
            and row['measurement_replay']['bit_exact_delivery']is True
            and row['physics_replay']['staging_reconstructed']is True,'COMPLETE_REPLAY')
        score=row['scoring'];lanes=score['lanes']
        require(score['block_id']==block['id'] and score['domain']==block['domain'] and score['total']==32
            and [r['lane'] for r in lanes]==list(range(32))
            and all(type(r['motion_pass'])is bool for r in lanes),'SCORE_LANES')
        groups={}
        for lane in lanes:
            name=lane_role(block,lane['lane'])[0]
            require(lane['subgroup']==name,'LANE_SUBGROUP')
            groups.setdefault(name,[]).append(lane['motion_pass'])
        require(set(score['subgroups'])==set(groups),'SUBGROUP_SET')
        for name,values in groups.items():
            group=score['subgroups'][name]
            require(group['total']==len(values) and group['successes']==sum(values),'SUBGROUP_COUNTS')
        successes=sum(r['motion_pass'] for r in lanes)
        require(type(score['successes'])is int and score['successes']==successes
            and score['motion_pass']is (successes==32),'BLOCK_COUNTS')
        if block['domain']=='target':
            targets+=1;require(successes==32,'TARGET_NOT_PASSED')
    require(targets>0,'NO_TARGETS')
    original=report['summary']['original']
    require(all(original[k]is True for k in ('replay_complete','saved_result_equal','saved_wall_equal','all_gates_pass'))
        and original['wall_motion']['all32']is True and original['wall_motion']['per_lane']==[True]*32,'ORIGINAL_NOT_PASSED')


def describe(journal,controller_key,store):
    journal.require_writer();state=journal.read();cursor=state['events'][controller_key]['cursor']
    request=cursor['baseline_validation']
    require(set(request)=={'parent_key','job_key','context'} and 'confirmation' not in cursor,'EXCLUSIVE_BRANCH')
    require(totals(state['budget'])['training_transitions']==0,'ZERO_TRAINING_BRANCH')
    parent=parent_decision(journal,request['parent_key'],store)
    require(parent['decision']['decision']=='PROPOSE_BASELINE_VALIDATION'
        and parent['job_event_key']==request['job_key'],'PARENT_DECISION')
    job=journal.canonical_job(request['job_key']);bound=json.loads(checked(job['analysis']).read_text())
    require(bound==bind_analysis(job,request['job_key'],bound['report'],store)
        and bound['schema_version']=='r6-bound-motion-analysis/1' and not bound['issues'],'EXACT_BASELINE_ANALYSIS')
    spec=json.loads(checked(bound['spec']).read_text());auth=json.loads(checked(bound['authorization']).read_text())
    require(spec['h1']['mode']==auth['mode']=='research-r6-evaluation','UNMODIFIED_BASELINE_MODE')
    p=auth['parameters'];baseline=base.baseline(p)
    require(not {'evaluation_selection','r6_training_plan'}.intersection(p),'BASELINE_ONLY')
    protocol_pin=p['r6']['protocol'];protocol=json.loads(checked(protocol_pin).read_text())
    require(protocol.get('frozen')is True and bound['protocol']==protocol_pin,'FROZEN_PROTOCOL')
    split_pin=protocol['split_manifest'];split=json.loads(checked(split_pin).read_text())
    review=base.accepted_review(p['protocol_review'],'PASS_R6_PROTOCOL',protocol_sha256=protocol_pin['sha256'])
    require(all(pin in review['evidence_pins'] for pin in (protocol_pin,split_pin)),'REVIEWED_SPLIT')
    require(split['schema_version']=='r6-split-manifest/1' and split['final']['seed_realization_generated']is False
        and p['r6']['split']=='development' and p['r6']['seed']==split['development']['seed'],'DEVELOPMENT_STREAM')
    report=json.loads(checked(bound['report']).read_text());development_pass(report,protocol)
    binding=json.loads(checked(bound['model_binding']).read_text())
    require(binding['checkpoint']==baseline['selected_model'] and binding['checkpoint_source_identity']==baseline['source_identity']
        and binding['method_hashes']==baseline['method_hashes'] and binding['ics_clearance_margin_m']==.15
        and binding['actor_input_extension']is False,'EXACT_BASELINE_METHOD')
    context=json.loads(checked(request['context']).read_text());snapshot_pin=context['snapshot_receipt']
    snapshot=json.loads(checked(snapshot_pin).read_text())
    require(spec['h1']['workload_root']==snapshot['snapshot']
        and spec['h1']['config_sha256']==snapshot['config']['sha256'],'EVALUATED_SNAPSHOT')
    verify_frozen(snapshot,json.loads(checked(snapshot['config']).read_text()))
    return dict(schema_version='r6-final-freeze/2',branch='BASELINE_SUFFICIENT',protocol=protocol_pin,
        recipe=p['baseline_bindings'],models={'baseline':baseline['selected_model']},
        selection_rule=dict(kind='unchanged_baseline',update=640,training_transitions=0,
            final_realization_indices=[0,1,2],all_original_and_registered_blocks_each_stream=True,
            select_using_holdout=False),
        evidence=dict(controller_key=controller_key,store_root=str(store.root.resolve()),
            parent_decision=state['events'][request['parent_key']]['decision_record'],baseline_analysis=job['analysis'],
            baseline_bindings=p['baseline_bindings'],split_manifest=split_pin,protocol_review=p['protocol_review'],
            snapshot=snapshot_pin,validator=file_pin(__file__),native_status=bound['native_status'],native_exit_code=bound['native_exit_code']))
