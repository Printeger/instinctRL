"""Collect every registered final assignment from closed, replayed originals.

No selection, dispatch or scientific acceptance. Complete negative results are
retained for independent review rather than treated as infrastructure failures.
"""
import json
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_analysis_binding import bind_analysis
from tools.r6_holdout_publication import validate_batch
from training_workloads.unified_h1.robustness import lane_role


def require(value,reason):
    if not value:raise RecoveryError('FINAL_COLLECTION_'+reason)


def gates(report,protocol):
    require(report['split']=='final' and report['scientific_acceptance']is False,'FINAL_REPORT')
    blocks=protocol['blocks'];rows=report['blocks']
    require(len(blocks)==len(rows)==16 and [r['block_id'] for r in rows]==[b['id'] for b in blocks]
        and len({b['id'] for b in blocks})==16,'FULL_BLOCK_SET')
    outcomes=[]
    for block,row in zip(blocks,rows):
        require(block['domain'] in ('target','stress') and row['domain']==block['domain']
            and block['lanes']==row['total']==32,'DOMAIN_DENOMINATOR')
        require(row['replay_complete']is True and row['saved_result_equal']is True
            and row['measurement_replay']['bit_exact_delivery']is True
            and row['physics_replay']['staging_reconstructed']is True,'COMPLETE_REPLAY')
        score=row['scoring'];lanes=score['lanes']
        require(score['block_id']==block['id'] and score['domain']==block['domain'] and score['total']==32
            and [r['lane'] for r in lanes]==list(range(32))
            and all(type(r['motion_pass'])is bool for r in lanes),'LANES')
        groups={}
        for lane in lanes:
            name=lane_role(block,lane['lane'])[0];require(lane['subgroup']==name,'SUBGROUP')
            groups.setdefault(name,[]).append(lane['motion_pass'])
        require(set(score['subgroups'])==set(groups),'SUBGROUP_SET')
        for name,values in groups.items():
            group=score['subgroups'][name]
            require(group['total']==len(values) and group['successes']==sum(values),'SUBGROUP_COUNTS')
        successes=sum(r['motion_pass'] for r in lanes)
        require(type(score['successes'])is int and score['successes']==successes
            and score['motion_pass']is (successes==32),'BLOCK_COUNTS')
        outcomes.append(dict(block_id=block['id'],domain=block['domain'],successes=successes,total=32,passed=successes==32))
    original=report['summary']['original'];wall=original['wall_motion']
    require(all(original[k]is True for k in ('replay_complete','saved_result_equal','saved_wall_equal'))
        and type(original['all_gates_pass'])is bool and len(wall['per_lane'])==32
        and all(type(x)is bool for x in wall['per_lane']) and wall['all32']is all(wall['per_lane']),'ORIGINAL_COMPLETE')
    targets=[r for r in outcomes if r['domain']=='target'];require(bool(targets),'TARGETS')
    return dict(blocks=outcomes,original_pass=original['all_gates_pass'] and wall['all32'],
        target_pass=all(r['passed'] for r in targets))


def collect(journal,batch_pin,job_keys,store):
    batch=validate_batch(journal,batch_pin)
    require(isinstance(job_keys,list) and len(job_keys)==len(batch['assignments'])
        and len(set(job_keys))==len(job_keys),'ALL_ASSIGNMENTS_ONCE')
    protocol=json.loads(checked(batch['protocol']).read_text());results=[];jobs=set()
    for index,key in enumerate(job_keys):
        entry=journal.canonical_job(key)
        require(entry['purpose']=='confirmation_delivery','PURPOSE')
        analysis=json.loads(checked(entry['analysis']).read_text())
        require(analysis==bind_analysis(entry,key,analysis['report'],store) and not analysis['issues'],'CLOSED_REPLAY')
        auth=json.loads(checked(analysis['authorization']).read_text())
        expected=dict(schema_version='r6-final-selection/1',batch=batch_pin,assignment_index=index)
        require(auth['mode']=='research-r6-final-evaluation' and auth['parameters']['final_selection']==expected,'ASSIGNMENT_AUTHORITY')
        report=json.loads(checked(analysis['report']).read_text());ident=report['final']
        assignment=batch['assignments'][index]
        require(ident['selection']==expected and ident['assignment']==assignment
            and ident['checkpoint']==assignment['checkpoint'] and report['seed']==assignment['evaluation_seed']
            and analysis['protocol']==batch['protocol'],'MODEL_REALIZATION')
        require(analysis['job_id'] not in jobs,'DISTINCT_JOBS');jobs.add(analysis['job_id'])
        results.append(dict(assignment_index=index,assignment=assignment,job_event_key=key,job_id=analysis['job_id'],
            analysis=entry['analysis'],report=analysis['report'],gates=gates(report,protocol),
            native_status=analysis['native_status'],native_exit_code=analysis['native_exit_code']))
    required_results=results if all(r['assignment']['model_role']=='baseline' for r in results) else [
        r for r in results if r['assignment']['model_role']!='baseline']
    return dict(schema_version='r6-final-collection/1',batch=batch_pin,results=results,collector=file_pin(__file__),
        required_assignments=[r['assignment_index'] for r in required_results],
        required_models_pass=all(r['gates']['original_pass'] and r['gates']['target_pass'] for r in required_results),
        all_models_pass=all(r['gates']['original_pass'] and r['gates']['target_pass'] for r in results),
        status='COMPLETE_FINAL_EVIDENCE_REQUIRES_INDEPENDENT_AUDIT',scientific_acceptance=False,dispatch_authorized=False)
