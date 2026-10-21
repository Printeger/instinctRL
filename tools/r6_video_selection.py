"""Deterministic representative video selections with original evidence bindings.

No renderer, simulator, model invocation, trajectory synthesis or dispatch.
Successful and failed conditions remain visibly distinct; full clips are selected.
"""
import json
from pathlib import Path
from tools.r6_recovery import checked, RecoveryError
from tools import r6_final_collection as collection


def require(value,reason):
    if not value:raise RecoveryError('VIDEO_SELECTION_'+reason)


def read(pin):return json.loads(checked(pin).read_text())


def select(final,reports,protocol):
    """First lane in protocol order, target before stress; pair with baseline."""
    rows=final['results'];require(len(rows)==len(reports) and len(rows) in (3,4),'COMPLETE_BATCH')
    blocks=protocol['blocks'];order=sorted(range(len(blocks)),key=lambda i:(blocks[i]['domain']!='target',i))
    clips={};missing=[]
    def add(index,block_index,lane,reason):
        block=blocks[block_index];key=(index,block['id'],lane)
        if key not in clips:
            scored=reports[index]['blocks'][block_index]['scoring']['lanes'][lane]
            clips[key]=dict(assignment_index=index,block_id=block['id'],domain=block['domain'],lane=lane,
                motion_pass=scored['motion_pass'],frame_count=block['steps'],first_policy_step=0,
                last_policy_step=block['steps']-1,selection_reasons=[])
        if reason not in clips[key]['selection_reasons']:clips[key]['selection_reasons'].append(reason)
    for i,(row,report) in enumerate(zip(rows,reports)):
        require(row['assignment_index']==i and report['final']['assignment']==row['assignment']
            and collection.gates(report,protocol)==row['gates'],'REPORT_BINDING')
        for passed in (True,False):
            found=None
            for b in order:
                for lane,score in enumerate(report['blocks'][b]['scoring']['lanes']):
                    if score['motion_pass']is passed:found=(b,lane);break
                if found is not None:break
            label='representative_success' if passed else 'representative_failure'
            if found is None:missing.append(dict(assignment_index=i,category=label,reason='NO_SUCH_OUTCOME_IN_COMPLETE_REGISTERED_BLOCKS'))
            else:add(i,*found,label)
    if len(rows)==4:
        require(rows[0]['assignment']['model_role']=='baseline'
            and [r['assignment']['model_role'] for r in rows[1:]]==['seed1','seed2','seed3'],'MODEL_ORDER')
        for clip in list(clips.values()):
            i=clip['assignment_index']
            if i:
                require(rows[i]['assignment']['evaluation_seed']==rows[0]['assignment']['evaluation_seed'],'PAIRED_SEED')
                b=next(i for i,block in enumerate(blocks) if block['id']==clip['block_id'])
                add(0,b,clip['lane'],'paired_baseline_for_assignment_'+str(i))
    require(bool(clips) and len(clips)<=14,'CLIP_CAP')
    return dict(clips=list(clips.values()),unavailable_categories=missing,
        selection_rule='FIRST_PROTOCOL_BLOCK_AND_LANE_TARGET_BEFORE_STRESS_PER_OUTCOME_PLUS_EXACT_BASELINE_PAIRS',
        full_trajectories=True,physics_reintegrated=False,scientific_acceptance=False)


def prepare(journal,batch_pin,job_keys,store):
    """Bind every selected source to the closed-job inventory and analyzed bytes."""
    final=collection.collect(journal,batch_pin,job_keys,store)
    batch=read(batch_pin);protocol=read(batch['protocol']);reports=[read(r['report']) for r in final['results']]
    plan=select(final,reports,protocol)
    cached={}
    for clip in plan['clips']:
        index=clip['assignment_index'];result=final['results'][index];bound=read(result['analysis'])
        auth=read(bound['authorization']);spec=read(bound['spec']);inventory=read(bound['inventory'])
        root=Path(auth['attempt'])/'results'/auth['mode']/'r6'/clip['block_id']
        pins={}
        for name in ('audit.bin','motion_traces.json','result.json'):
            path=str(root/name);matches=[r for r in inventory['files'] if r['path']==path]
            require(len(matches)==1,'RAW_INVENTORY')
            record=matches[0];pin={k:record[k] for k in ('path','sha256')}
            require(any(all(p.get(k)==record[k] for k in ('path','sha256','bytes')) for p in reports[index]['inputs']),'ANALYZED_RAW')
            if path not in cached:
                require(checked(pin).stat().st_size==record['bytes'],'RAW_BYTES');cached[path]=pin
            require(cached[path]==pin,'RAW_DRIFT');pins[name]=pin
        summary=read(pins['result.json'])
        require(summary['block_id']==clip['block_id'] and summary['run_id']==reports[index]['run_id']
            and summary['seed']==result['assignment']['evaluation_seed'] and summary['split']=='final'
            and type(summary['audit_records'])is int and summary['audit_records']>0,'RAW_IDENTITY')
        clip.update(job_event_key=result['job_event_key'],job_id=result['job_id'],assignment=result['assignment'],
            checkpoint=result['assignment']['checkpoint'],source_spec=bound['spec'],authorization=bound['authorization'],
            inventory=bound['inventory'],report=result['report'],source_snapshot=spec['cwd'],
            run_id=summary['run_id'],audit_records=summary['audit_records'],raw=pins)
    return dict(schema_version='r6-video-selection/1',status='VIDEO_SOURCES_SELECTED_NOT_RENDERED',batch=batch_pin,
        **plan,total_frames=sum(c['frame_count'] for c in plan['clips']),fps=10,dispatch_authorized=False,
        limitations=['Representative clips do not replace full-lane metrics.','Replay requires original recorded positions and orientations; no new dynamics or policy inference.',
            'No video, render resource reservation or scientific acceptance is produced by selection.'])
