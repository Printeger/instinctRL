"""Reserve new dual-volume storage before manifests exist, then bind it once.

No mounting, Store permission or job submission. The full storage charge stays
on this event, including failures. Job submission/inventory integration must
validate this owner before using it; this module alone grants no budget credit.
"""
import json
import re
from pathlib import Path
from rlflow.store import sha
from tools.r6_recovery import RecoveryError,checked,file_pin
from tools.r6_postrun import stop_paths
from tools import provision_r6_native_volume as native_tool
from tools import provision_r6_fixed_result_volume as result_tool
from training_workloads.unified_h1.result_volume import CONTROL_ALLOWANCE

DISK='output_actual_plus_unreleased_reservations_bytes'
ROLE='job-storage-allocation'


def require(value,reason):
    if not value:raise RecoveryError('STORAGE_ALLOCATION_'+reason)


def plans(request):
    require(set(request)=={'experiment_id','native_plan','result_plan','tools'},'REQUEST_FIELDS')
    require(bool(re.fullmatch(r'R6-[A-Z0-9-]{1,90}',request['experiment_id'])),'EXPERIMENT_ID')
    require(request['tools']==[file_pin(native_tool.__file__),file_pin(result_tool.__file__)],'TOOL_DRIFT')
    native=json.loads(checked(request['native_plan']).read_text())
    result=json.loads(checked(request['result_plan']).read_text())
    require(bool(re.fullmatch(r'J-[0-9a-f]{16}',native['volume_id'])),'PRODUCTION_ALLOCATION_ID')
    require(native==native_tool.plan(native['volume_id']),'NATIVE_PLAN')
    require(result==result_tool.plan(result['volume_id'],result['target'],result['maximum_backing_bytes'],
        result['minimum_free_bytes'],result['minimum_free_inodes']),'RESULT_PLAN')
    require(result['volume_id']==native['volume_id'],'VOLUME_PAIR')
    return native,result,native['maximum_backing_bytes']+result['maximum_backing_bytes']+CONTROL_ALLOWANCE


def validate(journal,key):
    state=journal.read();entry=state['events'][key];request=entry['request']
    native,result,cap=plans(request)
    identity=dict(goal_id=state['goal_id'],event_id='STORAGE:'+native['volume_id'],
        job_id='STORAGE:'+native['volume_id'],input_sha256=sha(request),role=ROLE)
    require(entry['kind']=='cpu' and entry['identity']==identity and sha(identity)==key,'IDENTITY')
    require(entry['resources']=={DISK:cap},'CHARGE')
    reservations=[r for r in state['budget']['reservations'] if r['identity']==identity]
    require(len(reservations)==1 and reservations[0]['resources']==entry['resources']
        and reservations[0]['purpose']==entry['purpose']
        and reservations[0]['event_key']==json.dumps(tuple(identity[k] for k in sorted(identity)),separators=(',',':'))
        and state['budget']['goal_id']==state['goal_id'],'RESERVATION')
    return entry,native,result,cap


def register(journal,experiment_id,native_plan,result_plan,*,purpose,free_disk_bytes):
    journal.guard();require(not stop_paths(journal),'STOP')
    request=dict(experiment_id=experiment_id,native_plan=native_plan,result_plan=result_plan,
        tools=[file_pin(native_tool.__file__),file_pin(result_tool.__file__)])
    native,result,cap=plans(request);state=journal.read()
    event=dict(goal_id=state['goal_id'],event_id='STORAGE:'+native['volume_id'],
        job_id='STORAGE:'+native['volume_id'],input_sha256=sha(request),role=ROLE)
    key=sha(event)
    for old_key,e in state['events'].items():
        if e['identity']['role']!=ROLE:continue
        old,n,r,_=validate(journal,old_key)
        overlap=(n['image']==native['image'] or r['image']==result['image']
                 or r['target']==result['target'] or old['request']['experiment_id']==experiment_id)
        require(not overlap or (old_key==key and old['purpose']==purpose),'ALLOCATION_REUSE')
    if key in state['events']:return key
    require(all(not Path(p['image']).parent.exists() and not Path(p['image']).parent.is_symlink()
                for p in (native,result)),'PREEXISTING_VOLUME')
    key,_=journal.intent(event,kind='cpu',request=request,resources={DISK:cap},
        purpose=purpose,free_disk_bytes=free_disk_bytes)
    return key


def bind(journal,key,spec_pin):
    """Claim the prepaid storage for one exact spec, never issue permission."""
    journal.require_writer();entry,native,result,cap=validate(journal,key)
    spec=json.loads(checked(spec_pin).read_text())
    require(spec['backend']=='real' and spec.get('h1',{}).get('mode')=='research-r6-evaluation','MODE')
    params=json.loads(checked(spec['pins']['authorization']).read_text())['parameters']
    require(params['experiment_id']==entry['request']['experiment_id']
        and params['output_cap_bytes']==cap
        and Path(spec['h1']['attempt'])/'results'==Path(result['target']),'SPEC_BINDING')
    for label,plan,tool in [('native_volume',native,native_tool),('result_volume',result,result_tool)]:
        pin=params[label];manifest=checked(pin)
        require(manifest==Path(plan['image']).parent/'volume.json','MANIFEST_PATH')
        record=json.loads(manifest.read_text())
        require(all(record.get(k)==v for k,v in plan.items())
            and record.get('implementation_sha256')==file_pin(tool.__file__)['sha256']
            and record.get('status')=='MOUNT_CREATED_NOT_ACCEPTED','MANIFEST_PLAN')
    from training_workloads.unified_h1.native_volume import validate as native_validate
    from training_workloads.unified_h1.result_volume import validate as result_validate
    native_validate(params['native_volume'],minimum_free_bytes=0)
    result_validate(params['result_volume'],output_cap_bytes=cap,expected_target=result['target'],minimum_free_bytes=0)
    binding=dict(spec=spec_pin,job_id='J-'+sha({'id':spec['id'],'spec':spec})[:16])
    if entry.get('storage_binding'):
        require(entry['storage_binding']==binding,'ALREADY_BOUND');return binding
    journal.guard();require(not stop_paths(journal),'STOP')
    state=journal.read();state['events'][key]['storage_binding']=binding;journal.save(state)
    return binding


def for_spec(journal,spec_pin,*,purpose,bind_new=False):
    """Find the one storage owner; fixed layouts may not omit preallocation."""
    spec=json.loads(checked(spec_pin).read_text())
    if spec['backend']!='real':return None
    params=json.loads(checked(spec['pins']['authorization']).read_text())['parameters']
    matches=[]
    for key,event in journal.read()['events'].items():
        if event['identity']['role']!=ROLE:continue
        entry,native,result,cap=validate(journal,key)
        expected_native=str(Path(native['image']).parent/'volume.json')
        expected_result=str(Path(result['image']).parent/'volume.json')
        if (entry['request']['experiment_id']==params.get('experiment_id')
            or params.get('native_volume',{}).get('path')==expected_native
            or params.get('result_volume',{}).get('path')==expected_result
            or str(Path(spec.get('h1',{}).get('attempt',''))/'results')==result['target']):
            matches.append((key,entry))
    require(len(matches)<=1,'OWNER_ALIAS')
    if not matches:
        if params.get('result_volume'):
            record=json.loads(checked(params['result_volume']).read_text())
            require(record.get('layout')!='fixed-results-v1','MISSING_OWNER')
        return None
    key,entry=matches[0]
    require(entry['purpose']==purpose,'PURPOSE_DRIFT')
    require(bind_new or bool(entry.get('storage_binding')),'UNBOUND_OWNER')
    bind(journal,key,spec_pin)
    return validate(journal,key)[0]


def owner(journal,job):
    """Validate the full storage charge without changing the job's reservation."""
    entry=for_spec(journal,job['request']['spec'],purpose=job['purpose'])
    if entry is None:return job
    require(job['resources'].get(DISK)==0,'DUPLICATE_JOB_CHARGE')
    require(entry['storage_binding']['job_id']==job['identity']['job_id'],'OWNER_JOB_DRIFT')
    return entry
