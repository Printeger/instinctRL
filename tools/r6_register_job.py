"""Reserve one already-authorized exact evaluation; never submit or issue permission.

Call after resource admission, immediately before submit. The first deadline is
immutable: recovery returns the same event, including an expired deadline.
"""
import json
import time

from rlflow import experiments
from rlflow.store import sha
from tools.r6_recovery import RecoveryError, checked
from tools.r6_job_handoff import no_model_workers
from tools.r6_postrun import stop_paths


def require(condition, reason):
    if not condition:raise RecoveryError('REGISTER_JOB_'+reason)


def register(journal, spec_pin, store, *, free_disk_bytes, purpose='development'):
    journal.require_writer()
    state=journal.read();spec=json.loads(checked(spec_pin).read_text())
    jid='J-'+sha({'id':spec['id'],'spec':spec})[:16]
    existing=[(k,e) for k,e in state['events'].items()
              if e['kind']=='job' and e['identity']['job_id']==jid]
    if existing:
        require(len(existing)==1,'ALIAS')
        key,entry=existing[0];request=entry['request'];identity=entry['identity']
        expected=dict(goal_id=state['goal_id'],event_id='JOB:'+jid,job_id=jid,
                      input_sha256=sha(request),role='job')
        require(identity==expected and sha(identity)==key and entry['purpose']==purpose
                and set(request)=={'spec','wait_deadline_unix'} and request['spec']==spec_pin,
                'REPLAY_DRIFT')
        reservation_key=json.dumps(tuple(identity[k] for k in sorted(identity)),separators=(',',':'))
        reservations=[r for r in state['budget']['reservations']
                      if r['identity']==identity or r['event_key']==reservation_key]
        require(state['budget']['goal_id']==state['goal_id'] and len(reservations)==1
            and reservations[0]['identity']==identity and reservations[0]['event_key']==reservation_key
            and reservations[0]['resources']==entry['resources'] and reservations[0]['purpose']==purpose,
            'RESERVATION_DRIFT')
        return key
    journal.guard();require(not stop_paths(journal),'STOP')
    require(state['budget']['capacity_pass'] is True,'CAPACITY_NOT_ACCEPTED')
    require(state.get('current_job') is None,'JOB_ACTIVE')
    no_model_workers(journal)
    require(jid not in store.replay()['jobs'],'EXISTING_STORE_JOB')
    experiments.validate(spec,store.root)
    experiments.authorized(store.replay(),spec,store.root)
    resources=dict(live_starts=1,live_wall_seconds=spec['resources']['wall_seconds']+180,
                   allocated_gpu_seconds=spec['resources']['wall_seconds']+180)
    if spec['backend']=='real':
        require(spec.get('h1',{}).get('mode')=='research-r6-evaluation','R6_MODE_REQUIRED')
        auth=json.loads(checked(spec['pins']['authorization']).read_text())
        resources.update(simulator_initializations=1,
            nontraining_transitions=spec['resources']['transitions'],
            output_actual_plus_unreleased_reservations_bytes=auth['parameters']['output_cap_bytes'])
    else:
        require(spec['backend']=='runner_cpu' and state['goal_id']=='R6-CPU-TEST-FIXTURE','CPU_FIXTURE_BOUNDARY')
        resources['output_actual_plus_unreleased_reservations_bytes']=1024**2
    require(not stop_paths(journal),'STOP')
    request=dict(spec=spec_pin,wait_deadline_unix=time.time()+spec['resources']['wall_seconds'])
    event=dict(goal_id=state['goal_id'],event_id='JOB:'+jid,job_id=jid,input_sha256=sha(request),role='job')
    key,_=journal.intent(event,kind='job',request=request,resources=resources,
        purpose=purpose,free_disk_bytes=free_disk_bytes)
    return key
