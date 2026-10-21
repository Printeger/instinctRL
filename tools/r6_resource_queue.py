"""Durable R6 resource-window accounting; never query a GPU or launch a job.

A bounded collector feeds samples to this journal. Reuses the exact V2 numeric
validator; queue observations never substitute for fresh AppLauncher admission.
"""
import json
import time
from pathlib import Path

from rlflow.jobs import identity
from rlflow.store import atomic,canonical,sha
from tools.r6_recovery import RecoveryError,checked,file_pin
from tools.r6_postrun import stop_paths
from training_workloads.unified_h1 import resource_admission as v2

ROOT=Path(__file__).resolve().parents[1]
SLOT_SECONDS=180
BACKOFF_SECONDS=60
OUTPUT_BYTES=1024**2
POLICY=dict(schema_version='r6-resource-queue/1',window_policy=v2.POLICY,
    reserved_seconds_per_window=SLOT_SECONDS,rejection_backoff_seconds=BACKOFF_SECONDS,
    maximum_sample_age_seconds=8,observation_is_launch_permission=False,
    interrupted_window_reused=False,implicit_refund=False)
TERMINAL={'BUSY','OBSERVATION_PASS','INVALID_WINDOW','INTERRUPTED','EXPIRED','STOPPED'}


def now():return dict(boot_id=identity(__import__('os').getpid())['boot_id'],monotonic_ns=time.monotonic_ns())


def tick(entry):
    clock=now();start=entry['queue_clock']
    if clock['boot_id']!=start['boot_id'] or clock['monotonic_ns']<start['monotonic_ns']:
        raise RecoveryError('QUEUE_CLOCK_REQUIRES_RECONCILIATION')
    return clock,(clock['monotonic_ns']-start['monotonic_ns'])/1e9


def validate(journal,key,state=None):
    state=journal.read() if state is None else state;entry=state['events'][key];request=entry['request']
    if (entry['kind']!='cpu' or entry['identity']['role']!='resource-window' or sha(entry['identity'])!=key
            or entry['identity']['input_sha256']!=sha(request) or request['policy']!=POLICY):
        raise RecoveryError('QUEUE_EVENT_IDENTITY')
    if request['review'] is not None:checked(request['review'])
    spec=json.loads(checked(request['spec']).read_text())
    job_id='J-'+sha({'id':spec['id'],'spec':spec})[:16]
    if (entry['identity']['job_id']!=job_id or entry['identity']['goal_id']!=state['goal_id']
            or entry['identity']['event_id']!='RESOURCE_WINDOW:'+job_id+':'+str(request['ordinal'])
            or type(request['ordinal']) is not int or request['ordinal']<0):
        raise RecoveryError('QUEUE_EXACT_SPEC_IDENTITY')
    if request['validator']!=file_pin(Path(v2.__file__)) or request['implementation']!=file_pin(Path(__file__)):
        raise RecoveryError('QUEUE_IMPLEMENTATION_DRIFT')
    aliases=[e for e in state['events'].values() if e['identity']['role']=='resource-window'
        and e['identity']['job_id']==job_id and e['request']['ordinal']==request['ordinal']]
    if len(aliases)!=1:raise RecoveryError('QUEUE_WINDOW_ALIAS')
    expected={'queue_seconds':SLOT_SECONDS,'output_actual_plus_unreleased_reservations_bytes':OUTPUT_BYTES}
    if entry['resources']!=expected:raise RecoveryError('QUEUE_RESERVATION')
    return entry


def observation_fresh(journal,key):
    """Validate the original complete PASS record, then check its current age.

    A past PASS is never rewritten as a different scientific observation.
    False permits only bounded re-observation, never submission.
    """
    entry=validate(journal,key)
    record=json.loads(checked(entry['window_record']).read_text())
    if (entry['queue_status']!=record['status'] or record['status']!='OBSERVATION_PASS'
            or record['event_key']!=key or record['request_sha256']!=sha(entry['request'])
            or record['window']['samples']!=entry['samples']):
        raise RecoveryError('QUEUE_RECORD')
    v2.validate_window(record['window'])
    clock=now();last=record['window']['samples'][-1]['monotonic_ns']
    return (record['finished_clock']['boot_id']==clock['boot_id']
            and 0<=clock['monotonic_ns']-last<=POLICY['maximum_sample_age_seconds']*10**9)


def begin(journal,spec_pin,*,ordinal,free_disk_bytes,purpose='development',review=None):
    journal.guard()
    if stop_paths(journal):raise RecoveryError('STOP_BEFORE_QUEUE')
    state=journal.read();spec=json.loads(checked(spec_pin).read_text())
    if type(ordinal) is not int or ordinal<0:raise RecoveryError('QUEUE_ORDINAL')
    if state.get('current_job') is not None or state.get('current_model_event') is not None:
        raise RecoveryError('QUEUE_NOT_IDLE')
    from tools.r6_job_handoff import no_model_workers
    no_model_workers(journal)
    if spec.get('backend')=='real':
        if state['budget']['capacity_pass'] is not True:raise RecoveryError('QUEUE_CAPACITY_NOT_ACCEPTED')
        accepted=json.loads(checked(review).read_text()) if review else {}
        if (accepted.get('decision')!='PASS_R6_RESOURCE_QUEUE' or accepted.get('policy')!=POLICY
                or accepted.get('implementation')!=file_pin(Path(__file__))
                or accepted.get('validator')!=file_pin(Path(v2.__file__)) or accepted.get('must_fix')!=[]):
            raise RecoveryError('QUEUE_REVIEW_REQUIRED')
    elif spec.get('backend')!='runner_cpu' or state['goal_id']!='R6-CPU-TEST-FIXTURE':
        raise RecoveryError('QUEUE_CPU_FIXTURE_BOUNDARY')
    job_id='J-'+sha({'id':spec['id'],'spec':spec})[:16]
    prior=[(k,e) for k,e in state['events'].items() if e['identity']['role']=='resource-window'
        and e['identity']['job_id']==job_id]
    for k,e in prior:
        if e['request']['ordinal']==ordinal:
            validate(journal,k,state)
            if e['request']['spec']!=spec_pin or e['request']['review']!=review or e['purpose']!=purpose:raise RecoveryError('QUEUE_REPLAY_DRIFT')
            return k  # Resume evidence; never reset its original clock.
    if ordinal!=len(prior):raise RecoveryError('QUEUE_ORDINAL_GAP')
    for k,e in state['events'].items():
        if e['identity']['role']=='resource-window' and e.get('queue_status') not in TERMINAL:
            raise RecoveryError('QUEUE_WINDOW_STILL_OPEN')
        if e['kind']=='cpu' and e.get('dispatch_claim') and not e.get('receipt'):
            raise RecoveryError('QUEUE_CPU_WORKER_UNRESOLVED')
    if prior:
        latest_key,latest=max(prior,key=lambda pair:pair[1]['request']['ordinal'])
        stale_pass=latest['queue_status']=='OBSERVATION_PASS' and not observation_fresh(journal,latest_key)
        if latest['queue_status'] not in {'BUSY','INTERRUPTED','EXPIRED'} and not stale_pass:
            raise RecoveryError('QUEUE_PRIOR_REQUIRES_DECISION')
        clock=now();end=latest.get('finished_clock',latest['queue_clock'])
        if clock['boot_id']!=end['boot_id']:raise RecoveryError('QUEUE_BOOT_RECONCILIATION_REQUIRED')
        if clock['monotonic_ns']<end['monotonic_ns']+BACKOFF_SECONDS*10**9:raise RecoveryError('QUEUE_LOCAL_BACKOFF')
        account_wait(journal,latest_key)
        state=journal.read()
    request=dict(spec=spec_pin,ordinal=ordinal,policy=POLICY,review=review,
        validator=file_pin(Path(v2.__file__)),implementation=file_pin(Path(__file__)))
    event=dict(goal_id=state['goal_id'],event_id='RESOURCE_WINDOW:'+job_id+':'+str(ordinal),job_id=job_id,
        input_sha256=sha(request),role='resource-window')
    # Include clock with the same transaction as the reservation. A crash
    # cannot create a new timer for an already charged queue event.
    from tools.r6_budget import reserve
    key=sha(event);budget=reserve(state['budget'],event_key=event,
        resources={'queue_seconds':SLOT_SECONDS,'output_actual_plus_unreleased_reservations_bytes':OUTPUT_BYTES},
        purpose=purpose,free_disk_bytes=free_disk_bytes)
    state['budget']=budget
    state['events'][key]=dict(identity=event,kind='cpu',request=request,
        resources={'queue_seconds':SLOT_SECONDS,'output_actual_plus_unreleased_reservations_bytes':OUTPUT_BYTES},
        purpose=purpose,status='INTENT',process=None,receipt=None,queue_status='COLLECTING',queue_clock=now(),samples=[])
    journal.save(state);return key


def account_wait(journal,key):
    """Charge actual waiting, including controller downtime, before another slot.

    The future local collector/backoff loop must call this on each wake and on
    STOP. It is accounting only and is allowed after STOP; it queries nothing.
    """
    journal.require_writer();state=journal.read();entry=validate(journal,key,state)
    stale_pass=entry['queue_status']=='OBSERVATION_PASS' and not observation_fresh(journal,key)
    if entry['queue_status'] not in {'BUSY','INTERRUPTED','EXPIRED','STOPPED'} and not stale_pass:
        raise RecoveryError('QUEUE_NOT_WAITING')
    if entry.get('wait_closed'):
        return entry['wait_accounted_seconds']
    clock,elapsed=tick(entry)
    stopped=bool(stop_paths(journal))
    previous=entry.get('wait_accounted_seconds',max(SLOT_SECONDS,
        json.loads(checked(entry['window_record']).read_text())['observed_elapsed_seconds']))
    if elapsed<=previous:
        if stopped:
            entry.update(wait_closed=True,wait_accounted_seconds=previous,queue_status='STOPPED',wait_finished_clock=clock)
            journal.save(state)
        return previous
    records=entry.setdefault('wait_records',[])
    if len(records)>=256:raise RecoveryError('QUEUE_WAIT_RECORD_BOUND')
    value=dict(event_key=key,request_sha256=sha(entry['request']),started_clock=entry['queue_clock'],
        observed_clock=clock,queue_seconds=elapsed,includes_controller_downtime=True)
    path=journal.directory/'resource-queue'/key/('wait-'+str(clock['monotonic_ns'])+'.json')
    data=canonical(value)+b'\n'
    if path.exists():
        if path.read_bytes()!=data:raise RecoveryError('QUEUE_WAIT_RECORD_DRIFT')
    else:atomic(path,data)
    pin=file_pin(path)
    from tools.r6_budget import observe
    state['budget']=observe(state['budget'],event_key=entry['identity'],resources={'queue_seconds':elapsed},evidence=pin)
    records.append(pin);entry['wait_accounted_seconds']=elapsed
    if stopped:entry.update(wait_closed=True,queue_status='STOPPED',wait_finished_clock=clock)
    journal.save(state)
    return elapsed


def append_sample(journal,key,ordinal,sample):
    journal.guard()
    if stop_paths(journal):raise RecoveryError('STOP_DURING_QUEUE')
    state=journal.read();entry=validate(journal,key,state)
    if entry['queue_status']!='COLLECTING':raise RecoveryError('QUEUE_WINDOW_NOT_COLLECTING')
    clock,elapsed=tick(entry)
    if elapsed>=SLOT_SECONDS:raise RecoveryError('QUEUE_SLOT_EXPIRED')
    if type(ordinal) is not int or not 0<=ordinal<v2.POLICY['sample_count']:raise RecoveryError('QUEUE_SAMPLE_ORDINAL')
    samples=entry['samples']
    if ordinal<len(samples):
        if samples[ordinal]!=sample:raise RecoveryError('QUEUE_SAMPLE_REPLAY_DRIFT')
        return
    if ordinal!=len(samples):raise RecoveryError('QUEUE_SAMPLE_GAP')
    stamp=sample.get('monotonic_ns')
    if (type(stamp) is not int or stamp<entry['queue_clock']['monotonic_ns']
            or not 0<=clock['monotonic_ns']-stamp<=POLICY['maximum_sample_age_seconds']*10**9):
        raise RecoveryError('QUEUE_STALE_OR_FUTURE_SAMPLE')
    if len(canonical(sample))>4096:raise RecoveryError('QUEUE_SAMPLE_SIZE_BOUND')
    samples.append(sample);journal.save(state)


def finish(journal,key,*,interrupted=False):
    """Persist raw full/partial window and deterministic result without queries."""
    journal.require_writer();state=journal.read();entry=validate(journal,key,state)
    path=journal.directory/'resource-queue'/key/'window.json'
    if any(not c.get('receipt') for c in entry.get('sample_claims',{}).values()):
        raise RecoveryError('QUEUE_SAMPLE_CLAIM_UNRESOLVED')
    if entry.get('window_record'):
        return json.loads(checked(entry['window_record']).read_text())
    # Handle crash after artifact but before state commit without recalculating
    # elapsed time, changing the result or creating a fresh observation.
    if path.exists():
        result=json.loads(path.read_text())
        if (result['event_key']!=key or result['request_sha256']!=sha(entry['request'])
                or result['window']['samples']!=entry['samples']):raise RecoveryError('QUEUE_RECORD_DRIFT')
    else:
        clock,elapsed=tick(entry);samples=entry['samples'];error=None;statistics=None
        window=dict(sample_count=len(samples),span_ns=samples[-1]['monotonic_ns']-samples[0]['monotonic_ns'] if samples else 0,samples=samples)
        if stop_paths(journal):status='STOPPED'
        elif elapsed>=SLOT_SECONDS:status='EXPIRED'
        elif interrupted:status='INTERRUPTED'
        else:
            try:statistics=v2.validate_window(window)['statistics'];status='OBSERVATION_PASS'
            except (RuntimeError,MemoryError) as exc:status='BUSY';error=str(exc)
            except ValueError as exc:status='INVALID_WINDOW';error=str(exc)
        result=dict(schema_version='r6-resource-queue-window/1',event_key=key,request_sha256=sha(entry['request']),
            policy=POLICY,window=window,status=status,error=error,statistics=statistics,
            started_clock=entry['queue_clock'],finished_clock=clock,observed_elapsed_seconds=elapsed,
            charged_queue_seconds=SLOT_SECONDS,launch_admission=False,live_starts=0,simulator_initializations=0,model_calls=0)
        data=canonical(result)+b'\n'
        if len(data)>65536:raise RecoveryError('QUEUE_RECORD_SIZE_BOUND')
        atomic(path,data)
    entry['queue_status']=result['status'];entry['finished_clock']=result['finished_clock']
    entry['window_record']=file_pin(path);entry['status']='CPU_QUEUE_WINDOW_RECORDED'
    if result['status']=='STOPPED':
        entry.update(wait_closed=True,wait_finished_clock=result['finished_clock'],
            wait_accounted_seconds=max(SLOT_SECONDS,result['observed_elapsed_seconds']))
    # Record an actual overrun conservatively; it can never become a refund.
    if result['observed_elapsed_seconds']>SLOT_SECONDS:
        from tools.r6_budget import observe
        state['budget']=observe(state['budget'],event_key=entry['identity'],
            resources={'queue_seconds':result['observed_elapsed_seconds']},evidence=entry['window_record'])
    journal.save(state);return result
