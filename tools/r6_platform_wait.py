"""Verify root-session turn boundaries against actual local job wait intervals.

This observes the platform log; it cannot disable automatic Goal continuation.
It never treats a missing log, open turn or partial capture as zero activity.
"""
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from rlflow.store import sha,canonical
from tools.r6_recovery import RecoveryError,checked,file_pin
from tools.r6_decision_packet import persist_exact

ROOT=Path(__file__).resolve().parents[1]
SESSION_ID='01a154cf-f2bb-79e3-8019-d6c78e8e0fc8'
SESSION=Path('/home/mint/.codex/sessions/2026/10/19/rollout-2026-10-19T23-37-56-01a154cf-f2bb-79e3-8019-d6c78e8e0fc8.jsonl')
CONTINUATION_ID='01a16275-c109-7302-84bf-4b916ea448e6'
CONTINUATION=Path('/home/mint/.codex/sessions/2026/10/22/rollout-2026-10-22T15-14-06-01a16275-c109-7302-84bf-4b916ea448e6.jsonl')
MAX_PREFIX=128*1024**2
CAP=256*1024


def require(ok,reason):
    if not ok:raise RecoveryError('PLATFORM_WAIT_'+reason)


def number(x):return type(x) in (int,float) and math.isfinite(x)


def enabled(state):
    return state.get('goal_id')!='R6-CPU-TEST-FIXTURE' or 'platform_session_fixture' in state


def binding(state):
    if state.get('goal_id')=='R6-CPU-TEST-FIXTURE':
        value=state.get('platform_session_fixture');require(isinstance(value,dict),'FIXTURE_REQUIRED')
        return value
    require(state.get('goal_id')=='R6-ROBUSTNESS-AUTO-20261019','GOAL')
    if state.get('platform_continuation_binding'):
        record=json.loads(checked(state['platform_continuation_binding']).read_text())
        require(record['previous_session_id']==SESSION_ID and record['session_id']==CONTINUATION_ID
            and record['repository']==str(ROOT) and record['anchor']['path']==str(CONTINUATION),
            'CONTINUATION_IDENTITY')
        checked(record['authority']);prefix_bytes(record['anchor'])
        return {k:record[k] for k in ('session_id','repository','anchor')}
    prefix=state.get('platform_automatic_wake_accounting',{}).get('source_prefix')
    require(isinstance(prefix,dict) and prefix.get('path')==str(SESSION),'SOURCE_REQUIRED')
    return dict(session_id=SESSION_ID,repository=str(ROOT),anchor=prefix)


def prefix_bytes(pin):
    require(set(pin)=={'path','bytes','sha256'} and type(pin['bytes']) is int
        and 0<pin['bytes']<=MAX_PREFIX,'PREFIX_FIELDS')
    p=Path(pin['path']);require(p.is_file() and not p.is_symlink() and p.resolve()==p,'SOURCE_PATH')
    with p.open('rb') as f:data=f.read(pin['bytes'])
    require(len(data)==pin['bytes'] and data.endswith(b'\n')
        and hashlib.sha256(data).hexdigest()==pin['sha256'],'PREFIX_DRIFT')
    return data


def scan(data,source):
    """Only the observed native ordered JSONL format is accepted."""
    turns=[];active=None;last_boundary=0;meta=None;seen=set();agents=False
    rows=data.splitlines();require(bool(rows),'EMPTY')
    for ordinal,line in enumerate(rows):
        require(len(line)<=8*1024**2,'RECORD_CAP')
        row=json.loads(line);require(type(row.get('ordinal')) is int and row['ordinal']==ordinal,'ORDINAL_GAP')
        payload=row.get('payload',{});kind=row.get('type')
        if kind=='session_meta':
            require(ordinal==0 and meta is None,'SESSION_METADATA');meta=payload
            require(meta.get('id')==source['session_id'] and meta.get('cwd')==source['repository'],'SESSION_IDENTITY')
        model_record=kind=='token_usage_record' or (kind=='response_item' and
            (payload.get('type') in ('reasoning','function_call','custom_tool_call') or payload.get('role')=='assistant'))
        require(not model_record or active is not None,'MODEL_ACTIVITY_WITHOUT_TURN')
        if kind=='response_item' and payload.get('type') in ('function_call','custom_tool_call'):
            if 'spawn_agent' in payload.get('name',''):agents=True
        if kind!='event_msg':continue
        event=payload.get('type')
        require(event!='turn_aborted','ABORTED_TURN_UNRESOLVED')
        if event not in ('task_started','task_complete'):continue
        timestamp=row.get('timestamp');require(isinstance(timestamp,str) and timestamp.endswith('Z'),'TIMESTAMP')
        when=datetime.fromisoformat(timestamp.replace('Z','+00:00')).timestamp()
        require(when>=last_boundary,'CLOCK_BACKWARDS');last_boundary=when
        turn=payload.get('turn_id');require(isinstance(turn,str) and bool(turn)
            and payload.get('root_turn_id')==turn,'ROOT_TURN')
        if event=='task_started':
            require(active is None and turn not in seen,'OVERLAPPING_OR_DUPLICATE_TURN')
            require(number(payload.get('started_at')) and abs(when-payload['started_at'])<2,'START_TIME')
            active=dict(turn_id=turn,started_at=when,finished_at=None);seen.add(turn);turns.append(active)
        else:
            require(active is not None and active['turn_id']==turn,'UNMATCHED_COMPLETION')
            require(number(payload.get('completed_at')) and abs(when-payload['completed_at'])<2,'END_TIME')
            active['finished_at']=when;active=None
    require(meta is not None and bool(turns),'NO_TURN_HISTORY')
    return dict(turns=turns,active_turn=active,external_agents_seen=agents,last_boundary=last_boundary)


def snapshot(state):
    source=binding(state);anchor=source['anchor'];prefix_bytes(anchor)
    path=Path(anchor['path'])
    with path.open('rb') as f:data=f.read(MAX_PREFIX+1)
    require(len(data)<=MAX_PREFIX,'SOURCE_CAP')
    # Concurrent partial writes are unknown, never evidence of inactivity.
    require(data.endswith(b'\n'),'PARTIAL_RECORD')
    require(len(data)>=anchor['bytes'] and hashlib.sha256(data[:anchor['bytes']]).hexdigest()==anchor['sha256'],'ANCHOR_DRIFT')
    pin=dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    return source,pin,scan(data,source)


def preflight(journal):
    """No charge or mutation. Called before queueing and exact submit effects."""
    journal.require_writer();state=journal.read()
    if not enabled(state):return
    _,_,observed=snapshot(state)
    require(observed['active_turn'] is None,'ROOT_TURN_ACTIVE_NO_DISPATCH')
    require(not observed['external_agents_seen'],'EXTERNAL_AGENT_COVERAGE_REQUIRED')
    # Avoid treating a just-ended turn as a stable handoff while the platform
    # immediately queues its next automatic continuation. This is a gate, not
    # a model polling loop; the local controller can retry after real handoff.
    require(time.time()-observed['last_boundary']>=2,'HANDOFF_NOT_SETTLED')
    from tools import r6_platform_rpc as rpc
    if rpc.enabled(state):
        record=rpc.observe(state);rpc.idle(state,record)
        return record


def assess(audit,observed,captured_at,coverage=None,*,rpc_proven=False):
    require(audit.get('schema_version')=='r6-local-wait-audit/1'
        and audit.get('scope')=='REGISTERED_LOCAL_MODEL_WORKERS_ONLY'
        and audit.get('complete') is True and audit.get('segments')
        and not audit.get('unobserved_restart_gap'),'LOCAL_WAIT_INCOMPLETE')
    windows=[];overlaps=set()
    for segment in audit['segments']:
        start=segment['started_at'];end=segment['last_checked_at']
        require(number(start) and number(end) and start<end<=captured_at
            and segment.get('finished_at')==end and segment['checks']>=2
            and segment['registered_model_workers_present'] is False
            and not segment['interrupted'],'LOCAL_WINDOW')
        require(any(t['finished_at'] is not None and t['finished_at']<=start for t in observed['turns']),'NO_PRIOR_COMPLETED_TURN')
        for t in observed['turns']:
            if t['started_at']<=end and (t['finished_at'] is None or t['finished_at']>=start):overlaps.add(t['turn_id'])
        windows.append(dict(started_at=start,finished_at=end))
    require(not observed['external_agents_seen'],'EXTERNAL_AGENT_COVERAGE_REQUIRED')
    # A collection of valid segments is not proof that the entire job was
    # observed. In particular, submission-before-observation and restart gaps
    # cannot disappear merely because the final waiter exited cleanly.
    covered=False
    if coverage is not None:
        start=coverage['started_at'];end=coverage['finished_at']
        require(number(start) and number(end) and start<end<=captured_at,'COVERAGE_INTERVAL')
        for t in observed['turns']:
            if t['started_at']<=end and (t['finished_at'] is None or t['finished_at']>=start):overlaps.add(t['turn_id'])
        cursor=start
        for window in windows:
            if window['started_at']>cursor:break
            cursor=max(cursor,window['finished_at'])
        else:covered=windows[0]['started_at']<=start and cursor>=end
    # Log age cannot establish freshness. Two live reads from the same pinned
    # server must bracket the complete interval with an unchanged completed
    # root turn and a paused Goal. Missing RPC evidence remains UNKNOWN.
    trace_complete=rpc_proven
    return dict(windows=windows,overlapping_root_turns=sorted(overlaps),
        job_wait_coverage_complete=covered,trace_completeness_proven=trace_complete,
        trace_completeness_reason='LIVE_APP_SERVER_UNCHANGED_COMPLETED_TURN' if rpc_proven
            else 'SUPPORTED_PLATFORM_FLUSH_OR_HEALTH_BOUNDARY_MISSING',
        platform_inactivity_proven=covered and trace_complete and not overlaps,
        scope='PINNED_ROOT_SESSION_AND_REGISTERED_LOCAL_WORKERS')


def coverage_for(job):
    claim=job.get('submission_claim',{});audit=job.get('local_wait_audit',{})
    if 'wait_coverage_start' not in claim or not job.get('closure') or not audit.get('complete'):
        return None
    closure=json.loads(checked(job['closure']).read_text())
    require(closure.get('event_key')==sha(job['identity'])
        and claim.get('job_id')==job['identity']['job_id']
        and closure.get('job_id')==job['identity']['job_id']
        and closure.get('request_sha256')==sha(job['request']),'CLOSURE_IDENTITY')
    return dict(started_at=claim['wait_coverage_start'],finished_at=audit['segments'][-1]['finished_at'],
        submission_claim_sha256=sha(claim),closure=job['closure'])


def capture(journal,key):
    """Persist raw-prefix identity after the exact waiter closes, before models."""
    journal.require_writer();state=journal.read()
    if not enabled(state):return
    job=journal.canonical_job(key,state=state)
    if job.get('platform_wait_evidence'):
        return verify(state,key)
    if not job.get('local_wait_audit'):return  # historical closure stays unknown
    require(job['local_wait_audit'].get('event_key')==key
        and job['local_wait_audit'].get('job_id')==job['identity']['job_id'],'WAIT_IDENTITY')
    from tools.r6_profile_preparation import reservation
    from tools.r6_job_handoff import no_model_workers
    journal.guard();no_model_workers(journal)
    name='PLATFORM_WAIT_CAPTURE:'+key
    previous=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(previous)<=1,'ALIAS')
    if previous:
        capture_key,entry=previous[0];request=entry['request']
        require(request['wait_audit']==job['local_wait_audit'],'WAIT_DRIFT')
    else:
        coverage=coverage_for(job)
        from tools import r6_platform_rpc as rpc
        rpc_start=job.get('submission_claim',{}).get('platform_wait_start');rpc_end=None;rpc_error=None
        if rpc_start and rpc.enabled(state):
            try:rpc_end=rpc.observe(state)
            except RecoveryError as error:rpc_error=str(error)
        source,pin,observed=snapshot(state);now=time.time()
        proven=rpc.pair_proves_inactivity(state,rpc_start,rpc_end,coverage,observed)
        result=assess(job['local_wait_audit'],observed,now,coverage,rpc_proven=proven)
        request=dict(job_key=key,job_id=job['identity']['job_id'],source=source,prefix=pin,
            captured_at=now,wait_audit=job['local_wait_audit'],coverage=coverage,
            platform_rpc_start=rpc_start,platform_rpc_end=rpc_end,platform_rpc_error=rpc_error,
            result=result,implementation=file_pin(__file__))
        require(len(canonical(request))<CAP//2,'CAP')
        event=dict(goal_id=state['goal_id'],event_id=name,job_id=job['identity']['job_id'],
            input_sha256=sha(request),role='platform-wait-capture')
        import shutil
        capture_key,_=journal.intent(event,kind='cpu',request=request,
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='confirmation_delivery',
            free_disk_bytes=shutil.disk_usage(journal.directory).free)
    reservation(journal.read(),capture_key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    path=journal.directory/'platform-wait'/capture_key/'evidence.json'
    persist_exact(path,canonical(dict(event_key=capture_key,**request))+b'\n')
    state=journal.read();state['events'][capture_key]['result']=file_pin(path)
    state['events'][key]['platform_wait_evidence']=file_pin(path);journal.save(state)
    return verify(state,key)


def verify(state,key):
    from tools.r6_profile_preparation import reservation
    job=state['events'][key];pin=job.get('platform_wait_evidence')
    if pin is None:return None
    record=json.loads(checked(pin).read_text());capture_key=record['event_key'];entry=state['events'][capture_key]
    require(entry.get('result')==pin and dict(event_key=capture_key,**entry['request'])==record,'RECORD')
    reservation(state,capture_key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    require(entry['identity']['role']=='platform-wait-capture' and entry['identity']['event_id']=='PLATFORM_WAIT_CAPTURE:'+key
        and record['job_key']==key and record['job_id']==job['identity']['job_id']
        and record['wait_audit']==job.get('local_wait_audit'),'JOB_BINDING')
    source=binding(state)
    # Later wake-accounting anchors may grow; the original capture's anchor
    # remains immutable and must identify this same root source.
    require(all(record['source'][n]==source[n] for n in ('session_id','repository'))
        and record['prefix']['path']==source['anchor']['path'],'SOURCE_BINDING')
    data=prefix_bytes(record['prefix']);anchor=record['source']['anchor'];prefix_bytes(anchor)
    require(anchor['path']==record['prefix']['path'] and len(data)>=anchor['bytes']
        and hashlib.sha256(data[:anchor['bytes']]).hexdigest()==anchor['sha256'],'ANCHOR_DRIFT')
    checked(record['implementation'])
    require(number(record['captured_at']) and record['captured_at']<=time.time(),'CAPTURE_TIME')
    require(record.get('coverage')==coverage_for(job),'COVERAGE_DRIFT')
    from tools import r6_platform_rpc as rpc
    require(record.get('platform_rpc_start')==job.get('submission_claim',{}).get('platform_wait_start'),'RPC_START_DRIFT')
    observed=scan(data,record['source'])
    proven=rpc.pair_proves_inactivity(state,record.get('platform_rpc_start'),record.get('platform_rpc_end'),record.get('coverage'),observed)
    result=assess(record['wait_audit'],observed,record['captured_at'],record.get('coverage'),rpc_proven=proven)
    require(result==record['result'],'RESULT_DRIFT')
    # A later append carrying delayed/backdated activity must invalidate an
    # earlier zero-call claim; an immutable prefix is not a license to hide it.
    _,_,current=snapshot(state)
    current_proof=rpc.pair_proves_inactivity(state,record.get('platform_rpc_start'),record.get('platform_rpc_end'),record.get('coverage'),current)
    require(assess(record['wait_audit'],current,record['captured_at'],record.get('coverage'),rpc_proven=current_proof)==result,'LATE_ACTIVITY_DRIFT')
    return dict(result,evidence=pin)
