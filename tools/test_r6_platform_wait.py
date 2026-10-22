"""Synthetic native session JSONL and actual Journal/files; no model or GPU."""
import hashlib,json
from datetime import datetime,timezone
from unittest.mock import Mock
import pytest
from tools import r6_platform_wait as platform,r6_delivery_evidence as delivery_module
from tools.test_r6_delivery_evidence import delivery,batch,run
from tools.r6_recovery import RecoveryError


def lines(events):
    return b''.join((json.dumps(dict(ordinal=i,**e))+'\n').encode() for i,e in enumerate(events))


def event(kind,when,turn='turn-1'):
    payload=dict(type=kind,turn_id=turn,root_turn_id=turn)
    payload['started_at' if kind=='task_started' else 'completed_at']=when
    return dict(type='event_msg',timestamp=datetime.fromtimestamp(when,timezone.utc).isoformat().replace('+00:00','Z'),payload=payload)


@pytest.fixture
def traced(delivery,tmp_path):
    b=delivery;b.trace=tmp_path/'platform-session.jsonl'
    b.rows=[dict(type='session_meta',payload=dict(id='CPU-ROOT',cwd=str(tmp_path))),event('task_started',10),event('task_complete',20)]
    data=lines(b.rows);b.trace.write_bytes(data)
    source=dict(session_id='CPU-ROOT',repository=str(tmp_path),anchor=dict(path=str(b.trace),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    with b.j.writer():
        state=b.j.read();state['platform_session_fixture']=source
        for key in b.keys:
            state['events'][key]['local_wait_audit']=dict(schema_version='r6-local-wait-audit/1',event_key=key,
                job_id=state['events'][key]['identity']['job_id'],scope='REGISTERED_LOCAL_MODEL_WORKERS_ONLY',
                complete=True,unobserved_restart_gap=False,segments=[dict(started_at=30,last_checked_at=40,finished_at=40,
                    checks=3,registered_model_workers_present=False,interrupted=False)])
        b.j.save(state)
    return b


def capture(b):
    with b.j.writer():
        for key in b.keys:platform.capture(b.j,key)
    return delivery_module.model_audit(b.j.read())


def test_idle_native_trace_without_flush_boundary_stays_unknown(traced):
    b=traced;out=capture(b);budget=b.j.read()['budget']
    assert not out['platform_inactivity_proven'] and out['pure_wait_model_calls']=='UNKNOWN'
    assert capture(b)==out and b.j.read()['budget']==budget
    result=run(b)
    assert 'PLATFORM_ZERO_MODEL_WAIT_AUDIT' in result['pending_requirements']
    saved=delivery_module.read(result['artifacts']['MODEL_USAGE_AUDIT.json'])
    assert not saved['platform_inactivity_proven'] and saved['pure_wait_model_calls']=='UNKNOWN'


def test_new_root_activity_during_wait_is_preserved_not_zero(traced):
    b=traced;b.rows.extend([event('task_started',35,'wake'),event('task_complete',45,'wake')]);b.trace.write_bytes(lines(b.rows))
    out=capture(b)
    assert not out['platform_inactivity_proven'] and out['pure_wait_model_calls']=='UNKNOWN'
    assert all(w['platform_evidence']['overlapping_root_turns']==['wake'] for w in out['waits'])


def test_root_turn_started_before_wait_is_not_missed(traced):
    b=traced;b.rows.append(event('task_started',25,'open'));b.trace.write_bytes(lines(b.rows))
    assert capture(b)['pure_wait_model_calls']=='UNKNOWN'
    with b.j.writer(),pytest.raises(RecoveryError,match='ROOT_TURN_ACTIVE'):platform.preflight(b.j)


def test_later_normal_turn_preserves_old_proof_but_delayed_activity_does_not(traced):
    b=traced;before=capture(b)
    b.rows.extend([event('task_started',50,'later'),event('task_complete',60,'later')]);b.trace.write_bytes(lines(b.rows))
    assert delivery_module.model_audit(b.j.read())==before
    b.rows=b.rows[:-2]+[event('task_started',35,'delayed'),event('task_complete',45,'delayed')];b.trace.write_bytes(lines(b.rows))
    with pytest.raises(RecoveryError,match='LATE_ACTIVITY_DRIFT'):delivery_module.model_audit(b.j.read())


@pytest.mark.parametrize('damage',['prefix','identity','partial','ordinal','unmatched','agents','gap','window','transplant','outside_turn'])
def test_incomplete_or_mismatched_evidence_cannot_claim_zero(traced,damage):
    b=traced
    if damage=='prefix':b.trace.write_bytes(b.trace.read_bytes().replace(b'CPU-ROOT',b'BAD-ROOT'))
    if damage=='identity':
        with b.j.writer():
            s=b.j.read();s['platform_session_fixture']['session_id']='wrong';b.j.save(s)
    if damage=='partial':b.trace.write_bytes(b.trace.read_bytes()+b'{')
    if damage=='ordinal':b.trace.write_bytes(b.trace.read_bytes()+b'{"ordinal":88,"type":"token_usage_record"}\n')
    if damage=='unmatched':b.rows.append(event('task_complete',45,'unknown'));b.trace.write_bytes(lines(b.rows))
    if damage=='agents':
        b.rows.extend([event('task_started',21,'agent-owner'),dict(type='response_item',payload=dict(type='function_call',name='collaboration.spawn_agent')),event('task_complete',22,'agent-owner')]);b.trace.write_bytes(lines(b.rows))
    if damage=='outside_turn':b.rows.append(dict(type='token_usage_record',payload={}));b.trace.write_bytes(lines(b.rows))
    if damage in ('gap','window'):
        with b.j.writer():
            s=b.j.read();a=s['events'][b.keys[0]]['local_wait_audit']
            if damage=='gap':a['unobserved_restart_gap']=True
            else:a['segments'][0]['last_checked_at']=29
            b.j.save(s)
    if damage=='transplant':
        capture(b)
        with b.j.writer():
            s=b.j.read();s['events'][b.keys[1]]['platform_wait_evidence']=s['events'][b.keys[0]]['platform_wait_evidence'];b.j.save(s)
    with pytest.raises(RecoveryError):capture(b)


def test_missing_capture_remains_unknown(traced):
    out=delivery_module.model_audit(traced.j.read())
    assert not out['platform_inactivity_proven'] and out['pure_wait_model_calls']=='UNKNOWN'


def test_crash_after_capture_intent_recovers_without_extra_charge(traced,monkeypatch):
    b=traced;persist=platform.persist_exact
    monkeypatch.setattr(platform,'persist_exact',Mock(side_effect=RuntimeError('capture write crash')))
    with b.j.writer(),pytest.raises(RuntimeError,match='capture write crash'):platform.capture(b.j,b.keys[0])
    budget=b.j.read()['budget'];monkeypatch.setattr(platform,'persist_exact',persist)
    with b.j.writer():out=platform.capture(b.j,b.keys[0])
    assert not out['platform_inactivity_proven'] and b.j.read()['budget']==budget


def test_production_cannot_use_arbitrary_idle_session(traced):
    b=traced
    with b.j.writer():
        s=b.j.read();s['goal_id']='R6-ROBUSTNESS-AUTO-20261019';b.j.save(s)
        with pytest.raises(RecoveryError,match='SOURCE_REQUIRED'):platform.preflight(b.j)


def test_idle_preflight_does_not_mutate_state(traced):
    b=traced;before=b.j.read()
    with b.j.writer():platform.preflight(b.j)
    assert b.j.read()==before


from tools.test_r6_submit_store import bound


def test_active_platform_rejects_before_submission_claim_and_store_write(bound):
    from tools.r6_submit import submit_once
    b=bound;p=b.journal.directory/'session.jsonl'
    rows=[dict(type='session_meta',payload=dict(id='CPU-ROOT',cwd=str(b.journal.directory))),event('task_started',10)]
    data=lines(rows);p.write_bytes(data)
    with b.journal.writer():
        state=b.journal.read();state['platform_session_fixture']=dict(session_id='CPU-ROOT',repository=str(b.journal.directory),
            anchor=dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));b.journal.save(state)
        before=b.journal.read();store=b.store.replay()
        with pytest.raises(RecoveryError,match='ROOT_TURN_ACTIVE'):
            submit_once(b.journal,b.key,b.store,queue_key=b.queue_key)
        assert b.journal.read()==before and b.store.replay()==store
    b.popen.assert_not_called()


def test_disjoint_segments_cannot_hide_inference_in_gap(traced):
    b=traced;a=b.j.read()['events'][b.keys[0]]['local_wait_audit']
    template=a['segments'][0]
    a['segments']=[dict(template,started_at=100,last_checked_at=110,finished_at=110),
                   dict(template,started_at=190,last_checked_at=200,finished_at=200)]
    b.rows.extend([event('task_started',120,'gap'),event('task_complete',180,'gap')])
    observed=platform.scan(lines(b.rows),platform.binding(b.j.read()))
    result=platform.assess(a,observed,210,dict(started_at=100,finished_at=200))
    assert not result['job_wait_coverage_complete']
    assert not result['platform_inactivity_proven']
    assert result['overlapping_root_turns']==['gap']


def test_first_observation_after_submit_is_not_complete_coverage(traced):
    a=traced.j.read()['events'][traced.keys[0]]['local_wait_audit']
    observed=platform.scan(lines(traced.rows),platform.binding(traced.j.read()))
    result=platform.assess(a,observed,50,dict(started_at=25,finished_at=40))
    assert not result['job_wait_coverage_complete']
    result=platform.assess(a,observed,50,dict(started_at=30,finished_at=40))
    assert result['job_wait_coverage_complete']
    assert not result['trace_completeness_proven']
    assert not result['platform_inactivity_proven']


def test_capture_binds_pre_submit_boundary_and_exact_closure(traced):
    from rlflow.store import sha
    b=traced;key=b.keys[0]
    with b.j.writer():
        s=b.j.read();job=s['events'][key]
        job['submission_claim']=dict(job_id=job['identity']['job_id'],wait_coverage_start=30)
        job['closure']=b.write('bound-closure.json',dict(event_key=key,job_id=job['identity']['job_id'],
            request_sha256=sha(job['request'])))
        b.j.save(s);result=platform.capture(b.j,key)
        assert result['job_wait_coverage_complete']
        assert not result['platform_inactivity_proven']
        s=b.j.read();s['events'][key]['submission_claim']['wait_coverage_start']=25;b.j.save(s)
        with pytest.raises(RecoveryError,match='COVERAGE_DRIFT'):platform.verify(b.j.read(),key)
