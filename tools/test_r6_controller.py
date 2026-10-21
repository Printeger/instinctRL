"""Controller cursor recovery with actual profile worker/allocation/materializer.

The first/next live runner and source/final scientific work are explicit stubs.
The profile fixture's frozen-source verifier is mocked. No GPU or paid model.
"""
import json
from unittest.mock import Mock
import pytest

from tools import r6_controller as c
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import Journal,RecoveryError,file_pin


@pytest.fixture
def controller(case,monkeypatch):
    b=case
    b.initial=b.write('initial.json',dict(plan_pin=b.context,master_pin=b.context,review_executable=file_pin(b.builder)))
    b.request=b.write('controller-request.json',dict(initial_request=b.initial,profile_context=b.context))
    b.first=Mock(return_value=dict(status='DECISION_PROPOSAL_RETURNED',event_key=b.key))
    monkeypatch.setattr(c.first,'start',b.first)
    b.source=Mock(return_value=dict(preparation={'fixture':'source'},snapshot={'fixture':'snapshot'}))
    monkeypatch.setattr(c.source,'prepare_source',b.source)
    b.next=Mock();monkeypatch.setattr(c.next_entry,'continue_job',b.next)
    return b


def run(b):
    with b.j.writer():return c.run(b.j,b.request,b.store,free_disk_bytes=500*1024**3)


def entry(b):
    return next(e for e in b.j.read()['events'].values() if e['identity']['role']==c.ROLE)


def test_actual_profile_and_materializer_wait_reuses_worker_and_budget(controller):
    b=controller;result=run(b);budget=b.j.read()['budget'];before=b.store.replay()
    assert result['status']=='WAITING_FOR_ADMIN_PROVISION' and not result['goal_complete']
    assert entry(b)['cursor']['phase']=='MATERIALIZE'
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    b.j=Journal(b.j.directory)
    assert run(b)==result and b.j.read()['budget']==budget and b.store.replay()==before
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    b.first.assert_called_once();b.next.assert_not_called();b.source.assert_not_called()


def test_crash_after_profile_receipt_before_cursor_advance_reuses_event(controller,monkeypatch):
    b=controller;original=c.save;crashed=[]
    def save(j,key,cursor,**kw):
        if cursor['phase']=='MATERIALIZE' and not crashed:
            crashed.append(True);raise RuntimeError('CPU cursor crash')
        return original(j,key,cursor,**kw)
    monkeypatch.setattr(c,'save',save)
    with pytest.raises(RuntimeError,match='cursor crash'):run(b)
    assert entry(b)['cursor']['phase']=='PROFILE'
    assert entry(b)['status']=='INCOMPLETE_EXCEPTION'
    assert run(b)['status']=='WAITING_FOR_ADMIN_PROVISION'
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    b.first.assert_called_once()


def test_intent_before_initial_cursor_can_recover(controller,monkeypatch):
    b=controller;original=b.j.save
    def save(state):
        if any(e.get('cursor') for e in state['events'].values()):raise RuntimeError('CPU before cursor')
        return original(state)
    monkeypatch.setattr(b.j,'save',save)
    with pytest.raises(RuntimeError,match='before cursor'):run(b)
    monkeypatch.setattr(b.j,'save',original)
    assert run(b)['status']=='WAITING_FOR_ADMIN_PROVISION'
    assert len([e for e in b.j.read()['events'].values() if e['identity']['role']==c.ROLE])==1


def test_missing_controller_reservation_blocks_resume(controller):
    b=controller;run(b)
    with b.j.writer():
        s=b.j.read();s['budget']['reservations']=[r for r in s['budget']['reservations'] if r['identity']['role']!=c.ROLE];b.j.save(s)
    with pytest.raises(RecoveryError,match='RESERVATION'):run(b)
    b.first.assert_called_once()


def test_changed_request_cannot_create_second_controller(controller):
    b=controller;run(b)
    b.request=b.write('other-request.json',dict(initial_request=b.initial,profile_context=b.initial))
    with pytest.raises(RecoveryError,match='INPUT_DRIFT'):run(b)
    assert len([e for e in b.j.read()['events'].values() if e['identity']['role']==c.ROLE])==1


def test_changed_implementation_blocks_resume(controller,monkeypatch):
    b=controller;run(b);monkeypatch.setattr(c,'implementation',lambda:{'different':True})
    with pytest.raises(RecoveryError,match='INPUT_DRIFT'):run(b)


def test_source_needed_has_one_worker_then_preserved_compatibility_boundary(controller):
    b=controller;b.builder.with_suffix('.settings').write_text('{"source":true}')
    result=run(b)
    assert result['status']=='SOURCE_COMPATIBILITY_REQUIRED'
    assert run(b)==result
    b.source.assert_called_once();b.next.assert_not_called()
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    assert not any(e['identity']['role']==c.profile.storage.ROLE for e in b.j.read()['events'].values())


def test_candidate_goes_directly_to_source_without_extra_profile_worker(controller,monkeypatch):
    b=controller;monkeypatch.setattr(c,'parent_for_job',lambda *a:(b.parent,{'decision':{'decision':'PROPOSE_CANDIDATE'}}))
    assert run(b)['status']=='SOURCE_COMPATIBILITY_REQUIRED'
    assert not b.builder.with_suffix('.count').exists()
    b.source.assert_called_once()


def test_final_request_is_incomplete_delivery_boundary_never_goal_complete(controller,monkeypatch):
    b=controller;monkeypatch.setattr(c,'parent_for_job',lambda *a:(b.parent,{'decision':{'decision':'REQUEST_FINAL_REVIEW'}}))
    result=run(b);assert result['status']=='FINAL_DELIVERY_REQUIRED' and result['goal_complete'] is False
    assert run(b)==result
    assert not b.builder.with_suffix('.count').exists()
    b.source.assert_not_called();b.next.assert_not_called()


def test_stop_after_preparation_blocks_materialization_without_recalling_model(controller):
    b=controller;run(b);(b.j.directory/'STOP').touch()
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    b.next.assert_not_called()


def test_existing_first_phase_delegates_stop_cleanup_to_original_runner(controller):
    b=controller;b.first.side_effect=RuntimeError('existing job in flight')
    with pytest.raises(RuntimeError):run(b)
    (b.j.directory/'STOP').touch()
    with pytest.raises(RuntimeError):run(b)
    assert b.first.call_count==2 and entry(b)['cursor']['phase']=='FIRST_JOB'
    assert not b.builder.with_suffix('.count').exists()


def test_ready_materializer_routes_to_next_entry_then_next_job_decision(controller,monkeypatch):
    b=controller
    monkeypatch.setattr(c.materializer,'materialize',Mock(return_value=dict(
        status='EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED',artifacts={'plan':b.context})))
    b.next.return_value=dict(status='DECISION_PROPOSAL_RETURNED',child_job_event_key='second-job')
    original_job=b.j.canonical_job
    monkeypatch.setattr(b.j,'canonical_job',lambda key,**kw:original_job(b.key if key=='second-job' else key,**kw))
    original_parent=c.parent_for_job
    monkeypatch.setattr(c,'parent_for_job',lambda j,key,s:
        ('second-parent',{'decision':{'decision':'REQUEST_FINAL_REVIEW'}}) if key=='second-job' else original_parent(j,key,s))
    assert run(b)['status']=='FINAL_DELIVERY_REQUIRED'
    assert entry(b)['cursor']['completed_jobs']==[b.key,'second-job']
    b.next.assert_called_once()
    assert b.next.call_args.args[1:5]==(b.parent,b.context,b.context,file_pin(b.builder))
    run(b);b.first.assert_called_once();b.next.assert_called_once()


def test_single_writer_rejects_second_journal(controller):
    b=controller
    with b.j.writer():
        with pytest.raises(BlockingIOError):
            with Journal(b.j.directory).writer():pass


@pytest.mark.parametrize('stop',[False,True])
def test_detached_process_wait_has_no_new_models_and_enforces_writer(controller,tmp_path,stop):
    """Actual detached controller process; only live wait/frozen verifier substituted."""
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path
    from rlflow.jobs import owns
    from tools.r6_job_handoff import no_model_workers
    b=controller
    fixture=b.write('detached-fixture.json',dict(request=b.request,store_state=b.store.replay(),
        batch=str(b.j.directory),root=str(b.root),job_key=b.key))
    ready=b.root/'wait-ready';release=b.root/'wait-release'
    script=r'''
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
from tools import r6_controller as c
from tools.r6_recovery import Journal,RecoveryError
v=json.loads(Path(sys.argv[1]).read_text());root=Path(v['root']);j=Journal(Path(v['batch']))
c.profile.verify_frozen=lambda *a:None
from tools import r6_analysis_binding
# Inherit the parent fixture's missing native cleanup receipt substitute.
r6_analysis_binding.cleanup_evidence=lambda *a:({},dict(status='FAILED',exit_code=-9))
c.profile.storage.native_tool.BASE=root/'native-volumes'
c.profile.storage.result_tool.BASE=root/'result-volumes'
c.profile.storage.result_tool.ATTEMPTS=root/'attempts'
def wait(*a,**kw):
 (root/'wait-ready').touch()
 deadline=time.monotonic()+10
 while not (root/'wait-release').exists():
  if (j.directory/'STOP').exists():raise RecoveryError('CPU owned wait STOP')
  if time.monotonic()>deadline:raise RuntimeError('CPU bounded fixture timeout')
  time.sleep(.02)
 return dict(status='DECISION_PROPOSAL_RETURNED',event_key=v['job_key'])
c.first.start=wait
# Real profile process, with a short test-only poll; production remains 15 s.
from tools.r6_batch import model_handoff
c.profile.model_handoff=lambda j,k:model_handoff(j,k,poll_seconds=.01)
with j.writer():
 result=c.run(j,v['request'],SimpleNamespace(root=root,replay=lambda:v['store_state']),free_disk_bytes=500*1024**3)
print(json.dumps(result))
'''
    before_models=len([e for e in b.j.read()['events'].values() if e['kind']=='model'])
    proc=subprocess.Popen([sys.executable,'-B','-c',script,fixture['path']],cwd=c.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
                 PYTHONPATH='.:tools/rlflow:isaaclab-training'),start_new_session=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+8
        while not ready.exists():
            assert proc.poll() is None
            assert time.monotonic()<deadline
            time.sleep(.02)
        assert os.getsid(proc.pid)==proc.pid
        assert owns(entry(b)['controller_process'])
        no_model_workers(b.j)
        assert len([e for e in b.j.read()['events'].values() if e['kind']=='model'])==before_models
        assert not b.builder.with_suffix('.count').exists()
        with pytest.raises(BlockingIOError):
            with Journal(b.j.directory).writer():pass
        if stop:(b.j.directory/'STOP').touch()
        else:release.touch()
        stdout,stderr=proc.communicate(timeout=20)
        if stop:
            assert proc.returncode!=0 and 'CPU owned wait STOP' in stderr
            assert entry(b)['status']=='INCOMPLETE_EXCEPTION'
            assert not b.builder.with_suffix('.count').exists()
        else:
            assert proc.returncode==0,stderr
            assert json.loads(stdout)['status']=='WAITING_FOR_ADMIN_PROVISION'
            assert b.builder.with_suffix('.count').read_text()=='call\n'
        assert not owns(entry(b)['controller_process'])
        with Journal(b.j.directory).writer():pass
    finally:
        if proc.poll() is None:proc.terminate();proc.wait(timeout=5)
