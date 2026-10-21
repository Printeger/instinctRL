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
    snapshot=dict(event_key='fixture-snapshot',fixture='snapshot')
    with b.j.writer():
        state=b.j.read();state['events']['fixture-snapshot']=dict(kind='cpu',identity={'role':'source-snapshot'},receipt=b.write('source-receipt.json',snapshot));b.j.save(state)
    b.source=Mock(return_value=dict(preparation={'fixture':'source'},snapshot=snapshot))
    b.compat=Mock(return_value=dict(status='EXACT_COMPATIBILITY_ACCEPTED_CONTEXT_READY',context=b.context))
    monkeypatch.setattr(c.compatibility,'prepare',b.compat)
    monkeypatch.setattr(c.source,'prepare_source',b.source)
    b.next=Mock();monkeypatch.setattr(c.next_entry,'continue_job',b.next)
    # Final science/execution remains a stub in these cursor-composition tests.
    b.final_inputs=Mock(return_value=dict(status='FINAL_INPUTS_AND_SNAPSHOT_READY_NOT_REVIEWED',snapshot=b.context,inputs=b.context))
    monkeypatch.setattr(c.final_inputs,'prepare',b.final_inputs)
    def final_prepare(j,*args,**kw):
        value=dict(status='FINAL_BATCH_CONTEXTS_READY_NOT_RESERVED',event_key='fixture-final-preparation')
        state=j.read();state['events']['fixture-final-preparation']={'kind':'cpu','identity':{'role':'final-batch-preparation'},'result':b.write('final-prepared.json',value)};j.save(state)
        return value
    b.final_prepare=Mock(side_effect=final_prepare);monkeypatch.setattr(c.final_preparation,'prepare',b.final_prepare)
    b.final_materialize=Mock(return_value=dict(status='FINAL_BATCH_PLANS_READY_NOT_AUTHORIZED',assignments=[],dispatch_authorized=False))
    monkeypatch.setattr(c.final_storage,'materialize_all',b.final_materialize)
    b.final_review=Mock(return_value=dict(status='FINAL_BATCH_PLANS_REVIEWED_NOT_AUTHORIZED',event_key='fixture-final-review',review=b.context))
    monkeypatch.setattr(c.final_plans_review,'prepare',b.final_review)
    b.final_sequence=Mock(return_value=dict(status='FINAL_BATCH_COLLECTED_REQUIRES_INDEPENDENT_AUDIT',collection={'fixture':True}))
    monkeypatch.setattr(c.final_sequence,'advance',b.final_sequence)
    b.delivery=Mock(return_value=dict(status='DELIVERY_EVIDENCE_PREPARED_NOT_ACCEPTED',scientific_acceptance=False))
    monkeypatch.setattr(c.delivery_evidence,'prepare',b.delivery)
    b.video_register=Mock(return_value='fixture-video-worker');monkeypatch.setattr(c.video_extraction,'register',b.video_register)
    b.video_handoff=Mock(return_value={'status':'VIDEO_TRAJECTORIES_READY_NOT_RENDERED'});monkeypatch.setattr(c.video_extraction,'handoff',b.video_handoff)
    def render_prepare(j,*args,**kw):
        value=dict(status='RENDER_INPUTS_READY_NOT_REVIEWED',event_key='fixture-render-preparation')
        state=j.read();state['events']['fixture-render-preparation']={'kind':'cpu','identity':{'role':'render-input-preparation'},'preparation':b.write('render-prepared.json',value)};j.save(state)
        return value
    b.render_prepare=Mock(side_effect=render_prepare);monkeypatch.setattr(c.render_preparation,'prepare',b.render_prepare)
    b.render_review=Mock(return_value=dict(status='RENDER_PLAN_REVIEWED_NOT_AUTHORIZED',event_key='fixture-render-review',review=b.context))
    monkeypatch.setattr(c.render_review,'prepare',b.render_review)
    def render_materialize(j,*args,**kw):
        value=dict(status='RENDER_EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED',event_key='fixture-render-materialization',plan=b.write('render-plan.json',{}))
        state=j.read();state['events']['fixture-render-materialization']={'kind':'cpu','identity':{'role':'render-input-materialization'},'materialization':b.write('render-materialized.json',value)};j.save(state)
        return value
    b.render_materialize=Mock(side_effect=render_materialize);monkeypatch.setattr(c.render_materialization,'materialize',b.render_materialize)
    b.render_issue=Mock(return_value=dict(status='EXACT_NEXT_JOB_AUTHORIZED_NOT_SUBMITTED'))
    monkeypatch.setattr(c.render_materialization,'issue',b.render_issue)
    b.render_run=Mock(return_value=dict(status='RENDER_JOB_CLOSED_REQUIRES_VIDEO_VALIDATION',event_key='fixture-render-job'))
    monkeypatch.setattr(c.render_runner,'run',b.render_run)
    b.video_validation_register=Mock(return_value='fixture-video-validation');monkeypatch.setattr(c.video_validation,'register',b.video_validation_register)
    b.video_validation_handoff=Mock(return_value=dict(status='VIDEOS_CPU_VALIDATED_REQUIRES_VISUAL_AUDIT'));monkeypatch.setattr(c.video_validation,'handoff',b.video_validation_handoff)
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


def test_source_needed_does_not_loop_when_changed_source_still_cannot_support_phase(controller):
    b=controller;b.builder.with_suffix('.settings').write_text('{"source":true}')
    for _ in range(2):
        with pytest.raises(RecoveryError,match='SOURCE_CHANGE_DID_NOT_RESOLVE'):run(b)
    b.source.assert_called_once();b.compat.assert_called_once();b.next.assert_not_called()
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    assert not any(e['identity']['role']==c.profile.storage.ROLE for e in b.j.read()['events'].values())


def test_candidate_source_and_compatibility_precede_one_profile_worker(controller,monkeypatch):
    b=controller;monkeypatch.setattr(c,'parent_for_job',lambda *a:(b.parent,{'decision':{'decision':'PROPOSE_CANDIDATE'}}))
    assert run(b)['status']=='WAITING_FOR_ADMIN_PROVISION'
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    b.source.assert_called_once();b.compat.assert_called_once()


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


@pytest.mark.parametrize('candidate',[False,True])
def test_ready_materializer_routes_to_next_entry_then_next_job_decision(controller,monkeypatch,candidate):
    b=controller
    monkeypatch.setattr(c.materializer,'materialize',Mock(return_value=dict(
        status='EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED',artifacts={'plan':b.context})))
    b.next.return_value=dict(status='DECISION_PROPOSAL_RETURNED',child_job_event_key='second-job')
    original_job=b.j.canonical_job
    monkeypatch.setattr(b.j,'canonical_job',lambda key,**kw:original_job(b.key if key=='second-job' else key,**kw))
    original_parent=c.parent_for_job
    monkeypatch.setattr(c,'parent_for_job',lambda j,key,s:
        ('second-parent',{'decision':{'decision':'REQUEST_FINAL_REVIEW'}}) if key=='second-job' else ((b.parent,{'decision':{'decision':'PROPOSE_CANDIDATE'}}) if candidate else original_parent(j,key,s)))
    assert run(b)['status']=='FINAL_DELIVERY_REQUIRED'
    assert entry(b)['cursor']['completed_jobs']==[b.key,'second-job']
    assert 'source_preparation' not in entry(b)['cursor']
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


def test_training_source_routes_to_training_reviews_then_profile(controller,monkeypatch):
    b=controller;monkeypatch.setattr(c,'parent_for_job',lambda *a:(b.parent,{'decision':{'decision':'PROPOSE_CANDIDATE'}}))
    b.source.return_value['preparation']['training_design']=b.context
    training=Mock(return_value=dict(status='TRAINING_REVIEWS_ACCEPTED_CONTEXT_READY',context=b.context,dispatch_authorized=False))
    monkeypatch.setattr(c.training,'prepare',training)
    assert run(b)['status']=='WAITING_FOR_ADMIN_PROVISION'
    training.assert_called_once();b.compat.assert_not_called();b.source.assert_called_once()


def test_training_review_interruption_keeps_cursor_for_exact_recovery(controller,monkeypatch):
    b=controller;monkeypatch.setattr(c,'parent_for_job',lambda *a:(b.parent,{'decision':{'decision':'PROPOSE_CANDIDATE'}}))
    b.source.return_value['preparation']['training_design']=b.context
    training=Mock(side_effect=RuntimeError('training review in flight'));monkeypatch.setattr(c.training,'prepare',training)
    with pytest.raises(RuntimeError,match='training review in flight'):run(b)
    assert entry(b)['cursor']['phase']=='TRAINING_PREPARATION'
    training.side_effect=None;training.return_value=dict(status='TRAINING_REVIEWS_ACCEPTED_CONTEXT_READY',context=b.context)
    assert run(b)['status']=='WAITING_FOR_ADMIN_PROVISION'
    b.source.assert_called_once();b.compat.assert_not_called()

def test_candidate_source_routes_to_candidate_reviews_then_profile(controller,monkeypatch):
    b=controller;monkeypatch.setattr(c,'parent_for_job',lambda *a:(b.parent,{'decision':{'decision':'PROPOSE_CANDIDATE'}}))
    b.source.return_value['preparation']['candidate_design']=b.context
    candidate=Mock(return_value=dict(status='CANDIDATE_REVIEWS_ACCEPTED_CONTEXT_READY',context=b.context,dispatch_authorized=False))
    monkeypatch.setattr(c.candidate,'prepare',candidate)
    assert run(b)['status']=='WAITING_FOR_ADMIN_PROVISION'
    candidate.assert_called_once();b.compat.assert_not_called();b.source.assert_called_once()


def test_candidate_review_interruption_keeps_cursor_for_exact_recovery(controller,monkeypatch):
    b=controller;monkeypatch.setattr(c,'parent_for_job',lambda *a:(b.parent,{'decision':{'decision':'PROPOSE_CANDIDATE'}}))
    b.source.return_value['preparation']['candidate_design']=b.context
    candidate=Mock(side_effect=RuntimeError('candidate review in flight'));monkeypatch.setattr(c.candidate,'prepare',candidate)
    with pytest.raises(RuntimeError,match='candidate review in flight'):run(b)
    assert entry(b)['cursor']['phase']=='CANDIDATE_PREPARATION'
    candidate.side_effect=None;candidate.return_value=dict(status='CANDIDATE_REVIEWS_ACCEPTED_CONTEXT_READY',context=b.context)
    assert run(b)['status']=='WAITING_FOR_ADMIN_PROVISION'
    b.source.assert_called_once();b.compat.assert_not_called()


@pytest.mark.parametrize('crash',[False,True])
def test_three_confirmation_jobs_in_order_recover_collection_and_do_not_reselect(controller,monkeypatch,crash):
    b=controller
    contexts=[b.write(f'confirmed-{i}.json',{'fixture':i}) for i in range(3)]
    inputs=Mock(return_value=dict(status='CONFIRMATION_INPUTS_PREPARED_NOT_REVIEWED',inputs=b.context))
    preparation=Mock(return_value=dict(status='THREE_CONFIRMATION_CONTEXTS_READY_NOT_RESERVED',contexts=contexts))
    monkeypatch.setattr(c.confirmation_inputs,'prepare',inputs);monkeypatch.setattr(c.confirmation,'prepare',preparation)
    profile=Mock(return_value=dict(status='PROFILE_AND_STORAGE_PLANS_READY_NOT_PROVISIONED',inputs=b.context))
    monkeypatch.setattr(c.profile,'prepare_profile',profile)
    monkeypatch.setattr(c.materializer,'materialize',Mock(return_value=dict(status='EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED',artifacts={'plan':b.context})))
    monkeypatch.setattr(c,'parent_for_job',lambda j,key,s:(b.parent if key==b.key else key+'-parent',{'decision':{'decision':'PROPOSE_CONFIRMATION'}}))
    original_job=b.j.canonical_job
    monkeypatch.setattr(b.j,'canonical_job',lambda key,**kw:original_job(b.key,**kw))
    b.next.side_effect=[dict(status='DECISION_PROPOSAL_RETURNED',child_job_event_key='confirm-'+str(i)) for i in range(3)]
    collected=[]
    def collect(j,key,context,store):
        collected.append((key,context));return dict(job_event_key=key,context=context,scientific_acceptance=False)
    monkeypatch.setattr(c.confirmation,'collect',collect)
    final_models=Mock(return_value=dict(status='FINAL_MODELS_BOUND_REVIEW_REQUIRED',freeze=b.context))
    monkeypatch.setattr(c.final_models,'prepare',final_models)
    final_review=Mock(return_value=dict(status='FINAL_MODEL_FREEZE_REVIEWED_NOT_RELEASED',freeze=b.context,review=b.context))
    monkeypatch.setattr(c.final_freeze_review,'prepare',final_review)
    holdout=Mock(return_value=dict(status='HOLDOUT_RELEASED_FINAL_BATCH_INPUTS_READY',batch=b.context))
    monkeypatch.setattr(c.holdout_publication,'prepare',holdout)
    original_save=c.save;failed=[]
    def save(j,key,cursor,**kw):
        if crash and len(cursor.get('confirmation',{}).get('results',[]))==1 and not failed:
            failed.append(True);raise RuntimeError('collection cursor crash')
        return original_save(j,key,cursor,**kw)
    monkeypatch.setattr(c,'save',save)
    if crash:
        with pytest.raises(RuntimeError,match='collection cursor crash'):run(b)
        assert entry(b)['cursor']['phase']=='DECIDE'
    result=run(b);assert result['status']=='FINAL_VISUAL_AUDIT_REQUIRED' and result['goal_complete']is False
    assert [call.args[2] for call in profile.call_args_list]==contexts
    assert [call.args[1] for call in b.next.call_args_list]==[b.parent,'confirm-0-parent','confirm-1-parent']
    assert [r['job_event_key'] for r in entry(b)['cursor']['confirmation']['results']]==['confirm-0','confirm-1','confirm-2']
    assert run(b)==result and b.next.call_count==3
    inputs.assert_called_once();preparation.assert_called_once();b.source.assert_not_called()
    final_models.assert_called_once()
    final_review.assert_called_once()
    holdout.assert_called_once()


def test_confirmation_input_interruption_keeps_exact_phase(controller,monkeypatch):
    b=controller;monkeypatch.setattr(c,'parent_for_job',lambda *a:(b.parent,{'decision':{'decision':'PROPOSE_CONFIRMATION'}}))
    inputs=Mock(side_effect=RuntimeError('confirmation inputs in flight'))
    monkeypatch.setattr(c.confirmation_inputs,'prepare',inputs)
    for _ in range(2):
        with pytest.raises(RuntimeError,match='inputs in flight'):run(b)
    assert entry(b)['cursor']['phase']=='CONFIRMATION_INPUTS'
    b.first.assert_called_once();b.source.assert_not_called();b.next.assert_not_called()


def test_confirmation_repair_decision_cannot_launch_next_seed(controller,monkeypatch):
    b=controller
    monkeypatch.setattr(c,'parent_for_job',lambda j,key,s:(b.parent,{'decision':{'decision':
        'PROPOSE_CONFIRMATION' if key==b.key else 'NEEDS_REPAIR'}}))
    monkeypatch.setattr(c.confirmation_inputs,'prepare',Mock(return_value=dict(status='CONFIRMATION_INPUTS_PREPARED_NOT_REVIEWED',inputs=b.context)))
    monkeypatch.setattr(c.confirmation,'prepare',Mock(return_value=dict(status='THREE_CONFIRMATION_CONTEXTS_READY_NOT_RESERVED',contexts=[b.context]*3)))
    monkeypatch.setattr(c.profile,'prepare_profile',Mock(return_value=dict(status='PROFILE_AND_STORAGE_PLANS_READY_NOT_PROVISIONED',inputs=b.context)))
    monkeypatch.setattr(c.materializer,'materialize',Mock(return_value=dict(status='EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED',artifacts={'plan':b.context})))
    original=b.j.canonical_job;monkeypatch.setattr(b.j,'canonical_job',lambda key,**kw:original(b.key,**kw))
    b.next.return_value=dict(status='DECISION_PROPOSAL_RETURNED',child_job_event_key='failed-confirmation')
    collect=Mock();monkeypatch.setattr(c.confirmation,'collect',collect)
    for _ in range(2):
        with pytest.raises(RecoveryError,match='CONFIRMATION_REQUIRES_REPAIR'):run(b)
    assert entry(b)['cursor']['phase']=='DECIDE' and entry(b)['cursor']['confirmation']['results']==[]
    b.next.assert_called_once();collect.assert_not_called()


def test_baseline_validation_routes_to_freeze_review_without_training_preparation(controller,monkeypatch):
    b=controller;monkeypatch.setattr(c,'parent_for_job',lambda *a:(b.parent,{'decision':{'decision':'PROPOSE_BASELINE_VALIDATION'}}))
    freeze=Mock(return_value=dict(status='FINAL_MODELS_BOUND_REVIEW_REQUIRED',freeze=b.context))
    review=Mock(return_value=dict(status='FINAL_MODEL_FREEZE_REVIEWED_NOT_RELEASED',freeze=b.context,review=b.context))
    monkeypatch.setattr(c.final_models,'prepare',freeze);monkeypatch.setattr(c.final_freeze_review,'prepare',review)
    holdout=Mock(return_value=dict(status='HOLDOUT_RELEASED_FINAL_BATCH_INPUTS_READY',batch=b.context))
    monkeypatch.setattr(c.holdout_publication,'prepare',holdout)
    result=run(b);assert result['status']=='FINAL_VISUAL_AUDIT_REQUIRED' and not result['goal_complete']
    assert entry(b)['cursor']['baseline_validation']==dict(parent_key=b.parent,job_key=b.key,context=b.context)
    assert run(b)==result;freeze.assert_called_once();review.assert_called_once()
    holdout.assert_called_once()
    b.source.assert_not_called();b.next.assert_not_called();assert not b.builder.with_suffix('.count').exists()


def test_holdout_publication_interruption_does_not_repeat_final_review(controller,monkeypatch):
    b=controller;monkeypatch.setattr(c,'parent_for_job',lambda *a:(b.parent,{'decision':{'decision':'PROPOSE_BASELINE_VALIDATION'}}))
    freeze=Mock(return_value=dict(status='FINAL_MODELS_BOUND_REVIEW_REQUIRED',freeze=b.context))
    review=Mock(return_value=dict(status='FINAL_MODEL_FREEZE_REVIEWED_NOT_RELEASED',freeze=b.context,review=b.context))
    monkeypatch.setattr(c.final_models,'prepare',freeze);monkeypatch.setattr(c.final_freeze_review,'prepare',review)
    publication=Mock(side_effect=RuntimeError('publication interrupted'))
    monkeypatch.setattr(c.holdout_publication,'prepare',publication)
    for _ in range(2):
        with pytest.raises(RuntimeError,match='publication interrupted'):run(b)
    assert entry(b)['cursor']['phase']=='HOLDOUT_PREPARATION'
    publication.side_effect=None;publication.return_value=dict(status='HOLDOUT_RELEASED_FINAL_BATCH_INPUTS_READY',batch=b.context)
    assert run(b)['status']=='FINAL_VISUAL_AUDIT_REQUIRED'
    freeze.assert_called_once();review.assert_called_once();b.next.assert_not_called()


@pytest.mark.parametrize('phase',['FINAL_PREPARATION','FINAL_MATERIALIZE','FINAL_PLANS_REVIEW','FINAL_SEQUENCE'])
def test_final_route_restart_does_not_repeat_prior_phases(controller,monkeypatch,phase):
    b=controller
    with b.j.writer():
        key,_=c.register(b.j,b.request,b.store,free_disk_bytes=500*1024**3)
        state=b.j.read();state['events'][key]['cursor'].update(phase='FINAL_EVALUATION_PREPARATION_REQUIRED',final_batch={'batch':b.context});b.j.save(state)
    target={'FINAL_PREPARATION':b.final_prepare,'FINAL_MATERIALIZE':b.final_materialize,
        'FINAL_PLANS_REVIEW':b.final_review,'FINAL_SEQUENCE':b.final_sequence}[phase]
    original=target.side_effect;target.side_effect=RuntimeError('final interruption')
    with pytest.raises(RuntimeError,match='final interruption'):run(b)
    assert entry(b)['cursor']['phase']==phase
    target.side_effect=original
    assert run(b)['status']=='FINAL_VISUAL_AUDIT_REQUIRED'
    b.final_inputs.assert_called_once()
    assert b.final_sequence.call_count==(2 if phase=='FINAL_SEQUENCE' else 1)
    b.first.assert_not_called();b.next.assert_not_called()


def test_final_missing_volumes_preserves_phase_and_reports_all(controller):
    b=controller
    with b.j.writer():
        key,_=c.register(b.j,b.request,b.store,free_disk_bytes=500*1024**3)
        state=b.j.read();state['events'][key]['cursor'].update(phase='FINAL_EVALUATION_PREPARATION_REQUIRED',final_batch={'batch':b.context});b.j.save(state)
    good=b.final_materialize.return_value
    b.final_materialize.return_value=dict(status='WAITING_FOR_ADMIN_PROVISION',missing=[{'assignment_index':0,'missing':['native','result']}])
    out=run(b);assert out['missing']==b.final_materialize.return_value['missing']
    assert entry(b)['cursor']['phase']=='FINAL_MATERIALIZE'
    b.final_review.assert_not_called();b.final_sequence.assert_not_called()
    b.final_materialize.return_value=good
    assert run(b)['status']=='FINAL_VISUAL_AUDIT_REQUIRED'
    b.final_inputs.assert_called_once();b.final_prepare.assert_called_once()


def test_delivery_interruption_reuses_final_sequence(controller):
    b=controller
    with b.j.writer():
        key,_=c.register(b.j,b.request,b.store,free_disk_bytes=500*1024**3)
        state=b.j.read();state['events'][key]['cursor'].update(phase='FINAL_SEQUENCE',final_materializations=b.context,
            final_plans_review={'event_key':'fixture-review','review':b.context});b.j.save(state)
    b.delivery.side_effect=RuntimeError('delivery interruption')
    with pytest.raises(RuntimeError,match='delivery interruption'):run(b)
    assert entry(b)['cursor']['phase']=='FINAL_DELIVERY_REQUIRED'
    b.delivery.side_effect=None
    assert run(b)['status']=='FINAL_VISUAL_AUDIT_REQUIRED'
    b.final_sequence.assert_called_once()


def test_video_worker_interruption_reconnects_registered_event(controller):
    b=controller
    with b.j.writer():
        key,_=c.register(b.j,b.request,b.store,free_disk_bytes=500*1024**3)
        state=b.j.read();state['events'][key]['cursor'].update(phase='FINAL_MEDIA_REQUIRED');b.j.save(state)
    b.video_handoff.side_effect=RuntimeError('video worker interruption')
    with pytest.raises(RuntimeError,match='video worker interruption'):run(b)
    assert entry(b)['cursor']['phase']=='VIDEO_EXTRACTION'
    b.video_handoff.side_effect=None
    assert run(b)['status']=='FINAL_VISUAL_AUDIT_REQUIRED'
    b.video_register.assert_called_once();assert b.video_handoff.call_count==2


@pytest.mark.parametrize('phase',['FINAL_RENDER_REQUIRED','RENDER_REVIEW','RENDER_MATERIALIZE','RENDER_ISSUE','RENDER_RUN'])
def test_render_restart_preserves_completed_phases(controller,phase):
    b=controller
    with b.j.writer():
        key,_=c.register(b.j,b.request,b.store,free_disk_bytes=500*1024**3)
        state=b.j.read();state['events'][key]['cursor'].update(phase='FINAL_RENDER_REQUIRED');b.j.save(state)
    stages=[b.render_prepare,b.render_review,b.render_materialize,b.render_issue,b.render_run]
    target=stages[['FINAL_RENDER_REQUIRED','RENDER_REVIEW','RENDER_MATERIALIZE','RENDER_ISSUE','RENDER_RUN'].index(phase)]
    original=target.side_effect;target.side_effect=RuntimeError('render interruption')
    with pytest.raises(RuntimeError,match='render interruption'):run(b)
    assert entry(b)['cursor']['phase']==phase
    target.side_effect=original
    out=run(b);assert out['status']=='FINAL_VISUAL_AUDIT_REQUIRED' and not out['goal_complete']
    assert run(b)==out
    for stage in stages:assert stage.call_count==(2 if stage is target else 1)
    b.first.assert_not_called();b.next.assert_not_called()


def test_render_missing_volume_and_resource_rejection_preserve_phase(controller):
    b=controller
    with b.j.writer():
        key,_=c.register(b.j,b.request,b.store,free_disk_bytes=500*1024**3)
        state=b.j.read();state['events'][key]['cursor'].update(phase='FINAL_RENDER_REQUIRED');b.j.save(state)
    original=b.render_prepare.side_effect
    b.render_prepare.side_effect=None;b.render_prepare.return_value=dict(status='WAITING_FOR_ADMIN_PROVISION',missing=['fixture-volume'])
    assert run(b)['missing']==['fixture-volume']
    assert entry(b)['cursor']['phase']=='FINAL_RENDER_REQUIRED';b.render_review.assert_not_called()
    b.render_prepare.side_effect=original
    b.render_run.return_value=dict(status='RESOURCE_OBSERVATION_REJECTED')
    assert run(b)['status']=='RESOURCE_OBSERVATION_REJECTED'
    assert entry(b)['cursor']['phase']=='RENDER_RUN'
    b.render_run.return_value=dict(status='RENDER_JOB_CLOSED_REQUIRES_VIDEO_VALIDATION')
    assert run(b)['status']=='FINAL_VISUAL_AUDIT_REQUIRED'
    b.render_review.assert_called_once();b.render_issue.assert_called_once()


def test_video_validation_interruption_reconnects_registered_event(controller):
    b=controller
    with b.j.writer():
        key,_=c.register(b.j,b.request,b.store,free_disk_bytes=500*1024**3)
        state=b.j.read();state['events'][key]['cursor'].update(phase='FINAL_VIDEO_VALIDATION_REQUIRED');b.j.save(state)
    b.video_validation_handoff.side_effect=RuntimeError('validation interrupted')
    with pytest.raises(RuntimeError,match='validation interrupted'):run(b)
    assert entry(b)['cursor']['phase']=='VIDEO_VALIDATION'
    b.video_validation_handoff.side_effect=None
    assert run(b)['status']=='FINAL_VISUAL_AUDIT_REQUIRED'
    b.video_validation_register.assert_called_once();assert b.video_validation_handoff.call_count==2
