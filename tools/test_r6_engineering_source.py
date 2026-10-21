"""Actual Git and fake source writer; scientific config/snapshot remain fixtures."""
import copy,json
import pytest
from tools.test_r6_source_preparation import prepared
from tools import r6_engineering_source as engine,r6_controller as controller
from tools.r6_recovery import RecoveryError,file_pin,Journal
FREE=500*1024**3

@pytest.fixture
def case(prepared):
    b=prepared
    initial=b.write('engine-initial.json',dict(plan_pin=b.report_pin,master_pin=b.report_pin,review_executable=file_pin(b.builder)))
    request=b.write('engine-controller.json',dict(initial_request=initial,profile_context=b.report_pin));b.request=request
    with b.j.writer():
        b.controller,_=controller.register(b.j,request,b.store,free_disk_bytes=FREE)
        state=b.j.read();state['events'][b.controller]['cursor'].update(phase='REPAIR_AND_REVIEW',parent_key=b.parent,job_key=b.key)
        b.cursor=copy.deepcopy(state['events'][b.controller]['cursor']);b.original=copy.deepcopy(state['events'][b.key]);b.j.save(state)
    return b


def run(b):
    with b.j.writer():return engine.prepare(b.j,b.controller,b.store,executable=file_pin(b.builder),free_disk_bytes=FREE)


def test_one_source_writer_freezes_proposal_and_replays_without_new_model(case):
    b=case;before=b.store.replay();out=run(b);state=b.j.read();budget=state['budget']
    assert out['status']=='ENGINEERING_SOURCE_READY_REQUIRES_INDEPENDENT_REVIEW'
    assert not out['dispatch_authorized'] and not out['goal_complete']
    assert state['events'][b.key]==b.original and state['events'][b.controller]['cursor']==b.cursor
    assert b.store.replay()==before
    source=state['events'][out['source_event_key']]
    assert source['source_input']['engineering_repair']==out['repair_input'] and source['purpose']=='repair'
    assert 'major_scientific_candidates' not in source['resources']
    card=json.loads(engine.checked(out['source_result']).read_text())['card']
    assert json.loads(engine.checked(card).read_text())['engineering_repair']['findings_addressed']
    b.j=Journal(b.j.directory);assert run(b)==out and b.j.read()['budget']==budget
    assert b.builder.with_suffix('.count').read_text()=='call\n'


def test_controller_automatically_repairs_but_does_not_self_approve_or_submit(case,monkeypatch):
    from unittest.mock import Mock
    b=case;next_job=Mock(side_effect=AssertionError('no reviewed repair yet'))
    monkeypatch.setattr(controller.next_entry,'continue_job',next_job)
    monkeypatch.setattr(controller.profile,'prepare_profile',Mock(side_effect=AssertionError('repair not reviewed')))
    monkeypatch.setattr(controller.engineering_review,'prepare',Mock(side_effect=RuntimeError('review boundary')))
    with b.j.writer():
        state=b.j.read();state['events'][b.controller]['cursor']=dict(phase='FIRST_JOB',completed_jobs=[]);b.j.save(state)
    monkeypatch.setattr(controller.first,'start',Mock(return_value=dict(status='DECISION_PROPOSAL_RETURNED',event_key=b.key)))
    with b.j.writer(),pytest.raises(RuntimeError,match='review boundary'):
        controller.run(b.j,b.request,b.store,free_disk_bytes=FREE)
    budget=b.j.read()['budget']
    with b.j.writer(),pytest.raises(RuntimeError,match='review boundary'):
        controller.run(b.j,b.request,b.store,free_disk_bytes=FREE)
    assert b.j.read()['budget']==budget
    assert b.j.read()['events'][b.controller]['cursor']['phase']=='ENGINEERING_REPAIR_REVIEW'
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    next_job.assert_not_called()


@pytest.mark.parametrize('setting,reason',[('omit_engineering','DESIGN_CARD'),('omit_findings','ENGINEERING_FINDINGS'),('incomplete','INCOMPLETE')])
def test_incomplete_or_unbound_fix_cannot_freeze_or_repeat_worker(case,setting,reason):
    b=case;b.builder.with_suffix('.settings').write_text(json.dumps({setting:True}))
    for _ in range(2):
        with pytest.raises(RecoveryError,match=reason):run(b)
    b.freeze.assert_not_called();assert b.builder.with_suffix('.count').read_text()=='call\n'
    assert b.j.read()['events'][b.controller]['cursor']==b.cursor


def test_restart_after_bound_input_before_worker_uses_same_input(case,monkeypatch):
    b=case;original=engine.source.prepare_source
    monkeypatch.setattr(engine.source,'prepare_source',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('before worker')))
    with pytest.raises(RuntimeError,match='before worker'):run(b)
    bound=b.j.read()['events'][b.controller]['engineering_source_input']
    monkeypatch.setattr(engine.source,'prepare_source',original);out=run(b)
    assert out['repair_input']==bound['repair_input']
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    assert sum(e['identity']['role']=='engineering-repair-input' for e in b.j.read()['events'].values())==1


def test_stop_prevents_new_worker(case):
    b=case;(b.j.directory/'STOP').touch();before=b.j.read()
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert b.j.read()==before and not b.builder.with_suffix('.count').exists()


def test_changed_cursor_and_frozen_confirmation_do_not_enter_workload_repair(case,monkeypatch):
    b=case;original=engine.source.prepare_source
    monkeypatch.setattr(engine.source,'prepare_source',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('before worker')))
    with pytest.raises(RuntimeError):run(b)
    monkeypatch.setattr(engine.source,'prepare_source',original)
    with b.j.writer():
        state=b.j.read();state['events'][b.controller]['cursor']['completed_jobs'].append('unexpected');b.j.save(state)
    with pytest.raises(RecoveryError,match='CURSOR_DRIFT'):run(b)
    with b.j.writer():
        state=b.j.read();state['events'][b.controller]['cursor']['confirmation']={};b.j.save(state)
    with pytest.raises(RecoveryError,match='FROZEN_PHASE'):run(b)
    assert not b.builder.with_suffix('.count').exists()
