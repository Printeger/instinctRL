"""Real journal edges; declared substitutes for parent verification/Store/runner."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rlflow.store import sha
from tools.r6_recovery import Journal,file_pin,RecoveryError
from tools import r6_next_job as next_job


@pytest.fixture
def case(tmp_path,monkeypatch):
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return file_pin(p)
    j=Journal(tmp_path/'batch')
    with j.writer():j.initialize(json.loads((Path(__file__).parent/'fixtures/r6_budget.json').read_text()))
    proposal={'decision':{'decision':'PROPOSE_CANDIDATE'}}
    parent_pin=write('decision.json',proposal);spec={'id':'CPU-NEXT'};spec_pin=write('spec.json',spec)
    child='J-'+sha({'id':spec['id'],'spec':spec})[:16]
    plan=dict(spec_pin=spec_pin,query_executable=spec_pin,decision_executable=spec_pin,
              queue_review=None,sampler_review=None,purpose='development',sandbox='read-only')
    plan_pin=write('plan.json',plan)
    def event(role,request):
        identity=dict(goal_id=j.read()['goal_id'],event_id=role,job_id='J-parent',input_sha256=sha(request),role=role)
        with j.writer():return j.intent(identity,kind='model',request=request,resources={'automatic_model_sessions':1},
            purpose='development',free_disk_bytes=500*1024**3)[0]
    parent=event('post-analysis-decision',{'fixture':'parent'})
    review=event('next-job-review',{'fixture':'review','schema':file_pin(next_job.SCHEMA),'sandbox':'read-only'})
    output=dict(decision='PASS_R6_NEXT_JOB',summary='CPU fixture',must_fix=[],changed_files=[],evidence_pins=[parent_pin,plan_pin],parent_decision_sha256=parent_pin['sha256'],plan_sha256=plan_pin['sha256'])
    review_pin=write('review.json',output)
    with j.writer():
        s=j.read();s['events'][parent]['decision_record']=parent_pin
        e=s['events'][review];e['status']='RECEIPT_RECORDED'
        e['receipt']=write('receipt.json',dict(event_key=review,request_sha256=sha(e['request']),
            process=e.get('process'),last_message=review_pin));j.save(s)
    parent_check=Mock(return_value=proposal);worker_check=Mock(return_value=output)
    monkeypatch.setattr(next_job,'record_decision',parent_check)
    monkeypatch.setattr(next_job,'verified_decision',worker_check)
    monkeypatch.setattr(next_job,'no_model_workers',Mock())
    monkeypatch.setattr(next_job.experiments,'validate',Mock())
    authorize=Mock();monkeypatch.setattr(next_job.experiments,'authorized',authorize)
    store=SimpleNamespace(root=tmp_path,replay=lambda:{'jobs':{}})
    runner=Mock(return_value={'status':'RESOURCE_OBSERVATION_REJECTED'})
    monkeypatch.setattr(next_job,'run_job',runner)
    return SimpleNamespace(j=j,parent=parent,review=review,review_pin=review_pin,plan_pin=plan_pin,
        plan=plan,child=child,store=store,runner=runner,write=write,authorize=authorize,
        parent_check=parent_check,worker_check=worker_check)


def call(b,fn=next_job.advance,**overrides):
    args=dict(parent_key=b.parent,plan_pin=b.plan_pin,review_key=b.review,review_pin=b.review_pin,
              store=b.store,free_disk_bytes=500*1024**3);args.update(overrides)
    with b.j.writer():return fn(b.j,**args)


def test_registered_edge_replays_one_reservation_and_rechecks_before_start(case):
    b=case;first=call(b);before=b.j.read()['budget'];second=call(b)
    assert first==second and before==b.j.read()['budget'] and b.runner.call_count==2
    assert b.parent_check.call_count==3  # register + both not-yet-submitted starts
    assert first['scientific_acceptance'] is False


def test_crash_after_edge_before_runner_recovers_same_plan(case):
    b=case;b.runner.side_effect=RuntimeError('crash before run')
    with pytest.raises(RuntimeError):call(b)
    before=b.j.read()['budget'];b.j=Journal(b.j.directory)
    b.runner.side_effect=None;call(b)
    assert b.j.read()['budget']==before
    changed=dict(b.plan,sandbox='danger-full-access')
    with pytest.raises(RecoveryError,match='PARENT_ALREADY_BOUND'):call(b,plan_pin=b.write('changed.json',changed))


def test_missing_store_permission_does_not_create_edge(case):
    b=case;before=b.j.read();b.authorize.side_effect=ValueError('no exact authority')
    with pytest.raises(ValueError,match='no exact authority'):call(b)
    assert b.j.read()==before;b.runner.assert_not_called()


def test_review_identity_tampering_rejected(case):
    b=case
    with b.j.writer():
        s=b.j.read();s['events'][b.review]['identity']['role']='post-analysis-decision';b.j.save(s)
    with pytest.raises(RecoveryError,match='REVIEW_IDENTITY'):call(b)
    b.runner.assert_not_called()


def test_new_start_rechecks_parent_after_edge_registration(case):
    b=case;call(b,next_job.register);b.parent_check.side_effect=ValueError('parent drift')
    with pytest.raises(ValueError,match='parent drift'):call(b)
    b.runner.assert_not_called()


def test_completed_edge_returns_without_second_runner(case):
    b=case;call(b,next_job.register)
    with b.j.writer():
        s=b.j.read();s['events']['fixture-job']=dict(kind='job',identity={'job_id':b.child,'role':'evaluation'});b.j.save(s)
    b.runner.return_value=dict(status='DECISION_PROPOSAL_RETURNED',event_key='fixture-job')
    first=call(b);(b.j.directory/'STOP').touch();second=call(b)
    assert first==second;b.runner.assert_called_once()


def test_stop_blocks_unsubmitted_edge_but_claim_reaches_existing_recovery(case):
    b=case;call(b,next_job.register);(b.j.directory/'STOP').touch()
    with pytest.raises(RecoveryError,match='STOP'):call(b)
    b.runner.assert_not_called()
    with b.j.writer():
        s=b.j.read();s['events']['fixture-job']=dict(kind='job',identity={'job_id':b.child,'role':'evaluation'},submission_claim={'fixture':True});b.j.save(s)
    call(b);b.runner.assert_called_once()  # Existing runner owns exact STOP cleanup.


def test_cached_completion_cannot_point_at_another_child(case):
    b=case;key=call(b,next_job.register)
    with b.j.writer():
        s=b.j.read()
        s['events']['wrong-job']=dict(kind='job',identity={'job_id':'J-other','role':'evaluation'})
        s['events'][key]['next_job_result']=dict(status='DECISION_PROPOSAL_RETURNED',
            handoff_event_key=key,child_job_event_key='wrong-job',scientific_acceptance=False)
        b.j.save(s)
    with pytest.raises(RecoveryError,match='CACHED_CHILD'):call(b)
    b.runner.assert_not_called()


def test_review_must_bind_exact_plan(case):
    b=case
    changed=dict(b.plan,sandbox='danger-full-access')
    with pytest.raises(RecoveryError,match='REVIEW_BINDING'):
        call(b,plan_pin=b.write('unreviewed.json',changed))
    b.runner.assert_not_called()


def test_replay_rejects_missing_reservation(case):
    b=case;key=call(b,next_job.register)
    with b.j.writer():
        s=b.j.read();identity=s['events'][key]['identity']
        s['budget']['reservations']=[r for r in s['budget']['reservations'] if r['identity']!=identity]
        b.j.save(s)
    with pytest.raises(RecoveryError,match='REPLAY_IDENTITY'):call(b)
    b.runner.assert_not_called()


def register_review(b):
    exe=b.write('fixture-executable.json',{})
    with b.j.writer():
        return next_job.register_review(b.j,b.parent,b.plan_pin,b.store,
            executable=exe,free_disk_bytes=500*1024**3)


def test_review_registration_replays_one_session_and_recovers_missing_prompt(case):
    b=case;key=register_review(b);before=b.j.read()['budget']
    prompt=Path(b.j.read()['events'][key]['request']['prompt']['path']);original=prompt.read_bytes()
    prompt.unlink()  # Crash after intent, before prompt persistence.
    assert register_review(b)==key and prompt.read_bytes()==original
    assert b.j.read()['budget']==before
    assert b.j.read()['events'][key]['request']['sandbox']=='read-only'


def test_real_receipt_and_schema_validator_accept_exact_review(case,monkeypatch):
    from tools.r6_batch import verified_decision
    b=case;key=register_review(b)
    with b.j.writer():
        s=b.j.read();e=s['events'][key]
        receipt=dict(event_key=key,request_sha256=sha(e['request']),process=None,
            last_message=b.review_pin,exit_code=0,error=None,schema_valid=True,
            events=b.write('events.json',{}),stderr=b.write('stderr.json',{}),
            cleanup=dict(status='VERIFIED_NO_OWNED_DESCENDANTS',stable_empty_scans=2,
                signalled_process_count=0,observed_identities=[]))
        e.update(receipt=b.write('full-receipt.json',receipt),status='RECEIPT_RECORDED');b.j.save(s)
    monkeypatch.setattr(next_job,'verified_decision',verified_decision)
    assert call(b,review_key=key)['status']=='RESOURCE_OBSERVATION_REJECTED'
    # Alter only output bytes and update both pins: real schema still rejects.
    bad=json.loads(Path(b.review_pin['path']).read_text());bad['changed_files']=['unauthorized.py']
    badpin=b.write('bad-review.json',bad)
    with b.j.writer():
        s=b.j.read();receipt['last_message']=badpin
        s['events'][key]['receipt']=b.write('bad-receipt.json',receipt);b.j.save(s)
    import jsonschema
    with pytest.raises(jsonschema.ValidationError):
        next_job.reviewed_plan(b.j,b.parent,b.plan_pin,key,badpin,b.store)


def test_failed_review_allows_separately_charged_revised_plan(case):
    b=case;first=register_review(b)
    b.plan_pin=b.write('revised-plan.json',dict(b.plan,sandbox='danger-full-access'))
    second=register_review(b)
    assert first!=second
    assert b.j.read()['events'][first]['request']['prompt']!=b.j.read()['events'][second]['request']['prompt']


@pytest.mark.parametrize('relative',['STOP','coordination/STOP','coordination/commander/STOP','coordination/executor/STOP'])
@pytest.mark.parametrize('phase',['before','during'])
def test_registered_workspace_stop_reaches_real_worker(case,monkeypatch,relative,phase):
    import sys
    from tools import r6_decision, r6_model_worker
    b=case;root=b.store.root
    # Redirect workspace to the fixture; never create a production STOP.
    monkeypatch.setattr(r6_decision,'ROOT',root)
    monkeypatch.setattr(r6_model_worker,'ROOT',root)
    target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
    started=root/'started'
    exe=root/'fake-reviewer'
    exe.write_text('#!'+sys.executable+'\nfrom pathlib import Path\nimport time\n'
        +'Path('+repr(str(started))+').touch()\n'
        +'Path('+repr(str(target))+').touch()\ntime.sleep(10)\n')
    exe.chmod(0o700)
    with b.j.writer():
        key=next_job.register_review(b.j,b.parent,b.plan_pin,b.store,
            executable=file_pin(exe),free_disk_bytes=500*1024**3)
    request=b.j.read()['events'][key]['request']
    assert str(target) in request['stop_paths']
    if phase=='before':target.touch()
    r6_model_worker.run(b.j.directory,key)
    receipt=json.loads((b.j.directory/'calls'/key/'receipt.json').read_text())
    assert ('STOP_BEFORE_MODEL_START' if phase=='before' else 'STOP_DURING_MODEL') in receipt['error']
    assert started.exists()==(phase=='during')
    from tools.r6_worker_processes import cleanup_verified
    if phase=='during':assert cleanup_verified(receipt['cleanup'])
    else:assert receipt['child_identity'] is None and receipt['cleanup'] is None
