"""Real Git and fake source/review CLI; native snapshot and live/compatibility stubs."""
import copy,json
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools.test_r6_source_preparation import prepared
from tools import r6_engineering_source as engine,r6_engineering_review as review,r6_controller as controller
from tools.r6_recovery import RecoveryError,file_pin,Journal
from tools.r6_batch import model_handoff
FREE=500*1024**3
REVIEW='''
if 'source_result' in p and 'model_cap' in p:
 settings=json.loads(Path(__file__).with_suffix('.settings').read_text())
 with Path(__file__).with_suffix('.review-count').open('a') as f:f.write('review\\n')
 checks={k:True for k in ['original_failure_preserved','all_findings_resolved','engineering_only','actor_boundary_preserved','scoring_and_split_preserved','no_holdout_tuning','source_and_snapshot_bound','fresh_attempt_required']}
 if settings.get('bad_check'):checks['all_findings_resolved']=False
 value={'decision':'MUST_FIX' if settings.get('reject') else 'PASS_R6_ENGINEERING_REPAIR',
 'reviewer':'r6-independent-engineering-repair',
 'packet_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
 'summary':'CPU fixture review','checks':checks,'changed_files':[],
 'must_fix':['Fixture unresolved repair'] if settings.get('reject') else [],
 'evidence_pins':[p[n] for n in ['source','repair_input','source_result','snapshot','card','config','parent']],
 'automatic_sessions_remaining':p['model_cap']-p['budget']['automatic_model_sessions']-1}
 if settings.get('omit_evidence'):value['evidence_pins'].pop()
 print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(value)}}))
 sys.exit(0)
'''

@pytest.fixture
def case(prepared,monkeypatch):
    b=prepared
    text=b.builder.read_text();marker="p=json.loads(sys.stdin.read().splitlines()[-1]);repo=Path(p['repository'])"
    assert marker in text
    b.builder.write_text(text.replace(marker,"p=json.loads(sys.stdin.read().splitlines()[-1])\n"+REVIEW+"\nrepo=Path(p['repository'])"))
    initial=b.write('review-initial.json',dict(plan_pin=b.report_pin,master_pin=b.report_pin,review_executable=file_pin(b.builder)))
    b.request=b.write('review-controller.json',dict(initial_request=initial,profile_context=b.report_pin))
    with b.j.writer():
        b.controller,_=controller.register(b.j,b.request,b.store,free_disk_bytes=FREE)
        state=b.j.read();state['events'][b.controller]['cursor'].update(phase='REPAIR_AND_REVIEW',parent_key=b.parent,job_key=b.key);b.j.save(state)
    monkeypatch.setattr(review,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    monkeypatch.setattr(review,'snapshot_pin',lambda j,out:b.write('snapshot-review-fixture.json',out['snapshot']))
    return b


def source(b):
    with b.j.writer():return engine.prepare(b.j,b.controller,b.store,executable=file_pin(b.builder),free_disk_bytes=FREE)


def run(b):
    with b.j.writer():return review.prepare(b.j,b.controller,b.store,executable=file_pin(b.builder),free_disk_bytes=FREE)


def test_actual_source_then_separate_review_is_charged_once_and_preserves_failure(case):
    b=case;original=copy.deepcopy(b.j.read()['events'][b.key]);source(b);out=run(b);budget=b.j.read()['budget']
    assert out['status']=='ENGINEERING_REPAIR_REVIEW_ACCEPTED' and not out['dispatch_authorized']
    b.j=Journal(b.j.directory);assert run(b)==out and b.j.read()['budget']==budget
    assert b.j.read()['events'][b.key]==original
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    assert b.builder.with_suffix('.review-count').read_text()=='review\n'
    assert b.j.read()['events'][out['event_key']]['purpose']=='repair'


def test_controller_reenters_compatibility_only_after_review_and_restarts_without_models(case,monkeypatch):
    b=case;compat=Mock(side_effect=RuntimeError('compatibility boundary'))
    monkeypatch.setattr(controller.compatibility,'prepare',compat)
    next_job=Mock(side_effect=AssertionError('no exact permission'))
    monkeypatch.setattr(controller.next_entry,'continue_job',next_job)
    monkeypatch.setattr(controller.first,'start',Mock(return_value=dict(status='DECISION_PROPOSAL_RETURNED',event_key=b.key)))
    with b.j.writer():
        state=b.j.read();state['events'][b.controller]['cursor'].update(phase='FIRST_JOB');b.j.save(state)
    for _ in range(2):
        with b.j.writer(),pytest.raises(RuntimeError,match='compatibility boundary'):
            controller.run(b.j,b.request,b.store,free_disk_bytes=FREE)
    cursor=b.j.read()['events'][b.controller]['cursor']
    assert cursor['phase']=='COMPATIBILITY' and cursor['engineering_review']['status']=='ENGINEERING_REPAIR_REVIEW_ACCEPTED'
    assert compat.call_args.args[2]==cursor['snapshot_receipt']
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    assert b.builder.with_suffix('.review-count').read_text()=='review\n';next_job.assert_not_called()


def test_controller_preserves_rejection_without_compatibility_or_repeat_review(case,monkeypatch):
    b=case;b.builder.with_suffix('.settings').write_text('{"reject":true}')
    compat=Mock(side_effect=AssertionError('must not advance'));monkeypatch.setattr(controller.compatibility,'prepare',compat)
    with b.j.writer():first=controller.run(b.j,b.request,b.store,free_disk_bytes=FREE)
    budget=b.j.read()['budget']
    with b.j.writer():second=controller.run(b.j,b.request,b.store,free_disk_bytes=FREE)
    assert first==second and first['status']=='ENGINEERING_REPAIR_REVIEW_REJECTED'
    assert b.j.read()['budget']==budget and b.j.read()['events'][b.controller]['cursor']['phase']=='ENGINEERING_REPAIR_REJECTED'
    assert b.builder.with_suffix('.review-count').read_text()=='review\n';compat.assert_not_called()


@pytest.mark.parametrize('setting,reason',[('bad_check','REVIEW_CHECKS'),('omit_evidence','REVIEW_EVIDENCE')])
def test_superficial_pass_cannot_be_accepted(case,setting,reason):
    b=case;source(b);b.builder.with_suffix('.settings').write_text(json.dumps({setting:True}))
    with pytest.raises(RecoveryError,match=reason):run(b)
    assert not any(e.get('review_result') for e in b.j.read()['events'].values())


def test_prompt_crash_after_model_intent_recovers_without_second_charge(case,monkeypatch):
    b=case;source(b);original=review.persist_exact
    monkeypatch.setattr(review,'persist_exact',Mock(side_effect=RuntimeError('prompt crash')))
    with pytest.raises(RuntimeError,match='prompt crash'):run(b)
    budget=b.j.read()['budget'];monkeypatch.setattr(review,'persist_exact',original);run(b)
    assert b.j.read()['budget']==budget
    assert b.builder.with_suffix('.review-count').read_text()=='review\n'


def test_changed_schema_cannot_create_a_second_review_for_same_packet(case,monkeypatch):
    b=case;source(b);original=review.persist_exact
    monkeypatch.setattr(review,'persist_exact',Mock(side_effect=RuntimeError('prompt crash')))
    with pytest.raises(RuntimeError):run(b)
    before=b.j.read();alternate=b.root/'alternate.schema.json';alternate.write_bytes(review.SCHEMA.read_bytes())
    monkeypatch.setattr(review,'persist_exact',original);monkeypatch.setattr(review,'SCHEMA',alternate)
    with pytest.raises(RecoveryError,match='MODEL_ALIAS'):run(b)
    assert b.j.read()==before


def test_cached_pass_is_not_authority_after_review_bytes_change(case,monkeypatch):
    b=case;compat=Mock(side_effect=RuntimeError('compatibility boundary'))
    monkeypatch.setattr(controller.compatibility,'prepare',compat)
    with b.j.writer(),pytest.raises(RuntimeError,match='compatibility boundary'):
        controller.run(b.j,b.request,b.store,free_disk_bytes=FREE)
    accepted=b.j.read()['events'][b.controller]['cursor']['engineering_review']
    Path(accepted['review']['path']).write_text('{}')
    with b.j.writer(),pytest.raises(RecoveryError,match='PIN_DRIFT'):
        controller.run(b.j,b.request,b.store,free_disk_bytes=FREE)
    assert compat.call_count==1


def test_stop_or_changed_source_blocks_review_before_model(case):
    b=case;out=source(b);before=b.j.read();(b.j.directory/'STOP').touch()
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert b.j.read()==before
    (b.j.directory/'STOP').unlink()  # fixture STOP only
    artifact=review.read(out['source_result']);Path(artifact['card']['path']).write_text('{}')
    with pytest.raises(RecoveryError):run(b)
    assert not b.builder.with_suffix('.review-count').exists()
