"""Real fake repair CLI/Journal/Store; baseline science and prior review are fixtures."""
import copy,json,sys,subprocess
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools.test_r6_initial_review_repair import ready,bound,failed
from tools import r6_baseline_failure as m,r6_controller as controller,r6_initial_baseline as initial
from tools.r6_batch import model_handoff
from tools.r6_recovery import RecoveryError,file_pin
FREE=500*1024**3
POPEN=subprocess.Popen
SCRIPT='''import json,sys
from pathlib import Path
p=json.loads(sys.stdin.read().splitlines()[-1]);settings=json.loads(Path(__file__).with_suffix('.settings').read_text())
with Path(__file__).with_suffix('.count').open('a') as f:f.write('call\\n')
if settings.get('fail'):raise SystemExit(9)
out=dict(decision=settings.get('decision','REPLACE_FAILED_ATTEMPT'),packet_sha256=p['packet_pin']['sha256'],
 summary='CPU fixture: supply exact missing evidence for independent re-review',next_plan='Re-review exact baseline',
 evidence_pins=[p['packet_pin'],p['failed_receipt']],changed_files=[],must_fix=[],timeout_seconds=settings.get('timeout',0),
 remaining_budget={'automatic_sessions_remaining':p['model_remaining']-1})
if out['decision']!='REPLACE_FAILED_ATTEMPT':out['must_fix']=['CPU fixture requires a concrete code repair']
if settings.get('bad_pin'):out['packet_sha256']='0'*64
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(out)}}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))
'''

@pytest.fixture
def case(ready,monkeypatch):
    b=ready;failed(b);b.oldkey=b.review;b.old=copy.deepcopy(b.journal.read()['events'][b.review])
    def scoped_popen(argv,*args,**kwargs):
        if isinstance(argv,list) and argv[:4]==[sys.executable,'-B','-m','tools.r6_model_worker']:
            return POPEN(argv,*args,**kwargs)
        return b.popen(argv,*args,**kwargs)
    monkeypatch.setattr(subprocess,'Popen',scoped_popen)
    b.builder=b.journal.directory.parent/'repair-cli';b.builder.write_text('#!'+sys.executable+'\n'+SCRIPT);b.builder.chmod(0o700)
    b.builder.with_suffix('.settings').write_text('{}')
    b.args=dict(plan_pin=b.plan_pin,master_pin=b.plan_pin,review_executable=file_pin(b.builder))
    b.request=b.write('controller-input.json',dict(initial_request=b.write('initial-input.json',b.args),profile_context=b.plan_pin))
    with b.journal.writer():b.controller,_=controller.register(b.journal,b.request,b.store,free_disk_bytes=FREE)
    monkeypatch.setattr(m,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    return b


def enter(b):
    with b.journal.writer():
        key=m.capture(b.journal,b.controller,b.args,b.store,free_disk_bytes=FREE)
        s=b.journal.read();s['events'][b.controller]['cursor'].update(phase='BASELINE_FAILURE_HANDOFF',baseline_failure_input=key);b.journal.save(s)
    return key


def run(b):
    with b.journal.writer():return m.prepare(b.journal,b.controller,b.store,free_disk_bytes=FREE)


def test_real_disposition_registers_new_review_and_preserves_failed_original(case):
    b=case;before=b.store.replay();enter(b);out=run(b)
    assert out['status']=='BASELINE_REPLACEMENT_REGISTERED'
    assert b.journal.read()['events'][b.oldkey]==b.old and b.store.replay()==before
    assert initial.active_review(b.journal,b.plan_pin)==out['replacement']
    assert b.journal.read()['events'][b.controller]['cursor']['phase']=='FIRST_JOB'
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    assert b.journal.read()['events'][out['replacement']]['purpose']=='repair'
    b.popen.assert_not_called()


def test_controller_catches_first_boundary_and_reenters_existing_first_job(case,monkeypatch):
    b=case;first=Mock(side_effect=[RecoveryError('INITIAL_BASELINE_REVIEW_MUST_FIX'),dict(status='RESOURCE_OBSERVATION_REJECTED')])
    monkeypatch.setattr(controller.first,'start',first)
    with b.journal.writer():out=controller.run(b.journal,b.request,b.store,free_disk_bytes=FREE)
    assert out['status']=='RESOURCE_OBSERVATION_REJECTED' and first.call_count==2
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    assert len(b.journal.read()['events'][b.controller]['baseline_failure_history'])==1


def test_reserved_input_recovers_before_cursor_save_without_extra_charge(case):
    b=case
    with b.journal.writer():
        first=m.capture(b.journal,b.controller,b.args,b.store,free_disk_bytes=FREE);budget=b.journal.read()['budget']
        assert m.capture(b.journal,b.controller,b.args,b.store,free_disk_bytes=FREE)==first
        assert b.journal.read()['budget']==budget


def test_replacement_registration_crash_replays_same_intent_not_model(case,monkeypatch):
    b=case;enter(b);original=initial.register_review
    def crash(*args,**kwargs):
        original(*args,**kwargs);raise RuntimeError('after replacement registration')
    monkeypatch.setattr(initial,'register_review',crash)
    with pytest.raises(RuntimeError,match='after replacement'):run(b)
    budget=b.journal.read()['budget'];monkeypatch.setattr(initial,'register_review',original)
    out=run(b)
    assert out['status']=='BASELINE_REPLACEMENT_REGISTERED' and b.journal.read()['budget']==budget
    assert b.builder.with_suffix('.count').read_text()=='call\n'


@pytest.mark.parametrize('setting',[dict(decision='REPAIR_CODE_REQUIRED'),dict(decision='EXTERNAL_BLOCKER')])
def test_unresolved_disposition_is_retained_without_repeated_model(case,setting):
    b=case;b.builder.with_suffix('.settings').write_text(json.dumps(setting));enter(b)
    first=run(b);budget=b.journal.read()['budget'];assert run(b)==first
    assert first['status']==setting['decision'] and b.journal.read()['budget']==budget
    assert initial.active_review(b.journal,b.plan_pin)==b.oldkey
    assert b.builder.with_suffix('.count').read_text()=='call\n'


@pytest.mark.parametrize('setting,reason',[(dict(fail=True),'MODEL_FAILED'),(dict(bad_pin=True),'OUTPUT_BINDING'),(dict(timeout=10),'NON_ANALYSIS_SETTINGS')])
def test_failed_or_invalid_disposition_never_registers_successor(case,setting,reason):
    b=case;b.builder.with_suffix('.settings').write_text(json.dumps(setting));enter(b)
    for _ in range(2):
        with pytest.raises(RecoveryError,match=reason):run(b)
    assert initial.active_review(b.journal,b.plan_pin)==b.oldkey
    assert b.builder.with_suffix('.count').read_text()=='call\n'


@pytest.mark.parametrize('damage',['stop','cleanup','missing','live','store'])
def test_unreconciled_state_blocks_new_repair_model(case,damage):
    from rlflow.jobs import identity
    import os
    b=case
    if damage=='store':
        enter(b);b.store.transact('DRIFT','fixture',lambda s:s.update(paused=True))
    elif damage=='stop':(b.journal.directory/'STOP').touch()
    else:
        with b.journal.writer():
            s=b.journal.read();e=s['events'][b.oldkey];r=json.loads(Path(e['receipt']['path']).read_text())
            if damage=='cleanup':r['cleanup']['stable_empty_scans']=0
            if damage=='live':r['child_identity']=identity(os.getpid())
            e['receipt']=None if damage=='missing' else b.write('damaged-receipt.json',r);b.journal.save(s)
    before=b.journal.read()
    if damage=='missing':
        with b.journal.writer():assert m.capture(b.journal,b.controller,b.args,b.store,free_disk_bytes=FREE) is None
    else:
        with pytest.raises(RecoveryError):
            if damage=='store':run(b)
            else:enter(b)
    assert b.journal.read()==before
    assert not b.builder.with_suffix('.count').exists()


def closed_controller(b,monkeypatch,*,timeout=0,decision_script=None):
    with b.j.writer():
        s=b.j.read();s['events'][b.key]['identity']['role']='job';b.j.save(s)
    builder=b.root/'baseline-repair-cli';script=SCRIPT
    if decision_script:
        marker="p=json.loads(sys.stdin.read().splitlines()[-1]);settings="
        replacement="text=sys.stdin.read();p=json.loads(text.splitlines()[-1])\nif 'packet_pin' not in p:\n import io\n sys.stdin=io.StringIO(text)\n exec("+repr(decision_script)+")\n raise SystemExit(0)\nsettings="
        assert marker in script;script=script.replace(marker,replacement)
    builder.write_text('#!'+sys.executable+'\n'+script);builder.chmod(0o700)
    builder.with_suffix('.settings').write_text(json.dumps(dict(timeout=timeout)))
    plan=b.write('baseline-plan.json',dict(spec_pin=b.spec_pin,query_executable=file_pin(builder),
        decision_executable=file_pin(builder),queue_review=None,sampler_review=None,purpose='development',sandbox='read-only'))
    b.args=dict(plan_pin=plan,master_pin=plan,review_executable=file_pin(builder))
    b.request=b.write('baseline-controller.json',dict(initial_request=b.write('baseline-initial.json',b.args),profile_context=plan))
    b.journal=b.j;b.builder=builder
    monkeypatch.setattr(initial,'EXPERIMENT',b.spec['id'])
    monkeypatch.setattr(m,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    with b.j.writer():b.controller,_=controller.register(b.j,b.request,b.store,free_disk_bytes=FREE)


def test_analysis_timeout_disposition_replaces_then_ordinary_finalizer_runs(monkeypatch):
    from tools.test_r6_cpu_analysis import CPUAnalysisTests
    b=CPUAnalysisTests('test_finalizer_registers_analysis_once_and_replay_preserves_state');b.setUp()
    try:
        b.register(body='time.sleep(1.2)',timeout=1)
        with pytest.raises(RecoveryError,match='FAILED_PRESERVED'):b.complete()
        oldkey=b.cpu_key;old=copy.deepcopy(b.j.read()['events'][oldkey])
        closed_controller(b,monkeypatch,timeout=3);enter(b);out=run(b)
        assert out['status']=='BASELINE_REPLACEMENT_REGISTERED'
        assert b.j.read()['events'][out['replacement']]['request']['timeout_seconds']==3
        b.finalize_analysis(timeout=None)
        assert b.count()==2 and b.j.read()['events'][oldkey]==old
        assert b.builder.with_suffix('.count').read_text()=='call\n'
    finally:b.doCleanups()


def test_non_timeout_analysis_failure_cannot_be_hidden_by_more_time(monkeypatch):
    from tools.test_r6_analysis_repair import AnalysisRepairTests
    b=AnalysisRepairTests('test_unchanged_input_cannot_blind_retry');b.setUp()
    try:
        b.fail();closed_controller(b,monkeypatch,timeout=20);enter(b)
        with pytest.raises(RecoveryError,match='ANALYSIS_SETTINGS'):run(b)
        assert m.analysis.active(b.j,b.key)==b.oldkey and b.count()==1
    finally:b.doCleanups()


def test_failed_decision_disposition_registers_then_normal_finalizer_resumes(monkeypatch):
    from tools.test_r6_decision_repair import DecisionRepairTests
    from tools.test_r6_decision import FAKE
    b=DecisionRepairTests('test_failed_call_replaced_and_controller_uses_new_proposal');b.setUp()
    try:
        b.fail();old=copy.deepcopy(b.j.read()['events'][b.oldkey])
        closed_controller(b,monkeypatch,decision_script=FAKE.format(python=sys.executable));b.exe=b.builder
        enter(b);out=run(b)
        assert out['status']=='BASELINE_REPLACEMENT_REGISTERED'
        result=b.finalize_decision()
        assert result['decision']['decision']=='NEEDS_REPAIR'
        assert b.j.read()['events'][b.oldkey]==old
        assert b.builder.with_suffix('.count').read_text()=='call\ncall\n'
    finally:b.doCleanups()


def test_initial_preflight_review_is_not_a_post_analysis_decision(ready):
    from tools.test_r6_initial_baseline import issue
    b=ready;out=issue(b)
    with b.journal.writer():
        s=b.journal.read();s['events']['closed-fixture']=dict(kind='job',identity=dict(job_id=out['job_id'],role='job'),request=dict(spec=b.plan['spec_pin']));b.journal.save(s)
        assert m.pre_analysis_model(b.journal,b.review,'closed-fixture')
        s=b.journal.read();edge=next(e for e in s['events'].values() if e['identity'].get('role')=='initial-baseline-authorization');edge.pop('authorization_result');b.journal.save(s)
        assert not m.pre_analysis_model(b.journal,b.review,'closed-fixture')


def test_snapshot_budget_change_does_not_spend_new_repair_call(case):
    b=case;enter(b)
    with b.journal.writer():
        s=b.journal.read();s['budget']['used']['automatic_model_sessions']+=1;b.journal.save(s)
    before=b.journal.read()
    with pytest.raises(RecoveryError,match='BUDGET_STALE'):run(b)
    assert b.journal.read()==before and not b.builder.with_suffix('.count').exists()


def test_rejected_initial_review_chain_can_precede_later_analysis_repair(case):
    from tools.test_r6_initial_review_repair import accept
    from tools.test_r6_initial_baseline import issue
    b=case;enter(b);out=run(b);accept(b,out['replacement']);issued=issue(b)
    with b.journal.writer():
        s=b.journal.read();s['events']['closed-fixture']=dict(kind='job',identity=dict(job_id=issued['job_id'],role='job'),request=dict(spec=b.plan['spec_pin']));b.journal.save(s)
        for key in [b.oldkey,out['disposition_key'],out['replacement']]:
            assert m.pre_analysis_model(b.journal,key,'closed-fixture')
