"""Actual Git completion recovery and real owned-process rejection, science mocked."""
import copy,os,subprocess,sys,time
from pathlib import Path
from unittest.mock import Mock
import pytest
from rlflow.jobs import identity
from tools import r6_terminal_audit as m
from tools.test_r6_final_delivery import ready,case,run,FREE
from tools import r6_publish_final_delivery as publication
from tools.r6_recovery import file_pin,RecoveryError

@pytest.fixture
def published(ready,monkeypatch):
    b=ready;out=run(b);pin=b.j.read()['events'][out['event_key']]['delivery']
    with b.j.writer():pub=publication.publish(b.j,pin,b.store)
    b.pub=pub;b.delivery=out;b.delivery_pin=pin
    with b.j.writer():
        state=b.j.read();state['events']['controller-fixture']=dict(kind='cpu',identity={'role':'r6-controller'},request={'store_root':str(b.store.root.resolve())},cursor=dict(final_delivery=pin,final_publication=pub));b.j.save(state)
    # Independent science/model-free wait evidence and native job closure are
    # synthetic for this completion transaction test; actual checks tested below.
    b.audit=dict(status='TERMINAL_REQUIREMENTS_VERIFIED_PENDING_COMMIT_AND_CONTROLLER_RECORD',controller_key='controller-fixture',delivery=pin,
        publication=b.j.read()['events'][out['event_key']]['publication'],scientific_status='BASELINE_SUFFICIENT',review=b.description['review'],review_receipt=b.description['review_receipt'],
        inventories=[],process_audit={'status':'CPU_FIXTURE'},model_usage=out['files']['MODEL_USAGE_AUDIT.json'],resources=out['files']['RESOURCE_AUDIT.json'],requirements=out['files']['REQUIREMENTS_AUDIT.json'],
        free_disk_floor_bytes=b.j.read()['budget']['disk_free_floor_bytes'],completion_condition='CPU fixture',audit_implementation=file_pin(m.__file__))
    b.auditor=Mock(return_value=b.audit);monkeypatch.setattr(m,'audit',b.auditor)
    return b


def finish(b):
    with b.j.writer():return m.finish(b.j,'controller-fixture',b.store)


def test_completion_is_recorded_only_after_exact_remote_publication_and_recheck(published):
    b=published;(b.repo/'unrelated.txt').write_text('preserve staged');b.git('add','unrelated.txt')
    budget=b.j.read()['budget'];out=finish(b);assert out['goal_complete']is True
    assert out['completion_commit']==b.git('rev-parse','HEAD')==b.git('rev-parse','origin/main')==b.git('rev-parse',out['completion_tag'])
    assert b.git('show','-s','--format=%P',out['completion_commit'])==b.pub['commit']
    assert b.git('diff','--cached','--name-only')=='unrelated.txt'
    assert b.j.read()['completion']==out and b.j.read()['events']['controller-fixture']['cursor']['phase']=='COMPLETE'
    assert finish(b)==out and b.j.read()['budget']==budget and b.auditor.call_count==4


def test_failed_push_leaves_canonical_state_incomplete_then_recovers_same_commit(published,monkeypatch):
    b=published;original=m.delivery.git
    def fail(*args):
        if args[0]=='push':raise RuntimeError('fixture failed push')
        return original(*args)
    monkeypatch.setattr(m.delivery,'git',fail)
    with pytest.raises(RuntimeError,match='failed push'):finish(b)
    assert not b.j.read().get('goal_complete') and 'completion' not in b.j.read()
    commit=b.git('rev-parse','HEAD');assert b.git('rev-parse','origin/main')==b.pub['commit']
    monkeypatch.setattr(m.delivery,'git',original);assert finish(b)['completion_commit']==commit


def test_post_push_evidence_drift_never_marks_complete(published):
    b=published;b.auditor.side_effect=[b.audit,dict(b.audit,scientific_status='R6_TARGET_DOMAIN_PASS')]
    with pytest.raises(RecoveryError,match='POST_PUBLICATION_DRIFT'):finish(b)
    assert not b.j.read().get('goal_complete')


def test_stop_prevents_completion_publication(published):
    b=published;(b.j.directory/'STOP').touch()
    with pytest.raises(RecoveryError):finish(b)
    assert not b.j.read().get('goal_complete') and b.git('rev-parse','HEAD')==b.pub['commit']


def test_actual_owned_process_and_unresolved_dispatch_are_not_silently_ignored(case):
    b=case;proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    try:
        ident=identity(proc.pid)
        with b.j.writer():
            state=b.j.read();state['events']['owned']=dict(kind='cpu',process=ident,request={},identity={'role':'fixture'});b.j.save(state)
            with pytest.raises(RecoveryError,match='OWNED_PROCESS_ALIVE'):m.workers(b.j)
        assert proc.poll()is None  # The final audit never kills even owned processes.
    finally:proc.terminate();proc.wait(timeout=5)
    with b.j.writer():
        state=b.j.read();state['events']['owned']['dispatch_claim']={'fixture':True};b.j.save(state)
        with pytest.raises(RecoveryError,match='UNRESOLVED_DISPATCH'):m.workers(b.j)
        state=b.j.read();state['events'].pop('owned');b.j.save(state)
        assert m.workers(b.j)['status']=='NO_KNOWN_OWNED_JOB_OR_WORKER_SURVIVORS'


def test_resource_sample_without_receipt_is_unresolved(case):
    b=case
    with b.j.writer():
        state=b.j.read();state['events']['sample']=dict(kind='cpu',request={},sample_claims={'0':{'request_sha256':'fixture'}},identity={'role':'fixture'});b.j.save(state)
        with pytest.raises(RecoveryError,match='UNRESOLVED_RESOURCE_SAMPLE'):m.workers(b.j)


@pytest.mark.parametrize('record',[{'pid':-1},{'pid':1,'pgid':1,'starttime':'x','boot_id':'unknown'},{}])
def test_malformed_identity_cannot_prove_dead(record):
    with pytest.raises(RecoveryError,match='PROCESS_IDENTITY'):m.dead(record)


@pytest.fixture
def auditable(ready,monkeypatch):
    b=ready
    # These substitutes represent previously independently reviewed science and
    # platform instrumentation; they are not evidence of a real R6 run.
    original=m.delivery.evidence.model_audit
    def usage(state):
        return dict(original(state),all_registered_local_waits_verified=True,platform_inactivity_proven=True,pure_wait_model_calls=0)
    monkeypatch.setattr(m.delivery.evidence,'model_audit',usage)
    d=b.description;decision=m.delivery.read(d['review']);decision.update(decision='PASS_R6_FINAL_AUDIT',must_fix=[])
    keys=[k for k,e in b.j.read()['events'].items() if e['kind']=='job'];assert len(keys)==1
    b.job_key=keys[0];d['jobs']['jobs'][0]['event_key']=b.job_key
    d['review']=b.write('final-verdict.json',decision)
    with b.j.writer():
        state=b.j.read();state['events'][b.job_key].update(closure=b.context,inventory=b.context,postrun=b.context);b.j.save(state)
    monkeypatch.setattr(b.j,'canonical_job',lambda key:b.j.read()['events'][key])
    monkeypatch.setattr(m,'record_inventory',Mock(return_value={'output_inventory_complete':True}))
    out=run(b);pin=b.j.read()['events'][out['event_key']]['delivery']
    with b.j.writer():
        pub=publication.publish(b.j,pin,b.store)
        state=b.j.read();state['events']['controller-fixture']=dict(kind='cpu',identity={'role':'r6-controller'},request={'store_root':str(b.store.root.resolve())},cursor={'final_delivery':pin,'final_publication':pub});b.j.save(state)
    b.out=out;b.pin=pin;b.pub=pub;return b


def test_full_terminal_audit_links_published_bytes_accounting_and_originals(auditable):
    b=auditable
    with b.j.writer():out=m.audit(b.j,'controller-fixture',b.store)
    assert out['scientific_status']=='BASELINE_SUFFICIENT'
    assert out['inventories']==[dict(event_key=b.job_key,inventory=b.context,closure=b.context,postrun=b.context)]
    assert not b.j.read().get('goal_complete')
    with b.j.writer():result=m.finish(b.j,'controller-fixture',b.store)
    assert result['goal_complete']is True
    with b.j.writer():assert m.finish(b.j,'controller-fixture',b.store)==result


@pytest.mark.parametrize('fault',['ledger','missing_closure','incomplete_inventory','extra_job','wait_proof','publication','tag'])
def test_terminal_audit_missing_evidence_remains_incomplete(auditable,monkeypatch,fault):
    b=auditable
    if fault=='incomplete_inventory':m.record_inventory.return_value={'output_inventory_complete':False}
    elif fault=='wait_proof':
        original=m.delivery.evidence.model_audit
        monkeypatch.setattr(m.delivery.evidence,'model_audit',lambda state:dict(original(state),platform_inactivity_proven=False))
    elif fault=='tag':b.git('tag','-f',b.pub['tag'],b.git('rev-parse',b.pub['commit']+'^'))
    else:
        with b.j.writer():
            s=b.j.read()
            if fault=='ledger':s['budget']['used']['queue_seconds']+=1
            elif fault=='missing_closure':s['events'][b.job_key].pop('closure')
            elif fault=='extra_job':s['events']['extra-job']=copy.deepcopy(s['events'][b.job_key])
            else:s['events']['controller-fixture']['cursor']['final_publication']=dict(b.pub,commit='0'*40)
            b.j.save(s)
    with b.j.writer(),pytest.raises(RecoveryError):m.finish(b.j,'controller-fixture',b.store)
    assert not b.j.read().get('goal_complete')
