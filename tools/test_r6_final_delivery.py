"""Real temporary Git/Journal/accounting; accepted science and review are explicit fixtures."""
import json,subprocess
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools import r6_final_delivery as m, r6_publish_final_delivery as publication
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import file_pin,RecoveryError

DESCRIBE=m.describe
FREE=500*1024**3

@pytest.fixture
def ready(case,tmp_path,monkeypatch):
    b=case;repo=tmp_path/'repo';repo.mkdir();remote=tmp_path/'remote.git'
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE).decode().strip()
    subprocess.run(['git','init','--bare','-q',str(remote)],check=True)
    git('init','-q','-b','main');git('config','user.name','CPU fixture');git('config','user.email','cpu@example.invalid');git('remote','add','origin',str(remote))
    for module in (m,m.profile,m.evidence,m.packet,m.review,publication):
        p=repo/'tools'/Path(module.__file__).name;p.parent.mkdir(exist_ok=True);p.write_bytes(Path(module.__file__).read_bytes());monkeypatch.setattr(module,'__file__',str(p))
    historic=repo/'docs/experiments/r6-robustness-auto/MODEL_USAGE_AUDIT.json';historic.parent.mkdir(parents=True);historic.write_text('historical bytes retained\n')
    git('add','.');git('commit','-qm','fixture reviewed implementation')
    monkeypatch.setattr(m,'ROOT',repo);b.repo=repo;b.git=git;b.historic=historic
    report=b.write('baseline-report.json',dict(blocks=[dict(block_id='fixture',domain='target',scoring=dict(successes=32,total=32))],summary=dict(original={'all_gates_pass':True})))
    baseline=[dict(event_key='baseline',job_id='J-fixture',mode='research-r6-evaluation',native_status='FAILED',native_exit_code=-9,report=report)]
    final=dict(results=[dict(assignment_index=i,assignment=dict(model_role='baseline',evaluation_seed=17+i),gates=dict(original_pass=True,target_pass=True,blocks=[dict(block_id='fixture',domain='target',successes=32,total=32)]),native_status='FAILED',native_exit_code=-9) for i in range(3)])
    review=b.write('accepted-review.json',dict(summary='CPU fixture independent verdict',requirements_audit={'diagnosis_decisions':{'reason':'CPU diagnosis'},'attribution_limits':{'reason':'CPU limitations'}}))
    media=b.write('media-manifest.json',dict(viewer=b.context,videos=[]))
    d=dict(controller_key='fixture',audit_packet=b.context,review_key=b.parent,review=review,review_receipt=b.j.read()['events'][b.parent]['receipt'],scientific_status='BASELINE_SUFFICIENT',final=final,branch='BASELINE_SUFFICIENT',models={'baseline':b.context},jobs=dict(jobs=baseline),baseline=baseline,decisions=[],video_manifest=media,requirements=b.context,requirements_audit={'fixture':'pass'},protocol=b.context,split_manifest=b.context,baseline_bindings=b.context,master_authority=b.context,scope=b.context)
    b.description=d;monkeypatch.setattr(m,'describe',Mock(return_value=d))
    return b


def run(b):
    with b.j.writer():return m.prepare(b.j,'fixture',b.store,free_disk_bytes=FREE)


def test_accounting_includes_review_and_own_reservation_without_history_rewrite(ready):
    b=ready;out=run(b);budget=b.j.read()['budget'];assert run(b)==out and b.j.read()['budget']==budget
    assert set(out['files'])==m.REQUIRED and not out['goal_complete']
    resources=m.read(out['files']['RESOURCE_AUDIT.json']);models=m.read(out['files']['MODEL_USAGE_AUDIT.json'])
    assert models['final_review_receipt']==b.description['review_receipt']
    assert any(c['event_key']==b.parent for c in models['calls'])
    assert resources['charged']['automatic_model_sessions']==models['automatic_sessions_charged']
    assert any(row['identity']['role']==m.ROLE and row['reserved']['output_actual_plus_unreleased_reservations_bytes']==m.CAP for row in resources['reservations'])
    assert b.historic.read_text()=='historical bytes retained\n'
    pin=b.j.read()['events'][out['event_key']]['delivery']
    with b.j.writer():assert m.verify(b.j,pin,b.store)==out


def test_exact_publication_and_replay_preserve_unrelated_staging(ready):
    b=ready;(b.repo/'unrelated.txt').write_text('keep\n');b.git('add','unrelated.txt')
    out=run(b);pin=b.j.read()['events'][out['event_key']]['delivery'];budget=b.j.read()['budget']
    with b.j.writer():published=publication.publish(b.j,pin,b.store)
    commit=b.git('rev-parse','HEAD')
    assert published['commit']==commit==b.git('rev-parse','origin/main')==b.git('rev-parse',published['tag'])
    assert b.git('diff','--cached','--name-only')=='unrelated.txt' and b.historic.read_text()=='historical bytes retained\n'
    assert not published['goal_complete']
    with b.j.writer():assert publication.publish(b.j,pin,b.store)==published
    assert b.git('rev-parse','HEAD')==commit and b.j.read()['budget']==budget


def test_crash_after_commit_recovers_without_duplicate_commit(ready,monkeypatch):
    b=ready;out=run(b);pin=b.j.read()['events'][out['event_key']]['delivery'];original=m.git
    def fail(*args):
        if args[0]=='push':raise RuntimeError('CPU interrupted before push')
        return original(*args)
    monkeypatch.setattr(m,'git',fail)
    with b.j.writer(),pytest.raises(RuntimeError,match='before push'):publication.publish(b.j,pin,b.store)
    commit=b.git('rev-parse','HEAD');monkeypatch.setattr(m,'git',original)
    with b.j.writer():assert publication.publish(b.j,pin,b.store)['commit']==commit
    assert b.git('rev-parse','origin/main')==commit


@pytest.mark.parametrize('fault',['stop','artifact','extra','ledger','wake','head','tag'])
def test_drift_prevents_final_publication(ready,fault):
    b=ready;out=run(b);pin=b.j.read()['events'][out['event_key']]['delivery'];base=b.git('rev-parse','HEAD')
    if fault=='stop':(b.j.directory/'STOP').touch()
    elif fault=='artifact':m.checked(out['files']['MODEL_USAGE_AUDIT.json']).write_text('{}')
    elif fault=='extra':(Path(out['folder'])/'unclaimed').touch()
    elif fault in ('ledger','wake'):
        with b.j.writer():
            s=b.j.read()
            if fault=='ledger':s['budget']['used']['queue_seconds']+=1
            else:s['platform_automatic_wake_accounting']={'count':1}
            b.j.save(s)
    elif fault=='head':b.git('commit','--allow-empty','-qm','unrelated head')
    else:b.git('tag','r6-final-delivery-'+out['event_key'][:16],base)
    with b.j.writer(),pytest.raises(RecoveryError):publication.publish(b.j,pin,b.store)
    assert b.git('ls-remote','origin','refs/heads/main')==''


@pytest.mark.parametrize('fault',[None,'review_status','review_decision','job_drift','baseline_missing','receipt_status'])
def test_delivery_requires_accepted_original_review_and_same_jobs(ready,monkeypatch,fault):
    import copy
    b=ready;d=b.description;rows=copy.deepcopy(d['baseline']);rows[0]['parameters']={'master_authority':b.context}
    jobs=dict(jobs=rows);jobs_pin=b.write('reviewed-jobs.json',jobs)
    if fault=='job_drift':jobs=dict(jobs=[dict(rows[0],native_exit_code=0)])
    if fault=='baseline_missing':
        rows[0]['mode']='research-r6-final-evaluation';jobs_pin=b.write('reviewed-jobs2.json',dict(jobs=rows));jobs=dict(jobs=rows)
    monkeypatch.setattr(m.evidence,'jobs_manifest',Mock(return_value=jobs))
    freeze=b.write('freeze.json',dict(models=d['models'],evidence=dict(baseline_bindings=b.context)))
    protocol=b.write('protocol.json',dict(split_manifest=b.context));batch=b.write('batch.json',dict(freeze=freeze,protocol=protocol))
    final=dict(d['final'],batch=batch)
    description=dict(controller_key='fixture',final=final,branch=d['branch'],video_manifest=d['video_manifest'],requirements=b.context)
    monkeypatch.setattr(m.packet,'verify',Mock(return_value=dict(description=description,artifacts={'TRAINING_AND_EVALUATION_MANIFEST.json':jobs_pin})))
    decision=dict(decision='MUST_FIX' if fault=='review_decision' else 'PASS_R6_FINAL_AUDIT',scientific_status='BASELINE_SUFFICIENT',requirements_audit={})
    monkeypatch.setattr(m.review,'verify',Mock(return_value=decision))
    scope=b.repo/'docs/experiments/r6-robustness-auto/GOAL_SCOPE_AND_MASTER_AUTHORITY.md';scope.write_text('CPU authority fixture')
    with b.j.writer():
        s=b.j.read();s['events']['fixture']=dict(kind='cpu',identity={'role':'r6-controller'},request={'store_root':str(b.store.root.resolve())},cursor=dict(final_audit_packet=b.context,final_independent_audit=dict(status='FINAL_AUDIT_MUST_FIX_REQUIRED' if fault=='review_status' else 'FINAL_AUDIT_ACCEPTED_PENDING_PUBLICATION',preparation=b.context,event_key=b.parent,review=d['review'])))
        if fault=='receipt_status':s['events'][b.parent]['status']='INTENT'
        b.j.save(s)
        if fault:
            with pytest.raises(RecoveryError):DESCRIBE(b.j,'fixture',b.store)
        else:
            result=DESCRIBE(b.j,'fixture',b.store)
            assert result['review_receipt']==d['review_receipt'] and result['protocol']==protocol


def test_nested_output_symlink_does_not_touch_external_file(ready):
    b=ready;out=run(b);folder=Path(out['folder'])/'reviews'
    (folder/'final-independent.json').unlink();folder.rmdir()
    other=b.root/'external';other.mkdir();target=other/'final-independent.json';target.write_text('preserve external bytes')
    folder.symlink_to(other,target_is_directory=True)
    with pytest.raises(RecoveryError,match='ARTIFACT_ALIAS'):run(b)
    assert target.read_text()=='preserve external bytes'
