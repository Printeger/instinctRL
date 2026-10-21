"""Real temporary Git/entropy/Journal; model provenance uses holdout fixture boundary."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools import r6_holdout_publication as p
from tools.test_r6_holdout_journal import setup
from tools.r6_recovery import RecoveryError


@pytest.fixture
def ready(setup,monkeypatch):
    j,request,write=setup;root=j.directory.parent
    monkeypatch.setattr(p,'ROOT',root)
    p.git('init','-q','-b','main');p.git('config','user.name','CPU Fixture');p.git('config','user.email','fixture@example.invalid')
    (root/'initial.txt').write_text('initial');p.git('add','initial.txt');p.git('commit','-qm','Initial fixture')
    freeze=json.loads(p.checked(request['freeze']).read_text())
    freeze['selection_rule']={'final_realization_indices':[0,1,2]}
    request['freeze']=write('freeze-v2.json',freeze)
    request['review']=write('review-v2.json',dict(decision='PASS_R6_FINAL_FREEZE',must_fix=[],reviewer='fixture',freeze_sha256=request['freeze']['sha256']))
    draw=Mock(return_value='12'*32);monkeypatch.setattr(p.holdout.secrets,'token_hex',draw)
    return j,request,write,draw


def run(b):
    j,request,_,_=b
    with j.writer():return p.prepare(j,request,free_disk_bytes=500*1024**3)


def test_commit_only_public_hash_then_release_three_baseline_streams_once(ready):
    j,request,write,draw=ready;root=j.directory.parent
    (root/'unrelated.txt').write_text('staged unrelated');p.git('add','unrelated.txt')
    (root/'initial.txt').write_text('unstaged unrelated')
    result=run(ready);budget=j.read()['budget'];assert run(ready)==result and j.read()['budget']==budget
    assert draw.call_count==1
    assert p.git('diff','--cached','--name-only')=='unrelated.txt'
    assert p.git('diff','--name-only')=='initial.txt'
    changed=p.git('diff-tree','--no-commit-id','--name-only','-r',result['publication_commit'])
    assert changed.endswith('/commitment.json') and '\n' not in changed
    assert 'nonce' not in json.loads((root/changed).read_text())
    assert not any('private' in name or 'realization' in name for name in p.git('ls-files').splitlines())
    batch=json.loads(p.checked(result['batch']).read_text())
    with j.writer():assert p.validate_batch(j,result['batch'])==batch
    assert [a['model_role'] for a in batch['assignments']]==['baseline']*3
    assert [a['realization_index'] for a in batch['assignments']]==[0,1,2]
    assert len({a['evaluation_seed'] for a in batch['assignments']})==3
    assert not batch['dispatch_authorized'] and not batch['scientific_acceptance']


def test_crash_after_git_commit_recovers_same_commit_without_redrawing(ready,monkeypatch):
    original=p.git;failed=[]
    def git(*args):
        value=original(*args)
        if args[0]=='commit' and not failed:failed.append(True);raise RuntimeError('after git commit')
        return value
    monkeypatch.setattr(p,'git',git)
    with pytest.raises(RuntimeError,match='after git commit'):run(ready)
    commit=original('rev-parse','HEAD');result=run(ready)
    assert result['publication_commit']==commit and original('rev-list','--count','HEAD')=='2'
    assert ready[3].call_count==1


def test_crash_after_release_recovers_identical_realization(ready,monkeypatch):
    original=p.persist_exact;failed=[]
    def persist(path,data):
        if path.name=='realization.json' and not failed:failed.append(True);raise RuntimeError('after release')
        return original(path,data)
    monkeypatch.setattr(p,'persist_exact',persist)
    with pytest.raises(RuntimeError,match='after release'):run(ready)
    j=ready[0];event=next(e for e in j.read()['events'].values() if e['identity']['role']==p.holdout.ROLE)
    release=copy.deepcopy(event['release']);run(ready)
    event=next(e for e in j.read()['events'].values() if e['identity']['role']==p.holdout.ROLE)
    assert event['release']==release and ready[3].call_count==1 and p.git('rev-list','--count','HEAD')=='2'


def test_stop_after_commit_prevents_release_but_preserves_recoverable_commit(ready,monkeypatch):
    original=p.git;j=ready[0]
    def git(*args):
        value=original(*args)
        if args[0]=='commit':(j.directory/'STOP').touch()
        return value
    monkeypatch.setattr(p,'git',git)
    with pytest.raises(RecoveryError,match='STOP'):run(ready)
    event=next(e for e in j.read()['events'].values() if e['identity']['role']==p.holdout.ROLE)
    assert 'release' not in event
    (j.directory/'STOP').unlink()  # Isolated test STOP only.
    result=run(ready);assert result['publication_commit']==original('rev-parse','HEAD') and ready[3].call_count==1


def test_unrelated_head_change_before_publication_is_not_silently_rebased(ready,monkeypatch):
    original=p.commit_publication
    monkeypatch.setattr(p,'commit_publication',Mock(side_effect=RuntimeError('before commit')))
    with pytest.raises(RuntimeError):run(ready)
    root=ready[0].directory.parent;(root/'other.txt').write_text('other');p.git('add','other.txt');p.git('commit','-qm','Other change')
    monkeypatch.setattr(p,'commit_publication',original)
    with pytest.raises(RecoveryError,match='HEAD_CHANGED'):run(ready)
    assert ready[3].call_count==1


def test_stop_before_prepare_has_no_effect(ready):
    j=ready[0];before=j.read();(j.directory/'STOP').touch()
    with pytest.raises(RecoveryError,match='STOP'):run(ready)
    assert j.read()==before;ready[3].assert_not_called();assert p.git('rev-list','--count','HEAD')=='1'


def test_paired_model_assignment_never_indexes_randomness_by_model():
    freeze=dict(branch='THREE_CONTINUATIONS',models={r:{'fixture':r} for r in ('baseline','seed1','seed2','seed3')})
    rows=p.assignments(freeze,dict(split='final',realization_seeds=[7,8,9]))
    assert [r['model_role'] for r in rows]==['baseline','seed1','seed2','seed3']
    assert {r['evaluation_seed'] for r in rows}=={7} and {r['realization_index'] for r in rows}=={0}


@pytest.mark.parametrize('seeds',[[7,7,9],[7,8],[True,8,9],[-1,8,9]])
def test_malformed_final_streams_rejected(seeds):
    with pytest.raises(RecoveryError):p.assignments({'branch':'BASELINE_SUFFICIENT'},dict(split='final',realization_seeds=seeds))


@pytest.mark.parametrize('fault',['model','seed','index','commit','protocol','release','nonce','acceptance'])
def test_batch_consumer_rejects_substitution_even_with_recomputed_file_pin(ready,fault):
    j,_,_,draw=ready;result=run(ready);pin=result['batch'];batch=json.loads(p.checked(pin).read_text())
    if fault=='model':batch['assignments'][0]['checkpoint']=batch['protocol']
    if fault=='seed':batch['assignments'][0]['evaluation_seed']+=1
    if fault=='index':batch['assignments'][0]['realization_index']=2
    if fault=='commit':batch['publication_commit']='0'*40
    if fault=='protocol':batch['protocol']=batch['freeze']
    if fault=='acceptance':batch['scientific_acceptance']=True
    if fault=='release':
        with j.writer():
            state=j.read();state['events'][batch['holdout_event_key']].pop('release');j.save(state)
    if fault=='nonce':
        path=p.checked(batch['realization']);realization=json.loads(path.read_text());realization['nonce']='34'*32
        path.write_text(json.dumps(realization));batch['realization']=p.file_pin(path)
    path=p.checked(pin);path.write_text(json.dumps(batch));pin=p.file_pin(path)
    with j.writer(),pytest.raises((RecoveryError,ValueError)):p.validate_batch(j,pin)
    assert draw.call_count==1
