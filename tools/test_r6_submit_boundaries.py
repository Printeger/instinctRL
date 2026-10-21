"""Real submit/Store regressions; synthetic resource clocks, no launched process."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rlflow import experiments
from tools import r6_resource_queue as queue
from tools.r6_recovery import RecoveryError
from tools.r6_submit import submit_once
from tools.test_r6_submit_store import bound


@pytest.fixture
def delayed(bound, monkeypatch):
    b=bound; clock=queue.now()
    monkeypatch.setattr(queue,'now',lambda:dict(clock))
    original=experiments.validate
    def validate(*args):
        result=original(*args)
        if b.journal.read()['events'][b.key].get('submission_claim'):
            clock['monotonic_ns']+=9*10**9
        return result
    monkeypatch.setattr(experiments,'validate',validate)
    def refresh():
        entries=[(k,e) for k,e in b.journal.read()['events'].items()
                 if e['identity']['role']=='resource-window']
        _,old=max(entries,key=lambda pair:pair[1]['request']['ordinal'])
        clock['monotonic_ns']=old['finished_clock']['monotonic_ns']+60*10**9
        key=queue.begin(b.journal,old['request']['spec'],
            ordinal=old['request']['ordinal']+1,free_disk_bytes=500*1024**3)
        for index in range(13):
            if index:clock['monotonic_ns']+=5*10**9
            sample=copy.deepcopy(old['samples'][index]);sample['monotonic_ns']=clock['monotonic_ns']
            queue.append_sample(b.journal,key,index,sample)
        queue.finish(b.journal,key)
        return key
    b.refresh=Mock(side_effect=refresh);b.clock=clock
    monkeypatch.setattr('rlflow.releases.freeze',lambda root:b.store.root)
    return b


def invoke(b,refresh=True):
    claim=b.journal.read()['events'][b.key].get('submission_claim')
    with b.journal.writer():
        return submit_once(b.journal,b.key,b.store,
            queue_key=claim['queue_key'] if claim else b.queue_key,
            refresh_observation=b.refresh if refresh else None)


def test_stale_after_real_submit_validation_cannot_reserve(delayed):
    b=delayed
    with pytest.raises(RecoveryError,match='QUEUE_STALE'):invoke(b,False)
    assert not b.store.replay()['jobs']
    b.popen.assert_not_called()
    with pytest.raises(RecoveryError,match='AMBIGUOUS_NO_RETRY'):invoke(b)
    b.refresh.assert_not_called()


@pytest.mark.parametrize('revoke',[False,True])
def test_full_authority_validation_precedes_supervisor_but_revocation_still_checked(delayed,monkeypatch,revoke):
    b=delayed; original=experiments.authorized; launched=[]; validations=[]
    def authorized(*args):
        assert not launched, 'expensive validation must precede supervisor timeout'
        validations.append(True)
        return original(*args)
    monkeypatch.setattr(experiments,'authorized',authorized)
    monkeypatch.setattr(experiments,'identity',lambda pid:dict(pid=pid,pgid=pid,
        starttime='0',boot_id='CPU-NOT-LIVE'))
    def popen(*args,**kwargs):
        launched.append(True)
        if revoke:
            b.store.transact('FIXTURE_REVOKE','late-revoke',
                lambda s:s['approvals'][b.approval].update(revoked=True))
        return SimpleNamespace(pid=99999999)
    b.popen.side_effect=popen
    invoke(b)
    d=b.store.root/'jobs'/b.jid
    assert len(validations)==3 # outer, reservation, final full pre-launch check
    assert (d/'cancel').exists()==revoke
    assert (d/'supervisor-release.json').exists()==(not revoke)
    if not revoke:
        assert json.loads((d/'supervisor-release.json').read_text())['job_id']==b.jid


def test_real_submit_refreshes_at_reserve_and_launch_and_never_retries(delayed,monkeypatch):
    b=delayed
    def freeze(root):
        claim=b.journal.read()['events'][b.key]['submission_claim']
        assert b.store.replay()['jobs'][b.jid]['status']=='START_INTENT'
        assert queue.observation_fresh(b.journal,claim['queue_key'])
        b.clock['monotonic_ns']+=9*10**9
        return b.store.root
    monkeypatch.setattr('rlflow.releases.freeze',freeze)
    def popen(*args,**kwargs):
        claim=b.journal.read()['events'][b.key]['submission_claim']
        assert queue.observation_fresh(b.journal,claim['queue_key'])
        raise OSError('injected before process launch')
    b.popen.side_effect=popen
    with pytest.raises(OSError,match='before process launch'):invoke(b)
    claim=b.journal.read()['events'][b.key]['submission_claim']
    assert [v['boundary'] for v in claim['admission_history']]==['reserve','launch']
    assert b.refresh.call_count==2
    assert len({b.queue_key,*[v['queue_key'] for v in claim['admission_history']]})==3
    b.refresh.side_effect=AssertionError('recovery cannot sample')
    result=invoke(b)
    assert invoke(b)==result
    assert b.popen.call_count==1
    assert result['queue_window']==claim['queue_window']


@pytest.mark.parametrize('boundary',['reserve','launch'])
@pytest.mark.parametrize('failure',['stale','stop','deadline'])
def test_failed_boundary_refresh_does_not_launch(delayed,monkeypatch,boundary,failure):
    b=delayed; refresh=b.refresh.side_effect
    if boundary=='launch':
        def freeze(root):
            b.clock['monotonic_ns']+=9*10**9
            return b.store.root
        monkeypatch.setattr('rlflow.releases.freeze',freeze)
    def fail():
        if boundary=='launch' and b.refresh.call_count==1:return refresh()
        if failure=='stale':return b.queue_key
        key=refresh()
        if failure=='stop':(b.journal.directory/'STOP').touch()
        else:
            end=b.journal.read()['events'][b.key]['request']['wait_deadline_unix']
            monkeypatch.setattr('tools.r6_submit.time.time',lambda:end+1)
        return key
    b.refresh.side_effect=fail
    with pytest.raises(RecoveryError,match='QUEUE_STALE|STOP|DEADLINE'):invoke(b)
    assert (b.jid in b.store.replay()['jobs'])==(boundary=='launch')
    b.popen.assert_not_called()
