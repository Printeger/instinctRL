from types import SimpleNamespace
import pytest
from tools import r5_w_batch as b

def test_operation_intent_is_durable_and_argv_is_copied(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'RUN',tmp_path)
    state={'operations':{}};args=['example','submit'];observed=[]
    def run(argv,**kwargs):
        assert b.read(tmp_path/'state.json')['operations']['op']['intent']
        observed.append(list(argv));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(b.subprocess,'run',run)
    b.operation(state,'op',args);args[1]='settle'
    assert state['operations']['op']['argv']==['example','submit']
    b.operation(state,'op',['example','submit']);assert len(observed)==1


def test_ambiguous_side_effect_never_reexecuted(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'RUN',tmp_path)
    state={'operations':{'op':{'intent':True}}}
    monkeypatch.setattr(b.subprocess,'run',lambda *a,**k:pytest.fail('must not execute'))
    with pytest.raises(AssertionError,match='AMBIGUOUS_OPERATION'):b.operation(state,'op',['submit'])


def test_stop_requests_exact_owned_cancel_once_and_preserves_wait_state(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'RUN',tmp_path/'run');monkeypatch.setattr(b,'CONTROL',tmp_path/'control')
    (tmp_path/'run').mkdir();jd=tmp_path/'control/jobs/J-owned';jd.mkdir(parents=True)
    entry={'job_id':'J-owned','done':False};state={'entries':{'entry':entry}};calls=[]
    def cancel(store,jid):
        assert jid=='J-owned'
        assert b.read(tmp_path/'run/state.json')['entries']['entry']['cancel_intent']
        calls.append(jid);(jd/'cancel').write_text('cancel')
    monkeypatch.setattr(b.experiments,'cancel',cancel)
    b.cancel_once(state,entry,object(),'J-owned');b.cancel_once(state,entry,object(),'J-owned')
    assert calls==['J-owned'] and entry['stop_seen'] and not entry['done']


def test_old_and_new_stop_are_both_honored(tmp_path,monkeypatch):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    monkeypatch.setattr(b,'ORIGINAL_RUN',old);monkeypatch.setattr(b,'RUN',new)
    monkeypatch.setattr(b.policy,'stopped',lambda _:[])
    for directory in (old,new):
        stop=directory/'STOP';stop.touch()
        assert str(stop) in b.stopped()
        stop.unlink()


def test_resume_rejects_live_predecessor_and_preserves_bytes(tmp_path,monkeypatch):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir()
    monkeypatch.setattr(b,'ORIGINAL_RUN',old);monkeypatch.setattr(b,'RESOURCE_RUN',new);monkeypatch.setattr(b,'RUN',new)
    state={'status':'STOPPED_EXCEPTION','controller_identity':{'pid':1},'model_calls':{}}
    b.save(old/'state.json',state);raw=(old/'state.json').read_bytes()
    plan={'run_directory':str(new),'predecessor_state':b.pin(old/'state.json')}
    monkeypatch.setattr(b,'owns',lambda _:True)
    with pytest.raises(AssertionError):b.verify_resumption(plan)
    assert (old/'state.json').read_bytes()==raw
    state['controller_identity']=None;state['model_calls']={'x':{'worker_identity':{'pid':2}}}
    b.save(old/'state.json',state);plan['predecessor_state']=b.pin(old/'state.json')
    monkeypatch.setattr(b,'owns',lambda i:bool(i))
    with pytest.raises(AssertionError):b.verify_resumption(plan)


def test_new_run_uses_shared_lock(tmp_path,monkeypatch):
    import fcntl
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir()
    monkeypatch.setattr(b,'ORIGINAL_RUN',old);monkeypatch.setattr(b,'RUN',new)
    with (old/'writer.lock').open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):b.run(tmp_path/'unused')
    assert not (new/'state.json').exists()


def test_terminal_state_never_reopened(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'ORIGINAL_RUN',tmp_path);monkeypatch.setattr(b,'RUN',tmp_path)
    b.save(tmp_path/'state.json',{'status':'STOPPED_EXCEPTION'})
    raw=(tmp_path/'state.json').read_bytes()
    with pytest.raises(AssertionError,match='preserve terminal'):b.run(tmp_path/'unused')
    assert (tmp_path/'state.json').read_bytes()==raw


def test_worker_receives_exact_run_directory(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'RUN',tmp_path)
    assert b.worker_argv(tmp_path/'calls/x')[-2:]==['--run-directory',str(tmp_path)]
