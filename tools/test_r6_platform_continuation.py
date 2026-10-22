"""Current continuation must not inherit the old thread's handoff identity."""
import copy,hashlib,json
import pytest
from tools import r6_platform_wait as wait,r6_platform_rpc as rpc
from tools.r6_recovery import file_pin,RecoveryError

@pytest.fixture
def continuation(tmp_path,monkeypatch):
    trace=tmp_path/'trace.jsonl';trace.write_bytes(b'{}\n')
    authority=tmp_path/'authority';authority.write_text('CPU user authority')
    monkeypatch.setattr(wait,'CONTINUATION',trace)
    doc=dict(previous_session_id=wait.SESSION_ID,session_id=wait.CONTINUATION_ID,repository=str(wait.ROOT),
        anchor=dict(path=str(trace),bytes=3,sha256=hashlib.sha256(b'{}\n').hexdigest()),authority=file_pin(authority))
    path=tmp_path/'binding.json';path.write_text(json.dumps(doc))
    state=dict(goal_id='R6-ROBUSTNESS-AUTO-20261019',platform_continuation_binding=file_pin(path),
        platform_automatic_wake_accounting={'count':196,'source_prefix':{'path':str(wait.SESSION)}})
    return state,doc,path

def test_new_identity_preserves_historical_accounting(continuation):
    state,doc,path=continuation;before=copy.deepcopy(state)
    assert wait.binding(state)=={k:doc[k] for k in ('session_id','repository','anchor')}
    assert state==before

@pytest.mark.parametrize('damage',['session','previous','path','authority','prefix'])
def test_bad_binding_cannot_fall_back_to_old_idle_thread(continuation,damage):
    state,doc,path=continuation
    if damage=='session':doc['session_id']=wait.SESSION_ID
    if damage=='previous':doc['previous_session_id']='unrelated'
    if damage=='path':doc['anchor']['path']=str(wait.SESSION)
    if damage=='authority':doc['authority']['sha256']='0'*64
    if damage=='prefix':doc['anchor']['sha256']='0'*64
    path.write_text(json.dumps(doc));state['platform_continuation_binding']=file_pin(path)
    with pytest.raises(RecoveryError):wait.binding(state)

def test_rpc_selects_current_objective_without_query(continuation,monkeypatch,tmp_path):
    state,_,_=continuation
    sock=tmp_path/'sock';sock.touch();alias=tmp_path/'alias';alias.symlink_to(sock)
    monkeypatch.setattr(rpc,'SOCKET_ALIAS',alias);monkeypatch.setattr(rpc,'SOCKET',sock)
    cfg=rpc.settings(state)
    assert cfg['source']['session_id']==wait.CONTINUATION_ID
    assert '7d738239-2883-4e11-9aab-785b93ffd910' in cfg['objective']
