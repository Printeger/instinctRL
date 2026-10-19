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
