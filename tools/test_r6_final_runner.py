"""Final runner composition with real Journal/Store and explicit close/replay substitutes."""
import json
from unittest.mock import Mock
import pytest
from tools import r6_run_job as runner,r6_finalize_job as finalizer
from tools.test_r6_submit_real_contract import make_bound
from tools.r6_recovery import file_pin,RecoveryError

@pytest.fixture
def ready(tmp_path,monkeypatch):
    b=make_bound(tmp_path,monkeypatch,'research-r6-final-evaluation',final_parameters={})
    b.spec_pin=b.journal.read()['events'][b.key]['request']['spec'];b.exe=b.spec['pins']['executable']
    # Existing executable fixture must be under Journal parent; use a local data-only pin.
    path=tmp_path/'decision-fixture';path.write_text('never executed');b.decision=file_pin(path)
    b.observe=Mock(return_value=dict(status='OBSERVATION_PASS',event_key=b.queue_key))
    b.close=Mock(return_value=dict(event_key=b.key,closure={'native_status':'FAILED'}))
    monkeypatch.setattr(runner,'wait_for_observation',b.observe);monkeypatch.setattr(runner,'register_and_close',b.close)
    b.finalize=Mock();monkeypatch.setattr(finalizer,'finalize',b.finalize)
    def replay(journal,key,store,**kwargs):
        state=journal.read()
        for name in ('inventory','postrun','analysis'):
            p=tmp_path/(name+'.json');p.write_text(json.dumps({'fixture':name}))
            state['events'][key][name]=file_pin(p)
        journal.save(state)
    b.replay=Mock(side_effect=replay);monkeypatch.setattr(finalizer,'analyze',b.replay)
    monkeypatch.setattr(runner,'finish_and_decide',Mock(side_effect=AssertionError('final must not call model decision')))
    monkeypatch.setattr(finalizer,'decide',Mock(side_effect=AssertionError('no per-model decision')))
    return b


def run(b,**kwargs):
    with b.journal.writer():return runner.run(b.journal,b.spec_pin,b.store,b.exe,b.decision,
        free_disk_bytes=500*1024**3,purpose=kwargs.get('purpose','confirmation_delivery'))


def test_final_job_closes_and_analyzes_without_model(ready):
    b=ready;before=len([e for e in b.journal.read()['events'].values() if e['kind']=='model'])
    result=run(b)
    assert result['status']=='FINAL_JOB_ANALYSIS_RETURNED' and result['event_key']==b.key
    assert result['closure']['native_status']=='FAILED' and not result['scientific_acceptance']
    assert result['result']['analysis']=={'fixture':'analysis'}
    b.finalize.assert_called_once();b.replay.assert_called_once()
    assert len([e for e in b.journal.read()['events'].values() if e['kind']=='model'])==before


def test_unprotected_purpose_cannot_queue_final_job(ready):
    with pytest.raises(RecoveryError,match='FINAL_PURPOSE'):run(ready,purpose='development')
    ready.observe.assert_not_called();ready.close.assert_not_called()


def test_analysis_failure_does_not_fall_back_to_model(ready):
    ready.replay.side_effect=RecoveryError('raw replay incomplete')
    with pytest.raises(RecoveryError,match='raw replay incomplete'):run(ready)
    ready.finalize.assert_called_once()


def test_existing_per_model_decision_is_not_reused_as_final_acceptance(ready):
    b=ready
    with b.journal.writer():
        state=b.journal.read();state['events']['bad-model']=dict(kind='model',identity=dict(job_id=b.jid,role='post-analysis-decision'))
        b.journal.save(state)
        with pytest.raises(RecoveryError,match='UNEXPECTED_MODEL_DECISION'):
            finalizer.finish_final(b.journal,b.key,b.store,b.exe,free_disk_bytes=500*1024**3)
    b.finalize.assert_not_called()
