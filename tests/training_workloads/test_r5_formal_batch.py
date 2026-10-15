"""No GPU/model calls: exercise fail-closed crash and side-effect boundaries."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from tools import r5_formal_batch as batch


def test_submit_never_repeats_existing_or_ambiguous():
    assert batch.can_submit({}, {'id':'existing'}, charged=True, admission=True) is False
    assert batch.can_submit({}, None, charged=False, admission=False)
    for e, charged, admission in [({'submit_intent':True},False,False), ({},True,False), ({},False,True)]:
        with pytest.raises(RuntimeError, match='AMBIGUOUS_SUBMIT'):
            batch.can_submit(e, None, charged=charged, admission=admission)


@pytest.mark.parametrize('count,files', [(1,[True,False]), (2,[False,True]), (0,[True,False]), (3,[True,True])])
def test_partial_postrun_never_requeries(count, files):
    with pytest.raises(RuntimeError, match='PARTIAL_POSTRUN'): batch.postrun_complete(count, files)


def test_complete_postrun_can_resume_without_query():
    assert batch.postrun_complete(2,[True,True])
    assert not batch.postrun_complete(0,[False,False])


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(x))


def test_known_close_requires_complete_actual_counters_and_first_gate(tmp_path):
    out = tmp_path/'results'; out.mkdir()
    raw={'status':'PASS_PENDING_CLOSE','primary_error':None,'cleanup_errors':[], 'counters':{
        'training_transitions':131072,'nontraining_transitions':0,'completed_updates':256,
        'real_optimizer_updates':256,'actor_optimizer_steps':2048,'critic_optimizer_steps':2048,'real_checkpoint_writes':5}}
    gate={'status':'PASS','sample_count':512,'starting_update':256,'first_absolute_update':257,
          'minibatches':8,'float32_atol':1e-6,'float32_rtol':1e-5,'policy_rng_consumed':False}
    write(out/'status.json',raw);write(out/'pilot_diagnostic/first_update_gate.json',gate)
    write(out/'cleanup_timeout.json',{'status':'HARD_TIMEOUT','cleanup':'app.close','timeout_seconds':9.5})
    for u in [272,288,320,384,512]:
        ck=out/'checkpoints'/f'u{u}.pt'; ck.parent.mkdir(exist_ok=True);ck.write_bytes(b'actualfixture')
        write(ck.with_suffix('.json'),{'path':str(ck),'sha256':batch.sha(ck)})
    receipt={'status':'FAILED','exit_code':-9};spec={'h1':{'mode':'research-formal-training'}}
    assert batch.science_complete(spec,receipt,out)['native_status']=='FAILED'
    gate['status']='FAILED';write(out/'pilot_diagnostic/first_update_gate.json',gate)
    with pytest.raises(RuntimeError,match='FIRST_ROLLOUT'):batch.science_complete(spec,receipt,out)
    gate['status']='PASS';write(out/'pilot_diagnostic/first_update_gate.json',gate)
    raw['counters']['completed_updates']=255;write(out/'status.json',raw)
    with pytest.raises(RuntimeError,match='COUNTERS'):batch.science_complete(spec,receipt,out)
    raw['status']='FAIL';raw['primary_error']={'message':'GPU utilization exceeds5'};write(out/'status.json',raw)
    with pytest.raises(RuntimeError,match='RESOURCE_FAILURE'):batch.science_complete(spec,receipt,out)


def fixture_batch(tmp_path, monkeypatch):
    static=tmp_path/batch.STATIC;write(static/'plan.json',{'pins':{},'model_call_limit':2,'experiment_id':batch.CASE,'poll_seconds':.001})
    b=batch.Batch(tmp_path);b.state={'status':'RUNNING','phase':'fixture','entries':{},'operations':{},'model_calls':{},'final_analysis':'fixture.md','plan':batch.pin(static/'plan.json')}
    monkeypatch.setattr(b,'verify_pins',lambda:None);monkeypatch.setattr(b,'guard',lambda:None)
    return b


def test_recover_existing_terminal_job_never_calls_submit_settle_or_queries(tmp_path, monkeypatch):
    from rlflow.store import sha
    from rlflow import experiments
    from tools import r5_formal_commander
    b=fixture_batch(tmp_path,monkeypatch);mode='research-formal-training';name='R0-'+mode
    spec={'id':'fixed','h1':{'attempt':str(tmp_path/'attempt'),'mode':mode},'resources':{'wall_seconds':1}}
    write(b.case/'inputs'/f'{name}.authorized.json',spec);jid='J-'+sha({'id':'fixed','spec':spec})[:16]
    job={'id':jid,'status':'FAILED','spec':spec,'identity':None}
    monkeypatch.setattr(b,'store',lambda:SimpleNamespace(replay=lambda:{'jobs':{jid:job},'paused':False}))
    monkeypatch.setattr(b,'stopped',lambda:[]);monkeypatch.setattr(experiments,'reconcile',lambda *_:job)
    monkeypatch.setattr(r5_formal_commander,'previous_quiescence',lambda *_:[])
    write(b.control/'jobs'/jid/'receipt.json',{'status':'FAILED'});write(b.control/'jobs'/jid/'lifeguard-ack.json',{})
    write(tmp_path/batch.BASE/'goal_budget.json',{'reservations':{'live_entries':[{'job_id':jid,'spec_id':'fixed','settled':True,'postrun_queries_used':2}]}})
    for i in range(2):write(b.case/f'{name}-postrun-{i}.json',{'job_id':jid,'exit_code':0})
    monkeypatch.setattr(batch,'science_complete',lambda *_:{'accepted':True})
    monkeypatch.setattr(b,'commander',lambda *_:pytest.fail('must not repeat any side effect'))
    b.run_entry('R0',mode);assert b.state['entries'][name]['done'];b.run_entry('R0',mode)


def test_fixed_order_only_two_model_boundaries(tmp_path,monkeypatch):
    b=fixture_batch(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(b,'run_entry',lambda g,m:calls.append((g,m)))
    monkeypatch.setattr(b,'call_model',lambda s:calls.append(('model',s)))
    monkeypatch.setattr(b,'verify_evaluations',lambda:calls.append(('CPU','raw-verification')))
    b.execute()
    assert calls==[('R0','research-formal-training'),('R1','research-formal-training'),('model','checkpoint-review'),
                   ('R0','research-formal-evaluation'),('R1','research-formal-evaluation'),('CPU','raw-verification'),('model','final-analysis')]
    assert b.state['status']=='COMPLETE';b.execute();assert len(calls)==7


def test_failure_stops_before_following_group_or_model(tmp_path,monkeypatch):
    b=fixture_batch(tmp_path,monkeypatch);calls=[]
    def fail(g,m):calls.append(g);raise RuntimeError('GPU_GATE_REFUSAL')
    monkeypatch.setattr(b,'run_entry',fail);monkeypatch.setattr(b,'call_model',lambda *_:pytest.fail('not allowed'))
    with pytest.raises(RuntimeError,match='GPU_GATE'):b.execute()
    assert calls==['R0'] and b.state['status']=='STOPPED'
    assert batch.read(b.run/'FAILURE.json')['no_automatic_retry']
    with pytest.raises(RuntimeError,match='BATCH_STOPPED'):b.execute()


def test_model_receipt_failure_does_not_spawn_again(tmp_path,monkeypatch):
    b=fixture_batch(tmp_path,monkeypatch);b.state['model_calls']['checkpoint-review']={'intent':True}
    write(b.run/'calls/checkpoint-review/receipt.json',{'exit_code':1,'error':'API_AUTH_DENIED'})
    monkeypatch.setattr(batch.subprocess,'Popen',lambda *_a,**_kw:pytest.fail('model may not retry'))
    with pytest.raises(RuntimeError,match='MODEL_EXIT_FAILURE'):b.call_model('checkpoint-review')


def test_atomic_state_failure_keeps_previous_json(tmp_path,monkeypatch):
    from rlflow import store
    p=tmp_path/'state.json';batch.write(p,{'phase':'before'})
    monkeypatch.setattr(store.os,'replace',lambda *_:(_ for _ in ()).throw(OSError('crash before rename')))
    with pytest.raises(OSError):batch.write(p,{'phase':'after'})
    assert batch.read(p)=={'phase':'before'}


def test_model_cannot_change_method_or_original_authority():
    assert batch.allowed_model_changes('checkpoint-review', [str(batch.RESULTS/'reviews/joint-actual-training.json'), str(batch.BASE/'goal_budget.json')])
    assert not batch.allowed_model_changes('checkpoint-review', ['docs/PROJECT_STATUS.md'])
    assert batch.allowed_model_changes('final-analysis', ['docs/PROJECT_STATUS.md'])
    for name in ['tools/r5_formal_commander.py','isaaclab-training/instinctlab/training/ppo.py',str(batch.BASE/'formal_r5_launch_manifest.json'),'docs/goal/user.md']:
        assert not batch.allowed_model_changes('final-analysis', [name])
