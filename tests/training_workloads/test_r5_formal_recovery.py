"""CPU-only adversarial tests of evidence and one-time recovery boundaries."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools import r5_formal_recovery as r
from tools import r5_formal_batch as b
from tools.analyze_r5_formal_binding_recovery import check_scalars


def previous():
    return {'status':'STOPPED','phase':'MODEL:checkpoint-review','model_call_limit':2,
        'model_calls':{'checkpoint-review':{'intent':True}},
        'entries':{g+'-research-formal-training':{'done':True} for g in ['R0','R1']}}


def test_only_exact_old_rejected_stage_can_recover():
    s=previous();r.validate_recovery_origin(s)
    for change in [dict(status='RUNNING'),dict(phase='ENTRY:R0-research-formal-training'),
        dict(recovery={}),dict(model_call_limit=3),dict(entries={}),
        dict(model_calls={'checkpoint-review':{'accepted':True}}),
        dict(model_calls={'checkpoint-review':{},'final-analysis':{}})]:
        with pytest.raises(RuntimeError):r.validate_recovery_origin({**s,**change})


def test_callback_field_and_loss_inclusion_are_required():
    updates=[{'completed_updates':u,'actor_optimizer_steps':u*8,'critic_optimizer_steps':u*8} for u in range(257,513)]
    mini=[{'update':u,'minibatch_ordinal':i,'actor_auxiliary_loss':.125,
        'actor_loss':1.,'value_loss':2.,'entropy':-1.,'total_loss':2.135}
        for u in range(257,513) for i in range(8)]
    cfg={'value_loss_coefficient':.5,'entropy_coefficient':.01}
    check_scalars(updates,mini,10,cfg)
    for field,value in [('actor_auxiliary_loss',0.),('total_loss',2.01),('actor_auxiliary_loss',float('nan'))]:
        changed=deepcopy(mini);changed[100][field]=value
        with pytest.raises(AssertionError):check_scalars(updates,changed,10,cfg)
    del mini[100]['actor_auxiliary_loss']
    with pytest.raises(KeyError):check_scalars(updates,mini,10,cfg)


def fixture(tmp_path,monkeypatch):
    x=object.__new__(r.RecoveryBatch);x.root=tmp_path;x.run=tmp_path/'run';x.run.mkdir()
    x.path=x.run/'state.json';x.interrupted=False;x.plan={'model_call_limit':3}
    b.write(tmp_path/r.PLAN,x.plan)
    x.state={'status':'ARMED','phase':'RECOVERY_ARMED','recovery':{'previous_failure':{}},
        'model_calls':{'checkpoint-review':{}},'final_analysis':'final.md'}
    monkeypatch.setattr(x,'guard',lambda:None)
    return x


def test_only_eval_after_review_then_single_final_model(tmp_path,monkeypatch):
    x=fixture(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(x,'call_model',lambda slot:calls.append(('model',slot)))
    monkeypatch.setattr(x,'require_review',lambda:calls.append(('accept','review')))
    monkeypatch.setattr(x,'run_entry',lambda group,mode:calls.append((group,mode)))
    monkeypatch.setattr(x,'verify_evaluations',lambda:calls.append(('CPU','verify')))
    x.execute()
    assert calls==[('model','checkpoint-recovery'),('accept','review'),
        ('R0','research-formal-evaluation'),('R1','research-formal-evaluation'),
        ('CPU','verify'),('model','final-analysis')]
    assert x.state['status']=='COMPLETE';x.execute();assert len(calls)==6


def test_recovery_cannot_call_training_even_after_review(tmp_path,monkeypatch):
    x=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(x,'require_review',lambda:None)
    monkeypatch.setattr(b.Batch,'run_entry',lambda *_:pytest.fail('no training side effect'))
    with pytest.raises(RuntimeError,match='NEVER_RUNS_TRAINING'):x.run_entry('R0','research-formal-training')


def test_rejected_extra_review_preserves_original_failure_and_never_retries(tmp_path,monkeypatch):
    x=fixture(tmp_path,monkeypatch);old=x.run/'FAILURE.json';old.write_bytes(b'original STOP')
    def stop(slot):raise RuntimeError('MODEL_REJECTED')
    monkeypatch.setattr(x,'call_model',stop)
    monkeypatch.setattr(x,'run_entry',lambda *_:pytest.fail('no eval before acceptance'))
    with pytest.raises(RuntimeError,match='MODEL_REJECTED'):x.execute()
    assert old.read_bytes()==b'original STOP'
    assert x.state['status']=='STOPPED' and Path(x.state['failure']['path']).parent.name==r.RECOVERY
    with pytest.raises(RuntimeError,match='NO_AUTOMATIC_RETRY'):x.execute()


def test_accepted_flag_alone_is_not_independent_evidence(tmp_path,monkeypatch):
    x=fixture(tmp_path,monkeypatch)
    with pytest.raises(RuntimeError,match='NOT_ACCEPTED'):x.require_review()
    x.state['model_calls']['checkpoint-recovery']={'accepted':True}
    monkeypatch.setattr(x,'verify_accepted_model',lambda _:None)
    p=x.checkpoint_review_path();p.parent.mkdir(parents=True);b.write(p,{'verdict':'PASS','must_fix':[]})
    with pytest.raises(RuntimeError,match='INDEPENDENTLY_ACCEPTED'):x.require_review()


def test_old_slot_cannot_spawn_again_and_limit_counts_failed_call(tmp_path,monkeypatch):
    x=fixture(tmp_path,monkeypatch)
    x.state['model_calls']={'checkpoint-review':{},'checkpoint-recovery':{},'final-analysis':{}}
    monkeypatch.setattr(x,'checkpoint_slot',lambda:'extra-forbidden')
    with pytest.raises(RuntimeError,match='MODEL_CALL_LIMIT'):x.call_model('extra-forbidden')


@pytest.mark.parametrize('problem',['existing-approval','partial-attempt','live-worker','changed-state'])
def test_arm_rejects_existing_effects_or_ownership_without_mutation(tmp_path,monkeypatch,problem):
    from tools import r5_formal_commander
    from rlflow import jobs
    x=fixture(tmp_path,monkeypatch);x.state=previous();x.case=tmp_path/'case'
    old=tmp_path/'old.json';b.write(old,x.state)
    x.plan['pins']={'previous_state':b.pin(old)}
    b.write(x.run/'calls/checkpoint-review/request.json',{'worker_identity':{'pid':77},'codex_identity':None})
    monkeypatch.setattr(jobs,'owns',lambda ident:bool(problem=='live-worker' and ident))
    monkeypatch.setattr(x,'store',lambda:None);monkeypatch.setattr(r5_formal_commander,'idle',lambda _:None)
    for g in ['R0','R1']:
        n=g+'-research-formal-evaluation'
        b.write(x.case/'inputs'/f'{n}-spec.proposed.json',{'h1':{'attempt':str(tmp_path/g)}})
    if problem=='existing-approval':b.write(x.case/'R0-research-formal-evaluation-approval.json',{})
    if problem=='partial-attempt':(tmp_path/'R0').mkdir()
    if problem=='changed-state':x.state['new_unreviewed_field']=True
    before=deepcopy(x.state)
    with pytest.raises(RuntimeError):x.arm()
    assert x.state==before and 'recovery' not in x.state
