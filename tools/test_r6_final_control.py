"""Final controller boundaries; publication and frozen interpreter are explicit fixtures."""
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools import r6_final_job as gate
from tools.r6_recovery import RecoveryError,file_pin
from tools.test_r6_job_resources import candidate_inputs
from tools.r6_job_resources import resources

@pytest.fixture
def batch(tmp_path,monkeypatch):
    def pin(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return file_pin(p)
    protocol=pin('protocol.json',dict(blocks=[dict(id=str(i)) for i in range(16)]))
    baseline=pin('baseline.json',{})
    freeze=pin('freeze.json',dict(evidence=dict(baseline_bindings=baseline)))
    assignment=dict(model_role='baseline',evaluation_seed=171,split='final')
    value=dict(protocol=protocol,freeze=freeze,assignments=[assignment])
    bp=pin('batch.json',value)
    verified=Mock(return_value=value);monkeypatch.setattr(gate.publication,'validate_batch',verified)
    p=dict(final_selection=dict(schema_version='r6-final-selection/1',batch=bp,assignment_index=0),
        baseline_bindings=baseline,evaluation_seed=20260909,r6=dict(protocol=protocol,seed=171,split='final',block_ids=[str(i) for i in range(16)]))
    return p,verified,assignment


def test_controller_requires_actual_published_batch(batch):
    p,verified,assignment=batch
    assert gate.validate('journal',p,'confirmation_delivery')==assignment
    verified.assert_called_once_with('journal',p['final_selection']['batch'])
    verified.side_effect=RecoveryError('publication no release')
    with pytest.raises(RecoveryError,match='no release'):gate.validate('journal',p,'confirmation_delivery')


@pytest.mark.parametrize('fault',['purpose','mixed','index','bool','seed','split','blocks','baseline','original'])
def test_final_batch_drift_rejected(batch,fault):
    p,verified,_=batch;purpose='confirmation_delivery'
    if fault=='purpose':purpose='development'
    elif fault=='mixed':p['r6_training_plan']={}
    elif fault=='index':p['final_selection']['assignment_index']=1
    elif fault=='bool':p['final_selection']['assignment_index']=False
    elif fault=='seed':p['r6']['seed']+=1
    elif fault=='split':p['r6']['split']='development'
    elif fault=='blocks':p['r6']['block_ids'].reverse()
    elif fault=='baseline':p['baseline_bindings']={}
    elif fault=='original':p['evaluation_seed']+=1
    with pytest.raises(RecoveryError):gate.validate('journal',p,purpose)


@pytest.mark.parametrize('bad',[None,'training_transitions','optimizer_updates','checkpoint_writes','nontraining_transitions','mode'])
def test_final_charge_never_becomes_training(bad):
    spec,auth=candidate_inputs();spec['h1']['mode']=auth['mode']='research-r6-final-evaluation'
    if bad=='mode':auth['mode']='research-r6-evaluation'
    elif bad:auth['budgets'][bad]=1
    if bad:
        with pytest.raises(RecoveryError):resources(spec,auth)
    else:
        r=resources(spec,auth);assert r['nontraining_transitions']==512 and 'training_transitions' not in r
        assert resources(spec,auth,prepaid=True)['output_actual_plus_unreleased_reservations_bytes']==0


def test_final_spec_and_submission_reach_provenance_gate(tmp_path,monkeypatch):
    from tools.test_r6_submit_real_contract import make_bound,invoke
    from tools.test_r6_build_spec import inputs
    from tools.r6_build_spec import build
    b=make_bound(tmp_path,monkeypatch,'research-r6-final-evaluation')
    args=inputs(b);spec=build(**args)
    assert spec['h1']['mode']=='research-r6-final-evaluation' and spec['seed']==19
    # inputs changed the auth bytes: restore original pin for the existing submission fixture.
    authpath=Path(b.spec['pins']['authorization']['path'])
    original=dict(parameters=dict(evaluation_seed=19,output_cap_bytes=1024**3),
        budgets=dict(wall_seconds=60,training_transitions=0,nontraining_transitions=512,optimizer_updates=0,checkpoint_writes=0),mode='research-r6-final-evaluation')
    authpath.write_text(json.dumps(original));assert file_pin(authpath)==b.spec['pins']['authorization']
    check=Mock(side_effect=RecoveryError('final publication not released'));monkeypatch.setattr(gate,'validate',check)
    before=b.store.replay()
    with pytest.raises(RecoveryError,match='not released'):invoke(b)
    check.assert_called_once();assert b.store.replay()==before;b.popen.assert_not_called()


def test_final_authority_program_uses_real_stage_budget(monkeypatch,request):
    import io,sys
    from tools.test_r6_final_contract import stage
    from tools import r6_prepare_authorization as proposal
    from training_workloads.unified_h1 import config,robustness_final_contract as contract
    b=stage.__wrapped__(monkeypatch,request)
    cfg=dict(schema_version=config.R6_FINAL_CONFIG_SCHEMA,method=dict(proposal_sha256='a'*64),resolved_identity=dict(config_sha256='b'*64))
    monkeypatch.setattr(config,'load_config',lambda:cfg)
    compat=Mock();monkeypatch.setattr(contract,'validate_compatibility',compat)
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(dict(parameters=b.p,attempt='/fixture'))))
    monkeypatch.setattr(sys,'argv',['fixture',str(Path.cwd())]);out=io.StringIO();monkeypatch.setattr(sys,'stdout',out)
    original=list(sys.path)
    try:exec(proposal.PROGRAM,{})
    finally:sys.path[:]=original
    auth=json.loads(out.getvalue())
    assert auth['mode']==contract.MODE and auth['budgets']['nontraining_transitions']==193600
    assert auth['budgets']['optimizer_updates']==0 and auth['budgets']['checkpoint_writes']==0
    compat.assert_called_once_with(b.p['checkpoint_compatibility'],b.p['final_selection'],cfg)


def test_final_issue_uses_registered_batch_and_replays(tmp_path,monkeypatch,batch):
    from tools.test_r6_submit_real_contract import make_bound
    from tools.test_r6_issue_next import ready,issue
    from rlflow.store import sha
    b=ready.__wrapped__(make_bound(tmp_path,monkeypatch,'research-r6-final-evaluation'),monkeypatch)
    p,verified,_=batch
    auth=b.observed['authorization'];auth['parameters'].update(p)
    ap=Path(b.newspec['pins']['authorization']['path']);ap.write_text(json.dumps(auth))
    b.newspec['pins']['authorization']=file_pin(ap);b.newspec['seed']=20260909
    sp=tmp_path/'final-spec.json';sp.write_text(json.dumps(b.newspec))
    b.plan.update(spec_pin=file_pin(sp),purpose='confirmation_delivery')
    pp=tmp_path/'final-plan.json';pp.write_text(json.dumps(b.plan));b.pp=file_pin(pp);b.args['plan_pin']=b.pp
    b.jid='J-'+sha(dict(id=b.newspec['id'],spec=b.newspec))[:16];b.review.return_value=(b.plan,b.newspec,b.jid)
    first=issue(b);state=b.store.replay();budget=b.journal.read()['budget']
    assert first['status']=='EXACT_NEXT_JOB_AUTHORIZED_NOT_SUBMITTED'
    assert issue(b)==first and b.store.replay()==state and b.journal.read()['budget']==budget
    assert verified.call_count==2;b.popen.assert_not_called()
    verified.side_effect=RecoveryError('publication drift')
    with pytest.raises(RecoveryError,match='publication drift'):issue(b)
    assert b.store.replay()==state


def test_final_submission_passes_provenance_before_single_process_attempt(tmp_path,monkeypatch,batch):
    from tools.test_r6_submit_real_contract import make_bound,invoke
    p,verified,_=batch
    b=make_bound(tmp_path,monkeypatch,'research-r6-final-evaluation',final_parameters=p)
    with pytest.raises(OSError,match='CPU stop before Popen'):invoke(b)
    verified.assert_called_once();b.popen.assert_called_once()
    assert b.journal.read()['events'][b.key]['submission_claim']['job_id']==b.jid
    assert b.store.replay()['jobs'][b.jid]['status']=='START_INTENT'
    recovered=invoke(b)
    assert recovered['job_id']==b.jid and recovered['automatic_resubmission'] is False
    b.popen.assert_called_once()
