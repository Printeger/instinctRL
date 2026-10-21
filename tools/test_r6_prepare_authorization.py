"""Proposal plumbing; frozen runtime validators are explicit fixture boundaries."""
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools import r6_prepare_authorization as m
from tools.r6_recovery import file_pin,RecoveryError

@pytest.fixture(params=['research-r6-evaluation','research-r6-training'])
def case(tmp_path,monkeypatch,request):
    mode=request.param
    def pin(name,obj):
        p=tmp_path/name;p.write_text(json.dumps(obj));return file_pin(p)
    config=pin('config.json',{'execution':{'python_path':file_pin(sys.executable)['path'],'python_sha256':file_pin(sys.executable)['sha256']}})
    identity=dict(config_sha256=config['sha256'],product_commit='1'*40,product_tree='2'*40,workload_commit='1'*40,workload_tree='2'*40)
    receipt=pin('receipt.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',config=config,snapshot=str(tmp_path),source_identity=identity))
    params={'experiment_id':'R6-FIXTURE'}
    if mode=='research-r6-training':params['r6_training_plan']={'fixture':True}
    parameters=pin('parameters.json',params)
    attempt=tmp_path/'attempt';(attempt/'results').mkdir(parents=True)
    checker=Mock();monkeypatch.setattr(m,'verify_frozen',checker)
    auth=dict(identity,parameters=params,attempt=str(attempt),mode=mode)
    run=Mock(return_value=SimpleNamespace(returncode=0,stdout=json.dumps(auth),stderr=''))
    monkeypatch.setattr(m.subprocess,'run',run)
    return SimpleNamespace(args=dict(snapshot_receipt=receipt,parameters_pin=parameters,attempt=str(attempt)),run=run,checker=checker,auth=auth)


def test_proposal_does_not_issue_and_checks_snapshot_before_process(case):
    r=m.prepare(**case.args)
    assert r['status']=='PROPOSED_NOT_ISSUED' and not r['store_permission_issued'] and not r['job_submitted']
    case.checker.assert_called_once();case.run.assert_called_once()
    args,kw=case.run.call_args;assert args[0][1:4]==['-I','-B','-c'] and kw['env']['CUDA_VISIBLE_DEVICES']==''


def test_snapshot_or_frozen_validator_failure_is_not_hidden(case):
    case.checker.side_effect=RecoveryError('inventory drift')
    with pytest.raises(RecoveryError,match='inventory drift'):m.prepare(**case.args)
    case.run.assert_not_called();case.checker.side_effect=None
    case.run.return_value=SimpleNamespace(returncode=1,stdout='',stderr='protocol not frozen')
    with pytest.raises(RecoveryError,match='protocol not frozen'):m.prepare(**case.args)


def test_subprocess_identity_drift_rejected(case):
    bad=dict(case.auth,config_sha256='0'*64)
    case.run.return_value.stdout=json.dumps(bad)
    with pytest.raises(RecoveryError,match='IDENTITY_DRIFT'):m.prepare(**case.args)


@pytest.mark.parametrize('training',[False,True])
def test_frozen_program_assembles_exact_budget_with_declared_validator_stubs(monkeypatch,training):
    import training_workloads.unified_h1.config as config
    if training:
        import training_workloads.unified_h1.robustness_training_contract as contract
    else:
        import training_workloads.unified_h1.robustness_contract as contract
    identity=dict(config_sha256='1'*64,product_commit='2'*40,product_tree='3'*40,workload_commit='2'*40,workload_tree='3'*40)
    c=dict(method={'proposal_sha256':'4'*64},resolved_identity=identity)
    p=dict(experiment_id='R6-FIXTURE',group='R1',checkpoint_compatibility={'fixture':True})
    if training:p['r6_training_plan']={'fixture':'exact-plan'}
    budget={'nontraining_transitions':0 if training else 193600,'training_transitions':512 if training else 0}
    monkeypatch.setattr(config,'load_config',lambda:c)
    monkeypatch.setattr(contract,'validate_parameters',lambda mode,value:value)
    monkeypatch.setattr(contract,'required_budget',lambda mode,value:budget)
    if not training:monkeypatch.setattr(contract,'baseline',lambda value:{'source_identity':{'origin':True}})
    compat=Mock();monkeypatch.setattr(contract,'validate_compatibility',compat)
    monkeypatch.setattr(sys,'argv',['fixture',str(__import__('pathlib').Path.cwd())])
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps({'parameters':p,'attempt':'/fixture'})))
    out=io.StringIO();monkeypatch.setattr(sys,'stdout',out)
    original=list(sys.path)
    try:exec(m.PROGRAM,{})
    finally:sys.path[:]=original
    a=json.loads(out.getvalue());assert a['budgets']==budget
    assert a['order_id']=='R6-FIXTURE-R1-'+contract.MODE.upper()
    assert compat.call_args.args[1]==(p['r6_training_plan'] if training else {'origin':True})
    assert all(a[k]==v for k,v in identity.items());compat.assert_called_once()

@pytest.mark.parametrize('consumed_mode',['research-r6-evaluation','research-r6-training'])
def test_either_consumed_mode_prevents_authority_reuse(case,consumed_mode):
    from pathlib import Path
    (Path(case.args['attempt'])/'results'/consumed_mode).mkdir()
    with pytest.raises(RecoveryError,match='ATTEMPT_CONSUMED'):m.prepare(**case.args)
    case.run.assert_not_called()


def test_returned_mode_cannot_switch_stage(case):
    bad=dict(case.auth,mode='research-r6-evaluation' if case.auth['mode']=='research-r6-training' else 'research-r6-training')
    case.run.return_value.stdout=json.dumps(bad)
    with pytest.raises(RecoveryError,match='IDENTITY_DRIFT'):m.prepare(**case.args)
