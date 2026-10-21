"""Frozen CPU transport; model/source payload checks exercised in final contract tests."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools import r6_final_review_inputs as m
from tools.r6_recovery import file_pin,RecoveryError

@pytest.fixture(params=[3,4])
def case(tmp_path,monkeypatch,request):
    def pin(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return file_pin(p)
    exe=file_pin(sys.executable);cp=pin('config.json',dict(execution=dict(python_path=exe['path'],python_sha256=exe['sha256'])))
    identity=dict(config_sha256=cp['sha256']);snapshot=pin('snapshot.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',config=cp,snapshot=str(tmp_path),source_identity=identity))
    bp=pin('batch.json',dict(assignments=[{}]*request.param));q=dict(batch=bp,origin_configs=[cp]*request.param)
    value=dict(source_identity=identity,payloads={str(i):{} for i in range(request.param)})
    run=Mock(return_value=SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(value)))
    monkeypatch.setattr(m,'verify_frozen',Mock());monkeypatch.setattr(m.subprocess,'run',run)
    return SimpleNamespace(snapshot=snapshot,q=q,value=value,run=run)


def test_all_final_origins_use_isolated_frozen_source(case):
    result=m.prepare(case.snapshot,case.q)
    assert result['payloads']==case.value['payloads'] and not result['dispatch_authorized']
    argv,kw=case.run.call_args;assert argv[0][1:4]==['-I','-B','-c']
    assert kw['env']['CUDA_VISIBLE_DEVICES']=='' and kw['timeout']==180
    assert json.loads(kw['input'])==case.q


@pytest.mark.parametrize('fault',['fields','origins','failure','identity','missing','review','size'])
def test_incomplete_or_accepted_payload_rejected(case,fault):
    b=case
    if fault=='fields':b.q['permission']=True
    elif fault=='origins':b.q['origin_configs']=[]
    elif fault=='failure':b.run.return_value.returncode=1
    elif fault=='identity':b.value['source_identity']={}
    elif fault=='missing':b.value['payloads'].pop('0')
    elif fault=='review':b.value['payloads']['0']['independent_review']={}
    elif fault=='size':b.run.return_value.stdout='x'*262145
    if fault!='size':b.run.return_value.stdout=json.dumps(b.value)
    with pytest.raises(RecoveryError):m.prepare(b.snapshot,b.q)


def test_frozen_program_builds_all_actual_compatibility_payloads(tmp_path,monkeypatch,request):
    import io
    from tools.test_r6_training_plan import case as training_fixture
    from tools.test_r6_training_contract import compatibility as compatibility_fixture
    from training_workloads.unified_h1 import config,robustness_final_contract as contract
    training=training_fixture.__wrapped__(tmp_path,monkeypatch)
    b,r,new=compatibility_fixture.__wrapped__(training,tmp_path)
    model=dict(source_identity=r['origin_source_identity'],checkpoint=r['checkpoint'])
    seen=[]
    def selected(value):
        seen.append(value);return dict(model=model,method_pin=None)
    monkeypatch.setattr(contract.selection,'validate',selected)
    cp=tmp_path/'snapshot/training_workloads/unified_h1/candidate_config.json';cp.parent.mkdir(parents=True)
    cfg=dict(new,schema_version=config.R6_FINAL_CONFIG_SCHEMA)
    cp.write_text(json.dumps({k:v for k,v in cfg.items() if k!='resolved_identity'}))
    cfg['resolved_identity']=dict(new['resolved_identity'],config_sha256=file_pin(cp)['sha256'])
    batch=tmp_path/'released.json';batch.write_text(json.dumps(dict(assignments=[{}]*4)));bp=file_pin(batch)
    q=dict(batch=bp,origin_configs=[r['origin_config']]*4)
    monkeypatch.setattr(config,'load_config',lambda:cfg)
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(q)));out=io.StringIO();monkeypatch.setattr(sys,'stdout',out)
    monkeypatch.setattr(sys,'argv',['fixture',str(tmp_path/'snapshot')]);before=list(sys.path)
    try:exec(m.PROGRAM,{})
    finally:sys.path[:]=before
    result=json.loads(out.getvalue());assert set(result['payloads'])=={'0','1','2','3'}
    assert {v['assignment_index'] for v in seen}=={0,1,2,3}
    assert all(p['schema_version']=='r6-final-compatibility/1' and p['checkpoint']==model['checkpoint'] for p in result['payloads'].values())


@pytest.mark.parametrize('schema',['unified-h1-r6-candidate-config.v1','unified-h1-r6-final-config.v1'])
def test_real_snapshot_preflight_admits_exact_committed_stage(tmp_path,monkeypatch,schema):
    import hashlib,subprocess
    from tools import prepare_r6_snapshot as snap
    repo=tmp_path/'repo';repo.mkdir();base=repo/'docs/experiments/r6-robustness-auto';base.mkdir(parents=True)
    target=repo/'coordination/research/workspaces/R6-FINAL-FIXTURE'
    source=repo/'source.py';source.write_text('fixture = True\n')
    tool=repo/'tools/prepare_r6_snapshot.py';tool.parent.mkdir();tool.write_bytes(Path(snap.__file__).read_bytes())
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    git('init','-q');git('add','.');git('-c','user.name=CPU fixture','-c','user.email=cpu@example.invalid','commit','-qm','source')
    original=git('rev-parse','HEAD')
    section=dict(path=str(target),candidate_commit='SELF',candidate_tree='SELF',branch='',base_commit=original,
        source_sha256={'source.py':hashlib.sha256(source.read_bytes()).hexdigest()})
    config=dict(schema_version=schema,product=dict(section),workload=dict(section));cp=base/'config.json';cp.write_text(json.dumps(config))
    git('add','.');git('-c','user.name=CPU fixture','-c','user.email=cpu@example.invalid','commit','-qm','config');head=git('rev-parse','HEAD')
    monkeypatch.setattr(snap,'ROOT',repo);monkeypatch.setattr(snap,'BASE',base)
    monkeypatch.setattr(snap,'required_source_sets',lambda value:({'source.py'},{'source.py'}))
    assert snap.preflight(cp,head)==(config,target,head)
    cp.write_text('{}')
    with pytest.raises(RecoveryError,match='NOT_COMMITTED'):snap.preflight(cp,head)
