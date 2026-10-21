"""Real temporary Git commit/tag/push recovery; published batch and snapshot execution are substitutes."""
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools import r6_final_inputs as m
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import Journal, file_pin, RecoveryError

FREE=500*1024**3

@pytest.fixture
def inputs(case,tmp_path,monkeypatch):
    b=case;repo=tmp_path/'repo';repo.mkdir();remote=tmp_path/'remote.git'
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE).decode().strip()
    subprocess.run(['git','init','--bare','-q',str(remote)],check=True)
    git('init','-q','-b','main');git('config','user.name','CPU fixture');git('config','user.email','cpu@example.invalid');git('remote','add','origin',str(remote))
    for module in (m,m.profile,m.publication,m.snapshot):
        p=repo/'tools'/Path(module.__file__).name;p.parent.mkdir(exist_ok=True);p.write_bytes(Path(module.__file__).read_bytes());monkeypatch.setattr(module,'__file__',str(p))
    (repo/'source.py').write_text('fixture = True\n')
    git('add','.');git('commit','-qm','fixture source');commit=git('rev-parse','HEAD')
    directory=repo/'docs/experiments/r6-robustness-auto/batch'
    old=b.j.read();b.j=Journal(directory)
    with b.j.writer():b.j.save(old)
    b.git=git;b.repo=repo
    monkeypatch.setattr(m,'ROOT',repo);monkeypatch.setattr(m.snapshot,'ROOT',repo)
    monkeypatch.setattr(m,'required_source_sets',lambda schema:({'source.py'},{'source.py'}))
    section=dict(path='old',candidate_commit='SELF',candidate_tree='SELF',branch='',base_commit=commit,source_sha256={})
    template=dict(schema_version='unified-h1-r6-training-config.v1',product=section,workload=section,
        future_entry=dict(modes=['main','research-r6-evaluation','research-r6-training']),actor={'frozen':True},method={'frozen':True})
    context=m.read(b.context)
    b.description=dict(batch=b.context,context=context,reference=m.read(context['baseline_profile']),template_config=b.write('template.json',template),
        origin_configs=[b.context]*3,wall_seconds=60,payload_bytes=128*1024**2)
    monkeypatch.setattr(m,'describe',Mock(return_value=b.description))
    def snap(j,cp,commit,**kw):
        assert kw['purpose']=='confirmation_delivery'
        value=dict(event_key='fixture-final-snapshot',status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY')
        state=j.read();state['events']['fixture-final-snapshot']={'kind':'cpu','identity':{'role':'source-snapshot'},'receipt':b.write('snapshot-receipt.json',value)};j.save(state)
        return value
    b.snapshot=Mock(side_effect=snap);monkeypatch.setattr(m.snapshot,'prepare',b.snapshot)
    return b


def run(b):
    with b.j.writer():return m.prepare(b.j,'controller',b.context,b.store,free_disk_bytes=FREE)


def test_commit_tag_push_and_replay_preserve_unrelated_files(inputs):
    b=inputs;(b.repo/'unrelated.txt').write_text('keep staged\n');b.git('add','unrelated.txt')
    out=run(b);budget=b.j.read()['budget'];assert run(b)==out and b.j.read()['budget']==budget
    assert b.git('diff','--cached','--name-only')=='unrelated.txt'
    assert b.git('diff-tree','--no-commit-id','--name-only','-r',out['source_commit']).endswith('/config.json')
    assert b.git('rev-parse',out['tag'])==out['source_commit']==b.git('rev-parse','origin/main')
    q=m.read(out['inputs']);assert len(set(q['experiment_ids']))==3
    assert all(re.fullmatch(r'R6-[A-Z0-9-]{1,90}',name) for name in q['experiment_ids'])
    assert q['wall_seconds']==60 and q['payload_bytes']==128*1024**2
    cfg=json.loads(b.snapshot.call_args.args[1].read_text())
    assert cfg['schema_version']==m.R6_FINAL_CONFIG_SCHEMA
    assert cfg['future_entry']['modes']==['main','research-r6-evaluation','research-r6-final-evaluation']
    assert cfg['actor']=={'frozen':True} and cfg['method']=={'frozen':True}


def test_crash_after_commit_reuses_exact_commit(inputs):
    b=inputs;b.snapshot.side_effect=RuntimeError('snapshot boundary interruption')
    with pytest.raises(RuntimeError):run(b)
    commit=b.git('rev-parse','HEAD');budget=b.j.read()['budget']
    with pytest.raises(RuntimeError):run(b)
    assert b.git('rev-parse','HEAD')==commit and b.j.read()['budget']==budget


@pytest.mark.parametrize('fault',['stop','tool','description','tag'])
def test_drift_refuses_snapshot(inputs,fault):
    b=inputs
    if fault in ('description','tag'):run(b)
    b.snapshot.reset_mock()
    if fault=='stop':(b.j.directory/'STOP').touch()
    elif fault=='tool':Path(m.__file__).write_text('changed tool')
    elif fault=='description':b.description['wall_seconds']=99
    else:
        tag=b.git('tag','--list','r6-final-inputs-*');b.git('tag','-f',tag,'HEAD^')
    with pytest.raises(RecoveryError):run(b)
    b.snapshot.assert_not_called()


@pytest.mark.parametrize('branch',['BASELINE_SUFFICIENT','THREE_CONTINUATIONS'])
def test_description_derives_original_origins_and_full_baseline_capacity(case,tmp_path,monkeypatch,branch):
    b=case;origin=tmp_path/'origin/training_workloads/unified_h1/candidate_config.json';origin.parent.mkdir(parents=True);origin.write_text('{}')
    original=file_pin(origin)
    baseline=b.write('baseline-binding-final.json',dict(snapshot={'snapshot':str(tmp_path/'origin')},source_identity={'config_sha256':original['sha256']}))
    protocol=b.write('full-protocol.json',dict(blocks=[{'id':'all-block'}]))
    context=m.read(b.context);reference=m.read(context['baseline_profile']);reference['baseline_bindings']=baseline;reference['r6'].update(protocol=protocol,block_ids=['all-block'])
    context['baseline_profile']=b.write('final-reference.json',reference)
    baseline_config=b.write('baseline-config.json',{'kind':'baseline'});trained_config=b.write('trained-config.json',{'kind':'trained'})
    context['snapshot_receipt']=b.write('baseline-snapshot.json',dict(config=baseline_config))
    ctx=b.write('initial-context.json',context)
    prepared=b.write('confirmation-preparation.json',dict(snapshot=b.write('training-snapshot.json',dict(config=trained_config))))
    params=dict(reference,result_volume=b.write('volume-manifest.json',dict(minimum_free_bytes=123456)))
    auth=b.write('baseline-auth.json',dict(mode='research-r6-evaluation',parameters=params))
    spec=b.write('baseline-spec-final.json',dict(pins={'authorization':auth}))
    initial=b.write('initial-final.json',dict(plan_pin=b.write('plan-final.json',dict(spec_pin=spec))))
    request=b.write('controller-final.json',dict(profile_context=ctx,initial_request=initial))
    with b.j.writer():
        state=b.j.read();state['events']['controller']={'kind':'cpu','identity':{'role':'r6-controller'},'request':{'inputs':request,'store_root':str(b.store.root.resolve())}};b.j.save(state)
    freeze=b.write('final-freeze-description.json',dict(branch=branch,evidence=dict(controller_key='controller',baseline_bindings=baseline,preparation=prepared)))
    value=dict(freeze=freeze,protocol=protocol,assignments=[{}]*(3 if branch=='BASELINE_SUFFICIENT' else 4));bp=b.write('batch-description.json',value)
    monkeypatch.setattr(m.publication,'validate_batch',Mock(return_value=value));monkeypatch.setattr(m.snapshot,'verify_frozen',Mock())
    with b.j.writer():out=m.describe(b.j,'controller',bp,b.store)
    assert out['origin_configs']==([original]*3 if branch=='BASELINE_SUFFICIENT' else [original]+[trained_config]*3)
    assert out['template_config']==(baseline_config if branch=='BASELINE_SUFFICIENT' else trained_config)
    assert out['payload_bytes']==123456 and out['wall_seconds']==reference['wall_seconds']
    params['r6']['block_ids']=[]
    # Replace the exact authorization pin through the explicit input chain.
    changed=b.write('partial-auth.json',dict(mode='research-r6-evaluation',parameters=params))
    b.write('baseline-spec-final.json',dict(pins={'authorization':changed}))
    with b.j.writer(),pytest.raises(RecoveryError):m.describe(b.j,'controller',bp,b.store)
