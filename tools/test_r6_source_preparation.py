"""Actual fake CLI and Git preparation; snapshot validation/creation are fixtures.

No paid model, GPU, production Store mutation or actual production snapshot.
Existing snapshot tests separately exercise real clone/freeze/recovery.
"""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from tools import r6_source_preparation as source
from tools import test_r6_decision as decision_fixture
from tools.r6_recovery import RecoveryError, file_pin
from tools.r6_batch import model_handoff

SCRIPT='''import hashlib,json,subprocess,sys
from pathlib import Path
p=json.loads(sys.stdin.read().splitlines()[-1]);repo=Path(p['repository'])
count=Path(__file__).with_suffix('.count')
with count.open('a') as f:f.write('call\\n')
settings=json.loads(Path(__file__).with_suffix('.settings').read_text())
def git(*args):return subprocess.check_output(['git','-C',str(repo),*args]).decode().strip()
def pin(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
out=Path(p['output_directory']);out.mkdir(parents=True,exist_ok=True)
config=out/'config.json';card=out/'card.json'
config.write_text(json.dumps(settings.get('config',{})))
design={'schema_version':'r6-source-design-card/1','parent_decision_sha256':p['parent']['sha256'],
 'experiment_id':'CPU-PREPARATION','hypothesis':'CPU evidence repair','comparison':'Original failed fixture',
 'source_scope':'CPU fixture only','planned_resources':{'live_starts':0}}
if settings.get('bad_card'):design['planned_resources']={'training_transitions':10**12}
card.write_text(json.dumps(design))
if settings.get('banned'):
 bad=repo/'tools/forbidden.py';bad.parent.mkdir();bad.write_text('not allowed')
git('add','.');git('-c','user.name=CPU fixture','-c','user.email=cpu@example.invalid','commit','-qm','prepared inputs')
commit=git('rev-parse','HEAD')
changed=git('diff','--name-only',p['base_commit'],commit).splitlines()
result={'decision':'SOURCE_READY','parent_decision_sha256':p['parent']['sha256'],
 'summary':'CPU source preparation fixture','source_commit':commit,'config':pin(config),'card':pin(card),
 'evidence_pins':[p['parent'],pin(config),pin(card)],'changed_files':changed,'must_fix':[],
 'remaining_budget':{'automatic_sessions_remaining':p['remaining_model_sessions']}}
if settings.get('incomplete'):
 result.update(decision='PREPARATION_INCOMPLETE',source_commit=None,config=None,card=None,must_fix=['fixture missing preparation'])
if settings.get('omit_changes'):result['changed_files']=[]
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(result)}}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))
'''


@pytest.fixture
def prepared(monkeypatch):
    b=decision_fixture.DecisionTests('test_fake_model_event_records_proposal_once_without_dispatch')
    b.setUp()
    try:
        b.parent=b.register();b.complete(b.parent)
        b.repo=source.repository(b.j);b.repo.mkdir()
        subprocess.run(['git','init','-q','-b','main',str(b.repo)],check=True)
        (b.repo/'seed').write_text('CPU seed')
        cfg=b.repo/'training_workloads/unified_h1/candidate_config.json'
        cfg.parent.mkdir(parents=True);cfg.write_text('{}')
        source.snapshot.git(b.repo,'add','.')
        source.snapshot.git(b.repo,'-c','user.name=CPU fixture','-c','user.email=cpu@example.invalid','commit','-qm','seed')
        b.builder=b.root/'fake_builder';b.builder.write_text('#!'+sys.executable+'\n'+SCRIPT);b.builder.chmod(0o700)
        b.builder.with_suffix('.settings').write_text('{}')
        b.preflight=Mock(return_value=({},b.root/'frozen','CPU'))
        b.freeze=Mock(return_value={'status':'CPU_SNAPSHOT_FIXTURE'})
        monkeypatch.setattr(source.snapshot,'preflight',b.preflight)
        monkeypatch.setattr(source.snapshot,'prepare',b.freeze)
        monkeypatch.setattr(source,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
        yield b
    finally:b.doCleanups()


def register(b):
    with b.j.writer():
        return source.register(b.j,b.parent,b.store,executable=file_pin(b.builder),free_disk_bytes=500*1024**3)


def run(b):
    with b.j.writer():
        return source.after_job(b.j,b.key,b.store,executable=file_pin(b.builder),free_disk_bytes=500*1024**3)


def test_real_git_and_fake_worker_bind_source_then_replay_once(prepared):
    b=prepared;first=run(b);budget=b.j.read()['budget']
    assert first['preparation']['status']=='SOURCE_INPUTS_READY_NOT_LIVE_AUTHORITY'
    assert not first['scientific_acceptance']
    assert run(b)==first
    assert b.j.read()['budget']==budget
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    b.freeze.assert_called_with(b.j,
        source.checked(first['preparation']['config']),first['preparation']['source_commit'])


def test_missing_prompt_before_dispatch_is_rebuilt_without_new_reservation(prepared):
    b=prepared;key=register(b);entry=b.j.read()['events'][key]
    path=source.checked(entry['request']['prompt']);path.unlink()
    budget=b.j.read()['budget']
    assert run(b)['snapshot']['status']=='CPU_SNAPSHOT_FIXTURE'
    assert b.j.read()['budget']==budget
    assert b.builder.with_suffix('.count').read_text()=='call\n'


@pytest.mark.parametrize('setting,reason',[('incomplete','INCOMPLETE'),('banned','CHANGE_SCOPE'),('omit_changes','CHANGED_FILES'),('bad_card','DESIGN_RESOURCES')])
def test_rejected_output_never_freezes_or_retries_worker(prepared,setting,reason):
    b=prepared;b.builder.with_suffix('.settings').write_text(json.dumps({setting:True}))
    with pytest.raises(RecoveryError,match=reason):run(b)
    with pytest.raises(RecoveryError,match=reason):run(b)
    b.freeze.assert_not_called()
    assert b.builder.with_suffix('.count').read_text()=='call\n'


def test_missing_input_metadata_is_ambiguous_and_not_duplicated(prepared):
    b=prepared;key=register(b)
    with b.j.writer():
        state=b.j.read();state['events'][key].pop('source_input');b.j.save(state)
    budget=b.j.read()['budget']
    with pytest.raises(RecoveryError,match='INPUT_MISSING_NO_RETRY'):run(b)
    assert b.j.read()['budget']==budget
    assert not b.builder.with_suffix('.count').exists()
    b.freeze.assert_not_called()


def test_output_drift_on_replay_blocks_freeze(prepared):
    b=prepared;result=run(b);b.freeze.reset_mock()
    source.checked(result['preparation']['card']).write_text('tampered')
    with pytest.raises(RecoveryError):run(b)
    b.freeze.assert_not_called()
    assert b.builder.with_suffix('.count').read_text()=='call\n'


def test_stop_before_registration_has_no_new_effect(prepared):
    b=prepared;(b.j.directory/'STOP').touch();before=b.j.read()
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert b.j.read()==before
    b.freeze.assert_not_called()


def test_missing_job_decision_never_registers_preparation(prepared):
    b=prepared
    with b.j.writer():
        state=b.j.read();state['events'].pop(b.parent);b.j.save(state)
    before=b.j.read()
    with pytest.raises(RecoveryError,match='EXACT_JOB_DECISION'):run(b)
    assert b.j.read()==before
    b.freeze.assert_not_called()


def test_budget_drift_before_dispatch_does_not_call_worker(prepared):
    b=prepared;register(b)
    with b.j.writer():
        state=b.j.read();state['budget']['used']['automatic_model_sessions']+=1;b.j.save(state)
    with pytest.raises(RecoveryError,match='BUDGET_MUTATED'):run(b)
    assert not b.builder.with_suffix('.count').exists()
    b.freeze.assert_not_called()


@pytest.mark.parametrize('drift',['budget','store','control'])
def test_worker_cannot_change_controller_authority(prepared,monkeypatch,drift):
    b=prepared;key=register(b)
    with b.j.writer():model_handoff(b.j,key,poll_seconds=.01)
    if drift=='budget':
        with b.j.writer():
            state=b.j.read();state['budget']['used']['automatic_model_sessions']+=1;b.j.save(state)
    elif drift=='store':
        original=b.store.replay
        monkeypatch.setattr(b.store,'replay',lambda:dict(original(),unexpected_mutation=True))
    else:monkeypatch.setattr(source,'control_identity',lambda:'different-control')
    with pytest.raises(RecoveryError,match='BUDGET_MUTATED|STORE_MUTATED|CONTROL_DRIFT'):run(b)
    b.freeze.assert_not_called()


def test_actual_charged_clone_freeze_and_recovery(prepared,monkeypatch):
    b=prepared;target=b.root/'frozen'
    # The fixture patched prepare on that module. Recover the original function
    # captured below at import time; keep actual Git/filesystem/Journal behavior.
    monkeypatch.setattr(source.snapshot,'prepare',ORIGINAL_PREPARE)
    monkeypatch.setattr(source.snapshot,'ROOT',b.repo)
    config=dict(product=dict(source_sha256={}),workload=dict(source_sha256={}),
        execution=dict(python_path=str(Path(sys.executable).resolve()),
                       python_sha256=file_pin(sys.executable)['sha256']))
    b.builder.with_suffix('.settings').write_text(json.dumps(dict(config=config)))
    monkeypatch.setattr(source.snapshot,'preflight',lambda path,commit:(config,target,commit))
    real_run=subprocess.run;validations=[]
    def command(argv,**kwargs):
        if argv[0]==config['execution']['python_path']:
            validations.append(True)
            commit=source.snapshot.git(b.repo,'rev-parse','HEAD').decode().strip()
            tree=source.snapshot.git(b.repo,'rev-parse','HEAD^{tree}').decode().strip()
            config_path=next((b.repo/'docs').rglob('config.json'))
            value={a+'_'+k:v for a in ('product','workload') for k,v in [('commit',commit),('tree',tree)]}
            value['config_sha256']=file_pin(config_path)['sha256']
            return subprocess.CompletedProcess(argv,0,json.dumps(value),'')
        return real_run(argv,**kwargs)
    monkeypatch.setattr(subprocess,'run',command)
    try:
        first=run(b);budget=b.j.read()['budget']
        assert first['snapshot']['status']=='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY'
        assert not target.stat().st_mode&0o222
        assert run(b)==first
        assert b.j.read()['budget']==budget
        assert len(validations)==1
        assert b.builder.with_suffix('.count').read_text()=='call\n'
        rows=[r for r in budget['reservations'] if r['identity']['role']=='source-snapshot']
        assert len(rows)==1 and rows[0]['resources']['output_actual_plus_unreleased_reservations_bytes']==2*1024**3
    finally:
        if target.exists():
            for p in [target,*target.rglob('*')]:
                if not p.is_symlink():p.chmod(p.stat().st_mode|0o700)


ORIGINAL_PREPARE=source.snapshot.prepare
