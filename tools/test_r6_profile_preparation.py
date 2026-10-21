"""Real fake CLI and storage-plan accounting; frozen verification is a fixture."""
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from tools import r6_profile_preparation as p
from tools import test_r6_decision as fixture
from tools.r6_batch import model_handoff
from tools.r6_recovery import file_pin,RecoveryError

SCRIPT='''import hashlib,json,sys
from pathlib import Path
v=json.loads(sys.stdin.read().splitlines()[-1]);packet=Path(v['packet_path'])
context=json.loads(Path(v['context']['path']).read_text())
profile=json.loads(Path(context['baseline_profile']['path']).read_text())
profile['experiment_id']='R6-PROFILE-CPU'
settings=json.loads(Path(__file__).with_suffix('.settings').read_text())
with Path(__file__).with_suffix('.count').open('a') as f:f.write('call\\n')
if settings.get('training'):
 profile.pop('r6');profile.pop('wall_seconds');profile.pop('protocol_review')
 profile['r6_training_plan']={'fixture':'training-plan'}
 profile['training_seed']=123
if settings.get('candidate'):
 profile['evaluation_selection']={'fixture':'reviewed-selection'}
 profile['training_seed']=456
if settings.get('managed'):profile['native_volume']={}
if settings.get('immutable'):profile['checkpoint_compatibility']={}
pin={'path':str(packet),'sha256':hashlib.sha256(packet.read_bytes()).hexdigest()}
r={'decision':'PROFILE_READY','packet_sha256':pin['sha256'],'summary':'CPU fixture',
 'profile_json':json.dumps(profile),'result_payload_bytes':128*1024**2,'minimum_free_inodes':128,
 'evidence_pins':[pin,v['parent'],v['context']],'changed_files':[],'must_fix':[],
 'remaining_budget':{'automatic_sessions_remaining':v['budget']['caps']['automatic_model_sessions']-v['budget']['charged']['automatic_model_sessions']-1}}
if settings.get('small'):r['result_payload_bytes']=32*1024**2
if settings.get('source'):r.update(decision='NEEDS_SOURCE_CHANGE',profile_json='',result_payload_bytes=0,minimum_free_inodes=0,must_fix=['CPU fixture: source contract does not support requested phase'])
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(r)}}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))
'''


@pytest.fixture
def case(monkeypatch):
    b=fixture.DecisionTests('test_fake_model_event_records_proposal_once_without_dispatch');b.setUp()
    try:
        b.parent=b.register();b.complete(b.parent)
        b.builder=b.root/'fake_profile';b.builder.write_text('#!'+sys.executable+'\n'+SCRIPT);b.builder.chmod(0o700)
        b.builder.with_suffix('.settings').write_text('{}')
        dummy=b.write('dummy.json',{})
        reference=dict(experiment_id='R6-REFERENCE',group='R1',training_seed=1,evaluation_seed=2,
            wall_seconds=60,master_authority=dummy,baseline_bindings=dummy,checkpoint_compatibility=dummy,
            protocol_review=dummy,extension_inputs=dummy,
            r6=dict(protocol=dummy,block_ids=['block'],output_quotas={'block':{'range_bytes':4096,'audit_bytes':4096}}))
        runner=dict(query_executable=file_pin(b.builder),decision_executable=file_pin(b.builder),
            queue_review=dummy,sampler_review=dummy,purpose='development',sandbox='read-only')
        b.context=b.write('context.json',dict(snapshot_receipt=b.write('snapshot.json',dict(
            status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',config=dummy)),
            baseline_profile=b.write('reference.json',reference),stage_id='CPU',conditions=[],runner=runner))
        monkeypatch.setattr(p,'verify_frozen',Mock())
        monkeypatch.setattr(p,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
        monkeypatch.setattr(p.storage.native_tool,'BASE',b.root/'native-volumes')
        monkeypatch.setattr(p.storage.result_tool,'BASE',b.root/'result-volumes')
        monkeypatch.setattr(p.storage.result_tool,'ATTEMPTS',b.root/'attempts')
        yield b
    finally:b.doCleanups()


def run(b):
    with b.j.writer():return p.prepare_profile(b.j,b.parent,b.context,b.store,
        executable=file_pin(b.builder),free_disk_bytes=500*1024**3)


def register(b):
    with b.j.writer():return p.register(b.j,b.parent,b.context,b.store,
        executable=file_pin(b.builder),free_disk_bytes=500*1024**3)


def test_real_worker_creates_plans_and_one_prepaid_allocation(case):
    b=case;before=b.store.replay();first=run(b);budget=b.j.read()['budget']
    assert first['status']=='PROFILE_AND_STORAGE_PLANS_READY_NOT_PROVISIONED'
    assert not first['dispatch_authorized']
    assert run(b)==first and b.j.read()['budget']==budget and b.store.replay()==before
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    entry,native,result,cap=p.storage.validate(b.j,first['storage_key'])
    assert cap==native['maximum_backing_bytes']+result['maximum_backing_bytes']+64*1024**2
    assert not Path(native['image']).exists() and not Path(result['image']).exists()
    inputs=json.loads(p.checked(first['inputs']).read_text());assert set(inputs)==p.materializer.FIELDS
    assert inputs['storage_key']==first['storage_key']


def test_unsupported_phase_routes_to_source_without_allocating_storage(case):
    b=case;b.builder.with_suffix('.settings').write_text('{"source":true}')
    first=run(b);assert first['status']=='SOURCE_PREPARATION_REQUIRED'
    assert run(b)==first
    assert not any(e['identity']['role']==p.storage.ROLE for e in b.j.read()['events'].values())
    assert b.builder.with_suffix('.count').read_text()=='call\n'


@pytest.mark.parametrize('setting,reason',[('managed','MANAGED_FIELDS'),('immutable','IMMUTABLE_PROFILE_BINDING'),('small','PAYLOAD_UNDERSIZED')])
def test_bad_profile_never_reserves_volume(case,setting,reason):
    b=case;b.builder.with_suffix('.settings').write_text(json.dumps({setting:True}))
    with pytest.raises(RecoveryError,match=reason):run(b)
    assert not any(e['identity']['role']==p.storage.ROLE for e in b.j.read()['events'].values())


def test_missing_prompt_replays_exact_packet_and_model_intent(case):
    b=case;key,_,_=register(b);budget=b.j.read()['budget']
    p.checked(b.j.read()['events'][key]['request']['prompt']).unlink()
    register(b)
    assert b.j.read()['budget']==budget
    assert run(b)['status']=='PROFILE_AND_STORAGE_PLANS_READY_NOT_PROVISIONED'
    assert b.builder.with_suffix('.count').read_text()=='call\n'


def test_preexisting_packet_crash_does_not_create_duplicate(case,monkeypatch):
    b=case;original=p.persist_exact
    monkeypatch.setattr(p,'persist_exact',Mock(side_effect=RuntimeError('CPU packet crash')))
    with pytest.raises(RuntimeError,match='packet crash'):register(b)
    packets=[e for e in b.j.read()['events'].values() if e['identity']['role']=='profile-input'];assert len(packets)==1
    monkeypatch.setattr(p,'persist_exact',original)
    assert run(b)['status']=='PROFILE_AND_STORAGE_PLANS_READY_NOT_PROVISIONED'
    assert len([e for e in b.j.read()['events'].values() if e['identity']['role']=='profile-input'])==1


def test_budget_drift_before_worker_is_rejected(case):
    b=case;register(b)
    with b.j.writer():
        state=b.j.read();state['budget']['used']['automatic_model_sessions']+=1;b.j.save(state)
    with pytest.raises(RecoveryError,match='STALE_BUDGET'):run(b)
    assert not b.builder.with_suffix('.count').exists()


def test_stop_before_new_event_has_no_effect(case):
    b=case;(b.j.directory/'STOP').touch();before=b.j.read()
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert b.j.read()==before


@pytest.mark.parametrize('kind',['packet','model'])
def test_missing_reservation_rejected_without_storage(case,kind):
    b=case;key,packet_key,packet_pin=register(b)
    with b.j.writer():
        model_handoff(b.j,key,poll_seconds=.01)
        state=b.j.read();identity=state['events'][packet_key if kind=='packet' else key]['identity']
        state['budget']['reservations']=[r for r in state['budget']['reservations'] if r['identity']!=identity]
        b.j.save(state)
        with pytest.raises(RecoveryError,match='RESERVATION'):
            p.record(b.j,key,packet_key,packet_pin,b.store,free_disk_bytes=500*1024**3)
    assert not any(e['identity']['role']==p.storage.ROLE for e in b.j.read()['events'].values())


def test_after_job_resolves_exact_parent_and_replays(case):
    b=case
    with b.j.writer():
        result=p.after_job(b.j,b.key,b.context,b.store,executable=file_pin(b.builder),free_disk_bytes=500*1024**3)
    assert result==run(b)
    assert b.builder.with_suffix('.count').read_text()=='call\n'


def test_candidate_cannot_bypass_charged_source_design(case,monkeypatch):
    b=case;original=p.parent_decision
    def candidate(*args,**kwargs):
        result=original(*args,**kwargs)
        result['decision']['decision']='PROPOSE_CANDIDATE'
        return result
    monkeypatch.setattr(p,'parent_decision',candidate)
    with pytest.raises(RecoveryError,match='CANDIDATE_SOURCE_REQUIRED'):run(b)
    assert not any(e['identity']['role']==p.storage.ROLE for e in b.j.read()['events'].values())


@pytest.mark.parametrize('fault',[None,'capacity','frozen','stop'])
def test_training_profile_validates_before_storage_and_replays(case,monkeypatch,fault):
    b=case;b.builder.with_suffix('.settings').write_text('{"training":true}')
    seen=[]
    def validate(snapshot,profile):
        seen.append((snapshot,profile))
        if fault=='frozen':raise RecoveryError('TRAINING_PROFILE_FROZEN_VALIDATION')
        if fault=='stop':(b.j.directory/'STOP').touch()
        return 64*1024**2 if fault=='capacity' else 128*1024**2
    monkeypatch.setattr(p.training_profile,'validate',validate)
    if fault:
        with pytest.raises(RecoveryError,match={'capacity':'TRAINING_PAYLOAD','frozen':'FROZEN_VALIDATION','stop':'STOP'}[fault]):run(b)
        assert not any(e['identity']['role']==p.storage.ROLE for e in b.j.read()['events'].values())
    else:
        first=run(b);budget=b.j.read()['budget'];assert run(b)==first
        assert b.j.read()['budget']==budget and b.builder.with_suffix('.count').read_text()=='call\n'
    assert seen and seen[0][1]['training_seed']==123


@pytest.mark.parametrize('fault',[None,'capacity','frozen','stop','mixed'])
def test_candidate_profile_validates_before_storage_and_replays(case,monkeypatch,fault):
    b=case;b.builder.with_suffix('.settings').write_text(json.dumps(dict(candidate=True,training=fault=='mixed')))
    seen=[]
    def validate(snapshot,profile,payload):
        seen.append((snapshot,profile,payload))
        if fault=='frozen':raise RecoveryError('CANDIDATE_PROFILE_FROZEN_VALIDATION')
        if fault=='stop':(b.j.directory/'STOP').touch()
        return payload-4096 if fault=='capacity' else payload
    monkeypatch.setattr(p.candidate_profile,'validate',validate)
    if fault:
        with pytest.raises(RecoveryError,match={'capacity':'CANDIDATE_PAYLOAD','frozen':'FROZEN_VALIDATION','stop':'STOP','mixed':'AMBIGUOUS_STAGE'}[fault]):run(b)
        assert not any(e['identity']['role']==p.storage.ROLE for e in b.j.read()['events'].values())
    else:
        first=run(b);budget=b.j.read()['budget'];assert run(b)==first
        assert b.j.read()['budget']==budget and b.builder.with_suffix('.count').read_text()=='call\n'
    if fault=='mixed':assert not seen
    else:assert seen and seen[0][1]['training_seed']==456


@pytest.mark.parametrize('field',['evaluation_selection','checkpoint_compatibility'])
def test_candidate_profile_cannot_replace_prepared_reviews(case,monkeypatch,field):
    b=case;context=json.loads(p.checked(b.context).read_text())
    reference=json.loads(p.checked(context['baseline_profile']).read_text())
    reference['evaluation_selection']={'fixture':'reviewed-selection' if field=='checkpoint_compatibility' else 'another-reviewed-selection'}
    context['baseline_profile']=b.write('candidate-reference.json',reference)
    b.context=b.write('candidate-context.json',context)
    b.builder.with_suffix('.settings').write_text(json.dumps(dict(candidate=True,immutable=field=='checkpoint_compatibility')))
    validator=Mock();monkeypatch.setattr(p.candidate_profile,'validate',validator)
    with pytest.raises(RecoveryError,match='IMMUTABLE_PROFILE_BINDING'):run(b)
    validator.assert_not_called()
    assert not any(e['identity']['role']==p.storage.ROLE for e in b.j.read()['events'].values())
