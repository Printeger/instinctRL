"""CPU fixtures only: no actual READY, approval, simulator or optimizer writes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.r5_formal_launch_gate import (
    BASE, GOAL, digest, require_formal_start, transfer_formal_reserve,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': digest(path)}


@pytest.fixture
def package(tmp_path):
    root=tmp_path; base=root/BASE
    master=save(base/'MASTER_AUTHORITY.json',{'id':GOAL})
    auth=save(base/'fixture-AUTH.json',{'fixture':'Not a real runtime authorization'})
    spec={'id':'CPU-FORMAL', 'h1':{'mode':'research-formal-training'},
          'pins':{'authorization':auth}, 'argv':['fixture'], 'cwd':'fixture','conditions':[]}
    specpath=base/'fixture-spec.json'; specpin=save(specpath,spec)
    templatepin=save(base/'training-spec-template.json',spec)
    manifest={'goal_id':GOAL,'status':'FROZEN_CONDITIONAL_FORMAL_FIRST_SEGMENT',
              'commander':save(root/'tools/r5_formal_commander.py',{'fixture':'Never executable'}),
              'launch_gate':save(root/'tools/r5_formal_launch_gate.py',{'fixture':'Never executable'}),
              'master_authority':master,'entries':[{'spec_id':spec['id'],
                'expected_authorization_sha256':auth['sha256'],
                'spec_template':templatepin,
                'additional_conditions':[{'evidence_id':'CPU-FINAL-REVIEW','binding':'final_review'}],
                'argv':spec['argv'],'cwd':spec['cwd'],'h1':spec['h1']}]}
    mp=save(base/'formal_r5_launch_manifest.json',manifest)
    rp=save(base/'reviews/final-readiness.json',{'verdict':'PASS','must_fix':[],
            'identities':{'formal_manifest_sha256':mp['sha256']}})
    spec['conditions'].append({'evidence_id':'CPU-FINAL-REVIEW','sha256':rp['sha256']})
    specpin=save(specpath,spec)
    permission={'goal_id':GOAL,'status':'CONDITIONAL_APPROVED_AT_READINESS',
                'manifest':mp,'independent_review':rp,'master_authority':master,
                'entries':[{'spec':specpin,'approval_id':'CPU-APPROVAL','authorization':auth}]}
    pp=save(base/'formal_conditional_permission.json',permission)
    ready={'goal_id':GOAL,'status':'READY_FOR_FORMAL_R5_TRAINING',
           'formal_training_submitted_at_readiness':False,
           'manifest':mp,'independent_review':rp,'conditional_permission':pp}
    save(base/'formal_r5_readiness.json',ready)
    return root,base,specpath


def test_explicit_later_start_and_exact_package_required(package):
    root,base,spec=package
    with pytest.raises(PermissionError,match='explicit later'):
        require_formal_start(root,spec,'CPU-APPROVAL')
    result=require_formal_start(root,spec,'CPU-APPROVAL',explicit_start=True)
    assert result['manifest_sha256']==digest(base/'formal_r5_launch_manifest.json')
    with pytest.raises(PermissionError,match='approval'):
        require_formal_start(root,spec,'WRONG',explicit_start=True)


@pytest.mark.parametrize('relative',['formal_r5_readiness.json','formal_r5_launch_manifest.json',
    'reviews/final-readiness.json','formal_conditional_permission.json','MASTER_AUTHORITY.json',
    'fixture-spec.json','fixture-AUTH.json'])
def test_missing_or_drifted_required_input_refuses_start(package,relative):
    root,base,spec=package
    target=base/relative
    if relative=='formal_r5_readiness.json':target.unlink()
    else:target.write_text('{}')
    with pytest.raises(PermissionError):
        require_formal_start(root,spec,'CPU-APPROVAL',explicit_start=True)


def test_updated_file_hash_cannot_hide_unaccepted_final_review(package):
    root,base,spec=package
    ready=json.loads((base/'formal_r5_readiness.json').read_text())
    permission=json.loads((base/'formal_conditional_permission.json').read_text())
    rp=save(base/'reviews/final-readiness.json',{'verdict':'MUST_FIX','must_fix':['CPU-FIX'],
        'identities':{'formal_manifest_sha256':ready['manifest']['sha256']}})
    permission['independent_review']=ready['independent_review']=rp
    ready['conditional_permission']=save(base/'formal_conditional_permission.json',permission)
    save(base/'formal_r5_readiness.json',ready)
    with pytest.raises(PermissionError,match='not accepted'):
        require_formal_start(root,spec,'CPU-APPROVAL',explicit_start=True)


def test_formal_current_tool_drift_refuses_start(package):
    root,base,spec=package
    (root/'tools/r5_formal_commander.py').write_text('changed')
    with pytest.raises(PermissionError,match='pin mismatch'):
        require_formal_start(root,spec,'CPU-APPROVAL',explicit_start=True)


def generated_evaluation(package):
    root,base,specpath=package
    spec=json.loads(specpath.read_text());spec['h1']['mode']='research-formal-evaluation'
    paths=[]
    for update in [272,288,320,384,512]:
        ck=base/f'CPU-only-u{update}.pt';ck.write_bytes(b'opaque fixture; never loaded as checkpoint')
        paths.append({'path':str(ck),'sha256':digest(ck),'update':update})
    auth={'parameters':{'checkpoints':paths},'frozen':'field'}
    spec['pins']['authorization']=save(base/'fixture-AUTH.json',auth)
    template=deepcopy(auth)
    for ck in template['parameters']['checkpoints']:ck['sha256']=None
    tp=save(base/'evaluation-template.json',template)
    rp=save(base/'CPU-training-review.json',{'verdict':'PASS','must_fix':[],
        'accepted_evaluation_checkpoints':paths})
    frozen_spec=deepcopy(spec);frozen_spec['conditions']=[]
    frozen_spec['pins']['authorization']['sha256']=None
    stp=save(base/'evaluation-spec-template.json',frozen_spec)
    manifest=json.loads((base/'formal_r5_launch_manifest.json').read_text())
    entry=manifest['entries'][0]
    entry.update(h1=spec['h1'],generated_checkpoint_inputs=True,authorization_template=tp,
        spec_template=stp,additional_conditions=[
            {'evidence_id':'CPU-ACCEPTED-TRAINING','binding':'checkpoint_acceptance'},
            {'evidence_id':'CPU-FINAL-REVIEW','binding':'final_review'}])
    entry.pop('expected_authorization_sha256')
    mp=save(base/'formal_r5_launch_manifest.json',manifest)
    review=json.loads((base/'reviews/final-readiness.json').read_text())
    review['identities']['formal_manifest_sha256']=mp['sha256']
    finalrp=save(base/'reviews/final-readiness.json',review)
    spec['conditions']=[{'evidence_id':'CPU-ACCEPTED-TRAINING','sha256':rp['sha256']},
                        {'evidence_id':'CPU-FINAL-REVIEW','sha256':finalrp['sha256']}]
    save(specpath,spec)
    permission=json.loads((base/'formal_conditional_permission.json').read_text())
    permission.update(manifest=mp,independent_review=finalrp,entries=[],allowed_generated_evaluation_spec_ids=[spec['id']])
    pp=save(base/'formal_conditional_permission.json',permission)
    ready=json.loads((base/'formal_r5_readiness.json').read_text())
    ready.update(manifest=mp,independent_review=finalrp,conditional_permission=pp)
    save(base/'formal_r5_readiness.json',ready)
    evidence={'CPU-ACCEPTED-TRAINING':{**rp,'accepted':True}}
    return evidence


def test_generated_future_evaluation_requires_accepted_real_checkpoint_binding(package):
    evidence=generated_evaluation(package)
    root,base,spec=package
    with pytest.raises(PermissionError,match='lack exact accepted'):
        require_formal_start(root,spec,'CPU-LATER-APPROVAL',explicit_start=True)
    result=require_formal_start(root,spec,'CPU-LATER-APPROVAL',explicit_start=True,accepted_evidence=evidence)
    assert result['generated_checkpoint_acceptance_sha256']==evidence['CPU-ACCEPTED-TRAINING']['sha256']
    evidence['CPU-ACCEPTED-TRAINING']['accepted']=False
    with pytest.raises(PermissionError,match='lack exact accepted'):
        require_formal_start(root,spec,'CPU-LATER-APPROVAL',explicit_start=True,accepted_evidence=evidence)


def test_generated_future_evaluation_cannot_change_frozen_noncheckpoint_field(package):
    evidence=generated_evaluation(package)
    root,base,specpath=package
    auth=json.loads((base/'fixture-AUTH.json').read_text());auth['frozen']='changed'
    spec=json.loads(specpath.read_text());spec['pins']['authorization']=save(base/'fixture-AUTH.json',auth);save(specpath,spec)
    with pytest.raises(PermissionError,match='other frozen fields'):
        require_formal_start(root,specpath,'CPU-LATER-APPROVAL',explicit_start=True,accepted_evidence=evidence)


def test_generated_future_checkpoint_drift_refuses_before_release(package):
    evidence=generated_evaluation(package)
    root,base,specpath=package
    (base/'CPU-only-u512.pt').write_bytes(b'changed')
    with pytest.raises(PermissionError,match='checkpoint file drift'):
        require_formal_start(root,specpath,'CPU-LATER-APPROVAL',explicit_start=True,accepted_evidence=evidence)


@pytest.mark.parametrize('change',[
    lambda s:s['pins'].update(dispatcher={'path':'wrong','sha256':'wrong'}),
    lambda s:s.update(resources={'starts':999}),
    lambda s:s.update(method='wrong'),
    lambda s:s.update(evaluation='wrong'),
    lambda s:s.update(runtime='wrong'),
    lambda s:s['conditions'].append({'evidence_id':'UNDECLARED','sha256':'wrong'}),
])
def test_generated_eval_complete_spec_tampering_is_rejected(package,change):
    evidence=generated_evaluation(package)
    root,base,specpath=package
    spec=json.loads(specpath.read_text());change(spec);save(specpath,spec)
    with pytest.raises(PermissionError,match='complete spec differs'):
        require_formal_start(root,specpath,'CPU-LATER-APPROVAL',explicit_start=True,accepted_evidence=evidence)


def ledger():
    reserve={'training_transitions':262144,'nontraining_transitions':329600,
             'live_starts':4,'simulator_initialization_reservations':4,
             'optimizer_updates':512,'actor_optimizer_steps':4096,'critic_optimizer_steps':4096,
             'training_checkpoint_writes':10,'allocated_gpu_seconds':21680,
             'wall_seconds_including_cleanup':21680,'output_peak_and_reservations_bytes':32*1024**3}
    return {'caps':{k:v*10 for k,v in reserve.items()},
            'stage_allocation':{k:0 for k in reserve},
            'reservations':{'formal_initial_segment':reserve,'confirmation_and_transition':{'untouched':True},
                'live_entries':[{'historical':'FAILED_INCOMPLETE_NO_REFUND'}]},
            'used':{'historical':123}}


def training_budget():
    return {'training_transitions':131072,'nontraining_transitions':0,'optimizer_updates':256,
            'checkpoint_writes':5,'wall_seconds':7200}


def test_exact_reserve_transfer_does_not_double_charge_or_refund_history():
    before=ledger(); original=deepcopy(before)
    after=transfer_formal_reserve(before,training_budget(),8*1024**3)
    assert before==original and after['used']==before['used']
    assert after['reservations']['live_entries']==before['reservations']['live_entries']
    assert after['reservations']['confirmation_and_transition']==before['reservations']['confirmation_and_transition']
    for key,value in original['reservations']['formal_initial_segment'].items():
        assert after['reservations']['formal_initial_segment'][key]+after['stage_allocation'][key]==value
    assert after['stage_allocation']['actor_optimizer_steps']==2048
    assert after['stage_allocation']['training_transitions']==131072


def test_all_four_formal_entries_leave_only_original_time_buffer():
    before=ledger(); after=before
    ev={'training_transitions':0,'nontraining_transitions':164800,'optimizer_updates':0,
        'checkpoint_writes':0,'wall_seconds':3600}
    for budget in [training_budget(),ev,training_budget(),ev]:
        after=transfer_formal_reserve(after,budget,8*1024**3)
    left=after['reservations']['formal_initial_segment']
    assert {k:v for k,v in left.items() if v}=={'allocated_gpu_seconds':40,'wall_seconds_including_cleanup':40}
    with pytest.raises(PermissionError,match='Insufficient'):
        transfer_formal_reserve(after,training_budget(),8*1024**3)


def test_insufficient_reserve_or_master_cap_does_not_mutate_ledger():
    before=ledger(); before['reservations']['formal_initial_segment']['live_starts']=0
    old=deepcopy(before)
    with pytest.raises(PermissionError,match='Insufficient'):
        transfer_formal_reserve(before,training_budget(),8*1024**3)
    assert old==before
    before=ledger(); before['caps']['training_transitions']=1; old=deepcopy(before)
    with pytest.raises(PermissionError,match='exceeds master'):
        transfer_formal_reserve(before,training_budget(),8*1024**3)
    assert before==old
