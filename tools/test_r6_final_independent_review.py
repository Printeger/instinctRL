"""Real fake-CLI independent receipt routing; scientific/image evidence is synthetic."""
import copy,json,sys
from unittest.mock import Mock
import pytest
from tools import r6_final_independent_review as r
from tools.test_r6_profile_preparation import case
from tools.r6_batch import model_handoff
from tools.r6_recovery import file_pin,RecoveryError

FREE=500*1024**3
SCRIPT='''import json,sys
from pathlib import Path
packet=json.loads(sys.stdin.read().splitlines()[-1]);settings=json.loads(Path(__file__).with_suffix('.settings').read_text())
with Path(__file__).with_suffix('.count').open('a') as f:f.write('call\\n')
d=json.loads(Path(__file__).with_suffix('.decision').read_text());d['remaining_budget']={'automatic_sessions_remaining':packet['caps']['automatic_model_sessions']-packet['budget']['automatic_model_sessions']-1}
if settings.get('reject'):
 d.update(decision='MUST_FIX',scientific_status='INCOMPLETE',must_fix=['CPU fixture: missing evidence'])
 d['requirements_audit']['media_visual']['status']='MUST_FIX';d['visual_audit']=[]
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(d)}}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))
'''

@pytest.fixture(params=['BASELINE_SUFFICIENT','THREE_CONTINUATIONS'])
def reviewed(case,monkeypatch,request):
    b=case;branch=request.param;count=3 if branch=='BASELINE_SUFFICIENT' else 4
    result=dict(required_models_pass=True,results=[dict(job_event_key=str(i),assignment={'evaluation_seed':17+i}) for i in range(count)])
    media=b.write('media.json',dict(visual_samples=[dict(index=0,image=b.context,frames=[0,199],video=b.context)]))
    jobs=b.write('jobs.json',dict(jobs=[dict(event_key=str(i),native_status='FAILED') for i in range(count)]))
    resources=b.write('resources.json',dict(charged={'training_transitions':0 if count==3 else 1024}))
    models=b.write('models.json',dict(all_registered_local_waits_verified=True,platform_inactivity_proven=True))
    b.data=dict(description=dict(branch=branch,final=result,video_manifest=media,checks=r.plans.CHECKS),artifacts={'TRAINING_AND_EVALUATION_MANIFEST.json':jobs,'RESOURCE_AUDIT_PRE_REVIEW.json':resources,'MODEL_USAGE_AUDIT_PRE_REVIEW.json':models})
    b.packet=b.write('audit-packet.json',b.data)
    b.decision=dict(decision='PASS_R6_FINAL_AUDIT',scientific_status='BASELINE_SUFFICIENT' if count==3 else 'R6_TARGET_DOMAIN_PASS',reviewer='r6-independent-final-audit',preparation_sha256=b.packet['sha256'],summary='CPU fixture only',changed_files=[],must_fix=[],evidence_pins=[b.packet],remaining_budget={'automatic_sessions_remaining':0},
        requirements_audit={k:dict(status='PASS',reason='CPU fixture only',evidence_pins=[b.context]) for k in r.plans.CHECKS},
        visual_audit=[dict(index=0,image=b.context,frames=[0,199],findings='CPU fixture only, no actual inspection')],
        native_failure_assessments=[dict(event_key=str(i),disposition='scientific_evidence_accepted',reason='CPU fixture only',evidence_pins=[b.context]) for i in range(count)])
    monkeypatch.setattr(r.plans,'verify',Mock(return_value=b.data))
    b.reviewer=b.root/'audit-reviewer';b.reviewer.write_text('#!'+sys.executable+'\n'+SCRIPT);b.reviewer.chmod(0o700)
    b.reviewer.with_suffix('.settings').write_text('{}');b.reviewer.with_suffix('.decision').write_text(json.dumps(b.decision))
    monkeypatch.setattr(r,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    return b


def run(b):
    with b.j.writer():return r.prepare(b.j,b.packet,b.store,executable=file_pin(b.reviewer),free_disk_bytes=FREE)


def test_receipt_replay_and_protected_final_review_budget(reviewed):
    b=reviewed;before=b.store.replay();out=run(b);budget=b.j.read()['budget']
    assert out['status']=='FINAL_AUDIT_ACCEPTED_PENDING_PUBLICATION' and not out['scientific_acceptance']
    assert run(b)==out and b.j.read()['budget']==budget and b.store.replay()==before
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'
    assert b.j.read()['events'][out['event_key']]['purpose']=='final_review'
    assert r.verify(b.j,out['event_key'],out['review'],b.packet,b.store)['decision']=='PASS_R6_FINAL_AUDIT'


def test_must_fix_is_persisted_as_repair_without_reinvocation(reviewed):
    b=reviewed;b.reviewer.with_suffix('.settings').write_text('{"reject":true}')
    out=run(b);assert out['status']=='FINAL_AUDIT_MUST_FIX_REQUIRED'
    assert run(b)==out and b.reviewer.with_suffix('.count').read_text()=='call\n'


@pytest.mark.parametrize('fault',['missing_check','visual_missing','visual_frame','native_missing','native_unusable','wrong_positive','bad_na','no_evidence','incomplete_negative'])
def test_acceptance_rejects_missing_or_contradictory_evidence(reviewed,fault):
    b=reviewed;d=copy.deepcopy(b.decision)
    if fault=='missing_check':d['requirements_audit'].pop('actor_boundary')
    elif fault=='visual_missing':d['visual_audit']=[]
    elif fault=='visual_frame':d['visual_audit'][0]['frames']=[0]
    elif fault=='native_missing':d['native_failure_assessments'].pop()
    elif fault=='native_unusable':d['native_failure_assessments'][0]['disposition']='preserved_unusable'
    elif fault=='wrong_positive':d['scientific_status']='R6_TARGET_DOMAIN_PASS' if d['scientific_status']=='BASELINE_SUFFICIENT' else 'BASELINE_SUFFICIENT'
    elif fault=='bad_na':d['requirements_audit']['actor_boundary']['status']='NOT_APPLICABLE'
    elif fault=='no_evidence':d['requirements_audit']['holdout']['evidence_pins']=[]
    else:d['scientific_status']='SCIENTIFIC_NEGATIVE_WITH_COMPLETE_EVIDENCE'
    with pytest.raises(RecoveryError):r.assess(d,b.packet)


def test_unknown_wait_evidence_cannot_be_claimed_pass(reviewed):
    b=reviewed;models=b.write('unknown-waits.json',dict(all_registered_local_waits_verified=True,platform_inactivity_proven=False))
    data=copy.deepcopy(b.data);data['artifacts']['MODEL_USAGE_AUDIT_PRE_REVIEW.json']=models
    pin=b.write('unknown-packet.json',data);d=copy.deepcopy(b.decision);d.update(preparation_sha256=pin['sha256'],evidence_pins=[pin])
    with pytest.raises(RecoveryError,match='UNPROVEN_ZERO_MODEL_WAIT'):r.assess(d,pin)


def test_missing_prompt_recovers_same_intent(reviewed,monkeypatch):
    b=reviewed;original=r.model_handoff
    monkeypatch.setattr(r,'model_handoff',Mock(side_effect=RuntimeError('interrupted')))
    with pytest.raises(RuntimeError):run(b)
    entry=next(e for e in b.j.read()['events'].values() if e['identity']['role']=='final-audit-review')
    budget=b.j.read()['budget'];r.checked(entry['request']['prompt']).unlink()
    monkeypatch.setattr(r,'model_handoff',original);run(b);assert b.j.read()['budget']==budget


def test_complete_negative_is_distinct_from_incomplete(reviewed):
    b=reviewed;data=copy.deepcopy(b.data);data['description']['final']['required_models_pass']=False
    pin=b.write('complete-negative.json',data);d=copy.deepcopy(b.decision)
    d.update(preparation_sha256=pin['sha256'],evidence_pins=[pin],scientific_status='SCIENTIFIC_NEGATIVE_WITH_COMPLETE_EVIDENCE')
    assert r.assess(d,pin)==d
    d['requirements_audit']['complete_results']['status']='MUST_FIX'
    with pytest.raises(RecoveryError,match='INCOMPLETE_ACCEPTANCE'):r.assess(d,pin)


def test_not_applicable_only_for_correct_scientific_branch(reviewed):
    b=reviewed;d=copy.deepcopy(b.decision)
    permitted=r.plans.BASELINE_NA if b.data['description']['branch']=='BASELINE_SUFFICIENT' else {'baseline_sufficient'}
    for key in permitted:d['requirements_audit'][key]['status']='NOT_APPLICABLE'
    assert r.assess(d,b.packet)==d
