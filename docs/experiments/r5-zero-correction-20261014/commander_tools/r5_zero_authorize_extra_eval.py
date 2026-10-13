"""Existing Commander exact issuance ONLY after a new explicit user exception.

This file is prepared now but must not run without the decision record. It
never infers permission from old diagnostic caps, elapsed time or reviewer PASS.
"""
from pathlib import Path
import json,hashlib,copy
from rlflow.store import Store
from rlflow import experiments,policy
from rlflow.jobs import owns
ROOT=Path('/home/mint/rl_dev/instinctRL-dev');BASE=ROOT/'docs/experiments/r5-zero-correction-20261014';CONTROL=ROOT/'coordination/research'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 assert not p.exists(),p
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
decision_path=BASE/'user_resource_decision_extra_evaluation.json'
assert decision_path.is_file(), 'required new user decision absent: no authority issuance'
proposal_path=BASE/'C1_EXTRA_EVALUATION_RESOURCE_PROPOSAL.json';proposal=read(proposal_path);decision=read(decision_path)
assert decision['decision']=='APPROVED' and decision['proposal_sha256']==sha(proposal_path)
assert decision['approved_caps']==proposal['proposed_caps_only_if_user_approves']
assert decision['fresh_entries_authorized']==1 and decision['automatic_retries']==0
review_path=BASE/'reviews/review-C0-Z-C1-failure-extra-prelive.json';review=read(review_path)
assert review['verdict']=='PASS' and not review['must_fix']
assert review['identities']['extra_resource_proposal_sha256']==sha(proposal_path)
spec_path=BASE/proposal['exact_spec']['path'];auth_path=BASE/proposal['expected_AUTH_unissued']['path']
assert sha(spec_path)==proposal['exact_spec']['sha256']==review['identities']['extra_spec_draft_sha256']
assert sha(auth_path)==proposal['expected_AUTH_unissued']['sha256']==review['identities']['extra_expected_AUTH_sha256']
budget=read(BASE/'goal_budget.json');assert budget['caps']==decision['approved_caps']
assert budget['used']['live_starts']==4 and budget['used']['simulator_initialization_reservations']==4
assert not policy.stopped(CONTROL)
store=Store(CONTROL);state=store.replay();assert not state['paused']
assert all(j['status'] in ['FAILED','SUCCEEDED','STOPPED'] and not owns(j.get('identity')) for j in state['jobs'].values())
spec=copy.deepcopy(read(spec_path));auth=read(auth_path)
for c in auth['parameters']['checkpoints']:assert sha(Path(c['path']))==c['sha256']
for eid,path,scope in [('E-R5-ZERO-EXTRA-CONDITIONAL-PRELIVE-20261014',review_path,'Role5conditionalfreshentryonly'),('E-R5-ZERO-USER-EXTRA-RESOURCE-DECISION-20261014',decision_path,'Explicitnewuser5thentry/resourceexception')]:
 evidence={'id':eid,'path':str(path),'sha256':sha(path),'accepted':True,'scope':scope,'producer':'Supervisor-Commander'}
 store.transact('EXTRA_ENTRY_CONDITION_ACCEPTED',eid+':'+sha(path),lambda s,e=evidence:s['evidence'].update({e['id']:e}),object_id=eid,producer='Supervisor-Commander')
 spec['conditions'].append({'evidence_id':eid,'sha256':evidence['sha256']})
actual=Path(spec['pins']['authorization']['path']);assert not actual.exists() and not actual.parent.parent.exists()
actual.parent.mkdir(parents=True);(actual.parent.parent/'results').mkdir();actual.write_bytes(auth_path.read_bytes());actual.chmod(0o444)
assert sha(actual)==spec['pins']['authorization']['sha256']
experiments.validate(spec,CONTROL);pid=experiments.propose(store,spec);shown=policy.show_proposal(store,pid)
state=policy.decide(store,pid,shown['sha256'],'APPROVED','New explicit user resourceexception SHA '+sha(decision_path)+'; independentRole5 SHA '+sha(review_path)+'. ExactlyonefreshC1originalZ u16/u64 entry; d257f9/w.05/checkpoints fixed,0training/update/writes,12800eval1810s16GiB, original5percentGPUgate, no automaticretry/no oldjobresume. Allpriorreservations/history retained. Commit/tag/push before single release.')
approval=next(a for a in state['approvals'].values() if a['proposal_id']==pid)
save(BASE/'inputs/C1-extra-evaluation-spec.authorized.json',spec)
save(BASE/'C1-extra-evaluation-commander-approval.json',{'group':'C1','mode':'pilot-evaluation','proposal_id':pid,'proposal_sha256':shown['sha256'],'approval':approval,'actual_authorization':{'path':str(actual),'sha256':sha(actual)},'user_resource_decision_sha256':sha(decision_path),'independent_extra_conditional_review_sha256':sha(review_path),'submitted':False})
print(json.dumps({'status':'EXTRA_EXACT_APPROVED_NOT_SUBMITTED','approval_id':approval['approval_id']}))
