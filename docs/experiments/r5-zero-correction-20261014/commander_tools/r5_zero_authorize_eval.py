import copy,hashlib,json,sys
from pathlib import Path
from rlflow.store import Store
from rlflow import experiments,policy

root=Path('/home/mint/rl_dev/instinctRL-dev');base=root/'docs/experiments/r5-zero-correction-20261014';control=root/'coordination/research'
group=sys.argv[1];assert group in ['C0','C1']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):assert not p.exists(),p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
review_path=base/f'reviews/review-{group}-training-actual.json';review=read(review_path)
assert review['verdict']=='PASS' and not review['must_fix']
assert review['identities']['source_commit']==read(base/'launch_inputs.fd-fix-proposed.json')['baseline']['snapshot_commit']
analysis=read(base/f'{group}-training-data-analysis.json');assert analysis['scientific_training_coverage_complete']
packet=read(base/'launch_inputs.fd-fix-proposed.json');row=next(row for row in packet['entries'] if row['group']==group and row['mode']=='pilot-evaluation')
auth=copy.deepcopy(row['authorization_proposal']);spec=copy.deepcopy(row['experiment_proposal']);checkpoints=review['accepted_evaluation_checkpoints']
assert [cp['update'] for cp in checkpoints]==[16,64]
bound=[]
for cp,original in zip(checkpoints,auth['parameters']['checkpoints']):
 assert cp['path']==original['path'] and sha(Path(cp['path']))==cp['sha256']
 proof=next(entry for entry in analysis['checkpoints'] if entry['label']==f"u{cp['update']}")
 assert proof['sha256']==cp['sha256'] and proof['source_identity']['config_sha256']==spec['h1']['config_sha256']
 bound.append({'path':cp['path'],'sha256':cp['sha256'],'update':cp['update']})
auth['parameters']['checkpoints']=bound;auth['status']='AUTHORIZED'
assert not policy.stopped(control)
eid=f'E-R5-ZERO-{group}-ACTUAL-TRAINING-20261014';store=Store(control)
evidence={'id':eid,'path':str(review_path),'sha256':sha(review_path),'accepted':True,
 'independent_handle':'/root/pilot_reviewer','scope':'Actual '+group+' scientific training/checkpoint acceptance; conditional exact originalZ evaluation admission only',
 'source_commit':review['identities']['source_commit'],'job_id':review['identities']['job_id'],'not_live_or_training_authority':True}
store.transact('INDEPENDENT_ACTUAL_TRAINING_ACCEPTED','pair-actual-training:'+evidence['sha256'],lambda s:s['evidence'].update({eid:evidence}),object_id=eid,producer='Supervisor-Commander')
actual=Path(spec['pins']['authorization']['path']);assert not actual.exists() and not actual.parent.parent.exists()
actual.parent.mkdir(parents=True);(actual.parent.parent/'results').mkdir()
actual.write_text(json.dumps(auth,ensure_ascii=False,indent=2)+'\n');actual.chmod(0o444)
spec['pins']['authorization']['sha256']=sha(actual)
prelive=read(base/'reviews/review-zero-prelive.json')
spec['conditions'].extend([{'evidence_id':'E-R5-ZERO-PRELIVE-20261014','sha256':sha(base/'reviews/review-zero-prelive.json')},
                         {'evidence_id':eid,'sha256':evidence['sha256']}])
experiments.validate(spec,control);pid=experiments.propose(store,spec);shown=policy.show_proposal(store,pid)
after=policy.decide(store,pid,shown['sha256'],'APPROVED',
 'Commander new exact '+group+' u16/u64 originalZ evaluation approval under user pair Goal; independent actualtraining/checkpoint acceptance '+evidence['sha256']+
 '. Correct actualcheckpoint source/config/hash, evalseed20260909,12800nontraining/0updates/0checkpointwrites. Trainingprocess remainsFAILED/INCOMPLETE withfullreservation; noautorerun. Commitinputs then STOP/idle/budget/disk/internalcoexistence before single release.')
approval=next(a for a in after['approvals'].values() if a['proposal_id']==pid)
save(base/f'inputs/{group}-evaluation-authorization.authorized.json',auth);assert sha(base/f'inputs/{group}-evaluation-authorization.authorized.json')==sha(actual)
save(base/f'inputs/{group}-evaluation-spec.authorized.json',spec)
save(base/f'{group}-evaluation-commander-approval.json',{'group':group,'mode':'pilot-evaluation','proposal_id':pid,'proposal_sha256':shown['sha256'],'approval':approval,
 'actual_authorization':{'path':str(actual),'sha256':sha(actual)},'accepted_actual_training_review':evidence,
 'checkpoint_source_binding':[next(e for e in analysis['checkpoints'] if e['label']==f"u{cp['update']}") for cp in checkpoints],
 'submitted':False,'remaining_budget_at_approval':read(base/'goal_budget.json')})
b=read(base/'goal_budget.json');b['evaluation_authorizations_issued']+=1
for call in b['role_calls']:
 if call['scope'].startswith('Actual '+group+' '):call.update(status='PASS',evidence=f'reviews/review-{group}-training-actual.json',sha256=sha(review_path))
(base/'goal_budget.json').write_text(json.dumps(b,ensure_ascii=False,indent=2)+'\n')
p=base/f'{group}-training-execution-summary.json';summary=read(p);summary['scientific_acceptance']='ACCEPTED_INDEPENDENT_ACTUAL_DATA_REVIEW';summary['independent_review']=evidence;summary['postrun_GPU_queries_performed']=2;summary['postrun_audit_receipts']=[f'{group}-training-postrun-gpu-{i}.json' for i in range(2)];p.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'group':group,'mode':'pilot-evaluation','approval':approval['approval_id'],'authorization_sha256':sha(actual),'submitted':False}))
