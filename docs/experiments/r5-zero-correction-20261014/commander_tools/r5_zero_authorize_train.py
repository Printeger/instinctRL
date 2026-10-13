import copy,hashlib,json,shutil
from pathlib import Path
from rlflow.store import Store
from rlflow import experiments,policy
from rlflow.releases import verify
from rlflow.jobs import owns

root=Path('/home/mint/rl_dev/instinctRL-dev');base=root/'docs/experiments/r5-zero-correction-20261014';control=root/'coordination/research'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 assert not p.exists(),p
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
packet=read(base/'launch_inputs.fd-fix-proposed.json');review_path=base/'reviews/review-zero-prelive.json';review=read(review_path)
assert review['verdict']=='PASS' and not review['must_fix']
assert sha(base/'launch_inputs.fd-fix-proposed.json')==review['identities']['launch_sha256']
assert not policy.stopped(control)
store=Store(control);initial=store.replay()
assert not initial['paused']
assert all(j['status'] in ['FAILED','SUCCEEDED','STOPPED'] and not owns(j.get('identity')) for j in initial['jobs'].values())
assert verify(Path(packet['entries'][0]['experiment_proposal']['cwd']).parents[1]/'control-releases'/packet['tools_release_content_id'])['id']==packet['tools_release_content_id']
assert shutil.disk_usage(root).free>64*1024**3
eid='E-R5-ZERO-PRELIVE-20261014'
evidence={'id':eid,'path':str(review_path),'sha256':sha(review_path),'accepted':True,'role_call':2,
 'independent_handle':'/root/pilot_reviewer','candidate_commit':review['identities']['source_commit'],
 'input_commit':review['identities']['input_commit'],'scope':'Exact C0/C1 training prelive; conditional evaluation template only',
 'not_live_or_training_authority':True}
store.transact('INDEPENDENT_PAIR_PRELIVE_ACCEPTED','r5-zero-prelive:'+evidence['sha256'],
 lambda s:s['evidence'].update({eid:evidence}),object_id=eid,producer='Supervisor-Commander')
records=[]
for row in packet['entries']:
 if row['mode']!='pilot-training':continue
 group=row['group'];spec=copy.deepcopy(row['experiment_proposal']);expected=base/row['expected_authorization']['path']
 assert sha(expected)==review['identities']['expected_training_AUTH_sha256'][group]
 actual=Path(spec['pins']['authorization']['path']);assert not actual.exists() and not actual.parent.parent.exists()
 actual.parent.mkdir(parents=True);(actual.parent.parent/'results').mkdir()
 actual.write_bytes(expected.read_bytes());actual.chmod(0o444)
 assert sha(actual)==spec['pins']['authorization']['sha256']
 spec['conditions'].append({'evidence_id':eid,'sha256':evidence['sha256']})
 experiments.validate(spec,control)
 pid=experiments.propose(store,spec);shown=policy.show_proposal(store,pid)
 after=policy.decide(store,pid,shown['sha256'],'APPROVED',
  'Commander exact approval under new user R5 zero-correction Goal; independent Role2 accepted. Same d257f9 quota-fixed newmethod baseline; solew0/.05 atsharedactorLR1e-4, group '+group+
  ',64updates/32768transitions, first actual512samplegate before BOTH optimizers, no retries. User task authority SHA '+sha(base/'user_task_authority.json')+
  '; require committed input pack and idle/STOP/budget/disk/coexistence/ownedcleanup before single release. Approval is not submission.')
 approvals=[a for a in after['approvals'].values() if a['proposal_id']==pid]
 assert len(approvals)==1 and approvals[0]['decision']=='APPROVED'
 approval=approvals[0]
 save(base/f'inputs/{group}-training-spec.authorized.json',spec)
 record={'group':group,'mode':'pilot-training','proposal_id':pid,'proposal_sha256':shown['sha256'],
  'approval':approval,'actual_authorization':{'path':str(actual),'sha256':sha(actual)},
  'independent_pre_review':evidence,'user_task_authority_sha256':sha(base/'user_task_authority.json'),
  'remaining_budget_before_submission':read(base/'goal_budget.json'), 'submitted':False}
 save(base/f'{group}-training-commander-approval.json',record);records.append(record)
b=read(base/'goal_budget.json');b['role_calls'][1].update(status='PASS',evidence='reviews/review-zero-prelive.json',sha256=sha(review_path));b['training_authorizations_issued']=2
(base/'goal_budget.json').write_text(json.dumps(b,ensure_ascii=False,indent=2)+'\n')
save(base/'training-authorization-summary.json',{'status':'EXACT_COMMANDER_APPROVED_NOT_SUBMITTED','training_groups':[{k:r[k] for k in ('group','proposal_id','proposal_sha256','actual_authorization')} for r in records],
 'production_commit':packet['baseline']['snapshot_commit'],'config_sha256':packet['baseline']['config_sha256'],
 'tool_release':packet['tools_release_content_id'],'actual_training_starts':0,'actual_optimizer_updates':0,'gpu_queries':0,'simulation_starts':0})
s_path=root/'coordination/PROJECT_STATE.json';s=read(s_path);g=s['current_goal'];g.update(training_authorization_issued=True,current_exact_release='P0-training-commander-approval.json',next_action='Commit/tag/push exactauthorizedinputs; validate idle/STOP/budget/disk then submit P0 once.')
g['independent_review']={'status':'PASS_PRELIVE_ONLY','role_call':2,'path':'../docs/experiments/r5-zero-correction-20261014/reviews/review-zero-prelive.json','sha256':sha(review_path)}
s['phase']='R5_ZERO_CORRECTION_EXACT_TRAINING_APPROVALS_ISSUED_NOT_SUBMITTED';s_path.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
index=base/'EVIDENCE_INDEX.md';index.write_text(index.read_text()+'\nIndependent Role2 PASS archived in reviews/review-zero-prelive.json. Existing Commander issued exact C0/C1 training approvals and actual AUTH files; training-authorization-summary.json and group approval/spec files preserve identity and expectedbytes. No submission/GPU query/optimizer yet. Evaluation remains unissued pending actualtraining acceptance.\n')
print(json.dumps({'status':'EXACT_COMMANDER_APPROVED_NOT_SUBMITTED','groups':[{'group':r['group'],'approval':r['approval']['approval_id']} for r in records],'gpu_queries':0}))
