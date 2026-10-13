import hashlib,json,shutil,subprocess,sys
from pathlib import Path
from rlflow.store import Store,sha as object_sha
from rlflow import experiments,policy
from rlflow.jobs import owns
from rlflow.releases import verify,files

root=Path('/home/mint/rl_dev/instinctRL-dev');base=root/'docs/experiments/r5-zero-correction-20261014';control=root/'coordination/research'
group='C1';mode='evaluation'
assert (base/'user_resource_decision_extra_evaluation.json').is_file(), 'requirednewuserdecisionabsent'
train,nontrain,updates,checkpoints,wall=(32768,0,64,7,3610) if mode=='training' else (0,12800,0,0,1810)
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
spec_path=base/'inputs/C1-extra-evaluation-spec.authorized.json';spec=read(spec_path);b=read(base/'goal_budget.json')
assert not any(row['spec_id']==spec['id'] for row in b['runs']), 'entry reservation already consumed; inspect existing handle instead'
assert not policy.stopped(control)
store=Store(control);s=store.replay();assert not s['paused']
assert all(job['status'] in ['FAILED','SUCCEEDED','STOPPED'] and not owns(job.get('identity')) for job in s['jobs'].values())
assert b['used']['live_starts']+1<=b['caps']['live_starts']
assert b['used']['training_transitions']+b['reserved']['training_transitions']+train<=b['caps']['training_transitions']
assert b['used']['nontraining_transitions']+b['reserved']['nontraining_transitions']+nontrain<=b['caps']['nontraining_transitions']
assert b['used']['optimizer_updates']+b['reserved']['optimizer_updates']+updates<=b['caps']['optimizer_updates']
assert b['used']['training_checkpoint_writes']+b['reserved']['training_checkpoint_writes']+checkpoints<=b['caps']['training_checkpoint_writes']
assert b['used']['wall_seconds_including_cleanup']+wall<=b['caps']['wall_seconds_including_cleanup']
assert b['used']['allocated_gpu_seconds']+wall<=b['caps']['allocated_gpu_seconds']
available=shutil.disk_usage(root).free;assert available>64*1024**3
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True)==subprocess.check_output(['git','rev-parse','origin/main'],cwd=root,text=True)
assert subprocess.check_output(['git','show','HEAD:'+str(spec_path.relative_to(root))],cwd=root)==spec_path.read_bytes()
packet=read(base/'launch_inputs.fd-fix-proposed.json');release=verify(control/'control-releases'/packet['tools_release_content_id'])
for name,path in files(root/'tools/rlflow').items():assert hashlib.sha256(path.read_bytes()).hexdigest()==release['files'][name]
experiments.validate(spec,control);approval=experiments.authorized(s,spec,control)
jid='J-'+object_sha({'id':spec['id'],'spec':spec})[:16];assert jid not in s['jobs']
admission={'status':'PASS_EXACT_SINGLE_ENTRY_ADMISSION','group':group,'mode':mode,'job_id':jid,'spec_id':spec['id'],
 'approval_id':approval['approval_id'],'authorization_sha256':spec['pins']['authorization']['sha256'],
 'input_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'source_commit':packet['baseline']['snapshot_commit'],'config_sha256':packet['baseline']['config_sha256'],
 'tool_release':packet['tools_release_content_id'],'STOP':[],'active_or_uncertain_jobs':0,
 'owned_previous_supervisors_alive':0,'available_disk_bytes':available,
 'GPU_queries':0,'coexistence':'DEFERRED_TO_EXISTING_INTERNAL_SIX_CALL_GATE_BEFORE_APP',
 'remaining_budget_before_reservation':{'caps':b['caps'],'used':b['used'].copy(),'reserved':b['reserved'].copy()}}
admission_path=base/'C1-extra-evaluation-release-admission.json';assert not admission_path.exists();save(admission_path,admission)
assert b['output_reservations_bytes']+16*1024**3<=b['caps']['output_bytes']
b['output_reservations_bytes']+=16*1024**3
entry={'group':group,'mode':'pilot-'+mode,'spec_id':spec['id'],'job_id':jid,'status':'RESERVED_BEFORE_SINGLE_SUBMISSION',
 'starts_charged':1,'simulator_initialization_reserved':1,'training_transitions_reserved':train,
 'nontraining_transitions_reserved':nontrain,'optimizer_updates_reserved':updates,'training_checkpoint_writes_reserved':checkpoints,
 'output_bytes_reserved':16*1024**3,'postrun_nvidia_smi_calls_used':0,'observed':{},'settlement':'PENDING_ACTUAL_FINAL_COLLECTOR'}
b['runs'].append(entry);b['used']['live_starts']+=1;b['used']['simulator_initialization_reservations']+=1
b['reserved']['training_transitions']+=train;b['reserved']['nontraining_transitions']+=nontrain;b['reserved']['optimizer_updates']+=updates;b['reserved']['training_checkpoint_writes']+=checkpoints
save(base/'goal_budget.json',b)
job=experiments.submit(store,spec)
assert job['id']==jid
save(base/'C1-extra-evaluation-started-job.json',job)
entry['status']=job['status'];save(base/'goal_budget.json',b)
state_path=root/'coordination/PROJECT_STATE.json';state=read(state_path);state['phase']='R5_ZERO_CORRECTION_'+group+'_'+mode.upper()+'_LIVE';state['current_goal'].update(current_job=jid,current_exact_release=approval['approval_id'],next_action='Poll exactlivejob; realgate/data/status/ownedcleanup then settle resources, noresubmit.')
save(state_path,state)
print(json.dumps({'status':job['status'],'group':group,'job_id':jid,'identity':job['identity'],'approval_id':job['approval_id']}))
