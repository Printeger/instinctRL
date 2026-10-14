"""Thin current-Goal adapters around existing rlflow exact approvals and jobs.

No scheduling, role substitution or automatic retry. Each action uses an explicit
immutable entry, independent review, delegated master and cumulative ledger.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'docs/experiments/r5-formal-readiness-auto';CONTROL=ROOT/'coordination/research'
FIELDS={'training_transitions':'training_transitions','nontraining_transitions':'nontraining_transitions','optimizer_updates':'optimizer_updates','training_checkpoint_writes':'checkpoint_writes'}
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v,*,new=False):
 if new:assert not p.exists(),p
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def previous_quiescence(control,state,*,probe=None):
 from rlflow.jobs import owns
 probe = owns if probe is None else probe
 observations=[]
 for jid,j in state['jobs'].items():
  assert j['status'] in ['FAILED','STOPPED','SUCCEEDED'] and not probe(j.get('identity')),jid
  d=control/'jobs'/jid;ackp=d/'lifeguard-ack.json'
  if ackp.exists():
   ack=read(ackp);clean=ack['descendant_cleanup'];assert ack['no_survivors'];boundary='NORMAL_ACK'
  else:
   # Exact historical exception only: the old watchdog recovery is already
   # independently accepted and must never impersonate a normal handshake.
   assert control.resolve()==CONTROL and jid=='J-afe4bda51f5206e5','Missing normal cleanup ACK: '+jid
   history=read(BASE/'HISTORICAL_CLEANUP_REUSE.json');assert history['job_id']==jid
   for pin in history['pins'].values():assert sha(Path(pin['path']))==pin['sha256']
   e=state['evidence'][history['accepted_evidence_id']];assert e['accepted'] and e['sha256']==history['pins']['review']['sha256'] and e['path']==history['pins']['review']['path']
   settlement=read(Path(history['pins']['settlement']['path']));assert settlement['job_id']==jid and settlement['normal_completion_ack_present'] is False and settlement['no_known_owned_survivors']
   clean=settlement['descendant_cleanup'];boundary='INDEPENDENTLY_ACCEPTED_HISTORICAL_KNOWN_OWNERSHIP_RECOVERY_NOT_NORMAL_ACK'
  assert clean['quiescent'] and clean['stable_scans']>=2 and not clean['survivors'] and not clean.get('timed_out',False),jid
  guard=read(d/'lifeguard-ready.json')['identity'];child=read(d/'process-identity.json')
  ledger=read(d/'descendant-ledger.json')
  ids=[j.get('identity'),guard,child]+clean['discovered']+ledger['descendants']
  assert not any(probe(i) for i in ids),jid
  observations.append({'job_id':jid,'boundary':boundary,'all_recorded_owned_absent':True})
 return observations

def idle(store):
 from rlflow import policy
 s=store.replay();assert not policy.stopped(CONTROL) and not s['paused']
 previous_quiescence(CONTROL,s)
 return s

def remaining_check(b,budget,output_cap):
 caps=b['caps'];used=b['used'];entries=b['reservations']['live_entries'];formal=b['reservations']['formal_initial_segment'];confirm=b['reservations']['confirmation_and_transition']
 for key in ['live_starts','simulator_initialization_reservations']:
  assert used[key]+1+formal[key]+confirm[key]<=caps[key],key
 for key,src in FIELDS.items():
  amount=used[key]+sum(e['reserved'][key] for e in entries)+budget[src]+formal[key]+confirm[key]
  assert amount<=caps[key],key
 for key in ['actor_optimizer_steps','critic_optimizer_steps']:
  amount=used[key]+sum(e['reserved'][key] for e in entries)+budget['optimizer_updates']*8+formal[key]+confirm[key]
  assert amount<=caps[key],key
 for key in ['allocated_gpu_seconds','wall_seconds_including_cleanup']:
  amount=used[key]+sum(e['reserved'][key] for e in entries)+budget['wall_seconds']+10+formal[key]+confirm[key]
  assert amount<=caps[key],key
 output=sum(e['output_cap_bytes'] for e in entries)+sum(s['reserved_bytes'] for s in b['reservations']['source_snapshots'])+output_cap+formal['output_peak_and_reservations_bytes']+confirm['output_peak_and_reservations_bytes']+2*1024**3
 assert output<=caps['output_peak_and_reservations_bytes'],'output reservation cap'
 stage=b['stage_allocation']
 for key,src in FIELDS.items():assert used[key]+sum(e['reserved'][key] for e in entries)+budget[src]<=stage[key],key+' stage'
 assert used['live_starts']+1<=stage['live_starts'],'stage starts'
 assert used['simulator_initialization_reservations']+1<=stage['simulator_initialization_reservations'],'stage init'
 for key in ['allocated_gpu_seconds','wall_seconds_including_cleanup']:
  assert used[key]+sum(e['reserved'][key] for e in entries)+budget['wall_seconds']+10<=stage[key],key+' stage'
 assert sum(e['output_cap_bytes'] for e in entries)+sum(s['reserved_bytes'] for s in b['reservations']['source_snapshots'])+output_cap+2*1024**3<=stage['output_peak_and_reservations_bytes'],'stage output'
 return {'conservative_output_reserved_bytes_including_confirmation_formal_and_metadata':output}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['issue','submit','settle','postrun']);ap.add_argument('experiment_id');ap.add_argument('group',choices=['R0','R1']);ap.add_argument('mode',choices=['research-training','research-evaluation']);ap.add_argument('--review',type=Path);ap.add_argument('--checkpoint-acceptance',type=Path);args=ap.parse_args()
 from rlflow.store import Store,sha as object_sha
 from rlflow import experiments,policy
 from rlflow.jobs import owns
 from rlflow.releases import verify,files
 store=Store(CONTROL);D=BASE/'experiments'/args.experiment_id;packetp=D/'launch_inputs.proposed.json';packet=read(packetp);row=next(r for r in packet['entries'] if r['group']==args.group and r['mode']==args.mode)
 name=args.group+'-'+args.mode;specp=D/'inputs'/(name+'.authorized.json');b=read(BASE/'goal_budget.json')
 def accept(eid,path,scope):
  e={'id':eid,'path':str(path),'sha256':sha(path),'accepted':True,'producer':'Supervisor-Commander','scope':scope}
  store.transact('FORMAL_RESEARCH_CONDITION_ACCEPTED',eid+':'+e['sha256'],lambda s:s['evidence'].update({eid:e}),object_id=eid,producer='Supervisor-Commander');return {'evidence_id':eid,'sha256':e['sha256']}
 if args.action=='issue':
  idle(store);assert args.review and args.review.is_file();review=read(args.review);assert review['verdict']=='PASS' and not review['must_fix']
  identities=review['identities'];assert packet['baseline']['snapshot_commit']==identities['source_commit']
  values=json.dumps(identities);assert sha(packetp) in values,'review must bind exact proposed packet'
  spec=deepcopy(row['experiment_proposal']);auth=deepcopy(row['authorization_proposal']);training=args.mode=='research-training'
  if training:
   expected=D/row['expected_authorization']['path'];assert sha(expected)==row['expected_authorization']['sha256'] and sha(expected) in values
   assert read(expected)==auth
  else:
   assert args.checkpoint_acceptance and args.checkpoint_acceptance.is_file();accepted=read(args.checkpoint_acceptance);assert accepted['verdict']=='PASS' and not accepted['must_fix']
   for ck in auth['parameters']['checkpoints']:
    f=Path(ck['path']);ck['sha256']=sha(f);assert ck['sha256'] in json.dumps(accepted)
   spec['conditions'].append(accept('E-'+args.experiment_id+'-ACTUAL-TRAINING',args.checkpoint_acceptance,'Independent actualpairedtraining/firstgate/checkpointSHA acceptance'))
  spec['conditions'].append(accept('E-'+args.experiment_id+'-PRELIVE',args.review,'Independent exact research method/profile/source prelive'))
  master=BASE/'MASTER_AUTHORITY.json';assert sha(master)==packet['master_authority']['sha256']
  state=store.replay();e=state['evidence']['E-R5-FORMAL-MASTER-DELEGATION-20261014'];assert e['accepted'] and e['sha256']==sha(master)
  actual=Path(spec['pins']['authorization']['path']);assert not actual.exists() and not actual.parent.parent.exists()
  actual.parent.mkdir(parents=True);(actual.parent.parent/'results').mkdir();save(actual,auth,new=True);actual.chmod(0o444);spec['pins']['authorization']['sha256']=sha(actual)
  if training:assert sha(actual)==row['expected_authorization']['sha256']
  experiments.validate(spec,CONTROL);pid=experiments.propose(store,spec);shown=policy.show_proposal(store,pid)
  st=policy.decide(store,pid,shown['sha256'],'APPROVED','Explicit masterdelegate '+sha(master)+'; independentexactreview '+sha(args.review)+'; exactcard '+packet['experiment_card']['sha256']+'. Preservefirst512gate/actorboundary/originalZ/STOP/coexistence/quota/cleanup. Commit/tag/normalpush before single release; no automaticretry; allmaster/confirmation/formalreserves protected.')
  approval=next(a for a in st['approvals'].values() if a['proposal_id']==pid);save(specp,spec,new=True);save(D/(name+'-approval.json'),{'proposal_id':pid,'proposal_sha256':shown['sha256'],'approval':approval,'actual_authorization':{'path':str(actual),'sha256':sha(actual)},'review_sha256':sha(args.review),'master_sha256':sha(master),'submitted':False},new=True)
  print(json.dumps({'status':'EXACT_DELEGATED_APPROVED_NOT_SUBMITTED','approval_id':approval['approval_id'],'spec':str(specp)}));return
 spec=read(specp);matches=[e for e in b['reservations']['live_entries'] if e['spec_id']==spec['id']]
 if args.action=='submit':
  assert not matches,'Already charged; poll exacthandle, never resubmit'
  idle(store);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT)==subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT)
  assert subprocess.check_output(['git','show','HEAD:'+str(specp.relative_to(ROOT))],cwd=ROOT)==specp.read_bytes()
  release=verify(Path(packet['baseline']['tool_release_path']))
  assert {n:sha(p) for n,p in files(ROOT/'tools/rlflow').items()}==release['files']
  auth=read(Path(spec['pins']['authorization']['path']));budget=auth['budgets'];p=auth['parameters'];remaining=remaining_check(b,budget,p['output_cap_bytes'])
  assert shutil.disk_usage(ROOT).free>30*1024**3+p['output_cap_bytes']
  experiments.validate(spec,CONTROL);approval=experiments.authorized(store.replay(),spec,CONTROL)
  jid='J-'+object_sha({'id':spec['id'],'spec':spec})[:16];assert jid not in store.replay()['jobs']
  save(D/(name+'-admission.json'),{'status':'PASS','input_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'job_id':jid,'spec_id':spec['id'],'approval_id':approval['approval_id'],'STOP':[],'old_jobs_terminal_noowned':True,'disk_free_bytes':shutil.disk_usage(ROOT).free,'remaining_check':remaining,'prelaunch_GPU_queries':0,'internal_coexistence':'Originalsixquerygate beforeApp'},new=True)
  reserved={k:budget[v] for k,v in FIELDS.items()};reserved.update(actor_optimizer_steps=budget['optimizer_updates']*8,critic_optimizer_steps=budget['optimizer_updates']*8,allocated_gpu_seconds=budget['wall_seconds']+10,wall_seconds_including_cleanup=budget['wall_seconds']+10)
  entry={'spec_id':spec['id'],'job_id':jid,'experiment_id':args.experiment_id,'group':args.group,'mode':args.mode,'status':'RESERVED_BEFORE_SUBMIT','reserved':reserved,'output_cap_bytes':p['output_cap_bytes'],'observed':{},'postrun_queries_used':0,'settled':False}
  b['reservations']['live_entries'].append(entry);b['used']['live_starts']+=1;b['used']['simulator_initialization_reservations']+=1;save(BASE/'goal_budget.json',b)
  job=experiments.submit(store,spec);assert job['id']==jid;entry['status']=job['status'];save(BASE/'goal_budget.json',b);save(D/(name+'-started-job.json'),job,new=True)
  s=read(ROOT/'coordination/PROJECT_STATE.json');s['current_goal'].update(current_job=jid,current_exact_release=approval['approval_id'],next_action='Poll this exactlivejob; no resubmit. Terminal+normalownedACK, settle/postrun, CPU science then continue Goal.');save(ROOT/'coordination/PROJECT_STATE.json',s)
  print(json.dumps({'job_id':jid,'status':job['status'],'identity':job['identity']}));return
 assert len(matches)==1;entry=matches[0];jd=CONTROL/'jobs'/entry['job_id'];receipt=read(jd/'receipt.json');assert receipt['status'] in ['FAILED','SUCCEEDED','STOPPED']
 job=store.replay()['jobs'][entry['job_id']];ack=read(jd/'lifeguard-ack.json');clean=ack['descendant_cleanup']
 assert ack['no_survivors'] and clean['quiescent'] and clean['stable_scans']>=2 and not clean['survivors']
 assert not owns(job['identity']) and not owns(read(jd/'lifeguard-ready.json')['identity']) and not owns(read(jd/'process-identity.json')) and not any(owns(i) for i in clean['discovered'])
 if args.action=='settle':
  assert not entry['settled'];job=experiments.reconcile(store,entry['job_id']);save(D/(name+'-final-job.json'),job,new=True)
  attempt=Path(spec['h1']['attempt']);out=attempt/'results'/args.mode;raw=read(out/'status.json');archive=D/'live'/name;assert not archive.exists();objects=[];copies=0
  originalbytes=sum(p.stat().st_size for folder in [jd,attempt] for p in folder.rglob('*') if p.is_file());assert originalbytes<=entry['output_cap_bytes']
  for category,folder in [('job',jd),('attempt',attempt)]:
   for f in sorted(folder.rglob('*')):
    assert not f.is_symlink()
    if not f.is_file():continue
    r={'original':str(f),'category':category,'relative':str(f.relative_to(folder)),'bytes':f.stat().st_size,'sha256':sha(f),'archived':None}
    if r['bytes']<=2*1024**2:
     assert copies+r['bytes']<=128*1024**2 and originalbytes+copies+r['bytes']<=entry['output_cap_bytes'];dest=archive/category/f.relative_to(folder);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest);assert sha(dest)==r['sha256'];r['archived']=str(dest);copies+=r['bytes']
    objects.append(r)
  inventory={'job_id':entry['job_id'],'roots':{'job':str(jd),'attempt':str(attempt)},'files':objects,'total_original_bytes':sum(r['bytes'] for r in objects),'total_archive_copy_bytes':copies,'coverage':'Every original job and entire attempt file; no history rewrite or replacement of big originals.'};assert inventory['total_original_bytes']+copies<=entry['output_cap_bytes'];save(archive/'INVENTORY.json',inventory,new=True)
  h1=job['h1_evidence'];c=h1['counters'];obs={k:c[v] for k,v in {'training_transitions':'training_transitions','nontraining_transitions':'nontraining_transitions','optimizer_updates':'real_optimizer_updates','training_checkpoint_writes':'real_checkpoint_writes','actor_optimizer_steps':'actor_optimizer_steps','critic_optimizer_steps':'critic_optimizer_steps'}.items()}
  for k,v in obs.items():assert type(v) is int and v>=0;b['observed'][k]=b['observed'].get(k,0)+v
  if h1['counter_completeness']=='FINAL':
   for k,v in obs.items():b['used'][k]+=v;entry['reserved'][k]=0
  for k,v in [('wall_seconds_including_cleanup',receipt['elapsed_seconds']),('allocated_gpu_seconds',receipt['allocated_seconds'])]:b['used'][k]+=v;entry['reserved'][k]=0
  outputbytes=inventory['total_original_bytes']+copies;b['used']['output_peak_and_reservations_bytes']+=outputbytes
  entry.update(status=job['status'],settled=True,observed=obs,native_exit_code=receipt['exit_code'],raw_status=raw['status'],formal_counter_completeness=h1['counter_completeness'],formal_transitions=h1['transitions'],owned_cleanup='NORMAL_ACK_2_STABLE_EMPTY_NO_SURVIVORS',output_observed_bytes=outputbytes,inventory=str((archive/'INVENTORY.json').relative_to(BASE)))
  save(BASE/'goal_budget.json',b);save(D/(name+'-execution-summary.json'),entry,new=True)
  s=read(ROOT/'coordination/PROJECT_STATE.json');s['current_goal'].update(current_job=None,current_exact_release=None,next_action='Exactly2boundpostrunqueries then realdataanalysis/independentreview; noblindretry.');save(ROOT/'coordination/PROJECT_STATE.json',s)
  print(json.dumps(entry));return
 assert entry['settled'] and entry['postrun_queries_used']==0
 from training_workloads.unified_h1.config import resource_binding_pins
 pins=resource_binding_pins();exe=Path(pins['nvidia_smi_path']);assert sha(exe)==pins['nvidia_smi_sha256']
 for i,q in enumerate(['--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu','--query-compute-apps=pid,process_name,used_gpu_memory']):
  entry['postrun_queries_used']+=1;save(BASE/'goal_budget.json',b);argv=[str(exe),q,'--format=csv,noheader,nounits'];r=subprocess.run(argv,capture_output=True,text=True,timeout=15)
  save(D/(name+f'-postrun-{i}.json'),{'job_id':entry['job_id'],'after_normal_owned_ACK':True,'ordinal':i,'argv':argv,'executable_sha256':pins['nvidia_smi_sha256'],'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},new=True);assert r.returncode==0
 summary_path=D/(name+'-execution-summary.json');summary=read(summary_path);summary['postrun_queries_used']=entry['postrun_queries_used'];save(summary_path,summary)
 print(json.dumps({'job_id':entry['job_id'],'postrun_queries_used':2}))
if __name__=='__main__':main()
