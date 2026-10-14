"""Freeze one registered research card with existing H1 inputs; never release."""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/experiments/r5-formal-readiness-auto'
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):
 assert not p.exists(),p
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('experiment_id');args=ap.parse_args();D=BASE/'experiments'/args.experiment_id
 from rlflow.store import Store
 from rlflow import policy
 from rlflow.jobs import owns
 from rlflow.releases import freeze,verify
 from training_workloads.unified_h1.budget import required_mode_budget
 C=ROOT/'coordination/research';s=Store(C).replay();assert not policy.stopped(C) and not s['paused']
 assert all(j['status'] in ['FAILED','STOPPED','SUCCEEDED'] and not owns(j.get('identity')) for j in s['jobs'].values())
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();configpath=D/'snapshot_config.json';cardpath=D/'EXPERIMENT_CARD.json'
 for p in [configpath,cardpath]:assert subprocess.check_output(['git','show',commit+':'+str(p.relative_to(ROOT))],cwd=ROOT)==p.read_bytes()
 config=read(configpath);card=read(cardpath);snapshot=Path(config['workload']['path']);assert not snapshot.exists()
 ledger=read(BASE/'goal_budget.json');reservation=2*1024**3
 reserved=ledger['reservations']['formal_initial_segment']['output_peak_and_reservations_bytes']+ledger['reservations']['confirmation_and_transition']['output_peak_and_reservations_bytes']+sum(x['reserved_bytes'] for x in ledger['reservations']['source_snapshots'])
 assert reserved+reservation+card['output_reservation_bytes']<=ledger['caps']['output_peak_and_reservations_bytes']
 assert shutil.disk_usage(ROOT).free>30*1024**3+reservation+card['output_reservation_bytes']
 ledger['reservations']['source_snapshots'].append({'path':str(snapshot),'reserved_bytes':reservation,'status':'RESERVED_BEFORE_CLONE','source_commit':commit});(BASE/'goal_budget.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
 subprocess.run(['git','clone','--no-hardlinks','--no-checkout','--local',str(ROOT),str(snapshot)],check=True)
 subprocess.run(['git','-C',str(snapshot),'-c','core.hooksPath=/dev/null','checkout','--detach',commit],check=True)
 (snapshot/'training_workloads/unified_h1/candidate_config.json').write_bytes(configpath.read_bytes())
 tree=subprocess.check_output(['git','-C',str(snapshot),'rev-parse','HEAD^{tree}'],text=True).strip()
 from training_workloads.unified_h1.b0_handoff import _candidate_diff
 overlay=hashlib.sha256(_candidate_diff(snapshot,commit)).hexdigest()
 release=verify(freeze(C/'control-releases',root=ROOT/'tools/rlflow'))
 template=read(ROOT/'docs/experiments/r5-zero-correction-20261014/launch_inputs.fd-fix-proposed.json')
 rows=[]
 for g in card['profiles']:
  for mode in ['research-training','research-evaluation']:
   training=mode=='research-training';old=next(r for r in template['entries'] if r['group']=='C1' and r['mode']==('pilot-training' if training else 'pilot-evaluation'))
   spec=deepcopy(old['experiment_proposal']);auth=deepcopy(old['authorization_proposal']);p=deepcopy(card['profiles'][g]);p['experiment_card']={'path':str(cardpath),'sha256':sha(cardpath)}
   # Exact evaluation parameters bind real checkpoints later; preserve unresolved draft.
   budget=required_mode_budget('research-training',p) if training else {'live_process_starts':1,'simulator_initializations':1,'initialization_ticks':10,'training_transitions':0,'nontraining_transitions':len(p['evaluation_updates'])*6400,'optimizer_updates':0,'checkpoint_writes':0,'explicit_resets':len(p['evaluation_updates']),'automatic_resets':len(p['evaluation_updates'])*6400,'wall_seconds':1800}
   attempt=C/'attempts'/f'{args.experiment_id}-{g}-{mode.upper()}-{commit[:12]}';assert not attempt.exists();authpath=attempt/'commands/AUTHORIZATION.json'
   trainattempt=C/'attempts'/f'{args.experiment_id}-{g}-RESEARCH-TRAINING-{commit[:12]}'
   if not training:p['checkpoints']=[{'path':str(trainattempt/f'results/research-training/checkpoints/u{u}.pt'),'sha256':None,'update':u} for u in p['evaluation_updates']]
   auth.update(status='AUTHORIZED',order_id=p['experiment_id']+'-'+g+'-'+mode.upper(),method_sha256=config['method']['proposal_sha256'],config_sha256=sha(configpath),product_commit=commit,product_tree=tree,workload_commit=commit,workload_tree=tree,attempt=str(attempt),mode=mode,budgets=budget,parameters=p)
   argv=[config['execution']['python_path'],'-B','-m','training_workloads.unified_h1.formal_unified_h1','--authorization',str(authpath),'--attempt',str(attempt),'--mode',mode]
   pin=lambda f:{'path':str(f),'sha256':sha(f)}
   spec.update(id=attempt.name,stage_id='R5-FORMAL-READINESS-AUTO-01',argv=argv,cwd=str(snapshot),method=config['method']['version'],seed=p['training_seed'] if training else p['evaluation_seed'],resources={'starts':1,'wall_seconds':budget['wall_seconds']+10,'gpu_seconds':budget['wall_seconds']+10,'transitions':budget['training_transitions']+budget['nontraining_transitions']},conditions=[{'evidence_id':'E-R5-FORMAL-MASTER-DELEGATION-20261014','sha256':p['master_authority']['sha256']}])
   spec['pins'].update(dispatcher=pin(snapshot/'training_workloads/unified_h1/local_dispatch.py'),product=pin(snapshot/'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py'),workload=pin(snapshot/'training_workloads/unified_h1/formal_unified_h1.py'),authorization={'path':str(authpath),'sha256':None})
   spec['h1'].update(workload_root=str(snapshot),attempt=str(attempt),mode=mode,config_sha256=sha(configpath))
   row={'group':g,'mode':mode,'experiment_proposal':spec,'authorization_proposal':auth,'missing_bindings':['independent exactprelive+delegatedCommander'] if training else ['accepted real checkpoint/node/source/SHA','independent actual training evidence','exact evaluation delegatedCommander']}
   if training:
    path=D/f'inputs/{g}-training-AUTH.proposed.json';save(path,auth);spec['pins']['authorization']['sha256']=sha(path);row['expected_authorization']={'path':str(path.relative_to(D)),'sha256':sha(path)}
   rows.append(row)
 baseline={'snapshot':str(snapshot),'snapshot_commit':commit,'snapshot_tree':tree,'config_sha256':sha(configpath),'snapshot_diff_sha256':overlay,'read_only':True,'tool_release_id':release['id'],'tool_release_path':str(C/'control-releases'/release['id'])}
 packet={'schema_version':'instinctrl-r5-formal-research-inputs/1','status':'PREPARED_PENDING_INDEPENDENT_EXACT_REVIEW','experiment_id':args.experiment_id,'baseline':baseline,'experiment_card':{'path':str(cardpath),'sha256':sha(cardpath)},'master_authority':{'path':str(BASE/'MASTER_AUTHORITY.json'),'sha256':sha(BASE/'MASTER_AUTHORITY.json')},'entries':rows,'first_rollout_gate':template['first_rollout_gate'],'stop_paths':template['stop_paths'],'generator_sha256':sha(Path(__file__))}
 for f in sorted(snapshot.rglob('*'),key=lambda p:len(p.parts),reverse=True):
  if not f.is_symlink():f.chmod(f.stat().st_mode&~0o222)
 snapshot.chmod(snapshot.stat().st_mode&~0o222)
 snapshotbytes=sum(p.stat().st_size for p in snapshot.rglob('*') if p.is_file());assert snapshotbytes<=reservation
 ledger=read(BASE/'goal_budget.json');ledger['reservations']['source_snapshots'][-1].update(status='FROZEN_READ_ONLY',observed_file_bytes=snapshotbytes,config_sha256=sha(configpath));(BASE/'goal_budget.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
 save(D/'launch_inputs.proposed.json',packet);save(D/'preparation-verification.json',{'status':'PASS_STATIC_PREPARATION_NOT_LIVE_APPROVAL','baseline':baseline,'source_hash_files':len(config['product']['source_sha256'])+len(config['workload']['source_sha256']),'tool_release_files':len(release['files']),'snapshot_observed_bytes':snapshotbytes,'snapshot_reserved_bytes':reservation,'oldjobs_terminal':len(s['jobs']),'GPU_queries':0,'newlive':0})
 print(json.dumps(baseline))
if __name__=='__main__':main()
