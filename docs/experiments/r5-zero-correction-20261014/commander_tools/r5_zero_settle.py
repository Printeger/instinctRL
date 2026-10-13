"""Use existing H1 reconciliation; preserve complete original inventory.

No GPU query. Calls must follow an actual terminal receipt/normal owned ACK.
"""
import hashlib,json,shutil,sys
from pathlib import Path
from rlflow.store import Store
from rlflow import experiments
from rlflow.jobs import owns

root=Path('/home/mint/rl_dev/instinctRL-dev');base=root/'docs/experiments/r5-zero-correction-20261014';control=root/'coordination/research'
group=sys.argv[1];mode=sys.argv[2];assert group in ['C0','C1'] and mode in ['training','evaluation']
def read(path):return json.loads(path.read_text())
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
spec=read(base/f'inputs/{group}-{mode}-spec.authorized.json');store=Store(control)
matches=[j for j in store.replay()['jobs'].values() if j['spec']==spec];assert len(matches)==1;job=matches[0];jid=job['id'];jobdir=control/'jobs'/jid
assert (jobdir/'receipt.json').is_file(), 'not terminal; keep polling same live handle'
receipt=read(jobdir/'receipt.json');assert receipt['status'] in ['FAILED','SUCCEEDED','STOPPED']
assert not owns(job['identity'])
ack=read(jobdir/'lifeguard-ack.json');cleanup=ack['descendant_cleanup']
assert ack['no_survivors'] and cleanup['quiescent'] and cleanup['stable_scans']>=2 and not cleanup['survivors']
assert not owns(read(jobdir/'lifeguard-ready.json')['identity'])
assert not owns(read(jobdir/'process-identity.json'))
assert not any(owns(identity) for identity in cleanup['discovered'])
job=experiments.reconcile(store,jid)
write(base/f'{group}-{mode}-final-job.json',job)
attempt=Path(spec['h1']['attempt']);output=attempt/'results'/spec['h1']['mode'];raw=read(output/'status.json')
inventory_dir=base/f'live-{group}-{mode}';assert not (inventory_dir/'INVENTORY.json').exists()
objects=[]
archive_bytes=0
assert sum(p.stat().st_size for folder in [jobdir,attempt] for p in folder.rglob('*') if p.is_file())<16*1024**3
for category,folder in [('job',jobdir),('attempt',attempt)]:
 assert folder.is_dir()
 for path in sorted(folder.rglob('*')):
  assert not path.is_symlink(), path
  if not path.is_file():continue
  size=path.stat().st_size;row={'category':category,'relative':str(path.relative_to(folder)),
   'original':str(path),'bytes':size,'sha256':sha(path),'archived':None}
  if size<=4*1024**2:
   assert archive_bytes+size<=256*1024**2, 'archive cap beforecopy'
   archive_bytes+=size
   archive=inventory_dir/category/path.relative_to(folder);archive.parent.mkdir(parents=True,exist_ok=True)
   assert not archive.exists();shutil.copyfile(path,archive);assert sha(archive)==row['sha256'];row['archived']=str(archive)
  objects.append(row)
inventory={'schema_version':'instinctrl-r5-pair-complete-original-inventory/1','job_id':jid,'group':group,'mode':spec['h1']['mode'],
 'roots':{'job':str(jobdir),'attempt':str(attempt)},'files':objects,
 'coverage':'Every regular file in actual original job and whole attempt including entire results, prelaunch and actualAUTH; big files retained at original paths with SHA/bytes.',
 'specialized_B0_seal_claimed':False,'total_original_bytes':sum(row['bytes'] for row in objects),
 'total_archive_copy_bytes':sum(row['bytes'] for row in objects if row['archived'])}
write(inventory_dir/'INVENTORY.json',inventory)
b=read(base/'goal_budget.json');entry=next(row for row in b['runs'] if row['job_id']==jid)
assert entry['status'] in ['RUNNING','RESERVED_BEFORE_SINGLE_SUBMISSION'] and not entry.get('actual_accounting_completed'), 'do not settle twice'
counters=job.get('h1_evidence',{}).get('counters',{})
observed={'training_transitions':counters.get('training_transitions'),'nontraining_transitions':counters.get('nontraining_transitions'),
 'optimizer_updates':counters.get('real_optimizer_updates'),'training_checkpoint_writes':counters.get('real_checkpoint_writes'),
 'actor_optimizer_steps':counters.get('actor_optimizer_steps'),'critic_optimizer_steps':counters.get('critic_optimizer_steps')}
for name,value in observed.items():
 if value is None:continue
 assert type(value) is int and value>=0
 b['observed'][name]+=value
complete=job['h1_evidence']['counter_completeness']=='FINAL'
reservation_fields={'training_transitions':'training_transitions_reserved','nontraining_transitions':'nontraining_transitions_reserved',
 'optimizer_updates':'optimizer_updates_reserved','training_checkpoint_writes':'training_checkpoint_writes_reserved'}
if complete:
 for key,entry_key in reservation_fields.items():
  b['reserved'][key]-=entry[entry_key];b['used'][key]+=observed[key];entry[entry_key]=0
entry.update(status=job['status'],native_exit_code=receipt['exit_code'],raw_status=raw.get('status'),
 formal_counter_completeness=job['h1_evidence']['counter_completeness'],
 formal_transitions=job['h1_evidence']['transitions'],observed=observed,
 settlement='FINAL_BY_EXISTING_H1_COLLECTOR' if complete else 'INCOMPLETE_FULL_ENTRY_RESERVATION_RETAINED',
 actual_accounting_completed=True,wall_seconds=receipt['elapsed_seconds'],allocated_gpu_seconds=receipt['allocated_seconds'],
 output_bytes_including_archive=inventory['total_original_bytes']+inventory['total_archive_copy_bytes'],
 owned_cleanup='NORMAL_ACK_TWO_STABLE_EMPTY_SCANS_NO_SURVIVORS', inventory=f'live-{group}-{mode}/INVENTORY.json')
b['used']['wall_seconds_including_cleanup']+=receipt['elapsed_seconds'];b['used']['allocated_gpu_seconds']+=receipt['allocated_seconds']
assert entry['output_bytes_including_archive']<=16*1024**3
b['used']['output_bytes']+=entry['output_bytes_including_archive'];write(base/'goal_budget.json',b)
summary={'group':group,'mode':spec['h1']['mode'],'job_id':jid,'native_status':job['status'],'exit_code':receipt['exit_code'],
 'raw_status':raw.get('status'),'primary_error':raw.get('primary_error'),'observed':observed,
 'formal_counter_completeness':entry['formal_counter_completeness'],'formal_transitions':entry['formal_transitions'],
 'retained_reservations':{key:entry[value] for key,value in reservation_fields.items()},
 'wall_seconds':receipt['elapsed_seconds'],'allocated_gpu_seconds':receipt['allocated_seconds'],
 'original_inventory_objects':len(objects),'original_bytes':inventory['total_original_bytes'],
 'owned_cleanup':entry['owned_cleanup'],'postrun_GPU_queries_performed':0,
 'scientific_acceptance':'PENDING_INDEPENDENT_ACTUAL_DATA_REVIEW'}
write(base/f'{group}-{mode}-execution-summary.json',summary)
s_path=root/'coordination/PROJECT_STATE.json';state=read(s_path);state['current_goal'].update(current_job=None,next_action='Two bound postrun audit calls after ownedquiescence; CPU actualdataanalysis; independent scientificacceptance. Noresubmit.')
state['phase']='R5_ZERO_CORRECTION_'+group+'_'+mode.upper()+'_TERMINAL_PENDING_ACTUAL_REVIEW';write(s_path,state)
print(json.dumps(summary,ensure_ascii=False))
