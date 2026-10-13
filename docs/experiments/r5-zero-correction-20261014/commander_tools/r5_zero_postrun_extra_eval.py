import hashlib,json,subprocess,sys
from pathlib import Path
from rlflow.jobs import owns

root=Path('/home/mint/rl_dev/instinctRL-dev');base=root/'docs/experiments/r5-zero-correction-20261014';control=root/'coordination/research'
group='C1';mode='evaluation'
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
b=read(base/'goal_budget.json');spec=read(base/'inputs/C1-extra-evaluation-spec.authorized.json');entry=next(row for row in b['runs'] if row['spec_id']==spec['id'])
assert entry['actual_accounting_completed'] and entry['postrun_nvidia_smi_calls_used']==0
jobdir=control/'jobs'/entry['job_id'];job=read(base/'C1-extra-evaluation-final-job.json')
assert not owns(job['identity'])
ack=read(jobdir/'lifeguard-ack.json');assert ack['no_survivors'] and ack['descendant_cleanup']['quiescent']
assert not owns(read(jobdir/'lifeguard-ready.json')['identity']) and not owns(read(jobdir/'process-identity.json'))
sys.path.insert(0,job['spec']['cwd'])
from training_workloads.unified_h1.config import resource_binding_pins
pins=resource_binding_pins();exe=Path(pins['nvidia_smi_path'])
assert hashlib.sha256(exe.read_bytes()).hexdigest()==pins['nvidia_smi_sha256']
queries=['--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu','--query-compute-apps=pid,process_name,used_gpu_memory']
for ordinal,query in enumerate(queries):
 path=base/f'C1-extra-evaluation-postrun-gpu-{ordinal}.json';assert not path.exists()
 # Debit before each actual call; a query failure still consumes its allowance.
 entry['postrun_nvidia_smi_calls_used']+=1;write(base/'goal_budget.json',b)
 argv=[str(exe),query,'--format=csv,noheader,nounits']
 result=subprocess.run(argv,capture_output=True,text=True,timeout=15,check=False)
 write(path,{'schema_version':'instinctrl-r5-pair-postrun-audit/1','group':group,'mode':mode,'job_id':entry['job_id'],
  'boundary':'After terminal receipt and normalACK two stable empty scans; no surviving owned identities',
  'ordinal':ordinal,'argv':argv,'executable_sha256':pins['nvidia_smi_sha256'],'exit_code':result.returncode,
  'stdout':result.stdout,'stderr':result.stderr})
 assert result.returncode==0
print(json.dumps({'group':group,'mode':mode,'postrun_queries_used':entry['postrun_nvidia_smi_calls_used']}))
