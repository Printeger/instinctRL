import json, math
from pathlib import Path
import torch
from tools.r6_evidence import checked,file_pin
from tools.r6_direct.import_legacy import stop_paths
from training_workloads.unified_h1.lossless_evidence import ChunkReader
from training_workloads.unified_h1 import pilot_contract
from tools.r6_training_probability import recompute
assert not stop_paths()
base=Path('/home/mint/rl_dev/instinctRL-dev')
c=base/'docs/experiments/r6-direct-training/TRAIN_CONTRACT_01.json'
plan=json.loads(c.read_text())['parameters']['r6_training_plan']
origin=torch.load(checked(plan['initialization']['checkpoint']),map_location='cpu',weights_only=True)
name='pilot_diagnostic/update_641_full_input.pt'
p=base/'coordination/research/attempts/R6-BASELINE-ROBUSTNESS-01/results/R6-DIRECT-TRAIN-01/results/research-r6-training/compressed_evidence'/ (name+'.r6z')
with ChunkReader(p,run_id=plan['experiment_id'],logical_name=name,maximum_raw_bytes=plan['storage']['maximum_raw_file_bytes']) as reader:
 data=torch.load(reader,map_location='cpu',weights_only=True)
gate=pilot_contract.probability_distribution_gate
measurements=[]
def capture(*args,**kw):
 rows={}
 for label,old,new in zip(['log_probability','alpha','beta'],args[::2],args[1::2]):
  failures=[{'index':i,'old':a,'recomputed':b,'absolute_error':abs(a-b)} for i,(a,b) in enumerate(zip(old,new)) if abs(a-b)>kw['atol']+kw['rtol']*abs(b)]
  rows[label]={'maximum_absolute_error':max(abs(a-b) for a,b in zip(old,new)),'failed_count':len(failures),'first_failures':failures[:8]}
 measurements.append(rows)
 return gate(*args,**kw)
pilot_contract.probability_distribution_gate=capture
try:
 result=recompute(origin,data,plan)
except (ValueError,RuntimeError) as e:result={'status':'FAIL','error':str(e)}
epsilon=origin['network_config']['action_log_epsilon']
comparisons={}
for dtype in (torch.float32,torch.float64):
 action=data['actions'].to(dtype).clamp(epsilon,1-epsilon)
 lp=torch.distributions.Beta(data['old_alpha'].to(dtype),data['old_beta'].to(dtype)).log_prob(action).sum(-1,keepdim=True)
 delta=(lp-data['old_log_probabilities']).abs()
 comparisons[str(dtype)]={'maximum_absolute_error':float(delta.max()),'failed_count':int((delta>1e-6+1e-5*lp.abs()).sum())}
out={'stored_concentrations_log_probability':comparisons,'result':result,'measurements':measurements,'contract':file_pin(c),'retained':file_pin(p),'origin':plan['initialization']['checkpoint'],'device':'cpu','tolerances_changed':False,'torch_version':torch.__version__,'threads':torch.get_num_threads()}
(base/'docs/experiments/r6-direct-training/continuation-20261101/PROBABILITY_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
