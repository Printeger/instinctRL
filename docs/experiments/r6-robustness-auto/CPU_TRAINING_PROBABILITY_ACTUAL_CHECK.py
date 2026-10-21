import json,time
from pathlib import Path
import torch
from tools.r6_recovery import checked
from tools.r6_training_probability import recompute
from instinctlab.training.networks import HistoryActor,NetworkConfig,LIDAR_SHAPE
start=time.monotonic()
pin=json.loads(Path('docs/experiments/r6-robustness-auto/BASELINE_BINDINGS.json').read_text())['selected_model']
def forbidden(*args,**kwargs):raise AssertionError('No CUDA query in offline replay')
for name in ('is_available','device_count','get_rng_state_all','manual_seed_all'):setattr(torch.cuda,name,forbidden)
origin=torch.load(checked(pin),map_location='cpu',weights_only=True)
actor=HistoryActor(NetworkConfig(**origin['network_config']));actor.load_state_dict(origin['actor'],strict=True);actor.eval()
policy=dict(lidar_grid=torch.zeros(1,*LIDAR_SHAPE).expand(512,-1,-1,-1),state_vec=torch.zeros(512,52));actions=torch.full((512,4),.5)
with torch.no_grad():
 out=actor({k:v[:32] for k,v in policy.items()});logp=actor.log_probability_from_output(out,actions[:32])
data=dict(schema_version='instinctrl-pilot-retained-input/1',update=641,policy=policy,actions=actions,old_alpha=out.alpha.repeat(16,1),old_beta=out.beta.repeat(16,1),old_log_probabilities=logp.repeat(16,1))
plan=dict(start_update=640,initialization={k:origin[k] for k in ('source_identity','method_hashes')})
rng=torch.get_rng_state().clone();result=recompute(origin,data,plan);assert torch.equal(rng,torch.get_rng_state())
print(json.dumps(dict(checkpoint=pin,replay=result,elapsed_seconds=time.monotonic()-start,scope='Actual frozen u640 weights; synthetic zero observations and constant actions, not a collected rollout or scientific result',scientific_acceptance=False)))
