"""CPU actor forward replay, real network and tensor evidence; no simulator."""
import copy
import pytest
import torch
from tools.r6_training_probability import recompute
from instinctlab.training.networks import HistoryActor,NetworkConfig,LIDAR_SHAPE
from instinctlab.training.checkpoint_h1 import CHECKPOINT_H1_SCHEMA,CHECKPOINT_H1_VERSION


@pytest.fixture
def sample():
    saved=torch.get_rng_state().clone()
    try:
        config=NetworkConfig(lidar_channels=(1,1,1),state_hidden=4,actor_hidden=4,critic_hidden=(4,4))
        actor=HistoryActor(config).eval()
        policy=dict(lidar_grid=torch.zeros(1,*LIDAR_SHAPE).expand(512,-1,-1,-1),state_vec=torch.zeros(512,52))
        actions=torch.full((512,4),.5)
        with torch.no_grad():
            out=actor({k:v[:32] for k,v in policy.items()})
            alpha=out.alpha.repeat(16,1);beta=out.beta.repeat(16,1)
            logp=actor.log_probability_from_output(out,actions[:32]).repeat(16,1)
        initial=dict(source_identity={'fixture':'source'},method_hashes={'fixture':'method'})
        plan=dict(start_update=640,initialization=initial)
        origin=dict(version=CHECKPOINT_H1_VERSION,schema=CHECKPOINT_H1_SCHEMA,**initial,network_config=config.manifest(),actor=copy.deepcopy(actor.state_dict()),
            counters=dict(update=640,transition=640*512,global_policy_step=640*16,actor_optimizer=640*8,critic_optimizer=640*8))
        data=dict(schema_version='instinctrl-pilot-retained-input/1',update=641,policy=policy,actions=actions,
            old_alpha=alpha,old_beta=beta,old_log_probabilities=logp)
        yield origin,data,plan
    finally:torch.set_rng_state(saved)


def test_recompute_all_samples_preserves_cpu_rng_and_never_queries_cuda(sample,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('CUDA forbidden')
    for name in ('is_available','device_count','get_rng_state_all','manual_seed_all'):monkeypatch.setattr(torch.cuda,name,forbidden)
    before=torch.get_rng_state().clone();result=recompute(*sample)
    assert result['status']=='PASS' and result['sample_count']==512
    assert result['maximum_absolute_errors']==dict(log_probability=0.,alpha=0.,beta=0.)
    assert torch.equal(before,torch.get_rng_state()) and not result['optimizer_restored']


@pytest.mark.parametrize('fault',['alpha','logp','actions','update','source','clock','weights','privileged','nonfinite','dtype'])
def test_mismatches_fail_without_rng_mutation(sample,fault):
    origin,data,plan=sample
    if fault=='alpha':data['old_alpha'][511,3]+=.1
    if fault=='logp':data['old_log_probabilities'][511,0]+=.1
    if fault=='actions':data['actions']=data['actions'][:1]
    if fault=='update':data['update']=642
    if fault=='source':origin['source_identity']={}
    if fault=='clock':origin['counters']['actor_optimizer']-=1
    if fault=='dtype':
        name=next(iter(origin['actor']));origin['actor'][name]=origin['actor'][name].double()
    if fault=='weights':next(iter(origin['actor'].values())).fill_(float('nan'))
    if fault=='privileged':data['policy']['velocity_truth']=torch.zeros(512,3)
    if fault=='nonfinite':data['old_beta'][0,0]=float('nan')
    before=torch.get_rng_state().clone()
    with pytest.raises((ValueError,TypeError,RuntimeError,FloatingPointError)):recompute(origin,data,plan)
    assert torch.equal(before,torch.get_rng_state())
