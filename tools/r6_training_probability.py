"""CPU replay of the exact initial actor on retained first-rollout inputs.

This does not restore Adam, reproduce GPU sampling, or establish learning gain.
"""

def recompute(origin,data,plan):
    import torch
    from instinctlab.training.networks import HistoryActor,NetworkConfig,validate_policy
    from instinctlab.training.checkpoint import _validate_model_state
    from instinctlab.training.checkpoint_h1 import CHECKPOINT_H1_SCHEMA,CHECKPOINT_H1_VERSION
    from training_workloads.unified_h1.pilot_contract import probability_distribution_gate
    def require(ok,reason):
        if not ok:raise ValueError(reason)
    initial=plan['initialization']
    require(origin['version']==CHECKPOINT_H1_VERSION and origin['schema']==CHECKPOINT_H1_SCHEMA,'PROBABILITY_ORIGIN_SCHEMA')
    require(origin['source_identity']==initial['source_identity'] and origin['method_hashes']==initial['method_hashes'],'PROBABILITY_ORIGIN_IDENTITY')
    expected=dict(update=plan['start_update'],transition=plan['start_update']*512,
        global_policy_step=plan['start_update']*16,actor_optimizer=plan['start_update']*8,critic_optimizer=plan['start_update']*8)
    require(origin['counters']==expected and all(type(v)is int for v in origin['counters'].values()),'PROBABILITY_ORIGIN_CLOCK')
    require(data['schema_version']=='instinctrl-pilot-retained-input/1' and type(data['update'])is int
        and data['update']==plan['start_update']+1,'PROBABILITY_INPUT_UPDATE')
    policy=data['policy'];validate_policy(policy)
    require(policy['state_vec'].shape[0]==512,'PROBABILITY_SAMPLE_COUNT')
    for name,shape in [('actions',(512,4)),('old_log_probabilities',(512,1)),('old_alpha',(512,4)),('old_beta',(512,4))]:
        value=data[name]
        require(isinstance(value,torch.Tensor) and value.shape==shape and value.dtype==torch.float32
            and value.device.type=='cpu' and bool(torch.isfinite(value).all()),'PROBABILITY_TENSOR_'+name)
    require(bool(((data['actions']>=0)&(data['actions']<=1)).all()),'PROBABILITY_ACTION_RANGE')
    require(bool((data['old_alpha']>0).all() and (data['old_beta']>0).all()),'PROBABILITY_CONCENTRATION')
    config=NetworkConfig(**origin['network_config'])
    require(max(*config.lidar_channels,config.state_hidden,config.actor_hidden,*config.critic_hidden)<=512,'PROBABILITY_NETWORK_BOUND')
    # Constructor initializes parameters; preserve the CPU generator explicitly.
    # Do not use fork_rng(), which can query CUDA even for an offline audit.
    rng=torch.get_rng_state().clone()
    try:
        actor=HistoryActor(config).cpu()
        _validate_model_state('probability origin actor',origin['actor'],actor.state_dict())
        actor.load_state_dict(origin['actor'],strict=True);actor.eval()
        alpha=[];beta=[];logp=[]
        with torch.no_grad():
            for start in range(0,512,32):
                output=actor({k:v[start:start+32] for k,v in policy.items()})
                alpha.append(output.alpha);beta.append(output.beta)
                logp.append(actor.log_probability_from_output(output,data['actions'][start:start+32]))
        values=[torch.cat(v) for v in (logp,alpha,beta)]
        probability_distribution_gate(data['old_log_probabilities'].flatten().tolist(),values[0].flatten().tolist(),
            data['old_alpha'].flatten().tolist(),values[1].flatten().tolist(),
            data['old_beta'].flatten().tolist(),values[2].flatten().tolist(),atol=1e-6,rtol=1e-5)
        errors={name:float((data[key]-actual).abs().max()) for name,key,actual in
            zip(('log_probability','alpha','beta'),('old_log_probabilities','old_alpha','old_beta'),values)}
    finally:torch.set_rng_state(rng)
    return dict(status='PASS',sample_count=512,starting_update=plan['start_update'],
        float32_atol=1e-6,float32_rtol=1e-5,maximum_absolute_errors=errors,
        cpu_actor_forward=True,cpu_rng_preserved=True,gpu_sampling_reproduced=False,optimizer_restored=False)
