"""Offline actor/critic/Adam load audit; CUDA RNG is preserved as opaque evidence.

No optimizer step, live restoration, physics reset or scientific acceptance.
"""

def audit(value):
    import torch
    from instinctlab.training.networks import HistoryActor,NetworkConfig
    from instinctlab.training.networks_h1 import PrivilegedCriticH1
    from instinctlab.training.ppo import PPOConfig,PPOTrainer
    from instinctlab.training.checkpoint import _validate_model_state,_finite_tree,CheckpointError
    from instinctlab.training.checkpoint_h1 import (CHECKPOINT_H1_KEYS,CHECKPOINT_H1_SCHEMA,CHECKPOINT_H1_VERSION,
        _validated_saved_optimizer,_validate_rng_payload,_validate_counter,_validate_clock_and_curriculum,_validate_r6_binding)
    def require(ok,reason):
        if not ok:raise CheckpointError(reason)
    require(isinstance(value,dict) and set(value)==CHECKPOINT_H1_KEYS,'AUDIT_CHECKPOINT_KEYS')
    require(value['schema']==CHECKPOINT_H1_SCHEMA and value['version']==CHECKPOINT_H1_VERSION,'AUDIT_CHECKPOINT_SCHEMA')
    counters=value['counters'];names={'update','transition','global_policy_step','actor_optimizer','critic_optimizer'}
    require(isinstance(counters,dict) and set(counters)==names,'AUDIT_COUNTER_SCHEMA')
    for name in names:_validate_counter(counters[name],name)
    update=counters['update'];require(update>0,'AUDIT_UPDATED_CHECKPOINT_REQUIRED')
    require(counters['actor_optimizer']==counters['critic_optimizer']==update*8,'AUDIT_ADAM_CLOCK')
    seed=_validate_counter(value['training_seed'],'seed');_validate_counter(value['reset_number'],'reset')
    _validate_clock_and_curriculum(update=update,transition=counters['transition'],policy_step=counters['global_policy_step'],curriculum=value['curriculum'],training_seed=seed)
    _validate_r6_binding(value);_finite_tree(value,'offline checkpoint')
    config=NetworkConfig(**value['network_config']);ppo=PPOConfig(**value['ppo_config'])
    require(max(*config.lidar_channels,config.state_hidden,config.actor_hidden,*config.critic_hidden)<=512,'AUDIT_NETWORK_BOUND')
    rng=torch.get_rng_state().clone()
    try:
        trainer=PPOTrainer(HistoryActor(config),PrivilegedCriticH1(config),ppo)
        counts={}
        for label in ('actor','critic'):
            network=getattr(trainer,label);optimizer=getattr(trainer,label+'_optimizer')
            _validate_model_state(label,value[label],network.state_dict())
            saved=_validated_saved_optimizer(label,value[label+'_optimizer'],optimizer,update*8)
            for item in saved['state'].values():
                require(bool((item['exp_avg_sq']>=0).all()),'AUDIT_NEGATIVE_ADAM_SECOND_MOMENT')
            network.load_state_dict(value[label],strict=True);optimizer.load_state_dict(saved)
            _validated_saved_optimizer(label,optimizer.state_dict(),optimizer,update*8)
            loaded=optimizer.state_dict()
            for key,item in saved['state'].items():
                for field,tensor in item.items():require(torch.equal(loaded['state'][key][field],tensor),'AUDIT_ADAM_LOAD_DRIFT')
            counts[label]=len(saved['state'])
        # Existing validator parses CPU/Python/NumPy/curriculum using private generators.
        # A cuda target here validates the retained CUDA list's schema only; no driver call.
        cuda=value['rng']['torch_cuda']
        _validate_rng_payload(value['rng'],target_device=torch.device('cuda' if cuda else 'cpu'))
    finally:torch.set_rng_state(rng)
    return dict(actor_critic_state_loaded=True,adam_state_loaded=True,adam_parameter_counts=counts,
        adam_step=update*8,cpu_rng_payload_parseable=True,cuda_rng_state_count=len(cuda),
        cuda_rng_semantics_verified=False,rng_restored=False,optimizer_step_executed=False,
        strict_checkpoint_restore_performed=False)
