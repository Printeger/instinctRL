"""Strict selected-policy restore for the candidate evaluation entry.

The formal authorization separately validates stage budget and source compatibility.
"""

def restore(backend,selection):
    import random
    import numpy as np
    import torch
    from instinctlab.training.networks import HistoryActor,NetworkConfig
    from instinctlab.training.networks_h1 import PrivilegedCriticH1
    from instinctlab.training.ppo import PPOConfig,PPOTrainer
    from instinctlab.training.checkpoint_h1 import load_checkpoint_h1
    from .research_contract import validate_pin
    from .robustness_evaluation_selection import validate,verify_loaded,require
    from .robustness_training_method import apply_method
    selected=validate(selection);model=selected['model']
    require(backend.authorization['mode']=='research-r6-candidate-evaluation'
        and backend.authorization['parameters'].get('evaluation_selection')==selection,'EXACT_AUTHORIZATION')
    require(not getattr(backend,'_r6_load_failed',False) and backend.trainer is None and backend.driver is None
        and backend.loop is None and backend.journal is None,'FRESH_BACKEND')
    path=validate_pin(model['checkpoint']);payload=torch.load(path,map_location='cpu',weights_only=True)
    network=NetworkConfig(**payload['network_config']);require(network==NetworkConfig(),'NETWORK_BOUNDARY')
    if model['ppo_config'] is not None:require(payload['ppo_config']==model['ppo_config'],'CHECKPOINT_PPO')
    device=torch.device(backend.environment.device);require(device.type in ('cpu','cuda'),'DEVICE')
    before=dict(cpu=torch.get_rng_state().clone(),python=random.getstate(),numpy=np.random.get_state(),
        curriculum=backend.curriculum_generator.get_state().clone())
    if device.type=='cuda':before['cuda']=torch.cuda.get_rng_state_all()
    try:
        trainer=PPOTrainer(HistoryActor(network).to(device),PrivilegedCriticH1(network).to(device),PPOConfig(**payload['ppo_config']))
        metadata=load_checkpoint_h1(path,trainer,expected_method_hashes=model['method_hashes'],
            expected_source_identity=model['source_identity'],curriculum_generator=backend.curriculum_generator)
        validate_pin(model['checkpoint']);verify_loaded(selected,metadata)
        method_binding=apply_method(trainer,backend.environment,selected['method'],
            resume=selection['role']=='trained_candidate',training_seed=model['training_seed'])
        trainer.actor.eval();trainer.critic.eval()
    except BaseException:
        torch.set_rng_state(before['cpu']);random.setstate(before['python']);np.random.set_state(before['numpy'])
        backend.curriculum_generator.set_state(before['curriculum'])
        if 'cuda' in before:torch.cuda.set_rng_state_all(before['cuda'])
        backend._r6_load_failed=True
        raise
    backend.trainer=trainer
    return dict(selection=selected,metadata=metadata,method_binding=method_binding,
        actor_input_extension=False,physical_history_continuity_claimed=False,optimizer_updates=0)
