"""Exact R6 method schema and strict checkpoint-to-runtime binding.

No permission issuance or simulator initialization. CPU cannot restore a CUDA
checkpoint's RNG; this module retains the existing strict serializer rule.
"""
import copy
import math
from .checkpoint_compatibility import pinned_json
from .robustness_contract import object_sha

PPO_FIELDS={'gamma','gae_lambda','clip_ratio','value_loss_coefficient','entropy_coefficient',
    'actor_learning_rate','critic_learning_rate','maximum_gradient_norm','advantage_epsilon','minibatch_size','epoch_count'}
WEIGHTS={'residual_neutrality_weight','anchor_weight','mean_prior_weight','ics_clearance_margin_m'}


def require(value,reason):
    if not value:raise PermissionError('R6_METHOD_'+reason)


def validate(value):
    require(isinstance(value,dict) and set(value)=={'schema_version','ppo'}|WEIGHTS
        and value['schema_version']=='r6-training-method/1','FIELDS')
    ppo=value['ppo'];require(isinstance(ppo,dict) and set(ppo)==PPO_FIELDS,'PPO_FIELDS')
    for k in WEIGHTS:
        require(type(value[k])in (float,int) and math.isfinite(value[k]) and value[k]>=0,'WEIGHT')
    for k,v in ppo.items():
        require(type(v)in (float,int) and math.isfinite(v),'PPO_FINITE')
        if k in ('minibatch_size','epoch_count'):
            require(type(v)is int and 1<=v<=(512 if k=='minibatch_size' else 8),'PPO_INTEGER')
        elif k in ('gamma','gae_lambda'):require(0<=v<=1,'PPO_DISCOUNT')
        elif k=='clip_ratio':require(1e-8<=v<=1,'PPO_CLIP')
        elif k in ('value_loss_coefficient','entropy_coefficient'):require(v>=0,'PPO_COEFFICIENT')
        else:require(v>=1e-12,'PPO_POSITIVE')
    require(ppo['epoch_count']*((512+ppo['minibatch_size']-1)//ppo['minibatch_size'])==8,'PPO_CLOCK')
    return copy.deepcopy(value)


def apply_method(trainer,env,method,*,resume,training_seed):
    from dataclasses import asdict
    from instinctlab.training.ppo import PPOConfig
    from .mean_prior import configure_trainer
    from .backend import _state_fingerprint
    method=validate(method)
    expected=PPOConfig(**method['ppo'],permutation_seed=training_seed if resume else trainer.config.permutation_seed)
    before=asdict(trainer.config)
    if resume:require(before==asdict(expected),'RESUME_PPO_DRIFT')
    weights_before=_state_fingerprint([trainer.actor.state_dict(),trainer.critic.state_dict(),
        trainer.actor_optimizer.state,trainer.critic_optimizer.state,trainer.update_counter])
    trainer.config=expected
    for optimizer,rate in ((trainer.actor_optimizer,expected.actor_learning_rate),(trainer.critic_optimizer,expected.critic_learning_rate)):
        if resume:require(all(group['lr']==rate for group in optimizer.param_groups),'RESUME_ADAM_LR')
        for group in optimizer.param_groups:group['lr']=rate
    env.set_h1_residual_neutrality_weight(method['residual_neutrality_weight'])
    env.set_h1_research_profile(pure_hover=False,anchor_weight=method['anchor_weight'])
    env.set_h1_ics_clearance_margin(method['ics_clearance_margin_m'])
    configure_trainer(trainer,method)
    actual=dict(residual_neutrality_weight=env._h1_residual_neutrality_weight,anchor_weight=env._h1_anchor_weight,
        ics_clearance_margin_m=env._h1_ics_clearance_margin_m,pure_hover=env._h1_pure_hover_training,
        pipeline_margins=[p.config.ics.clearance_margin_m for p in env._control_pipelines],
        actor_optimizer_lrs=[g['lr'] for g in trainer.actor_optimizer.param_groups],
        critic_optimizer_lrs=[g['lr'] for g in trainer.critic_optimizer.param_groups])
    require(actual['pipeline_margins'] and all(v==method['ics_clearance_margin_m'] for v in actual['pipeline_margins']),'ICS_READBACK')
    require(not actual['pure_hover'] and all(actual[k]==method[k] for k in WEIGHTS-{'mean_prior_weight'}),'ENV_READBACK')
    require(callable(trainer.actor_auxiliary),'AUXILIARY')
    weights_after=_state_fingerprint([trainer.actor.state_dict(),trainer.critic.state_dict(),
        trainer.actor_optimizer.state,trainer.critic_optimizer.state,trainer.update_counter])
    require(weights_before==weights_after,'MODEL_ADAM_MUTATION')
    trainer._r6_bound_method_sha256=object_sha(method)
    trainer._r6_bound_auxiliary=trainer.actor_auxiliary
    return dict(schema_version='r6-actual-method-binding/1',method_sha256=object_sha(method),
        before_ppo=before,actual_ppo=asdict(trainer.config),actual_environment=actual,
        mean_prior_weight=method['mean_prior_weight'],actor_adam_sha256=weights_after,
        unchanged_weights_adam_moments_and_counters=True,strict_resume=resume)


def verify_applied(trainer,env,method):
    from dataclasses import asdict
    method=validate(method)
    require(getattr(trainer,'_r6_bound_method_sha256',None)==object_sha(method),'BINDING_REQUIRED')
    require(all(asdict(trainer.config)[k]==v for k,v in method['ppo'].items()),'PPO_DRIFT')
    require(getattr(trainer,'_r6_bound_auxiliary',None) is trainer.actor_auxiliary and callable(trainer.actor_auxiliary),'AUXILIARY_DRIFT')
    require(not env._h1_pure_hover_training and env._h1_anchor_weight==method['anchor_weight']
        and env._h1_residual_neutrality_weight==method['residual_neutrality_weight']
        and env._h1_ics_clearance_margin_m==method['ics_clearance_margin_m'],'ENV_DRIFT')
    require(env._control_pipelines and all(p.config.ics.clearance_margin_m==method['ics_clearance_margin_m']
        for p in env._control_pipelines),'PIPELINE_DRIFT')
    for optimizer,key in ((trainer.actor_optimizer,'actor_learning_rate'),(trainer.critic_optimizer,'critic_learning_rate')):
        require(all(group['lr']==method['ppo'][key] for group in optimizer.param_groups),'ADAM_LR_DRIFT')


def restore_and_bind(backend,*,plan,execution_identity):
    import random
    import numpy as np
    import torch
    from instinctlab.training.networks import HistoryActor,NetworkConfig
    from instinctlab.training.networks_h1 import PrivilegedCriticH1
    from instinctlab.training.ppo import PPOConfig,PPOTrainer
    from instinctlab.training.checkpoint_h1 import load_checkpoint_h1
    from . import robustness_training_plan as contract
    from .research_contract import validate_pin
    p=contract.validate(plan);recipe=pinned_json(p['recipe']);method=validate(pinned_json(recipe['method']))
    from .direct_contract import training_inputs
    training_inputs(backend,p)
    require(execution_identity==recipe['source_identity'],'EXECUTION_IDENTITY')
    require(not getattr(backend,'_r6_load_failed',False) and backend.trainer is None and backend.driver is None and backend.loop is None and backend.journal is None,'FRESH_BACKEND')
    initial=p['initialization'];path=validate_pin(initial['checkpoint'])
    # Read metadata on CPU to construct the exact saved trainer configuration;
    # the existing loader independently validates all tensors and provenance.
    payload=torch.load(path,map_location='cpu',weights_only=True)
    network=NetworkConfig(**payload['network_config'])
    require(network==NetworkConfig(),'NETWORK_BOUNDARY')
    device=torch.device(backend.environment.device)
    require(device.type in ('cpu','cuda'),'DEVICE')
    before=dict(cpu=torch.get_rng_state().clone(),python=random.getstate(),numpy=np.random.get_state(),
                curriculum=backend.curriculum_generator.get_state().clone())
    if device.type=='cuda':before['cuda']=torch.cuda.get_rng_state_all()
    try:
        trainer=PPOTrainer(HistoryActor(network).to(device),PrivilegedCriticH1(network).to(device),PPOConfig(**payload['ppo_config']))
        metadata=load_checkpoint_h1(path,trainer,expected_method_hashes=initial['method_hashes'],
            expected_source_identity=initial['source_identity'],curriculum_generator=backend.curriculum_generator)
        validate_pin(initial['checkpoint'])
        contract.verify_loaded(p,metadata)
        binding=apply_method(trainer,backend.environment,method,
            resume=initial['kind']=='RESUME_R6_STREAM',training_seed=p['training_seed'])
    except BaseException:
        torch.set_rng_state(before['cpu']);random.setstate(before['python']);np.random.set_state(before['numpy'])
        backend.curriculum_generator.set_state(before['curriculum'])
        if 'cuda' in before:torch.cuda.set_rng_state_all(before['cuda'])
        # The caller must terminate on failure: environment setters may have
        # run. Never make a partly configured trainer available for rollout.
        backend._r6_load_failed=True
        raise
    backend.trainer=trainer;backend._r6_method_binding=binding
    hashes=dict(initial['method_hashes'],r6_recipe=p['recipe']['sha256'],r6_method=recipe['method']['sha256'],
                curriculum='curriculum-h1-r6-v1')
    return dict(metadata=metadata,method_hashes=hashes,source_identity=execution_identity)
