"""Audit saved RNG initialization against the exact origin and reviewed plan.

CPU seed derivation and resume byte identity do not prove GPU draws or rollout use.
"""

def audit(record,origin,plan):
    import random
    import numpy as np
    import torch
    from instinctlab.training.checkpoint_h1 import _validate_rng_payload,_json_tree
    from training_workloads.unified_h1 import robustness_training_plan as contract
    def require(ok,reason):
        if not ok:raise ValueError(reason)
    def equal(left,right):
        if isinstance(left,torch.Tensor):
            return isinstance(right,torch.Tensor) and left.shape==right.shape and left.dtype==right.dtype and torch.equal(left,right)
        if type(left)is not type(right):return False
        if isinstance(left,dict):return left.keys()==right.keys() and all(equal(v,right[k]) for k,v in left.items())
        if isinstance(left,(list,tuple)):return len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
        return left==right
    def valid(rng):
        _validate_rng_payload(rng,target_device=torch.device('cuda' if rng['torch_cuda'] else 'cpu'))
    require(origin['source_identity']==plan['initialization']['source_identity'] and origin['method_hashes']==plan['initialization']['method_hashes'],'RNG_ORIGIN_IDENTITY')
    binding=contract.verify_loaded(plan,origin)
    require(record['plan_sha256']==contract.object_sha(plan) and record['recipe']==plan['recipe']
        and record['binding']==binding and record['evidence_must_be_persisted_before_rollout'] is True,'RNG_RECORD_PLAN')
    saved=record['rng'];valid(origin['rng']);seed=plan['training_seed']
    require(saved['physical_continuity'] is False and type(saved['starting_update'])is int
        and saved['starting_update']==plan['start_update'],'RNG_RECORD_BOUNDARY')
    if binding['branch_new_rng']:
        require(saved['schema_version']=='r6-training-rng-branch/1' and saved['strict_rng_continuation'] is False
            and saved['origin_sha256']==plan['initialization']['checkpoint']['sha256']
            and type(saved['origin_training_seed'])is int and saved['origin_training_seed']==20260908
            and type(saved['branch_training_seed'])is int and saved['branch_training_seed']==seed,'RNG_BRANCH_IDENTITY')
        require(equal(saved['origin_rng'],origin['rng']),'RNG_ORIGIN_STATE_DRIFT')
        recipe=contract.pinned_json(plan['recipe']);method=contract.pinned_json(recipe['method'])
        require(saved['origin_ppo_config']==dict(method['ppo'],permutation_seed=origin['ppo_config']['permutation_seed'])
            and saved['branch_ppo_config']==dict(method['ppo'],permutation_seed=seed),'RNG_PPO_BINDING')
        current=saved['branch_rng'];valid(current)
        require(len(current['torch_cuda'])==len(origin['rng']['torch_cuda']),'RNG_CUDA_COUNT_DRIFT')
        generator=torch.Generator(device='cpu').manual_seed(seed)
        require(equal(current['torch_cpu'],generator.get_state()) and equal(current['curriculum'],generator.get_state()),'RNG_TORCH_SEED_DRIFT')
        require(equal(current['python'],_json_tree(random.Random(seed).getstate())),'RNG_PYTHON_SEED_DRIFT')
        state=np.random.RandomState(seed).get_state()
        expected=dict(algorithm=state[0],keys=state[1].tolist(),position=state[2],has_gauss=state[3],cached_gaussian=state[4])
        require(equal(current['numpy'],expected),'RNG_NUMPY_SEED_DRIFT')
    else:
        require(saved['schema_version']=='r6-training-rng-resume/1' and saved['strict_rng_continuation'] is True
            and type(saved['training_seed'])is int and saved['training_seed']==seed,'RNG_RESUME_IDENTITY')
        current=saved['restored_rng'];valid(current)
        require(equal(current,origin['rng']),'RNG_RESUME_STATE_DRIFT')
    return dict(status='PASS',kind=plan['initialization']['kind'],training_seed=seed,starting_update=plan['start_update'],
        cpu_seed_derivation_verified=binding['branch_new_rng'],origin_rng_equal=binding['branch_new_rng'],
        resume_rng_equal=not binding['branch_new_rng'],cuda_rng_state_count=len(current['torch_cuda']),
        cuda_seed_derivation_verified=False,actual_rollout_rng_use_verified=False,persistence_order_verified=False,
        physical_continuity=False)
