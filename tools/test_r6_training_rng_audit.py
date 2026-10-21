"""Real CPU branch generators and resume state; synthetic initial checkpoint identity."""
import copy
from dataclasses import asdict,replace
import random
import numpy as np
import pytest
import torch
from tools.test_r6_training_plan import case
from tools.test_r6_training_rng import restored,preserve_global_rng
from tools.r6_training_rng_audit import audit
from instinctlab.training.checkpoint_h1 import _rng_payload
from training_workloads.unified_h1.robustness_training_rng import prepare_plan_rng
from training_workloads.unified_h1 import robustness_training_plan as contract


def create(b,resume=False):
    t,g,meta=restored()
    meta['curriculum']=dict(version='curriculum-h1-v1',completed_policy_steps=640*16)
    if resume:
        b.plan.update(start_update=896,target_update=1152,checkpoint_updates=[1152])
        b.plan['initialization']=dict(kind='RESUME_R6_STREAM',checkpoint=b.ev,source_identity=b.identity,method_hashes={'r6_recipe':b.plan['recipe']['sha256']})
        t.update_counter=896;t.config=replace(t.config,permutation_seed=11)
        for optimizer in (t.actor_optimizer,t.critic_optimizer):
            for state in optimizer.state.values():state['step'].fill_(896*8)
        meta=dict(training_seed=11,counters=dict(update=896,transition=896*512,global_policy_step=896*16,actor_optimizer=896*8,critic_optimizer=896*8),
            curriculum=dict(version='curriculum-h1-r6-v1',completed_policy_steps=896*16,recipe_sha256=b.plan['recipe']['sha256'],origin_checkpoint_sha256=contract.BASELINE_SHA,branch_seed=11))
    origin=dict(meta,rng=_rng_payload(g,t),ppo_config=asdict(t.config),source_identity=b.plan['initialization']['source_identity'],method_hashes=b.plan['initialization']['method_hashes'])
    record=prepare_plan_rng(t,g,plan=b.plan,metadata=meta)
    return record,origin


@pytest.mark.parametrize('resume',[False,True])
def test_actual_branch_or_resume_record_matches_without_global_rng_mutation(case,resume,monkeypatch):
    record,origin=create(case,resume)
    before=torch.get_rng_state().clone();py=random.getstate();n=np.random.get_state()
    def forbidden(*args,**kwargs):raise AssertionError('CUDA forbidden')
    for name in ('is_available','device_count','get_rng_state_all','set_rng_state_all','manual_seed_all'):monkeypatch.setattr(torch.cuda,name,forbidden)
    result=audit(record,origin,case.plan)
    assert result['status']=='PASS' and result['resume_rng_equal']==resume and result['cpu_seed_derivation_verified']==(not resume)
    assert not result['actual_rollout_rng_use_verified'] and not result['cuda_seed_derivation_verified']
    assert torch.equal(before,torch.get_rng_state()) and random.getstate()==py
    after=np.random.get_state();assert n[0]==after[0] and np.array_equal(n[1],after[1]) and n[2:]==after[2:]


@pytest.mark.parametrize('fault',['plan','origin','torch','curriculum','python','numpy','ppo','seed','continuity','cuda_count'])
def test_branch_tampering_is_detected(case,fault):
    record,origin=create(case);assert audit(record,origin,case.plan)['status']=='PASS';r=record['rng']
    if fault=='plan':record['plan_sha256']='f'*64
    if fault=='origin':r['origin_rng']['torch_cpu']=torch.Generator().manual_seed(99).get_state()
    if fault in ('torch','curriculum'):r['branch_rng']['torch_cpu' if fault=='torch' else fault]=torch.Generator().manual_seed(99).get_state()
    if fault=='python':r['branch_rng']['python'][1][0]+=1
    if fault=='numpy':r['branch_rng']['numpy']['position']-=1
    if fault=='ppo':r['branch_ppo_config']['permutation_seed']=17
    if fault=='seed':r['branch_training_seed']=17
    if fault=='continuity':r['physical_continuity']=True
    if fault=='cuda_count':r['branch_rng']['torch_cuda']=[torch.zeros(16,dtype=torch.uint8)]
    with pytest.raises((ValueError,RuntimeError)):audit(record,origin,case.plan)


def test_resume_compares_cuda_bytes_without_claiming_seed_derivation(case):
    record,origin=create(case,True)
    origin['rng']['torch_cuda']=[torch.arange(16,dtype=torch.uint8)]
    record['rng']['restored_rng']['torch_cuda']=copy.deepcopy(origin['rng']['torch_cuda'])
    result=audit(record,origin,case.plan);assert result['resume_rng_equal'] and not result['cuda_seed_derivation_verified']
    record['rng']['restored_rng']['torch_cuda'][0][0]+=1
    with pytest.raises(ValueError,match='RESUME_STATE_DRIFT'):audit(record,origin,case.plan)
