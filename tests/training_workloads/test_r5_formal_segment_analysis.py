import pytest
import torch
from tools.analyze_r5_formal_segment import validate_update_logs, verify_probe
from instinctlab.training.ppo import deterministic_minibatches


def test_absolute_logs_keep_restored_optimizer_origin():
    updates=[{'completed_updates':u,'actor_optimizer_steps':u*8,'critic_optimizer_steps':u*8} for u in range(257,513)]
    mini=[{'update':u,'minibatch_ordinal':i,'actor_auxiliary_loss':0.} for u in range(257,513)for i in range(8)]
    validate_update_logs(updates,mini,0)
    updates[0]['actor_optimizer_steps']=8
    with pytest.raises(AssertionError):validate_update_logs(updates,mini,0)
    updates[0]['actor_optimizer_steps']=257*8;mini[-1]['update']=511
    with pytest.raises(AssertionError):validate_update_logs(updates,mini,0)


def test_actual257_probe_uses_restored_permutation_seed_and_original_relative_tolerance():
    data={'update':257,'actions':torch.full((512,4),.5),'advantages':torch.zeros(512,1),
          'returns':torch.ones(512,1),'old_log_probabilities':torch.zeros(512,1),'policy':{'state_vec':torch.zeros(512,52)}}
    data['policy']['state_vec'][256:,45]=.2
    mean=torch.full((512,4),.5);mean[:,1]+=torch.arange(512)*.0001
    row={'indices':torch.arange(512),'action':data['actions'],'advantage':data['advantages'],
         'return':data['returns'],'log_probability':data['old_log_probabilities'],'action_mean':mean}
    probe={'sample_count':512,'correction_heads':[1,2,3],'minibatches':[row]*8}
    mini=[]
    for ids in deterministic_minibatches(512,64,1,123+256):
        zero=data['policy']['state_vec'][ids,45:48].norm(dim=-1)<=.001
        loss=10*(zero*((mean[ids,1:4]-.5)/.1).square().sum(-1)).mean()
        mini.append({'actor_auxiliary_loss':float(loss)})
    verify_probe(data,probe,mini,{'mean_prior_weight':10},{'ppo_config':{'permutation_seed':123}})
    with pytest.raises(AssertionError):verify_probe(data,probe,mini,{'mean_prior_weight':10},{'ppo_config':{'permutation_seed':123-256}})
    data['update']=1
    with pytest.raises(AssertionError):verify_probe(data,probe,mini,{'mean_prior_weight':10},{'ppo_config':{'permutation_seed':123}})


def test_original_wilson_tuple_matches_json_producer_without_altering_score():
    import json
    from training_workloads.unified_h1.evaluation import case_outcome
    result=case_outcome([True]*32,['d'+str(i%8)for i in range(32)],expected_subgroups=['d'+str(i)for i in range(8)])
    producer=json.loads(json.dumps(result))
    assert isinstance(result['wilson_95'],tuple) and result!=producer
    assert json.loads(json.dumps(result,allow_nan=False))==producer
    producer['successes']=31
    assert json.loads(json.dumps(result,allow_nan=False))!=producer
