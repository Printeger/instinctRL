import pytest
from tools.r6_job_resources import resources, DISK
from tools.r6_recovery import RecoveryError


def inputs():
    return (dict(backend='real', h1=dict(mode='research-r6-training'),
                 resources=dict(wall_seconds=70, transitions=512)),
            dict(mode='research-r6-training', parameters=dict(output_cap_bytes=4096),
                 budgets=dict(training_transitions=512, nontraining_transitions=0, checkpoint_writes=2)))


def test_prepaid_changes_only_disk_and_keeps_training_charges():
    spec, auth=inputs()
    result=resources(spec,auth)
    assert result==dict(live_starts=1,simulator_initializations=1,live_wall_seconds=250,
        allocated_gpu_seconds=250,training_transitions=512,training_checkpoint_writes=2,
        output_actual_plus_unreleased_reservations_bytes=4096)
    assert resources(spec,auth,prepaid=True)==dict(result,**{DISK:0})


@pytest.mark.parametrize('field,value', [('training_transitions',0),('training_transitions',True),
    ('training_transitions',513),('nontraining_transitions',1),('nontraining_transitions',False),
    ('checkpoint_writes',0),('checkpoint_writes',True),('checkpoint_writes',None)])
def test_training_budget_cannot_fall_back_to_evaluation(field,value):
    spec,auth=inputs(); auth['budgets'][field]=value
    with pytest.raises(RecoveryError,match='TRAINING_BUDGET'):resources(spec,auth)


def test_training_requires_matching_authority_mode():
    spec,auth=inputs(); auth['mode']='research-r6-evaluation'
    with pytest.raises(RecoveryError,match='TRAINING_BUDGET'):resources(spec,auth)


def candidate_inputs():
    spec,auth=inputs()
    spec['h1']['mode']=auth['mode']='research-r6-candidate-evaluation'
    auth['budgets']=dict(nontraining_transitions=512,training_transitions=0,
                         optimizer_updates=0,checkpoint_writes=0)
    return spec,auth


def test_candidate_charges_evaluation_and_prepaid_disk_only():
    spec,auth=candidate_inputs()
    result=resources(spec,auth)
    assert result['nontraining_transitions']==512
    assert 'training_transitions' not in result and 'training_checkpoint_writes' not in result
    assert resources(spec,auth,prepaid=True)==dict(result,**{DISK:0})


@pytest.mark.parametrize('field,value',[(k,v) for k in
    ('training_transitions','optimizer_updates','checkpoint_writes') for v in (1,False,None)]+
    [('nontraining_transitions',0),('nontraining_transitions',True),('nontraining_transitions',513)])
def test_candidate_rejects_training_and_budget_drift(field,value):
    spec,auth=candidate_inputs();auth['budgets'][field]=value
    with pytest.raises(RecoveryError,match='CANDIDATE_BUDGET'):resources(spec,auth)


def test_candidate_requires_matching_authority_mode():
    spec,auth=candidate_inputs();auth['mode']='research-r6-evaluation'
    with pytest.raises(RecoveryError,match='CANDIDATE_BUDGET'):resources(spec,auth)
