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
