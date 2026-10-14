"""Synthetic compact-row fixtures test refusal, never claim simulator evidence."""
from copy import deepcopy

import pytest
import torch

from tools.analyze_r5_formal_research import replay_transition_rows
from training_workloads.unified_h1.transition_diagnostic import transition_commands


@pytest.fixture
def rows():
    commands=transition_commands()
    root=torch.zeros(32,13);root[:,2]=2.;root[:,3]=1.
    zero=torch.zeros(32,3).tolist()
    result=[{'kind':'T0','root_state_w':root.tolist(),'sensor_position_w_m':zero,'env_origins_w_m':zero}]
    for step,command in enumerate(commands):
        following=commands[min(step+1,111)]
        state=torch.zeros(32,52);state[:,45:48]=command
        nxt=state.clone();nxt[:,45:48]=following
        active=float(step<16 or step>=80)
        metrics={k:torch.zeros(32,1).tolist() for k in ['reset_code','usable_fraction','mean_absolute_error_m','huber_loss']}
        metrics['active']=torch.full((32,1),active).tolist()
        result.append({'kind':'STEP','step':step,'command_b':command.tolist(),'following_command_b':following.tolist(),
            'actor_state_vec':state.tolist(),'returned_state_vec':nxt.tolist(),'issued_action':torch.full((32,4),.5).tolist(),
            'reward':[0.]*32,'reset_mask':[False]*32,'root_state_w':root.tolist(),'actual_velocity_b':zero,
            'control':{'v_corr_b':zero,'v_final_b':zero,'v_final_w':zero,'anchor_metrics':metrics}})
    return result


def test_compact_replay_has_exact_counts_and_does_not_create_dense_data(rows):
    checks=replay_transition_rows(rows)
    assert len(checks)==112 and sum(r['samples'] for r in checks)==3584
    assert checks[0]['active_anchor_lanes']==checks[80]['active_anchor_lanes']==32
    assert checks[16]['active_anchor_lanes']==0
    assert all(r['performance_scoring']=='NOT_AN_ORIGINAL_TASK_ACCEPTANCE' for r in checks)


def test_missing_step_is_never_filled(rows):
    rows.pop(18)
    with pytest.raises(AssertionError,match='exact T0'):
        replay_transition_rows(rows)


def test_command_boundary_cannot_be_shifted_one_observation(rows):
    rows[16]['returned_state_vec'][0][45]=0.
    with pytest.raises(AssertionError,match='following command'):
        replay_transition_rows(rows)


def test_vertical_correction_cannot_be_interpreted_as_another_head(rows):
    rows[1]['issued_action'][0][3]=.6
    with pytest.raises(AssertionError,match='actual xyz decoder'):
        replay_transition_rows(rows)


def test_reset_values_are_not_silently_coerced(rows):
    rows[1]['reset_mask'][0]=1
    with pytest.raises(AssertionError,match='boolean reset'):
        replay_transition_rows(rows)


def test_nonzero_reference_must_be_inactive(rows):
    rows[17]['control']['anchor_metrics']['active'][0][0]=1.
    with pytest.raises(AssertionError,match='reference remains active'):
        replay_transition_rows(rows)


def test_reset_counts_are_retained_separately_without_splicing(rows):
    rows[30]['reset_mask'][2]=True;rows[81]['reset_mask'][3]=True
    checks=replay_transition_rows(rows)
    assert sum(c['actual_reset_lanes'] for c in checks)==2
    assert checks[29]['actual_reset_lanes']==checks[80]['actual_reset_lanes']==1
