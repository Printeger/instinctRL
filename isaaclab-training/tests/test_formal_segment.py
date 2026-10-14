"""Prospective formal contracts, warm-start wiring and additional accounting.

CPU tests do not execute the formal optimizer body or claim task performance.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from training_workloads.unified_h1.formal_segment import (
    CHECKPOINT_UPDATES, validate_parameters, required_budget, method_hashes,
    train_segment, evaluate_segment,
)
from training_workloads.unified_h1.driver import UnifiedH1TrainingLoop
from training_workloads.unified_h1.budget import ExecutionCounters
from training_workloads.unified_h1.schedule import ScheduleClock
from training_workloads.unified_h1.logging import LearningLog

ROOT = Path(__file__).resolve().parents[2]


def pin(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def refresh(p):
    path = Path(p['experiment_card']['path'])
    path.write_text(json.dumps({'experiment_id': p['experiment_id'], 'profiles': {
        p['group']: {k: v for k, v in p.items() if k not in ['experiment_card', 'checkpoints']}}}))
    p['experiment_card'] = pin(path)


@pytest.fixture
def profile(tmp_path):
    master = tmp_path / 'master.json'
    master.write_text(json.dumps({'id':'R5-FORMAL-READINESS-AUTO-01', 'status':'REGISTERED_USER_DELEGATION'}))
    origin = dict(experiment_id='R5-CPU-ORIGIN', group='R1', training_seed=20260908,
                  evaluation_seed=20260909, max_updates=256, checkpoint_updates=[1,256],
                  evaluation_updates=[256], command_profile='pure-hover', residual_neutrality_weight=.05,
                  anchor_weight=4., actor_lr=1e-4, critic_lr=3e-4, output_cap_bytes=8*1024**3,
                  master_authority=pin(master))
    oc = tmp_path / 'origin-card.json'
    oc.write_text(json.dumps({'experiment_id':origin['experiment_id'], 'profiles':{'R1':deepcopy(origin)}}))
    origin['experiment_card'] = pin(oc)
    checkpoint = tmp_path / 'u256.pt'; checkpoint.write_bytes(b'parameter fixture only, not a checkpoint')
    compatibility = tmp_path / 'compat.json'; compatibility.write_text('{}')
    p = dict(experiment_id='R5-CPU-FORMAL', group='R1', training_seed=20260908,
             evaluation_seed=20260909, start_update=256, target_update=512,
             checkpoint_updates=list(CHECKPOINT_UPDATES), evaluation_updates=list(CHECKPOINT_UPDATES),
             command_profile='mixed-center', residual_neutrality_weight=.05, anchor_weight=4.,
             actor_lr=1e-4, critic_lr=3e-4, output_cap_bytes=8*1024**3,
             master_authority=pin(master), experiment_card={'path':str(tmp_path/'formal-card.json')},
             phase_contract=pin(ROOT/'coordination/contracts/R5_FORMAL_MIXED_CENTER_V1.md'),
             checkpoint_source_identity={k:('a'*64 if k=='config_sha256' else 'b'*40) for k in
                 ['product_commit','product_tree','workload_commit','workload_tree','config_sha256']},
             checkpoint_compatibility=pin(compatibility), initialization_checkpoint={**pin(checkpoint),'update':256},
             origin_profile=origin)
    refresh(p)
    return p


def test_exact_additional_budget_and_method_origin(profile):
    b = required_budget('research-formal-training', profile)
    assert (b['training_transitions'], b['optimizer_updates'], b['checkpoint_writes']) == (131072,256,5)
    assert b['explicit_resets'] == 1 and b['nontraining_transitions'] == 0
    hashes = method_hashes({'method':{'hashes':{'method':'original'}}},profile)
    assert hashes['reset_distribution']=='center-only-v1' and hashes['curriculum']=='curriculum-h1-v1'
    assert hashes['origin_checkpoint_sha256']==profile['initialization_checkpoint']['sha256']
    # Only CURRENT intervention changes; both groups restore the SAME origin.
    q=deepcopy(profile);q.update(group='R0',residual_neutrality_weight=0.)
    assert method_hashes({'method':{'hashes':{'method':'original'}}},q)!=hashes
    assert q['initialization_checkpoint']==profile['initialization_checkpoint']


def mean_prior_profile(p):
    p['origin_profile'].update(mean_prior_contract=pin(ROOT/'coordination/contracts/R5_ZERO_COMMAND_MEAN_PRIOR_V1.md'), mean_prior_weight=10.)
    origin = p['origin_profile']
    oc = Path(origin['experiment_card']['path'])
    oc.write_text(json.dumps({'experiment_id':origin['experiment_id'], 'profiles':{'R1':{k:v for k,v in origin.items() if k!='experiment_card'}}}))
    origin['experiment_card']=pin(oc)
    p.update(mean_prior_contract=origin['mean_prior_contract'],mean_prior_weight=10.,
             phase_contract=pin(ROOT/'coordination/contracts/R5_FORMAL_MIXED_CENTER_MEAN_PRIOR_V1.md'))
    refresh(p)
    return p


def test_mean_prior_formal_phase_is_explicit_and_same_origin_for_both_arms(profile):
    p=mean_prior_profile(profile)
    assert validate_parameters('research-formal-training',p)==p
    q=deepcopy(p);q.update(group='R0',mean_prior_weight=0.);refresh(q)
    assert validate_parameters('research-formal-training',q)==q
    assert p['initialization_checkpoint']==q['initialization_checkpoint']
    assert p['origin_profile']==q['origin_profile']
    assert p['residual_neutrality_weight']==q['residual_neutrality_weight']==.05
    assert required_budget('research-formal-training',q)['optimizer_updates']==256
    a,b=[method_hashes({'method':{'hashes':{}}},x) for x in [p,q]]
    assert a['formal_phase']==p['phase_contract']['sha256'] and a['curriculum']=='curriculum-h1-v1'
    assert a['research_effective_method']!=b['research_effective_method']


def test_mean_prior_cannot_silently_resume_with_legacy_formal_loss(profile):
    p=mean_prior_profile(profile)
    p.pop('mean_prior_weight');p.pop('mean_prior_contract')
    p['phase_contract']=pin(ROOT/'coordination/contracts/R5_FORMAL_MIXED_CENTER_V1.md');refresh(p)
    with pytest.raises(PermissionError,match='explicit current formal'):
        validate_parameters('research-formal-training',p)


def test_mean_prior_warmup_and_intervention_must_match_registered_phase(profile):
    p=mean_prior_profile(profile);p['mean_prior_weight']=1.;refresh(p)
    with pytest.raises(PermissionError,match='phase/origin'):
        validate_parameters('research-formal-training',p)


@pytest.mark.parametrize('key,value', [('start_update',128),('target_update',1024),
    ('command_profile','pure-hover'),('training_seed',20261015),('anchor_weight',8.),
    ('checkpoint_updates',[272,512]),('actor_lr',3e-4)])
def test_registered_phase_cannot_be_changed_by_editing_card(profile,key,value):
    profile[key]=value;refresh(profile)
    with pytest.raises(PermissionError):validate_parameters('research-formal-training',profile)


def test_formal_evaluation_requires_all_actual_checkpoint_hashes(profile,tmp_path):
    with pytest.raises(PermissionError,match='fields'):required_budget('research-formal-evaluation',profile)
    rows=[]
    for u in CHECKPOINT_UPDATES:
        path=tmp_path/f'future-{u}.pt';path.write_bytes(b'parameter fixture')
        rows.append({**pin(path),'update':u})
    profile['checkpoints']=rows
    b=required_budget('research-formal-evaluation',profile)
    assert (b['nontraining_transitions'],b['explicit_resets'],b['optimizer_updates'])==(164800,25,0)
    profile['checkpoints'][0]['sha256']='0'*64
    with pytest.raises(PermissionError,match='drift'):required_budget('research-formal-evaluation',profile)


def test_warm_adam_totals_are_separate_from_new_segment_charge():
    counters=ExecutionCounters();driver=NS(counters=counters)
    loop=UnifiedH1TrainingLoop(driver,NS(),NS(),optimizer_step_origin=(2048,2048))
    counters.attempt_update(256);counters.complete_update(*loop._optimizer_deltas(2056,2056))
    assert counters.real_optimizer_updates==1 and counters.actor_optimizer_steps==8
    assert loop._optimizer_deltas(4096,4096)==(2048,2048)
    with pytest.raises(RuntimeError,match='below restored'):loop._optimizer_deltas(2047,2056)
    with pytest.raises(ValueError,match='origins'):UnifiedH1TrainingLoop(driver,NS(),NS(),optimizer_step_origin=(True,0))


def test_mixed_clock_and_center_reset_labels_are_independent(tmp_path):
    plan=ScheduleClock(4096).plan();assert plan.zero_mask.sum()==16 and plan.nonzero_mask.sum()==16
    root=torch.zeros(32,13);root[:,2]=2.
    transition=NS(reward=torch.zeros(32),reset_boundary=torch.ones(32,dtype=torch.bool))
    evidence={'_h1_driver':{'global_policy_step':4096,'pure_hover':False,'center_reset_only':True},
              'pre_reset':{'root_state_w':root,'task':{'actual_velocity_b':torch.zeros(32,3),'exact_clearance_m':torch.ones(32,1)}}}
    journal=LearningLog(tmp_path)
    journal._record_blocks_and_episodes(transition,evidence,plan.command_b,update=257,step=0)
    assert journal._episode_start_types[6:8]==['center','center']


def test_warm_entry_binds_original_method_restore_fresh_reset_and_gate(profile,tmp_path,monkeypatch):
    # Wiring test only. Loop body is intentionally replaced: no optimizer call,
    # synthetic training checkpoint or performance claim is made by this test.
    import training_workloads.unified_h1.driver as driver_module
    import training_workloads.unified_h1.config as config_module
    from training_workloads.unified_h1.research_contract import method_hashes as origin_hashes
    calls={};cfg={'method':{'hashes':{'method':'original'}}}
    monkeypatch.setattr(config_module,'load_config',lambda:cfg)
    trainer=NS(update_counter=256)
    def load(path,**kwargs):
        calls['load']=(path,kwargs)
        return {'training_seed':20260908,'curriculum':{'version':'curriculum-h1-research-pure-hover-v1'},
                'counters':{'global_policy_step':4096},'reset_number':129}
    env=NS(device='cpu',_h1_pure_hover_training=True,prepare_h1_step=lambda **kw:None,step=lambda a:None,
           set_h1_evaluation_case=lambda case:calls.update(case=case),
           set_h1_initial_command=lambda c,s:calls.update(initial_command=c.clone(),start=s))
    def reset(*,seed):
        calls['reset_seed']=seed
        return {'policy':{'state_vec':torch.zeros(32,52)}},{}
    class Loop:
        def __init__(self,*args,**kwargs):calls['loop']=kwargs
        def run_update(self,**kwargs):trainer.update_counter+=1
    monkeypatch.setattr(driver_module,'UnifiedH1TrainingLoop',Loop)
    backend=NS(research=profile,authorization={'mode':'research-formal-training','budgets':required_budget('research-formal-training',profile)},
        trainer=trainer,environment=env,output=tmp_path,counters=ExecutionCounters(),
        resource_check=lambda *a:None,_apply_research_profile=lambda:None,
        _load_evaluation_checkpoint=load,_preserving_reset=reset,_write_checkpoint=lambda *a:None)
    train_segment(backend)
    kwargs=calls['load'][1]
    assert kwargs['expected_method_hashes']==origin_hashes(cfg,profile['origin_profile'])
    assert kwargs['source_identity']==profile['checkpoint_source_identity'] and kwargs['restore_curriculum_generator']
    assert calls['start']==4096 and calls['initial_command'].ne(0).any()
    assert calls['reset_seed']==20260908+130 and backend._checkpoint_reset_offset==129
    assert backend.driver.clock.pure_hover is False and calls['loop']['optimizer_step_origin']==(2048,2048)
    assert calls['loop']['diagnostic'].starting_update==256
    saved=json.loads((tmp_path/'formal_initialization.json').read_text())
    assert saved['physical_history_anchor_actuator_continuity_claimed'] is False


def test_formal_actions_reject_other_entry_modes():
    backend=NS(authorization={'mode':'research-training'})
    with pytest.raises(PermissionError,match='exact mode'):train_segment(backend)
    with pytest.raises(PermissionError,match='exact mode'):evaluate_segment(backend)


@pytest.mark.parametrize('group,weight',[('R0',0.),('R1',10.)])
def test_restored_mean_prior_hook_uses_current_formal_arm_before_updates(profile,tmp_path,monkeypatch,group,weight):
    from training_workloads.unified_h1 import mean_prior
    p=mean_prior_profile(profile);p.update(group=group,mean_prior_weight=weight);refresh(p)
    calls=[];configure=mean_prior.configure_trainer
    def install(trainer,phase):
        calls.append((trainer.update_counter,phase['mean_prior_weight'],phase['origin_profile']['mean_prior_weight']))
        configure(trainer,phase)
        assert trainer.actor_auxiliary is not None
    monkeypatch.setattr(mean_prior,'configure_trainer',install)
    test_warm_entry_binds_original_method_restore_fresh_reset_and_gate(p,tmp_path,monkeypatch)
    assert calls==[(256,weight,10.)]
