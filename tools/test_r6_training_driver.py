"""CPU tensor driver with changing attitude/reset; real counters, fake physics."""
import math
from types import SimpleNamespace
import pytest
import torch
from instinctlab.control.lee_velocity_controller import rotate_world_to_body_wxyz
from training_workloads.unified_h1.robustness_training_driver import R6WorldCommandClock,R6TrainingDriver

START=640*16

def quat(yaw):
    q=torch.zeros(32,4);q[:,0]=math.cos(yaw/2);q[:,3]=math.sin(yaw/2);return q

def command(step):
    c=torch.zeros(32,3)
    c[:,0]=.2 if step==START else 0.
    c[:,1]=0. if step==START else .2
    return c

class Env:
    def __init__(self,*,reset=False,stale=False):
        self.q=quat(0);self.reset=reset;self.stale=stale;self.calls=[];self.scene=SimpleNamespace(env_origins=torch.zeros(32,3))
        self.obs=self.observation(command(START))
    def observation(self,body):
        state=torch.zeros(32,52);state[:,45:48]=body
        return {'policy':{'state_vec':state}}
    def stage_r6_world_command(self,value):self.world=value.clone();self.calls.append('world')
    def prepare_h1_step(self,**kw):self.prepared=kw;self.calls.append('prepare')
    def step(self,action):
        self.calls.append('step');self.q=quat(math.pi/2 if not self.reset else -math.pi/2)
        body=self.prepared['next_command_b'] if self.stale else rotate_world_to_body_wxyz(self.q,self.world)
        self.obs=self.observation(body)
        done=torch.full((32,),self.reset,dtype=torch.bool)
        return self.obs,torch.zeros(32),done,torch.zeros(32,dtype=torch.bool),{'fixture':'post-step'}

def build(env,**kwargs):
    clock=R6WorldCommandClock(completed_policy_steps=START,final_policy_step=START+16,
        recipe_sha256='a'*64,branch_seed=11,command_source=command,attitude_source=lambda:env.q)
    return R6TrainingDriver(env,env.obs,clock=clock,**kwargs)

@pytest.mark.parametrize('reset',[False,True])
def test_actual_next_attitude_controls_history_command_and_audit(reset):
    env=Env(reset=reset);driver=build(env,transition_limit=512,automatic_reset_limit=512)
    driver.step(torch.zeros(32,4))
    assert env.calls==['world','prepare','step']
    body=rotate_world_to_body_wxyz(env.q,command(START+1))
    assert torch.equal(driver.observation['policy']['state_vec'][:,45:48],body)
    row=driver.take_evidence()[0]['_h1_driver']
    assert torch.equal(row['next_command_b'],body)
    assert not torch.equal(row['next_command_b'],row['prepared_next_command_b'])
    assert driver.counters.training_transitions==32 and driver.counters.automatic_resets==(32 if reset else 0)
    assert driver.clock.state_dict()==dict(version='curriculum-h1-r6-v1',completed_policy_steps=START+1,
        recipe_sha256='a'*64,origin_checkpoint_sha256='aaef7d40de50032cf2c61e0bc7a6e6f277a2f65e252284f3af58c0f25bae5429',branch_seed=11)
    assert set(driver.observation['policy'])=={'state_vec'} and driver.observation['policy']['state_vec'].shape==(32,52)
    driver.step(torch.zeros(32,4));assert driver.counters.training_transitions==64


def test_stale_prephysics_attitude_command_rejected_after_accounting():
    env=Env(stale=True);driver=build(env)
    with pytest.raises(ValueError,match='obs_'):driver.step(torch.zeros(32,4))
    assert driver.counters.training_transitions==32


def test_budget_rejects_before_staging_or_physics():
    env=Env();driver=build(env,transition_limit=0)
    with pytest.raises(RuntimeError,match='budget exhausted'):driver.step(torch.zeros(32,4))
    assert env.calls==[] and driver.clock.completed_policy_steps==START


@pytest.mark.parametrize('bad',['vertical','fast','nan','attitude'])
def test_invalid_world_or_attitude_never_touches_environment(bad):
    env=Env();driver=build(env)
    value=command(START)
    if bad=='vertical':value[:,2]=.1
    elif bad=='fast':value[:,0]=.4
    elif bad=='nan':value[0,0]=float('nan')
    else:env.q[:,0]=2
    driver.clock.command_source=lambda step:value
    with pytest.raises(ValueError,match='R6_CLOCK'):driver.step(torch.zeros(32,4))
    assert not env.calls


def test_clock_finite_and_reset_cannot_rewind_it():
    env=Env();driver=build(env)
    for _ in range(16):driver.step(torch.zeros(32,4))
    driver.clock.reset_lane(0)
    with pytest.raises(ValueError,match='EXHAUSTED'):driver.step(torch.zeros(32,4))
    assert driver.counters.training_transitions==512 and env.calls.count('step')==16

from tools.test_r6_training_plan import case

def test_clock_from_reviewed_plan_binds_exact_seed_recipe_and_bounds(case):
    env=Env()
    clock=R6WorldCommandClock.from_plan(case.plan,command_source=command,attitude_source=lambda:env.q)
    assert clock.completed_policy_steps==case.plan['start_update']*16
    assert clock.final_policy_step==case.plan['target_update']*16
    assert clock.state_dict()['recipe_sha256']==case.plan['recipe']['sha256']
    assert clock.branch_seed==case.plan['training_seed']
