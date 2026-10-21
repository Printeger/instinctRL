"""R6 world-command training adapter on the existing rollout/PPO driver.

Only the converted command enters the existing actor command field. Attitude,
world commands and curriculum identity remain setup/control/audit data. The
caller supplies the frozen deterministic command source and installed R6 plant
and measurement adapters; this module grants no training authority.
"""
from dataclasses import dataclass
import re
import torch
from instinctlab.control.lee_velocity_controller import rotate_world_to_body_wxyz
from .schedule import StepPlan
from .driver import UnifiedH1Driver
from .robustness_contract import BASELINE_SHA


@dataclass(frozen=True)
class WorldStepPlan(StepPlan):
    world_command: torch.Tensor
    next_world_command: torch.Tensor


class R6WorldCommandClock:
    @classmethod
    def from_plan(cls,plan,*,command_source,attitude_source):
        from .robustness_training_plan import validate
        p=validate(plan)
        return cls(completed_policy_steps=p['start_update']*16,final_policy_step=p['target_update']*16,
            recipe_sha256=p['recipe']['sha256'],branch_seed=p['training_seed'],
            command_source=command_source,attitude_source=attitude_source)

    def __init__(self,*,completed_policy_steps,final_policy_step,recipe_sha256,branch_seed,
                 command_source,attitude_source):
        if (type(completed_policy_steps)is not int or completed_policy_steps<640*16
                or type(final_policy_step)is not int or final_policy_step<=completed_policy_steps
                or final_policy_step%16 or completed_policy_steps%16):
            raise ValueError('R6_CLOCK_BOUNDS')
        if not isinstance(recipe_sha256,str) or re.fullmatch('[0-9a-f]{64}',recipe_sha256) is None:
            raise ValueError('R6_CLOCK_RECIPE')
        if type(branch_seed)is not int or not 0<=branch_seed<2**31 or branch_seed==20260908:
            raise ValueError('R6_CLOCK_BRANCH')
        if not callable(command_source) or not callable(attitude_source):raise TypeError('R6_CLOCK_SOURCES')
        self.completed_policy_steps=completed_policy_steps;self.final_policy_step=final_policy_step
        self.recipe_sha256=recipe_sha256;self.branch_seed=branch_seed
        self.command_source=command_source;self.attitude_source=attitude_source;self.pure_hover=False

    def attitude(self,*,device,dtype):
        q=self.attitude_source()
        if (not isinstance(q,torch.Tensor) or q.shape!=(32,4) or q.device!=torch.device(device)
                or q.dtype!=dtype or not bool(torch.isfinite(q).all())
                or bool((torch.linalg.vector_norm(q,dim=-1)-1).abs().gt(1e-5).any())):
            raise ValueError('R6_CLOCK_ACTUAL_ATTITUDE')
        return q.detach().clone()

    def world(self,step,*,device,dtype):
        value=self.command_source(step)
        if (not isinstance(value,torch.Tensor) or value.shape!=(32,3) or value.device!=torch.device(device)
                or value.dtype!=dtype or not bool(torch.isfinite(value).all())
                or bool(value[:,2].ne(0).any()) or bool(torch.linalg.vector_norm(value,dim=-1).gt(.300001).any())):
            raise ValueError('R6_CLOCK_WORLD_COMMAND')
        return value.detach().clone()

    def plan(self,*,device='cpu',dtype=torch.float32):
        step=self.completed_policy_steps
        if step>=self.final_policy_step:raise ValueError('R6_CLOCK_EXHAUSTED')
        q=self.attitude(device=device,dtype=dtype)
        current=self.world(step,device=device,dtype=dtype)
        following=self.world(step+1,device=device,dtype=dtype)
        body=rotate_world_to_body_wxyz(q,current)
        zero=torch.linalg.vector_norm(body,dim=-1)<=1e-3
        return WorldStepPlan(step,body,rotate_world_to_body_wxyz(q,following),zero,~zero,None,current,following)

    def complete_step(self):
        if self.completed_policy_steps>=self.final_policy_step:raise ValueError('R6_CLOCK_EXHAUSTED')
        self.completed_policy_steps+=1

    def state_dict(self):
        return dict(version='curriculum-h1-r6-v1',completed_policy_steps=self.completed_policy_steps,
            recipe_sha256=self.recipe_sha256,origin_checkpoint_sha256=BASELINE_SHA,branch_seed=self.branch_seed)

    def reset_lane(self,env_id):
        if type(env_id)is not int or not 0<=env_id<32:raise ValueError('R6_CLOCK_LANE')
        # Physical reset never changes the registered global curriculum clock.


class R6TrainingDriver(UnifiedH1Driver):
    def __init__(self,environment,observation,*,clock,**kwargs):
        if not isinstance(clock,R6WorldCommandClock):raise TypeError('R6_TRAINING_CLOCK')
        if kwargs.get('step_kind','training')!='training':raise ValueError('R6_TRAINING_STEP_KIND')
        if not callable(getattr(environment,'stage_r6_world_command',None)):
            raise TypeError('R6_TRAINING_PLANT_ADAPTER')
        super().__init__(environment,observation,clock=clock,**kwargs)

    def prepare_step(self,plan):
        # The shared measurement boundary rotates c_(t+1) with the actual new
        # attitude after physics/reset; c_t still drives the current action.
        self.environment.stage_r6_world_command(plan.next_world_command)
        super().prepare_step(plan)

    def next_observation_command(self,plan):
        q=self.clock.attitude(device=plan.next_world_command.device,dtype=plan.next_world_command.dtype)
        return rotate_world_to_body_wxyz(q,plan.next_world_command)

    def command_evidence(self,plan):
        return dict(command_frame='world converted at actual observation boundary',
            command_world=plan.world_command.detach().cpu(),next_command_world=plan.next_world_command.detach().cpu(),
            prepared_next_command_b=plan.next_command_b.detach().cpu(),
            recipe_sha256=self.clock.recipe_sha256,branch_seed=self.clock.branch_seed)
