"""AST-extracted case runner lifecycle on CPU mocks; no actual physics."""
import ast
from pathlib import Path
from typing import Any
import pytest
import torch

def test_every_case_reset_rebuilds_observation_exactly_once_before_first_step():
    source=Path('isaaclab-training/instinctlab/envs/unified_h1_direct_env.py')
    method=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='run_frozen_enhanced_b0_zts')
    module=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Runner',bases=[],keywords=[],decorator_list=[],body=[method])],type_ignores=[]))
    ns={'torch':torch,'Any':Any};exec(compile(module,str(source),'exec'),ns)
    env=ns['Runner']();events=[];ready=False
    env._enhanced_b0_registered=True;env._enhanced_b0_active=False;env._enhanced_b0_batches=0
    env.device='cpu';env.num_envs=32;env._command_b=torch.zeros(32,3);env._actions=torch.zeros(32,4)
    def reset_indices(indices):
        nonlocal ready
        ready=False;events.append(('reset',env.case))
    def observations():
        nonlocal ready
        ready=True;events.append(('actual_observation',env.case))
    def reset(seed):
        assert seed==(20260909 if env.case=='Z' else None)
        reset_indices(None);observations()
    def step(action):
        assert ready, 'cleared actor history used before actual reset observation'
        coordinate=env._enhanced_b0_coordinate
        env._enhanced_b0_coordinate=(*coordinate[:2],20);env._enhanced_b0_batches+=20
        if coordinate[1]==0:events.append(('first_step',env.case))
        return None,None,None,None,None
    env.set_h1_evaluation_case=lambda case:setattr(env,'case',case)
    env.reset=reset;env._reset_idx=reset_indices;env._get_observations=observations
    env.prepare_b0_step=lambda **kw:None;env.step=step
    env._enhanced_b0_accounting={key:(lambda *args:None) for key in ('before_reset','before_step','after_step')}
    report=env.run_frozen_enhanced_b0_zts(seed=20260909,cases=('Z','T','S'),target_velocity_b_m_s=(0.,0.,0.),retries=0)
    assert events==[(event,case) for case in ('Z','T','S') for event in ('reset','actual_observation','first_step')]
    assert report['physics_rows']==403200 and not env._enhanced_b0_active
