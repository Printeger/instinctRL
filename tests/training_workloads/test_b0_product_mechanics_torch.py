"""Execute the product snapshot method with mocked articulation and real CPU tensors.

This tier is executable mock integration, not actual Isaac Sim integration.
"""
import ast
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any,Mapping

import torch
import pytest
from instinctlab.control.lee_velocity_controller import rotate_world_to_body_wxyz
from instinctlab.control.rotor_actuation import ROTOR_NAMES
from training_workloads.unified_h1.b0_enhanced import reconstruct_com,validate_row,EvidenceError,quaternion_rotate


@pytest.mark.parametrize("origin,rotated",[(0.,False),(100.,False),(700.,True)])
def test_product_snapshot_uses_link_velocity_once_and_preserves_controller_feedback(origin,rotated):
    path=Path('isaaclab-training/instinctlab/envs/unified_h1_direct_env.py')
    tree=ast.parse(path.read_text())
    selected=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ('_enhanced_b0_physics_rows','_cross3','_tolist')]
    wrapper=ast.ClassDef(name='Snapshot',bases=[],keywords=[],decorator_list=[],body=selected)
    module=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]))
    # Fixture quaternions are exactly identity; no alternative production math is installed.
    namespace={'torch':torch,'Any':Any,'Mapping':Mapping,'Sequence':list,'OrderedDict':OrderedDict,
               'ROTOR_NAMES':ROTOR_NAMES,'rotate_world_to_body_wxyz':rotate_world_to_body_wxyz,
               'quat_apply':lambda q,v:torch.tensor([quaternion_rotate(q[i].tolist(),v[i].tolist()) for i in range(q.shape[0])],dtype=v.dtype)}
    exec(compile(module,str(path),'exec'),namespace)
    env=namespace['Snapshot']()
    body=torch.zeros(32,5,13);body[:,:,3]=1;body[:,:,12]=2
    body[:,:,0]=origin
    if rotated:
        body[:,:,3]=2**-.5;body[:,:,6]=2**-.5
    hybrid=body.clone();hybrid[:,:,8]=.2  # body_state_w: velocities already at COM.
    com=torch.zeros(32,5,7);com[:,:,0]=.1;com[:,:,3]=1
    root=torch.zeros(32,13);root[:,3]=1
    control_root=root.clone();control_root[:,8]=.2
    data=NS(body_link_state_w=body,body_state_w=hybrid,body_com_pose_b=com,
            default_mass=torch.ones(32,5),default_inertia=torch.eye(3).reshape(1,1,9).repeat(32,5,1),
            root_link_state_w=root,root_state_w=control_root)
    env._robot=NS(data=data,body_names=['base']+list(ROTOR_NAMES),
        root_physx_view=NS(get_masses=lambda:torch.ones(32,5),get_inertias=lambda:torch.eye(3).reshape(1,1,9).repeat(32,5,1)))
    data.default_mass.fill_(99.)  # Defaults must not be mistaken for actual plant measurements.
    env.device='cpu';env.cfg=NS(sim=NS(dt=.005));env._command_b=torch.zeros(32,3)
    env._lee=NS(desired_rotation_w=torch.eye(3).repeat(32,1,1),collective_thrust_n=torch.ones(32,1))
    rotor=NS(normalized_commands=torch.full((32,4),.5),target_throttle=torch.full((32,4),.5),
             throttle=torch.full((32,4),.5),thrusts_n=torch.ones(32,4),reaction_moments_nm=torch.zeros(32,4))
    env._wrench=NS(rotor=rotor);env._enhanced_b0_coordinate=('Z',0,0)
    env._enhanced_b0_previous_system={};env._enhanced_b0_previous_velocity={};env._enhanced_b0_episode_ids={}
    env._enhanced_b0_last_snapshots={};env._enhanced_b0_reset_boundaries={}
    env._enhanced_b0_cache_unavailable=set();env._enhanced_b0_cache_generation=1
    before_body=body.clone();before_root=control_root.clone()
    rows=env._enhanced_b0_physics_rows()
    assert len(rows)==32
    for row in rows:
        derived=reconstruct_com(row['bodies'])
        expected=torch.tensor([-1.,0.,0.] if rotated else [0.,1.,0.])
        torch.testing.assert_close(torch.tensor(derived['linear_momentum_kg_m_s']),expected,atol=1e-6,rtol=1e-6)
        assert row['root_state']['linear_velocity']==[0.,0.,0.]
        assert row['response']['velocity_world_m_s']==[0.,0.,0.]
        assert abs(row['controller']['velocity_error_b_m_s'][1]+.2)<1e-7
        validate_row(row)
    assert torch.equal(before_body,body) and torch.equal(before_root,control_root)
    # A continuous next physics sample must match serialized momentum differencing.
    body[:,:,7]+=.01
    env._enhanced_b0_coordinate=('Z',0,1)
    following=env._enhanced_b0_physics_rows()
    for first,next_row in zip(rows,following):
        validate_row(next_row)
        from training_workloads.unified_h1.b0_causal_analyzer import derive_mechanics
        assert derive_mechanics(first,next_row)['momentum_wrench'] is not None
    import copy
    for kind in ('com','momentum','wrench'):
        bad=copy.deepcopy(rows[0])
        if kind=='com':bad['system']['combined_com_world_m']=list(bad['system']['combined_com_world_m']);bad['system']['combined_com_world_m'][0]+=.001
        elif kind=='momentum':bad['system']['linear_momentum_kg_m_s']=list(bad['system']['linear_momentum_kg_m_s']);bad['system']['linear_momentum_kg_m_s'][0]+=.001
        else:bad['wrenches']['rotor']['force_n']=list(bad['wrenches']['rotor']['force_n']);bad['wrenches']['rotor']['force_n'][0]+=.001
        with pytest.raises(EvidenceError):validate_row(bad)
