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


def snapshot_fixture(origin=0.,rotated=False,misaligned=False):
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
    if misaligned:
        body[:,1:,3]=2**-.5;body[:,1:,5]=2**-.5
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
    if misaligned:rotor.reaction_moments_nm[:]=torch.tensor([.01,.02,.03,.04])
    env._wrench=NS(rotor=rotor);env._enhanced_b0_coordinate=('Z',0,0)
    env._enhanced_b0_previous_system={};env._enhanced_b0_previous_velocity={};env._enhanced_b0_episode_ids={}
    env._enhanced_b0_last_snapshots={};env._enhanced_b0_reset_boundaries={}
    env._enhanced_b0_cache_unavailable=set();env._enhanced_b0_cache_generation=1
    before_body=body.clone();before_root=control_root.clone()
    return env,body,control_root


@pytest.mark.parametrize("origin,rotated,misaligned",[(0.,False,False),(100.,False,False),(700.,True,False),(100.,False,True)])
def test_product_snapshot_uses_link_velocity_once_and_preserves_controller_feedback(origin,rotated,misaligned):
    env,body,control_root=snapshot_fixture(origin,rotated,misaligned)
    rotor=env._wrench.rotor
    before_body=body.clone();before_root=control_root.clone()
    rows=env._enhanced_b0_physics_rows()
    assert len(rows)==32
    for row in rows:
        derived=reconstruct_com(row['bodies'])
        expected=torch.tensor([0.,.2,0.] if misaligned else ([-1.,0.,0.] if rotated else [0.,1.,0.]))
        torch.testing.assert_close(torch.tensor(derived['linear_momentum_kg_m_s']),expected,atol=1e-6,rtol=1e-6)
        assert row['root_state']['linear_velocity']==[0.,0.,0.]
        assert row['response']['velocity_world_m_s']==[0.,0.,0.]
        assert abs(row['controller']['velocity_error_b_m_s'][1]+.2)<1e-7
        validate_row(row)
    if misaligned:
        from instinctlab.control.runtime_adapter import TaslabArticulationWrenchAdapter
        adapter=object.__new__(TaslabArticulationWrenchAdapter)
        adapter._forces=torch.zeros(32,5,3);adapter._torques=torch.zeros_like(adapter._forces)
        forces,torques=adapter._wrench_buffers(rotor)
        # Independently reconstruct the actual adapter's staged body wrenches.
        force_total=torch.zeros(3,dtype=torch.float64);torque_total=force_total.clone()
        center=torch.tensor(rows[0]['system']['combined_com_world_m'],dtype=torch.float64)
        for index in range(5):
            q=body[0,index,3:7].double().tolist()
            f=torch.tensor(quaternion_rotate(q,forces[0,index].double().tolist()))
            t=torch.tensor(quaternion_rotate(q,torques[0,index].double().tolist()))
            force_total+=f;torque_total+=torch.cross(body[0,index,:3].double()-center,f.double(),dim=0)+t
        torch.testing.assert_close(torch.tensor(rows[0]['wrenches']['rotor']['force_n'],dtype=torch.float64),force_total,atol=1e-7,rtol=1e-7)
        torch.testing.assert_close(torch.tensor(rows[0]['wrenches']['rotor']['torque_at_com_nm'],dtype=torch.float64),torque_total,atol=1e-7,rtol=1e-7)
        assert abs(rows[0]['wrenches']['rotor']['torque_at_com_nm'][2]-.1)<1e-7
        import copy
        bad=copy.deepcopy(rows[0]);bad['rotors'][0]['reaction_direction_world']=[1.,0.,0.]
        with pytest.raises(EvidenceError,match='reaction direction'):validate_row(bad)
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


@pytest.mark.parametrize("bad,branch",[(0.,"INERTIA_NONPOSITIVE_DIAGONAL"),(float('nan'),"INERTIA_SHAPE_OR_NONFINITE")])
def test_product_first_derivation_failure_preserves_unvalidated_inputs(bad,branch):
    env,body,root=snapshot_fixture()
    raw=torch.eye(3).reshape(1,1,9).repeat(32,5,1);raw[0,3,0]=bad
    env._robot.root_physx_view.get_inertias=lambda:raw
    rng=torch.get_rng_state().clone();before=body.clone()
    with pytest.raises(EvidenceError) as caught:env._enhanced_b0_physics_rows()
    context=caught.value.measurement_failure
    assert context['status']=='UNVALIDATED' and not context['normal_physics_row']
    assert context['coordinate']=={'case':'Z','policy_step':0,'physics_step':0,'lane':0}
    assert context['failure']['body_index']==3 and context['failure']['body_name']==ROTOR_NAMES[2]
    assert context['failure']['failure_branch']==branch and len(context['bodies'])==5
    actual=context['bodies'][3]['inertia_at_com_body_kg_m2'][0][0]
    assert actual==0. if bad==0. else actual=={'availability':'NONFINITE','representation':'nan'}
    assert context['tensor_metadata']['inertia']=={'dtype':'torch.float32','shape':[32,5,9]}
    assert not env._enhanced_b0_previous_system and not env._enhanced_b0_last_snapshots
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(before,body)
