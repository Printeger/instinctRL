"""CPU plant fixtures verify R6 reset writes and additive wrench staging."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from instinctlab.observations.robustness_physics import R6PhysicsMixin,distributed_pulse
from instinctlab.control.lee_velocity_controller import quaternion_to_rotation_matrix_wxyz

ROOT=Path(__file__).resolve().parents[1]


class Base:
    def _apply_external_disturbance(self):pass
    def _policy_sensor_frame(self,ranges,valid,time):return ranges,valid,time


class Env(R6PhysicsMixin,Base):
    def __init__(self,task='pulse_hold',origin=1000.,steps=330):
        self.num_envs=32;self._h1_evaluation_case=None
        self.cfg=SimpleNamespace(ray_caster=SimpleNamespace(offset=SimpleNamespace(pos=(.02,.01,.1))))
        self.scene=SimpleNamespace(env_origins=torch.full((32,3),origin))
        root=torch.zeros(32,13);root[:,3]=1
        body=torch.zeros(32,5,13);body[:,:,3]=1
        self.mass=torch.tensor([[1.,.04,.04,.04,.04]]*32)
        self._robot=SimpleNamespace(data=SimpleNamespace(root_state_w=root,body_link_state_w=body),
            root_physx_view=SimpleNamespace(get_masses=lambda:self.mass),
            write_root_pose_to_sim=lambda pose,ids:root.__setitem__((ids,slice(0,7)),pose),
            write_root_velocity_to_sim=lambda velocity,ids:root.__setitem__((ids,slice(7,13)),velocity),
            set_external_force_and_torque=self.set_wrench)
        self._wrench=SimpleNamespace(forces_body_n=torch.full((32,5,3),.4),torques_body_nm=torch.full((32,5,3),.03))
        self.sim=SimpleNamespace(current_time_step_index=40);self.reset_rows=[];self.force_rows=[]
        block=dict(id='P-fixture',task=task,steps=steps,pulse_acceleration_m_s2=.5)
        if task=='wall_modes':block.update(id='W-fixture',pulse_acceleration_m_s2=0.,sensor_to_wall_distance_m=1.3)
        self.configure_r6_physics(block=block,seed=17,split='development',reset_sink=self.reset_rows.append,
            force_sink=self.force_rows.append,expected_mass_kg=1.16)
    def set_wrench(self,forces,torques,**kw):
        assert kw=={'is_global':False}
        self.staged=(forces.clone(),torques.clone())
    def reset(self,ids=None):
        if ids is None:ids=torch.arange(32)
        self._reset_r6_physics(ids)
        root=self._robot.data.root_state_w
        self._ray_caster=SimpleNamespace(data=SimpleNamespace(pos_w=root[:,:3].clone(),quat_w=root[:,3:7].clone()))
        self._policy_sensor_frame(None,None,None)
    def tick(self):
        # Actual base action path rewrites rotor-only forces before this hook.
        self.set_wrench(self._wrench.forces_body_n,self._wrench.torques_body_nm,is_global=False)
        self._apply_external_disturbance();self.sim.current_time_step_index+=1


class PhysicsTests(unittest.TestCase):
    def test_world_force_and_zero_com_moment_with_different_body_attitudes(self):
        mass=torch.tensor([[1.,.1,.2,.3,.4]],dtype=torch.float64)
        angles=torch.arange(5,dtype=torch.float64)*.3
        q=torch.stack((torch.cos(angles/2),torch.zeros(5),torch.zeros(5),torch.sin(angles/2)),dim=-1)[None]
        accel=torch.tensor([[.3,.4,0.]],dtype=torch.float64)
        rotor=torch.ones(1,5,3,dtype=torch.float64);torque=rotor*.03
        forces,moments,world,local=distributed_pulse(mass,q,accel,rotor,torque)
        rotation=quaternion_to_rotation_matrix_wxyz(q.reshape(-1,4)).reshape(1,5,3,3)
        reconstructed=torch.matmul(rotation,(forces-rotor).unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(reconstructed,world,atol=1e-14))
        self.assertTrue(torch.allclose(world.sum(dim=1),mass.sum()*accel))
        coms=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.]]],dtype=torch.float64)
        combined=(coms*mass[:,:,None]).sum(dim=1)/mass.sum()
        moment=torch.linalg.cross(coms-combined[:,None],world,dim=-1).sum(dim=1)
        self.assertLess(float(moment.abs().max()),1e-14)
        self.assertTrue(torch.equal(moments,torque));self.assertTrue(torch.equal(rotor,torch.ones_like(rotor)))

    def test_rotated_reset_observed_sensor_position_and_float_world_precision(self):
        e=Env(task='wall_modes');e.reset();self.assertEqual(len(e.reset_rows),32)
        self.assertEqual(e._r6_physics['ready'],set(range(32)))
        self.assertTrue(any(row['position_atol_m']>1e-5 for row in e.reset_rows))
        for row in e.reset_rows:
            lane=row['lane'];axis=0 if lane<16 else 1
            self.assertAlmostEqual(row['actual']['sensor_position_local_m'][axis],4.7,places=3)
            self.assertEqual(row['actual']['root_linear_velocity_w_m_s'],[0.,0.,0.])

    def test_observed_reset_mismatch_is_saved_and_poisoned(self):
        e=Env();e._reset_r6_physics(torch.arange(32));root=e._robot.data.root_state_w
        e._ray_caster=SimpleNamespace(data=SimpleNamespace(pos_w=root[:,:3].clone()+.1,quat_w=root[:,3:7].clone()))
        with self.assertRaisesRegex(ValueError,'RESET_POSITION_MISMATCH'):e._policy_sensor_frame(None,None,None)
        self.assertEqual(len(e.reset_rows),1);self.assertTrue(e._r6_physics['failed'])
        with self.assertRaises(RuntimeError):e.tick()

    def test_exact_40_tick_pulse_and_no_residual_force(self):
        e=Env();e.reset()
        for _ in range(1042):e.tick()
        active=[r for r in e.force_rows if r['active']]
        self.assertEqual([r['block_tick'] for r in active],list(range(1000,1040)))
        impulse=sum((r['additional_world_n'].sum(dim=1)*.005 for r in active))
        directions=torch.tensor([s['direction_world'] for s in e._r6_physics['specs']])
        self.assertTrue(torch.allclose(impulse,1.16*.5*.2*directions,atol=1e-6))
        self.assertTrue(torch.equal(e.staged[0],e._wrench.forces_body_n))
        self.assertTrue(torch.equal(e.staged[1],e._wrench.torques_body_nm))
        self.assertEqual(e.sim.current_time_step_index,1082)

    def test_mass_and_clock_drift_and_incomplete_finish_fail(self):
        e=Env();e.reset();e.mass[0,0]+=.01
        with self.assertRaisesRegex(RuntimeError,'mass differs'):e.tick()
        e=Env();e.reset();e.tick();e.sim.current_time_step_index+=1
        with self.assertRaisesRegex(RuntimeError,'clock discontinuity'):e.tick()
        with self.assertRaisesRegex(RuntimeError,'incomplete'):e.finish_r6_physics()

    def test_zero_level_and_selected_reset_never_rewind_block_clock(self):
        e=Env(task='wall_modes');e.reset();e.tick();e.reset(torch.tensor([3]));e.tick()
        self.assertEqual(e._r6_physics['ticks'],2)
        self.assertEqual(len(e.reset_rows),33)
        self.assertFalse(any(r['active'] for r in e.force_rows))
        self.assertTrue(all(not bool(r['additional_world_n'].any()) for r in e.force_rows))

    def test_finish_requires_last_physics_tick_and_quota_rejects_extra_action(self):
        e=Env(task='wall_modes',steps=1);e.reset()
        for _ in range(20):e.tick()
        with self.assertRaisesRegex(RuntimeError,'tick quota'):e.tick()
        e.sim.current_time_step_index-=1
        with self.assertRaisesRegex(RuntimeError,'no completed physics tick'):e.finish_r6_physics()
        e.sim.current_time_step_index+=1;e.finish_r6_physics()
        self.assertIsNone(e._r6_physics)

    def test_actual_hooks_follow_rotor_application_and_base_reset(self):
        text=(ROOT/'isaaclab-training/instinctlab/envs/taslab_direct_env.py').read_text()
        cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='TaslabDirectEnv')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_action')
        code=ast.unparse(method)
        self.assertLess(code.index('self._wrench_adapter.apply'),code.index('self._apply_external_disturbance'))
        text=(ROOT/'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py').read_text()
        cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='UnifiedH1DirectEnv')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_reset_idx_impl')
        code=ast.unparse(method)
        self.assertLess(code.index('super()._reset_idx'),code.index('self._reset_r6_physics'))
        self.assertLess(code.index('self._reset_r6_physics'),code.index('self._h1_evaluation_case'))


if __name__=='__main__':unittest.main()
