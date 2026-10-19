import copy
from pathlib import Path
import tempfile
import unittest
import torch
from tools.test_r6_physics_adapter import Env
from training_workloads.unified_h1.logging import _jsonable
from training_workloads.unified_h1.robustness_audit import AuditLog
from training_workloads.unified_h1.robustness_physics_replay import verify_force,verify_reset_record,replay_staging


class PhysicsReplayTests(unittest.TestCase):
    def force(self,tick=1000):
        env=Env();env.reset();env._r6_physics['ticks']=tick
        generator=torch.Generator().manual_seed(17)
        q=torch.randn(32,5,4,generator=generator);q/=torch.linalg.vector_norm(q,dim=-1,keepdim=True)
        env._robot.data.body_link_state_w[:,:,3:7]=q
        env._wrench.forces_body_n.copy_(torch.randn(32,5,3,generator=generator))
        env.tick()
        return _jsonable(env.force_rows[0]),env._r6_physics['block']

    def verify(self,row,block):
        return verify_force(row,block,expected_tick=row['block_tick'],previous_sim_tick=None,expected_mass_kg=1.16)

    def test_independent_rotation_and_window_endpoints(self):
        for tick in (0,999,1000,1039,1040,6599):
            row,block=self.force(tick);_,impulse=self.verify(row,block)
            self.assertEqual(bool((impulse!=0).any()),1000<=tick<1040)

    def test_force_world_local_rotor_and_torque_corruption(self):
        for field in ('additional_world_n','additional_local_n','rotor_forces_local_n','staged_forces_local_n','staged_torques_local_nm'):
            row,block=self.force();row[field][0][0][0]+=.01
            with self.subTest(field=field),self.assertRaises(ValueError):self.verify(row,block)

    def test_clock_dt_mass_and_active_corruption(self):
        for field,value in [('sim_tick',-1),('physics_dt_s',.01),('active',False),('dtype','torch.float16')]:
            row,block=self.force();row[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.verify(row,block)
        row,block=self.force();row['mass_kg'][0][0]+=.01
        with self.assertRaisesRegex(ValueError,'PHYSICS_MASS'):self.verify(row,block)

    def test_reset_recomputed_from_seed_and_observed_world(self):
        env=Env(task='wall_modes');env.reset()
        for row in _jsonable(env.reset_rows):
            self.assertEqual(verify_reset_record(row,env._r6_physics['block'],seed=17,split='development',mount_position_m=(.02,.01,.1)),row['lane'])

    def test_reset_tolerance_not_trusted(self):
        env=Env();env.reset();row=_jsonable(env.reset_rows[0]);row['position_atol_m']=1.
        with self.assertRaisesRegex(ValueError,'RESET_TOLERANCE'):
            verify_reset_record(row,env._r6_physics['block'],seed=17,split='development',mount_position_m=(.02,.01,.1))

    def test_reset_seed_and_local_coordinate_corruption(self):
        env=Env();env.reset();row=_jsonable(env.reset_rows[0]);block=env._r6_physics['block']
        with self.assertRaisesRegex(ValueError,'RESET_SPEC'):
            verify_reset_record(row,block,seed=18,split='development',mount_position_m=(.02,.01,.1))
        row['actual']['root_position_local_m'][0]+=.001
        with self.assertRaisesRegex(ValueError,'RESET_WORLD_LOCAL'):
            verify_reset_record(row,block,seed=17,split='development',mount_position_m=(.02,.01,.1))

    def archive(self,folder,*,omit=False):
        env=Env(task='wall_modes',steps=1);env.reset();block=copy.deepcopy(env._r6_physics['block'])
        for _ in range(20):env.tick()
        completion=env.finish_r6_physics();path=Path(folder)/'audit.bin'
        with AuditLog(path,run_id='CPU',block_id=block['id'],maximum_bytes=4*1024**2) as log:
            for row in env.reset_rows:log.append('reset',row)
            for row in env.force_rows[:-1] if omit else env.force_rows:log.append('force',row)
            count=log.records
        return path,block,count,completion

    def replay(self,path,block,count,completion):
        return replay_staging(path,block,run_id='CPU',seed=17,split='development',expected_records=count,
            expected_mass_kg=1.16,mount_position_m=(.02,.01,.1),completion=completion)

    def test_archive_roundtrip_and_final_tick(self):
        with tempfile.TemporaryDirectory() as folder:
            path,block,count,completion=self.archive(folder)
            result=self.replay(path,block,count,completion)
            self.assertTrue(result['staging_reconstructed']);self.assertFalse(result['physical_integration_verified'])
            self.assertEqual((result['force_records'],result['reset_records']),(20,32))
            completion['completed_sim_tick']-=1
            with self.assertRaisesRegex(ValueError,'FINAL_TICK_NOT_COMPLETED'):self.replay(path,block,count,completion)

    def test_missing_force_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path,block,count,completion=self.archive(folder,omit=True)
            with self.assertRaisesRegex(ValueError,'FORCE_COVERAGE'):self.replay(path,block,count,completion)


if __name__=='__main__':unittest.main()
