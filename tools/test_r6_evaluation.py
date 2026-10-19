"""CPU substitute exercises real R6 workload, archive and scorer; no physics claims."""
import copy
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from typing import Any,Mapping
import torch
from instinctlab.control.lee_velocity_controller import rotate_world_to_body_wxyz,quaternion_to_rotation_matrix_wxyz
from instinctlab.observations.robustness import RangeDelivery
from training_workloads.unified_h1.budget import ExecutionCounters
from training_workloads.unified_h1.robustness_evaluation import evaluate_block
from training_workloads.unified_h1.robustness_audit import AuditLog,read_audit
from training_workloads.unified_h1.range_archive import read_archive


class Env:
    device='cpu'
    def __init__(self,reset_last=False):
        self._command_b=torch.zeros(32,3);self.root=torch.zeros(32,13);self.root[:,3]=1;self.root[:,2]=2
        self._robot=SimpleNamespace(data=SimpleNamespace(root_state_w=self.root));self.scene=SimpleNamespace(env_origins=torch.zeros(32,3))
        self.masks={k:torch.zeros(32,1,dtype=torch.bool) for k in ('collision','below_height','above_height')}
        self._task=SimpleNamespace(termination=SimpleNamespace(masks=self.masks));self.reset_last=reset_last
    def set_h1_evaluation_case(self,case):assert case is None
    def configure_r6_physics(self,**kw):self.physics=kw;self.step_index=0
    def configure_r6_measurements(self,**kw):
        self.measure=kw;self.chain=RangeDelivery(lanes=32,rays=2,seed=kw['seed'],stream_id=kw['stream_id'],sigma_m=kw['sigma_m'],delay_cycles=kw['delay_cycles'])
    def set_h1_initial_command(self,body,index):self._command_b.copy_(body)
    def stage_r6_world_command(self,world):self.next_world=world.clone()
    def observe(self):
        self._command_b.copy_(rotate_world_to_body_wxyz(self.root[:,3:7],self.next_world))
        state=torch.zeros(32,52);state[:,45:48]=self._command_b
        frame=self.chain.deliver(torch.full((32,2),2.),torch.ones(32,2,dtype=torch.bool),torch.full((32,1),self.step_index*.1),torch.full((32,1),self.step_index in self.measure['drop_steps'],dtype=torch.bool))
        self.measure['audit_sink'](self.step_index,frame)
        return {'policy':{'state_vec':state,'lidar_grid':torch.zeros(32,1)},'critic':torch.zeros(32,1)}
    def reset(self):
        for lane in range(32):self.physics['reset_sink']({'lane':lane,'fixture_only':True})
        self._refresh_task_snapshot();return self.observe(),{}
    def _refresh_task_snapshot(self):self._evaluation_ray_snapshot=SimpleNamespace(ray_starts_w=self.root[:,:3,None].transpose(1,2).clone())
    def prepare_h1_step(self,**kw):assert torch.equal(kw['command_b'],self._command_b)
    def step(self,action):
        rotation=quaternion_to_rotation_matrix_wxyz(self.root[:,3:7])
        world=torch.matmul(rotation,self._command_b[:,:,None]).squeeze(-1)
        self.root[:,7:10]=world;self.root[:,:3]+=world*.1
        for tick in range(20):self.physics['force_sink']({'block_tick':20*self.step_index+tick,'fixture_only':True})
        self.step_index+=1;yaw=self.step_index*.001
        self.root[:,3]=math.cos(yaw/2);self.root[:,6]=math.sin(yaw/2)
        self._refresh_task_snapshot()
        done=torch.zeros(32,dtype=torch.bool)
        if self.reset_last and self.step_index==330:done[0]=True
        extras={'pre_reset':{'root_state_w':self.root.clone(),'task':{'termination':{'masks':self.masks}},'control':{}},
                'evaluation_ray_source':{'ray_starts_w':self._evaluation_ray_snapshot.ray_starts_w},'reward_diagnostics':{}}
        return self.observe(),torch.zeros(32),done,torch.zeros_like(done),extras
    def finish_r6_physics(self):assert self.step_index==self.physics['block']['steps']
    def finish_r6_measurements(self):assert self.step_index+1==self.measure['frame_limit']


class Actor:
    def eval(self):pass
    def __call__(self,policy):
        assert set(policy)=={'state_vec','lidar_grid'}
        return SimpleNamespace(mean_action=torch.zeros(32,4))


class EvaluationTests(unittest.TestCase):
    def test_actual_backend_entry_rejects_old_authority_and_unfrozen_protocol(self):
        root=Path(__file__).resolve().parents[1]
        source=root/'training_workloads/unified_h1/backend.py'
        cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='UnifiedH1LiveBackend')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_r6_blocks')
        namespace=dict(Path=Path,Mapping=Mapping,Any=Any,__package__='training_workloads.unified_h1',
            sha256_file=lambda path:hashlib.sha256(path.read_bytes()).hexdigest())
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),namespace)
        evaluate=namespace['evaluate_r6_blocks']
        with self.assertRaisesRegex(PermissionError,'new exact'):
            evaluate(SimpleNamespace(authorization={'mode':'research-formal-evaluation'}))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'protocol.json';path.write_text('{"frozen":false}')
            b=SimpleNamespace(authorization={'mode':'research-r6-evaluation','parameters':{'r6':{'protocol':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}})
            with self.assertRaisesRegex(PermissionError,'not independently frozen'):evaluate(b)

    def setup_backend(self,folder,**kw):
        e=Env(**kw);c=ExecutionCounters()
        b=SimpleNamespace(environment=e,output=Path(folder),counters=c,trainer=SimpleNamespace(actor=Actor()),
            authorization={'budgets':{'nontraining_transitions':10560,'automatic_resets':10560,'explicit_resets':1}},resource_check=lambda stage,ordinal:None)
        def reset(seed):c.reserve_reset(automatic=False,maximum=1);return e.reset()
        b._preserving_reset=reset;return b
    def block(self):return dict(id='S-fixture',task='stop',steps=330,lanes=32,domain='target',speed_m_s=.2,noise_std_m=0.,delay_cycles=1,drop_policy_steps=[70,180])
    def run_block(self,b):return evaluate_block(b,self.block(),run_id='CPU-FIXTURE',seed=17,split='development',expected_mass_kg=1.16,range_quota_bytes=8*1024**2,audit_quota_bytes=16*1024**2)

    def test_full_lane_raw_world_commands_archive_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.setup_backend(tmp);r=self.run_block(b);folder=Path(tmp)/'r6/S-fixture'
            self.assertEqual(r['scoring']['successes'],32);self.assertFalse(r['scientific_acceptance'])
            self.assertEqual(r['force_records'],6600);self.assertEqual(r['measurement_frames'],331)
            self.assertEqual(r['counters']['nontraining_transitions'],10560);self.assertEqual(r['counters']['explicit_resets'],1)
            rows=list(read_audit(folder/'audit.bin',run_id='CPU-FIXTURE',block_id='S-fixture',expected_records=r['audit_records']))
            self.assertEqual(sum(x['kind']=='step_complete' for x in rows),330)
            frames=list(read_archive(folder/'measurements.bin',run_id='CPU-FIXTURE',block_id='S-fixture',expected_frames=331))
            self.assertTrue(frames[70]['drop'].all());self.assertTrue(frames[180]['drop'].all());self.assertFalse(frames[69]['drop'].any())
            traces=json.loads((folder/'motion_traces.json').read_text());self.assertEqual(len(traces),32)
            with self.assertRaises(FileExistsError):self.run_block(b)

    def test_last_sample_reset_keeps_32_lane_failure_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=self.run_block(self.setup_backend(tmp,reset_last=True))
            self.assertEqual(r['scoring']['total'],32);self.assertEqual(r['scoring']['successes'],31)
            self.assertEqual(r['scoring']['lanes'][0]['reason'],'INVALID_EVIDENCE')
            self.assertEqual(r['counters']['automatic_resets'],1)

    def test_transition_cap_stops_before_second_physical_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.setup_backend(tmp);b.authorization['budgets']['nontraining_transitions']=32
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):self.run_block(b)
            self.assertEqual(b.environment.step_index,1);self.assertEqual(b.counters.nontraining_transitions,32)
            self.assertFalse((Path(tmp)/'r6/S-fixture/result.json').exists())
            self.assertTrue((Path(tmp)/'r6/S-fixture/audit.bin').stat().st_size>0)

    def test_metadata_combined_limit_rejects_before_either_write(self):
        from training_workloads.unified_h1 import robustness_audit as audit
        from training_workloads.scale_003.config import canonical_bytes
        traces=[{'lane':i,'rows':[{'value':1.2345678901234567}]} for i in range(32)]
        summary={'status':'CPU_FIXTURE'}
        count=sum(len(canonical_bytes(v)) for v in (traces,summary))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with patch.object(audit,'BLOCK_METADATA_CAP_BYTES',count-1):
                with self.assertRaisesRegex(ValueError,'METADATA_QUOTA_BEFORE_WRITE'):
                    audit.write_block_metadata(folder,traces,summary)
            self.assertEqual(list(folder.iterdir()),[])
            with patch.object(audit,'BLOCK_METADATA_CAP_BYTES',count):
                self.assertEqual(audit.write_block_metadata(folder,traces,summary),count)
            self.assertEqual(json.loads((folder/'motion_traces.json').read_text()),traces)
            self.assertEqual(sum(p.stat().st_size for p in folder.iterdir()),count)

    def test_metadata_writer_retains_global_write_quota(self):
        from training_workloads.unified_h1.robustness_audit import write_block_metadata
        from training_workloads.scale_003 import persistence as p
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);quota=p.OutputWriteQuota(folder,1)
            with patch.object(p,'_output_quota',quota):
                with self.assertRaises(p.PersistenceError):write_block_metadata(folder,[{'lane':0}],{})
            self.assertFalse((folder/'result.json').exists())
            self.assertFalse((folder/'motion_traces.json').exists())

    def test_audit_quota_corruption_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'audit.bin'
            with AuditLog(p,run_id='cpu',block_id='b',maximum_bytes=512) as log:
                log.append('fixture',{'value':1.25});size=p.stat().st_size
                with self.assertRaisesRegex(ValueError,'QUOTA_BEFORE_WRITE'):log.append('large',{'values':list(range(1000))})
                self.assertEqual(p.stat().st_size,size)
                with self.assertRaises(FileExistsError):AuditLog(p,run_id='cpu',block_id='b',maximum_bytes=512)
            self.assertEqual(next(read_audit(p,run_id='cpu',block_id='b',expected_records=1))['payload']['value'],1.25)
            data=p.read_bytes();p.write_bytes(data[:-1])
            with self.assertRaisesRegex(ValueError,'TRUNCATED'):list(read_audit(p,run_id='cpu',block_id='b',expected_records=1))


if __name__=='__main__':unittest.main()
