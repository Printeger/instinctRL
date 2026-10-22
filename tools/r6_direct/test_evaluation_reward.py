"""Real CPU observation history and reward gate; no simulator/GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import tempfile
import torch
from tools.test_r6_measurement_adapter import Env
from instinctlab.task.evaluation_reward import compute_evaluation_reward
from instinctlab.task.rewards_h1 import compute_command_risk, MeasurementCoverageUnavailable

CONTRACT=dict(execution_mode='local-direct',operation='baseline',mode='research-r6-evaluation',budgets={'training_transitions':0})

class EvaluationRewardTests(unittest.TestCase):
    def risk(self,e,o):
        return compute_evaluation_reward(lambda:compute_command_risk(o['policy']['lidar_grid'],torch.zeros(2,3)),context=e._r6_measurements['reward_context'])

    def test_startup_next_frame_drop_and_actor_identity(self):
        e=Env();e.install(evaluation_contract=CONTRACT,drop_steps=[2]);e.reset();o=e._get_observations()
        strict=Env();strict.install(drop_steps=[2]);strict.reset();ref=strict._get_observations()
        self.assertTrue(torch.equal(o['policy']['lidar_grid'],ref['policy']['lidar_grid']))
        self.assertIsNone(self.risk(e,o))
        with self.assertRaises(MeasurementCoverageUnavailable):self.risk(strict,ref)
        for distance in (3.,4.,5.):
            o=e.step(distance);ref=strict.step(distance)
            self.assertTrue(torch.equal(o['policy']['lidar_grid'],ref['policy']['lidar_grid']))
            self.assertIsNotNone(self.risk(e,o))
        self.assertEqual(float(e.rows[2][1]['acquisition_time'][0]),0.)
        self.assertEqual([x[0] for x in e.rows],[0,1,2,3])

    def test_bad_producer_and_received_bad_coverage_still_raise(self):
        for delay in (0,1):
            e=Env();e.valid.zero_();e.ranges.fill_(40.);e.install(delay_cycles=delay,evaluation_contract=CONTRACT);e.reset();o=e._get_observations()
            with self.assertRaises(MeasurementCoverageUnavailable):self.risk(e,o)

    def test_training_contract_rejected_before_installation(self):
        for contract in (dict(CONTRACT,operation='train'),dict(CONTRACT,budgets={'training_transitions':32})):
            e=Env()
            with self.assertRaises(ValueError):e.install(evaluation_contract=contract)
            self.assertIsNone(getattr(e,'_r6_measurements',None))
        from instinctlab.training.rollout_h1 import collect_h1_transition
        e=Env();e.install(evaluation_contract=CONTRACT)
        with self.assertRaisesRegex(ValueError,'evaluation-only'):
            collect_h1_transition(e,None,None,None)

    def test_actual_reward_method_preserves_snapshots_and_command_advance(self):
        # Execute the actual method, replacing only simulator-specific producers.
        path=Path('isaaclab-training/instinctlab/envs/unified_h1_direct_env.py')
        cls=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='UnifiedH1DirectEnv')
        method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_get_rewards')
        def unavailable(**kw):raise MeasurementCoverageUnavailable('CPU startup')
        ns=dict(torch=torch,compute_reward_h1=unavailable,build_critic_h1_state=lambda **kw:NS(privileged=torch.zeros(2,18)),_detached_snapshot=lambda v:v,OrderedDict=dict)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),ns)
        e=Env();e.install(evaluation_contract=CONTRACT);e.reset();e._get_observations()
        z=torch.zeros(2,1);z3=torch.zeros(2,3);b=z.bool()
        e._task=NS(termination=NS(masks=dict(collision=b,below_height=b,above_height=b)),actual_velocity_b=z3,root_state_w=e.root,exact_clearance_m=z,collision=b)
        e._control=NS(v_gov_b=z3,v_final_b=z3,v_corr_b=z3,anchor_metrics=dict(huber_loss=z,active=z,usable_fraction=z),ics_metrics={'beta':z})
        e._robot.data.root_state_w=e.root;e._h1_next_command_b=torch.ones(2,3);e._h1_residual_neutrality_weight=0.;e._h1_anchor_weight=4.;e._reward_previous_v_final_b=z3;e.scene=NS(env_origins=z3);e.reset_buf=b.flatten();e.extras={};e._h1_global_policy_step=0
        e._evaluation_ray_snapshot=NS(ray_starts_w=z3,ray_directions_w=z3,accepted_ray_hits_w=z3,accepted_beam_valid=b,source_step=z)
        e.cfg=NS(ray_caster=NS(mesh_prim_paths=['CPU'],max_distance=40.))
        result=ns['_get_rewards'](e)
        self.assertTrue(torch.equal(result,torch.zeros(2)))
        self.assertFalse(e.extras['reward_valid']);self.assertIsNone(e.extras['pre_reset']['reward'])
        self.assertEqual(e.extras['reward_components'],{})
        from training_workloads.unified_h1.robustness_audit import AuditLog, read_audit
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audit.bin'
            with AuditLog(path,run_id='CPU',block_id='startup',maximum_bytes=65536) as audit:
                audit.append('step_complete',e.extras)
            row=list(read_audit(path,run_id='CPU',block_id='startup',expected_records=1))[0]['payload']
            self.assertFalse(row['reward_valid'])
            self.assertFalse(row['reward_diagnostics']['learning_eligible'])
            self.assertIsNone(row['pre_reset']['reward'])
        self.assertIn('evaluation_ray_source',e.extras)
        self.assertTrue(torch.equal(e._command_b,torch.ones(2,3)));self.assertIsNone(e._h1_next_command_b);self.assertEqual(e._h1_global_policy_step,1)

if __name__=='__main__':unittest.main()
