"""Execute actual observation-method AST with CPU tensor producers, no Isaac."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from instinctlab.observations.actor import ActorObservationHistory,RAY_COUNT
from instinctlab.observations.robustness_adapter import R6MeasurementMixin
from instinctlab.observations.robustness_physics import R6PhysicsMixin
from instinctlab.control.pipeline import extract_lidar_history

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'isaaclab-training/instinctlab/envs/taslab_direct_env.py'
cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='TaslabDirectEnv')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in
         {'_get_observations','_policy_sensor_frame','_reset_policy_sensor_frame'}]
namespace={'Any':object,'POLICY_DT_S':.1,'POLICY_KEYS':('lidar_grid','state_vec'),
           'rotate_world_to_body_wxyz':lambda q,v:v,
           'build_critic_state':lambda **kw:SimpleNamespace(privileged=kw['exact_clearance_m'].clone())}
for node in methods:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),namespace)
Base=type('ActualObservationMethods',(),{n.name:namespace[n.name] for n in methods})


class Env(R6MeasurementMixin,Base):
    def __init__(self):
        self.num_envs=2;self._actions=torch.zeros(2,1);self._h1_evaluation_case=None
        self.root=torch.zeros(2,13);self.root[:,3]=1
        self.episode_length_buf=torch.zeros(2,dtype=torch.long)
        self._history=ActorObservationHistory();self._history_initialized=False
        self._robot=SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=torch.zeros(2,3),projected_gravity_b=torch.tensor([[0.,0.,-1.]]*2)))
        self._command_b=torch.zeros(2,3);self._previous_issued_v_final_b=torch.zeros(2,3)
        self.ranges=torch.full((2,RAY_COUNT),2.);self.valid=torch.ones_like(self.ranges,dtype=torch.bool)
        self.clearance=torch.full((2,1),1.234);self.rows=[]
    def _validated_root_state(self):return self.root
    def _read_sensor_producers(self):return self.ranges,self.valid,self.clearance,torch.zeros(2,1,dtype=torch.bool)
    def reset(self,indices=None):
        if indices is None:indices=torch.arange(2)
        self.episode_length_buf[indices]=0
        self._history.reset(indices) if self._history_initialized else self._history.reset()
        self._reset_policy_sensor_frame(indices)
    def install(self,**kw):
        settings=dict(seed=17,stream_id='CPU/S',sigma_m=0.,delay_cycles=1,drop_steps=[],frame_limit=4,
                      audit_sink=lambda sequence,frame:self.rows.append((sequence,frame)))
        settings.update(kw);self.configure_r6_measurements(**settings)
    def step(self,value):
        self.episode_length_buf+=1;self.ranges.fill_(value);return self._get_observations()


class AdapterTests(unittest.TestCase):
    def test_world_command_uses_actual_boundary_attitude_before_history_update(self):
        class Combined(R6PhysicsMixin,Env):pass
        e=Combined();e._robot.data.root_state_w=e.root
        e._validate_h1_command=lambda name,value:None # Shape validation belongs to live H1 environment.
        e._r6_physics=dict(pending=set(),failed=False,world_command=None)
        command=torch.tensor([[.2,0.,0.]]*2)
        e.stage_r6_world_command(command);first=e._get_observations()
        e.episode_length_buf+=1;e.root[:,3]=2**-.5;e.root[:,6]=2**-.5
        e.stage_r6_world_command(command);second=e._get_observations()
        expected=torch.tensor([[0.,-.2,0.]]*2)
        self.assertTrue(torch.allclose(second['policy']['state_vec'][:,45:48],expected,atol=1e-6))
        self.assertTrue(torch.equal(second['policy']['state_vec'][:,32:35],first['policy']['state_vec'][:,45:48]))
        self.assertTrue(torch.equal(e._command_b,second['policy']['state_vec'][:,45:48]))

    def test_default_path_identity_and_unchanged_privileged_critic(self):
        e=Env();out=e._get_observations()
        ranges,_,_=extract_lidar_history(out['policy']['lidar_grid'])
        self.assertTrue(torch.equal(ranges[:,-1].reshape(2,-1),e.ranges))
        self.assertTrue(torch.equal(out['critic'],e.clearance))
        self.assertEqual(set(out['policy']),{'lidar_grid','state_vec'})

    def test_actual_observation_path_delivers_delay_and_drop_to_shared_history(self):
        e=Env();e.install(drop_steps=[2]);e.reset()
        first=e._get_observations();self.assertFalse(bool(e.rows[0][1]['frame_available'].any()))
        e.step(3.);dropped=e.step(4.)
        ranges,valid,_=extract_lidar_history(dropped['policy']['lidar_grid'])
        self.assertTrue(torch.equal(ranges[:,-1].reshape(2,-1),torch.full_like(e.ranges,2.)))
        self.assertTrue(torch.equal(e.rows[2][1]['acquisition_time'],torch.zeros(2,1)))
        self.assertTrue(torch.equal(dropped['critic'],e.clearance))
        final=e.step(5.);self.assertEqual([r[0] for r in e.rows],[0,1,2,3])
        self.assertTrue(torch.equal(e.rows[3][1]['delivered_ranges'],torch.full_like(e.ranges,4.)))
        e.finish_r6_measurements()

    def test_full_reset_required_and_partial_reset_clears_only_selected_delay(self):
        e=Env();e.install()
        with self.assertRaisesRegex(RuntimeError,'reset all lanes'):e._get_observations()
        e.reset();e._get_observations();e.step(3.)
        # Automatic reset at the next policy boundary: other lanes advance.
        e.episode_length_buf+=1;e.reset(torch.tensor([0]));e._get_observations()
        self.assertFalse(bool(e.rows[-1][1]['frame_available'][0]))
        self.assertTrue(bool(e.rows[-1][1]['frame_available'][1]))

    def test_audit_sink_cannot_mutate_actor_frame(self):
        e=Env();e.install(delay_cycles=0,audit_sink=lambda sequence,frame:frame['delivered_ranges'].zero_())
        e.reset();out=e._get_observations()
        ranges,_,_=extract_lidar_history(out['policy']['lidar_grid'])
        self.assertTrue(torch.equal(ranges[:,-1].reshape(2,-1),e.ranges))

    def test_sink_failure_poison_and_frame_quota(self):
        e=Env();e.install(audit_sink=lambda s,f:(_ for _ in ()).throw(OSError('disk quota')));e.reset()
        with self.assertRaises(OSError):e._get_observations()
        with self.assertRaisesRegex(RuntimeError,'failed measurement'):e._get_observations()
        with self.assertRaisesRegex(RuntimeError,'incomplete'):e.finish_r6_measurements()
        e=Env();e.install(frame_limit=1);e.reset();e._get_observations()
        with self.assertRaisesRegex(RuntimeError,'frame quota'):e.step(3.)

    def test_original_suite_guard_and_dormant_method_resolution(self):
        e=Env();e._h1_evaluation_case='W'
        with self.assertRaisesRegex(ValueError,'original suite'):e.install()
        e._h1_evaluation_case=None;e.install();e.reset();e._h1_evaluation_case='W'
        with self.assertRaisesRegex(RuntimeError,'contaminate'):e._get_observations()
        text=(ROOT/'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py').read_text()
        c=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='UnifiedH1DirectEnv')
        self.assertEqual([n.id for n in c.bases],['R6PhysicsMixin','R6MeasurementMixin','TaslabDirectEnv'])
        reset=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_reset_idx')
        self.assertIn('_reset_policy_sensor_frame',ast.unparse(reset))


if __name__=='__main__':unittest.main()
