"""Focused CPU positive/negative diagnostic checks; no simulator imports."""
from collections import OrderedDict
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'evaluation_workloads/eval_002'))
sys.path.insert(0,'/home/mint/rl_dev/worktrees/instinctRL-blackwell/isaaclab-training')
import torch
import scene
import constants as C
import diagnostic_evaluator as D
D.torch=torch
from semantic_validator import validate_semantic, validate_semantic_mirror, validate_closed_kit_log, KIT_SHUTDOWN_MARKERS
from checkpoint_audit import audit_checkpoint
from io_utils import canonical_bytes, canonical_value, atomic_canonical_replace
from summary import summarize
from instinctlab.evaluation.audit import build_evaluation_record
from instinctlab.evaluation.metrics import EVALUATION_INPUT_KEYS
from instinctlab.evaluation.runtime import SOURCE_MANIFEST
from instinctlab.observations.audit import audit_actor_observation
from instinctlab.observations.actor import ActorObservationHistory


def fixture(case='tracking'):
    t,n=256,2
    module=lambda h:OrderedDict(sha256=h,parameter_count=1,all_finite=True)
    opt=OrderedDict(sha256='d'*64,state_count=1,step_min=2000,step_max=2000,all_steps_equal=True)
    before=OrderedDict(actor=module(C.ACTOR_SHA256),critic=module('c'*64),actor_optimizer=copy.deepcopy(opt),critic_optimizer=copy.deepcopy(opt))
    inputs=OrderedDict()
    bools={'anchor_valid','clearance_valid','collision','ics_emergency','terminated','truncated','termination_collision','termination_below_height','termination_above_height','termination_timeout','observability_valid'}
    for k in EVALUATION_INPUT_KEYS:
        shape=(t,n,3) if k in EVALUATION_INPUT_KEYS[:5] else (t,n)
        dtype=torch.bool if k in bools else torch.long if k in ('termination_reason','observability_source') else torch.float32
        inputs[k]=torch.zeros(shape,dtype=dtype)
    inputs['commanded_velocity_b_m_s']=torch.tensor([scene.command_at(case,i) for i in range(t)]).unsqueeze(1).repeat(1,n,1)
    inputs['actual_velocity_b_m_s'].fill_(10.) # Deliberately poor finite policy.
    inputs['position_w_m'].fill_(1000.)
    inputs['minimum_clearance_m'].fill_(1.)
    inputs['clearance_valid'].fill_(True)
    inputs['ics_beta'].fill_(1.)
    inputs['observability_valid'].fill_(True)
    inputs['observability_value'].fill_(1.)
    inputs['observability_source'].fill_(2)
    rows=[]; actions=[]
    for i in range(t):
        state=torch.zeros(n,52); state[:,-7:-4]=inputs['commanded_velocity_b_m_s'][i]
        returned=state.clone();returned[:,-7:-4]=torch.tensor(scene.command_at(case,i+1)).repeat(n,1)
        action=torch.full((n,4),.5);actions.append(action)
        root=torch.zeros(n,13);root[:,:3]=inputs['position_w_m'][i]
        pre=dict(reset_mask=torch.zeros(n,dtype=torch.bool),episode_length=torch.full((n,),i+1,dtype=torch.long),
            task=dict(root_state_w=root,actual_velocity_b=inputs['actual_velocity_b_m_s'][i].clone(),exact_clearance_m=torch.ones(n,1),collision=torch.zeros(n,1,dtype=torch.bool),termination=dict(reason_code=torch.zeros(n,1,dtype=torch.long))),
            control=dict(v_final_b=torch.zeros(n,3)))
        rows.append(dict(global_index=i,command_b=inputs['commanded_velocity_b_m_s'][i].clone(),state_vec=state,returned_state_vec=returned,action=action,
            pre_reset=pre,returned_episode_length=torch.full((n,),i+1,dtype=torch.long),visibility=dict(floor=[1,1],x_wall=[1,1],y_wall=[1,1])))
    policy=OrderedDict(
        lidar_grid=torch.zeros(n,12,59,360), state_vec=torch.zeros(n,52)
    )
    provenance=OrderedDict(
        product_commit=C.PRODUCT_COMMIT,
        fixture_id=f'eval-002-floor-corner-{case}',
        generator_id='evaluation_workloads/eval_002/diagnostic_evaluator.py',
        seed=C.ENVIRONMENT_SEED,
    )
    canonical_record=build_evaluation_record(
        OrderedDict(policy=policy,evaluation=inputs),provenance
    ).canonical_json
    runtime=OrderedDict(contract=OrderedDict(num_envs=2,policy_steps=256,samples=512,rays_per_environment=21240,ray_samples=10874880),policy_hashes=['a'*64]*t,action_hashes=[D.tensor_hash(a) for a in actions],
        source_manifest=copy.deepcopy(SOURCE_MANIFEST),evaluation_inputs=inputs,reset_masks=torch.zeros(t,n,dtype=torch.bool),returned_episode_lengths=torch.arange(1,t+1).unsqueeze(1).repeat(1,n),
        observability_details=dict(numerical_rank=torch.full((t,n),3)),canonical_record=canonical_record)
    points,indices=scene.geometry()
    geom=dict(mesh_path=scene.MESH_PATH,points=points,indices=indices)
    geom['sha256']=hashlib.sha256(json.dumps(geom,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    resources=OrderedDict(device_name='NVIDIA GeForce RTX 5070',device_capability=[12,0],total_memory_bytes=12*1024**3,free_memory_bytes=2*1024**3,memory_allocated_bytes=1,max_memory_allocated_bytes=1)
    value=OrderedDict(schema_version='eval-002-formal-semantic.v2',scope=OrderedDict(environment='TaslabDirectEnv combined floor and corner',case=case,performance_gate=False),
      identity=OrderedDict(product_commit=C.PRODUCT_COMMIT,product_tree=C.PRODUCT_TREE,checkpoint_sha256=C.CHECKPOINT_SHA256,checkpoint_counter=250,source_sha256=C.SOURCE_SHA256),
      campaign=OrderedDict(action_mode=C.ACTION_MODE,environment_seed=C.ENVIRONMENT_SEED,inference_seed=C.INFERENCE_SEED,num_envs=2,policy_steps=256,decimation=20,physics_steps_per_environment=5120,environment_transitions=512,rays_per_environment_per_step=21240,ray_samples=10874880),
      checkpoint_state_before=before,checkpoint_state_after=copy.deepcopy(before),gradients=OrderedDict(actor_grad_tensor_count=0,critic_grad_tensor_count=0,global_gradient_enabled=False),
      training_claims=OrderedDict(training_performed=False,optimizer_updates=0,checkpoint_writes=0,one_million_training=False,merge=False,push=False),
      actor_boundary=OrderedDict(actor_keys=['lidar_grid','state_vec'],privileged_actor_input=False,action_source='actor(policy).mean_action',critic_influenced_action=False,evaluation_influenced_action=False,action_passed_unchanged=True))
    for name in ['action','alpha','beta','entropy','log_probability','reward']:
        value[name+'_diagnostics']=D.diagnostic_summary(actions,name)
    value.update(termination_reason_counts=torch.tensor([512,0,0,0,0]),reset_count=0,cuda_resources=OrderedDict(before=copy.deepcopy(resources),after=copy.deepcopy(resources)),raycaster_provenance=OrderedDict(),runtime_evaluation=runtime,geometry=geom,
        initial_observability=dict(numerical_rank=[3,3],sigma_min=[1.,1.],singular_values=[[1.,1.,1.]]*2,surface_hits=dict(floor=[1,1],x_wall=[1,1],y_wall=[1,1])),initial_rays_sha256='0'*64,
        rows=rows,resources_per_step=[copy.deepcopy(resources) for _ in range(t)],diagnostic_summary=summarize(runtime,rows),physics_accounting={})
    return value


class Tests(unittest.TestCase):
    def test_geometry_and_negative_surfaces(self):
        p,i=scene.geometry();scene.validate_geometry(p,i)
        for bad_p,bad_i in [(p[:1849],i[:10584]),(p,i[:-6]),(p,list(reversed(i)))]:
            with self.assertRaises(ValueError):scene.validate_geometry(bad_p,bad_i)

    def test_schedule_history_and_reset_boundaries(self):
        class Base:
            def __init__(self):
                self.device='cpu';self.num_envs=2;self._command_b=torch.zeros(2,3)
                self.history=ActorObservationHistory();self.calls=0;self.seen=[]
            def _get_observations(self):
                self.calls+=1
                self._policy=self.history.update(torch.ones(2,21240),torch.ones(2,21240,dtype=torch.bool),torch.full((2,1),self.schedule_index*.1),torch.full((2,1),self.schedule_index*.1),torch.zeros(2,3),torch.tensor([[0.,0.,-1.]]).repeat(2,1),self._command_b,torch.zeros(2,3))
                return self._policy
            def step(self,action):
                self.seen.append((self._command_b.clone(),self._policy['state_vec'].clone()))
                if self.schedule_index in (32,64,96,128,160,192,224):self.history.reset()
                return self._get_observations()
        class Env(scene.CommandBoundaryMixin,Base):
            diagnostic_case='tracking';schedule_index=0
        env=Env();obs=env._get_observations()
        for index in range(256):
            expected=torch.tensor(scene.command_at('tracking',index)).repeat(2,1)
            self.assertTrue(torch.equal(obs['state_vec'][:,-7:-4],expected));audit_actor_observation(obs)
            obs=env.step(None)
            self.assertTrue(torch.equal(env.seen[-1][0],expected))
            self.assertTrue(torch.equal(env.seen[-1][1][:,-7:-4],expected))
        self.assertEqual(env.calls,257)
        with self.assertRaises(RuntimeError):env.step(None)
        with self.assertRaises(ValueError):scene.command_at('tracking',257)
        obs['pose']=torch.zeros(2,3)
        with self.assertRaises(ValueError):audit_actor_observation(obs)

    def test_positive_poor_policy_and_unavailable(self):
        for case in C.RUN_IDS:
            v=fixture(case);validate_semantic(v,run_id=case)
            anchor=v['diagnostic_summary']['reductions'][0]['metrics'][3]
            self.assertFalse(anchor['available']);self.assertIsNone(anchor['value'])
            self.assertEqual(anchor['denominator'],0)
            canonical=json.loads(v['runtime_evaluation']['canonical_record'])
            if case=='station':
                unavailable=canonical['metrics']['command_preservation_ratio']
                self.assertEqual(
                    (unavailable['available'],unavailable['value'],unavailable['numerator'],
                     unavailable['denominator'],unavailable['valid_count']),
                    (False,None,None,0,0),
                )

    def test_canonical_metric_record_mutations(self):
        for case in C.RUN_IDS:
            original=fixture(case)
            canonical=original['runtime_evaluation']['canonical_record']

            changed=copy.deepcopy(original)
            record=json.loads(canonical,object_pairs_hook=OrderedDict)
            record['metrics']['tracking_rmse_m_s']['value']=999.0
            changed['runtime_evaluation']['canonical_record']=json.dumps(
                record,ensure_ascii=True,allow_nan=False,separators=(',',':')
            )
            with self.assertRaises(ValueError):validate_semantic(changed,run_id=case)

            changed=copy.deepcopy(original)
            record=json.loads(canonical,object_pairs_hook=OrderedDict)
            record['metrics']['tracking_rmse_m_s']['value']=float('nan')
            changed['runtime_evaluation']['canonical_record']=json.dumps(
                record,ensure_ascii=True,allow_nan=True,separators=(',',':')
            )
            with self.assertRaises(ValueError):validate_semantic(changed,run_id=case)

            changed=copy.deepcopy(original)
            record=json.loads(canonical,object_pairs_hook=OrderedDict)
            record['metrics']['invented_metric']=record['metrics'].pop('tracking_rmse_m_s')
            changed['runtime_evaluation']['canonical_record']=json.dumps(
                record,ensure_ascii=True,allow_nan=False,separators=(',',':')
            )
            with self.assertRaises(ValueError):validate_semantic(changed,run_id=case)

            changed=copy.deepcopy(original)
            metric=json.loads(canonical,object_pairs_hook=OrderedDict)[
                'metrics'
            ]['tracking_rmse_m_s']
            value_token='"value":'+json.dumps(metric['value'])
            duplicate='"value":999.0,'+value_token
            changed_record=canonical.replace(value_token,duplicate,1)
            self.assertNotEqual(changed_record,canonical)
            changed['runtime_evaluation']['canonical_record']=changed_record
            with self.assertRaises(ValueError):validate_semantic(changed,run_id=case)

    def test_mutations(self):
        v=fixture()
        mutations=[lambda x:x.pop('geometry'),lambda x:x['identity'].__setitem__('checkpoint_sha256','0'*64),
          lambda x:x['campaign'].__setitem__('environment_seed',1),lambda x:x['campaign'].__setitem__('num_envs',3),
          lambda x:x['campaign'].__setitem__('environment_transitions',513),lambda x:x['geometry'].__setitem__('mesh_path','/World/wrong'),
          lambda x:x['initial_observability']['surface_hits'].__setitem__('floor',[0,1]),
          lambda x:x['initial_observability'].__setitem__('numerical_rank',[2,3]),
          lambda x:x['rows'][32]['command_b'].zero_(),lambda x:x['rows'][31]['returned_state_vec'].zero_(),
          lambda x:x['rows'][3]['action'].fill_(2.),lambda x:x['rows'][3]['action'].fill_(float('nan')),
          lambda x:x['checkpoint_state_after']['actor'].__setitem__('sha256','0'*64),
          lambda x:x['checkpoint_state_after']['actor_optimizer'].__setitem__('step_max',2001),
          lambda x:x['actor_boundary'].__setitem__('privileged_actor_input',True),
          lambda x:x['resources_per_step'][0].__setitem__('free_memory_bytes',True),
          lambda x:x['resources_per_step'][0].__setitem__('free_memory_bytes',0),
          lambda x:x['diagnostic_summary']['reductions'][0]['metrics'][0].__setitem__('value',0.),
          lambda x:x['runtime_evaluation']['evaluation_inputs']['actual_velocity_b_m_s'].fill_(float('inf'))]
        for mutation in mutations:
            x=copy.deepcopy(v);mutation(x)
            with self.assertRaises((ValueError,KeyError,TypeError)):validate_semantic(x,run_id='tracking')

    def test_checkpoint_and_mirror(self):
        self.assertEqual(audit_checkpoint(C.CHECKPOINT_PATH)['status'],'PASS')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'mirror.json';value=OrderedDict(value=torch.tensor([1.]))
            p.write_bytes(canonical_bytes(canonical_value(value)));validate_semantic_mirror(value,p)
            p.write_bytes(b'{"value":0}\n')
            with self.assertRaises(ValueError):validate_semantic_mirror(value,p)
            p.write_bytes(b'{"value":0,"value":0}\n')
            with self.assertRaises(ValueError):validate_semantic_mirror(value,p)
            p.write_bytes(b'\n'.join(KIT_SHUTDOWN_MARKERS));validate_closed_kit_log(p)
            p.write_bytes(b'\n'.join(reversed(KIT_SHUTDOWN_MARKERS)))
            with self.assertRaises(ValueError):validate_closed_kit_log(p)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    output=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors))
    if 'EVAL_002_TEST_OUTPUT' in os.environ:
        from attempt_binding import binding_from_environment,bound_path
        binding=binding_from_environment()
        path=bound_path(binding,Path(os.environ['EVAL_002_TEST_OUTPUT']),'results/eval002_cpu_tests.json',must_exist=False)
        atomic_canonical_replace(path,output)
    print(json.dumps(output));raise SystemExit(0 if result.wasSuccessful() else 1)
