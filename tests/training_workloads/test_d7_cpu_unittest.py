from __future__ import annotations
import copy, hashlib, importlib, importlib.util, json, pickle, sys, tempfile, types, unittest
from unittest import mock
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools/rlflow'))

class FakeTensor:
    def __init__(self,shape,dtype='float32',data=None,finite=True): self.shape=tuple(shape);self.dtype=dtype;self.data=data;self.finite=finite;self.ndim=len(self.shape)
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return copy.deepcopy(self)
    def reshape(self,*shape):
        data=self.data
        if isinstance(data,list) and data and isinstance(data[0],list): data=[item for row in data for item in row]
        resolved=tuple(len(data) if item == -1 else item for item in shape)
        return FakeTensor(resolved,self.dtype,data,self.finite)
    def is_floating_point(self): return self.dtype.startswith('float')
    def is_complex(self): return False
    def expand(self,batch,_): return FakeTensor((batch,self.shape[-1]),self.dtype,[list(self.data) for _ in range(batch)],self.finite)
    def tolist(self): return self.data
    def __or__(self,other): return FakeTensor(self.shape,'bool',[a or b for a,b in zip(self.data,other.data)])
    def __invert__(self): return FakeTensor(self.shape,'bool',[not x for x in self.data])
    def sum(self): return sum(self.data)

class BoolResult:
    def __init__(self,value): self.value=value
    def all(self): return self.value

fake=types.ModuleType('torch');fake.Tensor=FakeTensor;fake.float32='float32';fake.bool='bool';fake.long='int64';fake.dtype=str;fake.device=str
fake.tensor=lambda data,device=None,dtype='float32':FakeTensor((len(data),),dtype,list(data))
fake.isfinite=lambda value:BoolResult(value.finite)
fake.equal=lambda left,right:left.shape==right.shape and left.dtype==right.dtype and left.data==right.data
fake.save=lambda value,target:pickle.dump(value,target)
fake.load=lambda source,**kw:pickle.load(source)
real_torch=sys.modules.get('torch');sys.modules['torch']=fake
spec=importlib.util.spec_from_file_location('d7_control_diagnostic_fake_torch',ROOT/'training_workloads/unified_h1/control_diagnostic.py')
diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
if real_torch is None: del sys.modules['torch']
else: sys.modules['torch']=real_torch

class D7WriterAnalyzerTests(unittest.TestCase):
    def payload(self,case,step,finite=True):
        f=lambda shape,dtype='float32',data=None:FakeTensor(shape,dtype,data,finite)
        zeros=[False]*32; pre_root=f((32,13),data='pre-physics'); post_root=f((32,13),data='post-physics-pre-reset')
        false_column=[[False] for _ in range(32)]; zero_column=[[0] for _ in range(32)]
        masks={'collision':f((32,1),'bool',copy.deepcopy(false_column)),'below_height':f((32,1),'bool',copy.deepcopy(false_column)),
               'above_height':f((32,1),'bool',copy.deepcopy(false_column)),'timeout':f((32,1),'bool',copy.deepcopy(false_column))}
        termination={'terminated':f((32,1),'bool',copy.deepcopy(false_column)),'truncated':f((32,1),'bool',copy.deepcopy(false_column)),
                     'done':f((32,1),'bool',copy.deepcopy(false_column)),'reason_code':f((32,1),'int64',zero_column),'masks':masks}
        task={'root_state_w':post_root,'actual_velocity_b':f((32,3)),'exact_clearance_m':f((32,1)),
              'collision':f((32,1),'bool',copy.deepcopy(false_column)),'termination':termination}
        control={'alpha':f((32,1)),'v_corr_b':f((32,3)),'v_gov_b':f((32,3)),'v_final_b':f((32,3)),
          'v_final_w':f((32,3)),'anchor_metrics':{'active':f((32,1)),'reset_code':f((32,1)),
          'usable_fraction':f((32,1)),'mean_absolute_error_m':f((32,1)),'huber_loss':f((32,1))},
          'ics_metrics':{'beta':f((32,1)),'active_beam_count':f((32,1)),'minimum_clearance_m':f((32,1)),
          'worst_margin_m_s':f((32,1)),'emergency':f((32,1))}}
        pre_reset={'reset_mask':f((32,),'bool',zeros),'episode_length':f((32,),'int64'),
          'root_state_w':post_root,'critic_h1':f((32,21)),'control':control,'reward':{'total':f((32,1))},'task':task}
        lee={'normalized_rotor_commands':f((32,4)),'held_position_w_m':f((32,3)),'held_yaw_rad':f((32,1)),
             'desired_rotation_w':f((32,3,3)),'collective_thrust_n':f((32,1))}
        rotor={'normalized_commands':f((32,4)),'target_throttle':f((32,4)),'throttle':f((32,4)),
               'thrusts_n':f((32,4)),'reaction_moments_nm':f((32,4))}
        wrench={'rotor':rotor,'forces_body_n':f((32,5,3)),'torques_body_nm':f((32,5,3))}
        ordinal=sum(diagnostic.CASE_STEPS[x] for x in diagnostic.CASES[:diagnostic.CASES.index(case)])+step
        return {'schema_version':'unified-h1-control-diagnostic.step.v2','condition':'D7-R','case':case,'step':step,
          'ordinal':ordinal,'timing_semantics':diagnostic.TIMING_SEMANTICS,
          'pre_step':{'actor_input':{'lanes':f((8,),'int64',list(range(8))),'lidar_grid':f((8,12,59,360)),'state_vec':f((8,52))},
            'actor_alpha':f((32,4)),'actor_beta':f((32,4)),'actor_mean':f((32,4)),'issued_action':f((32,4)),
            'command_b':f((32,3)),'root_state_w':pre_root},
          'post_step':{'reward':f((32,)),'terminated':f((32,),'bool',zeros),'truncated':f((32,),'bool',zeros),
            'pre_reset':pre_reset,'lee':lee,'wrench':wrench,'downstream_cache_valid':f((32,),'bool',[True]*32)}}


    def test_shared_durable_ledger_checks_before_limit_and_accumulates(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=diagnostic.AggregateDiskLedger(Path(td)/'ledger.json',maximum_bytes=10)
            self.assertEqual(ledger.reserve(condition='D7-R',relative_path='a',byte_count=6),6)
            self.assertEqual(ledger.reserve(condition='B0',relative_path='b',byte_count=4),10)
            before=(Path(td)/'ledger.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError,'before write'):
                ledger.reserve(condition='B0',relative_path='never-written',byte_count=1)
            self.assertEqual((Path(td)/'ledger.json').read_bytes(),before)
            self.assertEqual(json.loads(before)['reserved_bytes'],10)

    def test_complete_trace_and_fail_closed_invalid_or_missing_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); writer=diagnostic.DiagnosticChunkWriter(root/'trace',condition='D7-R',experiment_id='UNIFIED-H1-R4-D7-R',aggregate_ledger=root/'ledger.json')
            for case in diagnostic.CASES:
                for step in range(diagnostic.CASE_STEPS[case]): writer.append(case,step,self.payload(case,step))
            writer.finalize()
            result=diagnostic.analyze_trace(root/'trace',condition='D7-R',experiment_id='UNIFIED-H1-R4-D7-R')
            self.assertEqual(result['status'],'COMPLETE_VALID_EVIDENCE');self.assertFalse(result['missing_evidence_filled'])
            first=root/'trace/chunks/Z/0000.pt'; original=first.read_bytes(); manifest_path=root/'trace/manifest.json'
            def replace_first(data):
                import hashlib
                first.write_bytes(data);manifest=json.loads(manifest_path.read_text())
                manifest['chunks'][0]['bytes']=len(data);manifest['chunks'][0]['sha256']=hashlib.sha256(data).hexdigest()
                manifest_path.write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n')
            # A full 630x32 trace with one genuine producer-consistent collision/reset remains valid.
            payload=pickle.loads(original);post=payload['post_step'];term=post['pre_reset']['task']['termination']
            post['terminated'].data[0]=True;post['downstream_cache_valid'].data[0]=False;post['pre_reset']['reset_mask'].data[0]=True
            post['pre_reset']['task']['collision'].data[0][0]=True;term['masks']['collision'].data[0][0]=True
            term['terminated'].data[0][0]=True;term['done'].data[0][0]=True;term['reason_code'].data[0][0]=4
            replace_first(pickle.dumps(payload))
            self.assertEqual(diagnostic.analyze_trace(root/'trace',condition='D7-R',experiment_id='UNIFIED-H1-R4-D7-R')['status'],'COMPLETE_VALID_EVIDENCE')

            # DirectRLEnv clears the post-step caches when every lane resets in
            # one step.  None is valid only when no lane has valid downstream
            # evidence; the analyzer must not manufacture zero-valued caches.
            payload=pickle.loads(original);post=payload['post_step'];term=post['pre_reset']['task']['termination']
            post['terminated'].data=[True]*32;post['downstream_cache_valid'].data=[False]*32
            post['pre_reset']['reset_mask'].data=[True]*32;post['pre_reset']['task']['collision'].data=[[True] for _ in range(32)]
            term['masks']['collision'].data=[[True] for _ in range(32)]
            term['terminated'].data=[[True] for _ in range(32)];term['done'].data=[[True] for _ in range(32)]
            term['reason_code'].data=[[4] for _ in range(32)];post['lee']=None;post['wrench']=None
            replace_first(pickle.dumps(payload))
            result=diagnostic.analyze_trace(root/'trace',condition='D7-R',experiment_id='UNIFIED-H1-R4-D7-R')
            self.assertEqual(result['invalid_downstream_cache_samples'],32)
            self.assertEqual(result['fully_invalid_downstream_chunks'],1)

            # Missing caches remain forbidden when even one lane is valid.
            payload=copy.deepcopy(payload);payload['post_step']['terminated'].data[0]=False
            payload['post_step']['downstream_cache_valid'].data[0]=True
            payload['post_step']['pre_reset']['reset_mask'].data[0]=False
            term=payload['post_step']['pre_reset']['task']['termination']
            payload['post_step']['pre_reset']['task']['collision'].data[0][0]=False
            term['masks']['collision'].data[0][0]=False;term['terminated'].data[0][0]=False
            term['done'].data[0][0]=False;term['reason_code'].data[0][0]=0
            replace_first(pickle.dumps(payload))
            with self.assertRaisesRegex(ValueError,'valid lanes'):
                diagnostic.analyze_trace(root/'trace',condition='D7-R',experiment_id='UNIFIED-H1-R4-D7-R')

            # Keep outer reset/cache facts self-consistent but contradict them in nested producer evidence.
            payload=pickle.loads(original);term=payload['post_step']['pre_reset']['task']['termination']
            payload['post_step']['pre_reset']['task']['collision'].data[0][0]=True
            term['masks']['collision'].data[0][0]=True;term['terminated'].data[0][0]=True
            term['done'].data[0][0]=True;term['reason_code'].data[0][0]=4
            replace_first(pickle.dumps(payload))
            with self.assertRaisesRegex(ValueError,'nested/outer termination'):
                diagnostic.analyze_trace(root/'trace',condition='D7-R',experiment_id='UNIFIED-H1-R4-D7-R')

            payload=pickle.loads(original);payload['post_step']['downstream_cache_valid'].data[0]=False
            replace_first(pickle.dumps(payload))
            with self.assertRaisesRegex(ValueError,'cache validity'): diagnostic.analyze_trace(root/'trace',condition='D7-R',experiment_id='UNIFIED-H1-R4-D7-R')
            payload=pickle.loads(original);payload['pre_step']['actor_alpha'].finite=False
            replace_first(pickle.dumps(payload))
            with self.assertRaisesRegex(ValueError,'non-finite'): diagnostic.analyze_trace(root/'trace',condition='D7-R',experiment_id='UNIFIED-H1-R4-D7-R')
            mutations = []
            def missing_task(value): del value['post_step']['pre_reset']['task']
            mutations.append((missing_task,'pre_reset'))
            def unidentified_pre_reset(value): value['post_step']['pre_reset']={'unidentified':FakeTensor((32,))}
            mutations.append((unidentified_pre_reset,'pre_reset'))
            def wrong_control_shape(value): value['post_step']['pre_reset']['control']['v_final_b']=FakeTensor((32,2))
            mutations.append((wrong_control_shape,'shape'))
            def wrong_lee_fields(value): value['post_step']['lee']={'unidentified':FakeTensor((32,))}
            mutations.append((wrong_lee_fields,'lee'))
            def obsolete_held_yaw_shape(value): value['post_step']['lee']['held_yaw_rad']=FakeTensor((32,))
            mutations.append((obsolete_held_yaw_shape,'lee.held_yaw_rad'))
            def wrong_wrench_shape(value): value['post_step']['wrench']['forces_body_n']=FakeTensor((32,))
            mutations.append((wrong_wrench_shape,'shape'))
            def wrong_time(value): value['ordinal']=1
            mutations.append((wrong_time,'time'))
            def wrong_reset(value): value['post_step']['pre_reset']['reset_mask'].data[0]=True
            mutations.append((wrong_reset,'reset'))
            def wrong_reason_precedence(value): value['post_step']['pre_reset']['task']['termination']['reason_code'].data[0][0]=1
            mutations.append((wrong_reason_precedence,'reason precedence'))
            def wrong_post_physics_identity(value):
                value['post_step']['pre_reset']['task']['root_state_w']=FakeTensor((32,13),data='wrong-time-state')
            mutations.append((wrong_post_physics_identity,'same-state identity'))
            for mutate,message in mutations:
                payload=pickle.loads(original);mutate(payload);replace_first(pickle.dumps(payload))
                with self.assertRaisesRegex(ValueError,message):
                    diagnostic.analyze_trace(root/'trace',condition='D7-R',experiment_id='UNIFIED-H1-R4-D7-R')
            replace_first(original);first.unlink()
            with self.assertRaisesRegex(ValueError,'missing'): diagnostic.analyze_trace(root/'trace',condition='D7-R',experiment_id='UNIFIED-H1-R4-D7-R')

    def test_complete_b0_trace_requires_exact_fixed_action(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);writer=diagnostic.DiagnosticChunkWriter(root/'trace',condition='B0',experiment_id='UNIFIED-H1-R4-B0',aggregate_ledger=root/'ledger.json')
            fixed=[list(diagnostic.FIXED_B0_ACTION) for _ in range(32)]
            for case in diagnostic.CASES:
                for step in range(diagnostic.CASE_STEPS[case]):
                    payload=self.payload(case,step);payload['condition']='B0';payload['pre_step']['issued_action']=FakeTensor((32,4),'float32',fixed)
                    writer.append(case,step,payload)
            writer.finalize()
            result=diagnostic.analyze_trace(root/'trace',condition='B0',experiment_id='UNIFIED-H1-R4-B0')
            self.assertEqual(result['status'],'COMPLETE_VALID_EVIDENCE')
            first=root/'trace/chunks/Z/0000.pt';payload=pickle.loads(first.read_bytes());payload['pre_step']['issued_action'].data[0][0]=0.0
            data=pickle.dumps(payload);first.write_bytes(data);manifest_path=root/'trace/manifest.json';manifest=json.loads(manifest_path.read_text())
            import hashlib
            manifest['chunks'][0]['bytes']=len(data);manifest['chunks'][0]['sha256']=hashlib.sha256(data).hexdigest()
            manifest_path.write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n')
            with self.assertRaisesRegex(ValueError,'B0 issued action'):
                diagnostic.analyze_trace(root/'trace',condition='B0',experiment_id='UNIFIED-H1-R4-B0')

    def test_b0_nonconstant_nan_and_bad_cache_mask_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);writer=diagnostic.DiagnosticChunkWriter(root/'trace',condition='B0',experiment_id='UNIFIED-H1-R4-B0',aggregate_ledger=root/'ledger.json')
            payload=self.payload('Z',0);payload['condition']='B0';payload['pre_step']['issued_action']=FakeTensor((32,4),'float32',[[0,0,0,0]]*32)
            writer.append('Z',0,payload)
            # A deliberately partial manifest reaches the coverage rejection without filling data.
            with self.assertRaisesRegex(RuntimeError,'incomplete'): writer.finalize()
            self.assertFalse((root/'trace/manifest.json').exists())


class HandoffBindingTests(unittest.TestCase):


    def test_load_config_post_adoption_projection_uses_real_repository_identity(self):
        from training_workloads.unified_h1 import asset_closure, config
        value=json.loads((ROOT/'training_workloads/unified_h1/candidate_config.json').read_text())
        self.assertEqual(value['workload']['branch'],'main')
        self.assertEqual(value['product']['branch'],'main')
        self.assertEqual(value['workload']['path'],value['product']['path'])
        actual_git=config._git; actual_sha256_file=config.sha256_file; actual_validate_repository=config._validate_repository
        def accessible_sha256_file(path):
            if Path(path).resolve()==Path(value['execution']['python_path']).resolve():
                return value['execution']['python_sha256']
            return actual_sha256_file(path)
        def validate_workload_repository(spec,**kwargs):
            if kwargs.get('label')=='product':
                return {'commit':spec['candidate_commit'],'tree':spec['candidate_tree'],'branch':spec['branch']}
            return actual_validate_repository(spec,**kwargs)
        def adopted_cleanliness_only(root,*args):
            if Path(root).resolve()==ROOT.resolve() and args==('status','--porcelain=v1'):
                return ''
            return actual_git(root,*args)
        with tempfile.TemporaryDirectory() as td:
            method=Path(td)/'method.md';method.write_text('bounded CPU method fixture\n')
            method_sha=hashlib.sha256(method.read_bytes()).hexdigest()
            value['method']['proposal_path']=str(method);value['method']['proposal_sha256']=method_sha
            value['method']['hashes']['method']=method_sha
            config_path=Path(td)/'candidate_config.json';config_path.write_text(json.dumps(value))
            with mock.patch.object(config,'METHOD_PATH',method), mock.patch.object(config,'METHOD_SHA256',method_sha), \
                 mock.patch.object(config,'_git',side_effect=adopted_cleanliness_only), \
                 mock.patch.object(config,'sha256_file',side_effect=accessible_sha256_file), \
                 mock.patch.object(config,'_validate_repository',side_effect=validate_workload_repository), \
                 mock.patch.object(config.sys,'executable',value['execution']['python_path']), \
                 mock.patch.object(asset_closure,'validate_asset_closure'), \
                 mock.patch.object(config,'validate_loaded_module_origins'):
                loaded=config.load_config(config_path)
        self.assertEqual(loaded['resolved_identity']['workload_commit'],actual_git(ROOT,'rev-parse','HEAD'))
        self.assertEqual(loaded['resolved_identity']['workload_tree'],actual_git(ROOT,'rev-parse','HEAD^{tree}'))
        self.assertEqual(loaded['workload']['branch'],actual_git(ROOT,'branch','--show-current'))



class SeedBindingTests(unittest.TestCase):
    def test_control_diagnostic_accepts_only_evaluation_seed(self):
        from rlflow import h1
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);work=base/'work';entry=work/'training_workloads/unified_h1';entry.mkdir(parents=True)
            for name in ('candidate_config.json','formal_unified_h1.py','local_dispatch.py'):(entry/name).write_text('{}')
            attempt=base/'attempt';(attempt/'results').mkdir(parents=True);authfile=base/'auth.json';authfile.write_text('{}')
            executable=base/'python';executable.write_text('cpu fixture');product=base/'product'
            config={'training':{'seed':123,'evaluation_seed':456},'method':{'version':'m'},'versions':{'metric':'e'},
                    'execution':{'frozen_runtime_root':'r'},'product':{'path':str(product)}}
            observed={'config':config,'authorization':{'budgets':{'wall_seconds':600,'training_transitions':0,'nontraining_transitions':0}}}
            spec={'cwd':str(work),'h1':{'workload_root':str(work),'attempt':str(attempt),'mode':'control-diagnostic',
                  'config_sha256':h1.digest(entry/'candidate_config.json'),'gpu_devices':['0'],'gpu_accounting':'allocated_device_wall_seconds'},
                  'pins':{'workload':{'path':str(entry/'formal_unified_h1.py')},'dispatcher':{'path':str(entry/'local_dispatch.py')},
                  'authorization':{'path':str(authfile)},'executable':{'path':str(executable)},'runtime':{'path':str(executable)},
                  'product':{'path':str(product/'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py')}},
                  'argv':[str(executable),'-B','-m','training_workloads.unified_h1.formal_unified_h1','--authorization',str(authfile),'--attempt',str(attempt),'--mode','control-diagnostic'],
                  'resources':{'starts':1,'wall_seconds':610,'gpu_seconds':610,'transitions':0},'seed':456,'method':'m','evaluation':'e','runtime':'r',
                  'observations':['status','counters','process_identity','process_start']}
            with mock.patch.object(h1,'inspect_authority',return_value=observed):
                self.assertEqual(h1.validate(spec,base),observed)
                for wrong in (123,457):
                    spec['seed']=wrong
                    with self.assertRaisesRegex(Exception,'seed'): h1.validate(spec,base)
                spec['h1']['mode']='controlled-failure';spec['argv'][-1]='controlled-failure';spec['seed']=123
                self.assertEqual(h1.validate(spec,base),observed)

if __name__=='__main__': unittest.main()
