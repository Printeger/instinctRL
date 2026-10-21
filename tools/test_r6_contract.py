"""CPU registration counterexamples. Fixture reviews are not live approvals."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from training_workloads.unified_h1.robustness_contract import MODE,validate_parameters,required_budget
from training_workloads.unified_h1 import research_contract
from training_workloads.unified_h1.robustness_contract import validate_compatibility

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/experiments/r6-robustness-auto'

def pin(path):return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


class ContractTests(unittest.TestCase):
    def test_lifecycle_dispatches_only_to_registered_r6_wrapper(self):
        from training_workloads.unified_h1.live_adapter import _dispatch
        backend=SimpleNamespace()
        with patch('training_workloads.unified_h1.robustness_contract.evaluate_registered') as run:
            _dispatch(backend,{'mode':MODE,'parameters':{}})
            run.assert_called_once_with(backend)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        def write(name,value):
            path=self.root/name;path.write_text(json.dumps(value));return pin(path)
        self.write=write
        protocol=json.loads((BASE/'TEST_PROTOCOL.json').read_text());protocol['frozen']=True
        protocol_pin=write('fixture-protocol.json',protocol)
        capacity=write('fixture-capacity.json',dict(capacity_pass=True,protocol_sha256=protocol_pin['sha256'],maximum_job_output_bytes=40*1024**3,maximum_job_wall_seconds=3600,fixture_only=True))
        review=write('fixture-review.json',dict(decision='PASS_R6_PROTOCOL',protocol_sha256=protocol_pin['sha256'],reviewer='CPU_FIXTURE_NOT_LIVE_REVIEW',must_fix=[],capacity_report=capacity))
        ids=[b['id'] for b in protocol['blocks']]
        self.p=dict(experiment_id='R6-CPU-FIXTURE',group='R1',training_seed=20260908,evaluation_seed=20260909,
            output_cap_bytes=40*1024**3,wall_seconds=3600,master_authority=pin(BASE/'MASTER_AUTHORITY.json'),
            baseline_bindings=pin(BASE/'BASELINE_BINDINGS.json'),checkpoint_compatibility=write('fixture-compatibility.json',{}),
            protocol_review=review,r6=dict(protocol=protocol_pin,block_ids=ids,run_id='CPU-FIXTURE',seed=17,split='development',expected_mass_kg=1.181517923,
                output_quotas={key:dict(range_bytes=2*1024**3,audit_bytes=16*1024**2) for key in ids}))
        self.card()
    def tearDown(self):self.tmp.cleanup()
    def card(self):
        payload={k:v for k,v in self.p.items() if k!='experiment_card'}
        self.p['experiment_card']=self.write('fixture-card.json',dict(experiment_id=self.p['experiment_id'],profiles={'R1':payload}))
    def test_full_baseline_budget_and_research_dispatch(self):
        self.assertEqual(validate_parameters(MODE,self.p),self.p)
        b=research_contract.required_budget(MODE,self.p)
        self.assertEqual(b['nontraining_transitions'],193600);self.assertEqual(b['explicit_resets'],21)
        self.assertEqual([b[k] for k in ('training_transitions','optimizer_updates','checkpoint_writes')],[0,0,0])
        self.assertEqual(b['simulator_initializations'],1)
    def test_original_seed_and_baseline_cannot_be_relabelled(self):
        self.p['evaluation_seed']=20261020;self.card()
        with self.assertRaisesRegex(PermissionError,'seed identity'):validate_parameters(MODE,self.p)
        self.p['evaluation_seed']=20260909;self.p['baseline_bindings']['sha256']='0'*64
        with self.assertRaisesRegex(PermissionError,'baseline bindings'):validate_parameters(MODE,self.p)

    def test_native_volume_requires_exact_card_and_runtime_validation(self):
        self.p['native_volume']={'path':'/fixture/volume.json','sha256':'1'*64}
        self.p['extension_inputs']={'path':'/fixture/extensions.json','sha256':'2'*64}
        self.p['result_volume']={'path':'/fixture/result.json','sha256':'3'*64}
        with patch('training_workloads.unified_h1.native_volume.validate',return_value={'fixture':True}) as check, patch('training_workloads.unified_h1.extension_inputs.validate',return_value='/fixture/installed'), patch('training_workloads.unified_h1.result_volume.validate',return_value={'fixture':True}) as result_check:
            with self.assertRaisesRegex(PermissionError,'card/profile mismatch'):validate_parameters(MODE,self.p)
            self.card()
            self.assertEqual(validate_parameters(MODE,self.p),self.p)
            check.assert_called_with(self.p['native_volume'])
            payload=64*1024**2+sum(sum(q.values())+16*1024**2 for q in self.p['r6']['output_quotas'].values())
            result_check.assert_called_with(self.p['result_volume'],output_cap_bytes=self.p['output_cap_bytes'],expected_target=None,minimum_free_bytes=payload)
            # A large nominal job cap cannot override insufficient mounted space.
            result_check.side_effect=ValueError('RESULT_FREE_SPACE')
            with self.assertRaisesRegex(ValueError,'RESULT_FREE_SPACE'):validate_parameters(MODE,self.p)
        with patch('training_workloads.unified_h1.native_volume.validate',side_effect=ValueError('MOUNT_ABSENT')):
            with self.assertRaisesRegex(ValueError,'MOUNT_ABSENT'):validate_parameters(MODE,self.p)

    def test_fixed_dual_volume_payload_covers_all_blocks_without_old_double_deduction(self):
        payload=64*1024**2+sum(sum(q.values())+16*1024**2 for q in self.p['r6']['output_quotas'].values())
        self.p.update(native_volume={'path':'/fixture/native','sha256':'1'*64},
                      result_volume={'path':'/fixture/results','sha256':'2'*64},
                      extension_inputs={'path':'/fixture/extensions','sha256':'3'*64},
                      output_cap_bytes=payload+1610612736+64*1024**2+96*1024**2)
        self.card()
        with patch('training_workloads.unified_h1.native_volume.validate'),patch('training_workloads.unified_h1.extension_inputs.validate'),patch('training_workloads.unified_h1.result_volume.validate',return_value={'payload_cap_bytes':payload}) as volume:
            self.assertEqual(validate_parameters(MODE,self.p),self.p)
            volume.return_value={'payload_cap_bytes':payload-1}
            with self.assertRaisesRegex(PermissionError,'components exceed'):validate_parameters(MODE,self.p)
    def test_baseline_cannot_omit_failed_domain_or_duplicate_blocks(self):
        self.p['r6']['block_ids'].pop();self.card()
        with self.assertRaisesRegex(PermissionError,'all frozen blocks'):validate_parameters(MODE,self.p)
    def test_no_training_mode_and_no_unknown_fields(self):
        with self.assertRaisesRegex(PermissionError,'fields/mode'):validate_parameters('research-r6-training',self.p)
        self.p['max_updates']=128
        with self.assertRaisesRegex(PermissionError,'fields/mode'):validate_parameters(MODE,self.p)
    def test_initial_baseline_entry_cannot_reveal_final_holdout(self):
        self.p['r6']['split']='final';self.card()
        with self.assertRaisesRegex(PermissionError,'random stream'):validate_parameters(MODE,self.p)
    def test_component_and_kernel_file_limits(self):
        self.p['output_cap_bytes']=4*1024**3;self.card()
        with self.assertRaisesRegex(PermissionError,'components exceed'):validate_parameters(MODE,self.p)
        self.p['output_cap_bytes']=40*1024**3
        self.p['r6']['output_quotas'][self.p['r6']['block_ids'][0]]['range_bytes']=2*1024**3+1;self.card()
        with self.assertRaisesRegex(PermissionError,'per-file ceiling'):validate_parameters(MODE,self.p)
    def test_capacity_or_independent_review_missing(self):
        self.p['protocol_review']=self.write('bad-review.json',dict(decision='PASS_CPU_SCOPE',reviewer='root',must_fix=[]))
        with self.assertRaisesRegex(PermissionError,'review missing'):validate_parameters(MODE,self.p)
    def test_mass_nonfinite_or_drift_rejected(self):
        for mass in (float('nan'),float('inf'),True,1.2):
            self.p['r6']['expected_mass_kg']=mass
            with self.assertRaises(PermissionError):validate_parameters(MODE,self.p)
    def test_exact_card_pin_drift_is_not_accepted(self):
        Path(self.p['experiment_card']['path']).write_text('{}')
        with self.assertRaisesRegex(PermissionError,'pin drift'):validate_parameters(MODE,self.p)

    def test_compatibility_rejects_original_metric_change(self):
        metric='training_workloads/unified_h1/evaluation.py'
        old=dict(versions={},actor={},training={},execution={},method=dict(version='same',proposal_sha256='same',hashes={}),
            product=dict(path=str(self.root),source_sha256={}),workload=dict(path=str(self.root),source_sha256={metric:'a'*64}))
        new=copy.deepcopy(old);new['workload']['source_sha256'][metric]='b'*64
        old_pin=self.write('old-config.json',old);new_pin=self.write('new-config.json',new)
        def identity(pin):return dict(product_commit='1'*40,product_tree='2'*40,workload_commit='3'*40,workload_tree='4'*40,config_sha256=pin['sha256'])
        origin=identity(old_pin);execution=identity(new_pin)
        record=dict(schema_version='r6-baseline-compatibility/1',origin_source_identity=origin,execution_source_identity=execution,
            origin_config=old_pin,execution_config=new_pin,changed_product_sources={},independent_review={})
        with self.assertRaisesRegex(PermissionError,'original evaluator source changed'):
            validate_compatibility(self.write('compatibility-metric-change.json',record),origin,{**new,'resolved_identity':execution})


    def test_newly_manifested_dependency_must_match_original_checkout(self):
        metric='training_workloads/unified_h1/evaluation.py'
        dependency='isaaclab-training/instinctlab/observations/audit.py'
        for name in (metric,dependency):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('CPU original bytes')
        metric_sha=pin(self.root/metric)['sha256'];dep_sha=pin(self.root/dependency)['sha256']
        old=dict(versions={},actor={},training={},execution={},method=dict(version='same',proposal_sha256='same',hashes={}),
                 product=dict(path=str(self.root),source_sha256={}),workload=dict(path=str(self.root),source_sha256={metric:metric_sha}))
        new=copy.deepcopy(old);new['product']['source_sha256'][dependency]=dep_sha
        old_pin=self.write('old-deps.json',old);new_pin=self.write('new-deps.json',new)
        def identity(p):return dict(product_commit='1'*40,product_tree='2'*40,workload_commit='3'*40,workload_tree='4'*40,config_sha256=p['sha256'])
        origin=identity(old_pin);execution=identity(new_pin)
        record=dict(schema_version='r6-baseline-compatibility/1',origin_source_identity=origin,execution_source_identity=execution,
                    origin_config=old_pin,execution_config=new_pin,changed_product_sources={dependency:dict(origin=None,execution=dep_sha)},independent_review={})
        record_pin=self.write('compat-deps.json',record)
        with patch('training_workloads.unified_h1.robustness_contract.accepted_review',return_value={}):
            self.assertEqual(validate_compatibility(record_pin,origin,{**new,'resolved_identity':execution}),origin)
            (self.root/dependency).write_text('changed source')
            with self.assertRaises((ValueError,PermissionError)):
                validate_compatibility(record_pin,origin,{**new,'resolved_identity':execution})

    def test_missing_or_changed_r6_resource_policy_is_rejected(self):
        protocol=json.loads(Path(self.p['r6']['protocol']['path']).read_text())
        changed=copy.deepcopy(protocol);changed.pop('resource_admission')
        self.p['r6']['protocol']=self.write('missing-resource-policy.json',changed);self.card()
        with self.assertRaisesRegex(PermissionError,'prelaunch resource policy differs'):
            validate_parameters(MODE,self.p)
        changed=copy.deepcopy(protocol);changed['resource_admission']['window_policy']['peak_utilization_max']=99
        self.p['r6']['protocol']=self.write('changed-resource-policy.json',changed);self.card()
        with self.assertRaisesRegex(PermissionError,'prelaunch resource policy differs'):
            validate_parameters(MODE,self.p)


if __name__=='__main__':unittest.main()
