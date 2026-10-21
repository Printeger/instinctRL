"""CPU clock/sample/launcher substitutes only; no resource tool or simulator."""
import copy
from functools import partial
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from training_workloads.unified_h1 import robustness_prelaunch as r
from training_workloads.unified_h1.resource_admission import select_gate
from training_workloads.scale_003 import resources


class PrelaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.exe=self.root/'fixture-executable';self.exe.write_text('CPU only; never executed')
        self.pins=dict(nvidia_smi_path=str(self.exe),nvidia_smi_sha256=hashlib.sha256(self.exe.read_bytes()).hexdigest())
        self.clock=100*10**9;self.calls=0;self.values=[3]*13

    def sleep(self,seconds):self.clock+=int(seconds*10**9)

    def sample(self):
        u=self.values[self.calls];self.calls+=1
        return dict(monotonic_ns=self.clock,gpu=dict(resources.GPU_IDENTITY,memory_used_mib=560,
            free_memory_mib=11667,utilization_percent=u),compute_apps=[])

    def gate(self,**kwargs):
        return r.run_gate(self.root,mode=kwargs.pop('mode','research-r6-evaluation'),attempt_id='CPU',authorization_sha256='a'*64,
            resource_pins=self.pins,sample_provider=kwargs.pop('sample_provider',self.sample),
            sleeper=self.sleep,clock_ns=lambda:self.clock,**kwargs)

    def test_complete_fresh_window_and_authorization_binding(self):
        self.values=[3]*6+[12]+[3]*6;result=self.gate()
        self.assertEqual(self.calls,13);self.assertEqual(result['window']['span_ns'],60*10**9)
        self.assertEqual(result['status'],'PASS');self.assertFalse(result['queue_receipt_used'])
        self.assertEqual(result['phase'],'ACTUAL_PRELAUNCH_FRESH_WINDOW')
        self.assertEqual(result['authorization_sha256'],'a'*64)

    def test_render_requires_the_same_complete_fresh_window(self):
        result=self.gate(mode='research-r6-video-replay')
        self.assertEqual(self.calls,13)
        self.assertEqual(result['window']['span_ns'],60*10**9)
        self.assertEqual(result['mode'],'research-r6-video-replay')
        self.assertFalse(result['queue_receipt_used'])

    def test_old_queue_samples_are_rejected_and_preserved(self):
        old=self.sample();self.calls=0;self.clock+=10**9
        with self.assertRaisesRegex(ValueError,'STALE_OR_FUTURE'):self.gate(sample_provider=lambda:old)
        fail=json.loads((self.root/'prelaunch_resource_gate_failure.json').read_text())
        self.assertEqual(fail['observed_window']['samples'],[old]);self.assertEqual(fail['completed_samples'],1)

    def test_future_timestamp_rejected(self):
        def future():
            sample=self.sample();sample['monotonic_ns']+=1;return sample
        with self.assertRaisesRegex(ValueError,'STALE_OR_FUTURE'):self.gate(sample_provider=future)

    def test_busy_full_window_preserved_and_launcher_not_constructed(self):
        self.values=[6]*13;launches=[]
        gate=partial(r.run_gate,resource_pins=self.pins,sample_provider=self.sample,sleeper=self.sleep,clock_ns=lambda:self.clock)
        with self.assertRaisesRegex(RuntimeError,'sustained'):
            resources.start_after_coexistence_gate(self.root,mode='research-r6-evaluation',attempt_id='CPU',authorization_sha256='a'*64,
                gate_runner=gate,launcher_factory=lambda p:launches.append(p))
        self.assertEqual(launches,[]);self.assertEqual(self.calls,13)
        fail=json.loads((self.root/'prelaunch_resource_gate_failure.json').read_text())
        self.assertEqual(len(fail['observed_window']['samples']),13)

    def test_only_fresh_pass_reaches_launcher_factory(self):
        gate=partial(r.run_gate,resource_pins=self.pins,sample_provider=self.sample,sleeper=self.sleep,clock_ns=lambda:self.clock)
        launches=[]
        def launch(path):launches.append(self.calls);return SimpleNamespace(app='CPU_APP_SUBSTITUTE')
        with patch.object(resources,'validate_shutdown_runtime_sources',return_value={}), \
             patch.object(resources,'_r6_capture_binding',return_value={'test_only':'CPU_CAPTURE_SUBSTITUTE'}):
            result,launcher,app=resources.start_after_coexistence_gate(self.root,mode='research-r6-evaluation',attempt_id='CPU',authorization_sha256='a'*64,
                gate_runner=gate,launcher_factory=launch,resource_pins=self.pins)
        self.assertEqual(launches,[13]);self.assertEqual(app,'CPU_APP_SUBSTITUTE')

    def test_partial_query_failure_preserved(self):
        def sample():
            if self.calls==2:raise OSError('CPU query failure')
            return self.sample()
        with self.assertRaisesRegex(OSError,'CPU query failure'):self.gate(sample_provider=sample)
        fail=json.loads((self.root/'prelaunch_resource_gate_failure.json').read_text())
        self.assertEqual(fail['completed_samples'],2);self.assertEqual(fail['error_type'],'OSError')

    def test_stop_before_collection_has_zero_queries(self):
        stop=self.root/'STOP';stop.touch()
        with self.assertRaisesRegex(RuntimeError,'STOP'):self.gate(stop_paths=[stop])
        self.assertEqual(self.calls,0)

    def test_stop_during_collection_preserves_exact_samples(self):
        stop=self.root/'STOP'
        def sample():
            value=self.sample()
            if self.calls==3:stop.touch()
            return value
        with self.assertRaisesRegex(RuntimeError,'STOP'):self.gate(stop_paths=[stop],sample_provider=sample)
        fail=json.loads((self.root/'prelaunch_resource_gate_failure.json').read_text());self.assertEqual(fail['completed_samples'],3)

    def test_collection_deadline_prevents_next_query(self):
        def slow():self.clock+=181*10**9;return self.sample()
        with self.assertRaisesRegex(TimeoutError,'DEADLINE'):self.gate(sample_provider=slow)
        self.assertEqual(self.calls,1)

    def test_r6_selection_cannot_fall_back_to_legacy_gate(self):
        sentinel=object();params={'master_authority':{'path':str(self.root/'docs/experiments/r6/master.json')}}
        with patch('training_workloads.unified_h1.robustness_contract.validate_parameters',return_value=params) as validate:
            selected=select_gate({'mode':'research-r6-evaluation','parameters':params},self.pins,sentinel)
        self.assertIs(selected.func,r.run_gate);validate.assert_called_once()
        self.assertIs(select_gate({'mode':'research-formal-evaluation','parameters':{}},self.pins,sentinel),sentinel)

    def test_missing_review_prevents_r6_gate_selection(self):
        with self.assertRaises(PermissionError):select_gate({'mode':'research-r6-evaluation','parameters':{}},self.pins,object())

    def test_bad_timestamp_failure_keeps_original_sample(self):
        value=self.sample();value.pop('monotonic_ns')
        with self.assertRaisesRegex(ValueError,'STALE_OR_FUTURE'):self.gate(sample_provider=lambda:value)
        fail=json.loads((self.root/'prelaunch_resource_gate_failure.json').read_text())
        self.assertEqual(fail['observed_window']['samples'],[value]);self.assertIsNone(fail['observed_window']['span_ns'])


    def test_real_collector_queries_share_remaining_deadline(self):
        times=iter([100*10**9,108*10**9,109*10**9]);timeouts=[]
        def query(argv,**kwargs):
            timeouts.append(kwargs['timeout'])
            text='0, NVIDIA GeForce RTX 5070, 580.173.02, 12227, 560, 3' if len(timeouts)==1 else ''
            return SimpleNamespace(returncode=0,stdout=text,stderr='')
        with patch.object(resources.time,'monotonic_ns',side_effect=lambda:next(times)),patch.object(resources.subprocess,'run',side_effect=query):
            sample=resources.collect_resource_sample(resource_pins=self.pins,deadline_monotonic_ns=110*10**9)
        self.assertEqual(timeouts,[10,2]);self.assertEqual(sample['monotonic_ns'],109*10**9)

    def test_deadline_before_second_query_prevents_it(self):
        times=iter([100*10**9,111*10**9]);calls=[]
        def query(argv,**kwargs):
            calls.append(argv);return SimpleNamespace(returncode=0,stdout='0, NVIDIA GeForce RTX 5070, 580.173.02, 12227, 560, 3',stderr='')
        with patch.object(resources.time,'monotonic_ns',side_effect=lambda:next(times)),patch.object(resources.subprocess,'run',side_effect=query):
            with self.assertRaisesRegex(TimeoutError,'deadline before query'):
                resources.collect_resource_sample(resource_pins=self.pins,deadline_monotonic_ns=110*10**9)
        self.assertEqual(len(calls),1)


    def test_legacy_three_sample_gate_keeps_original_threshold(self):
        pins=dict(self.pins,eval_002_gpu_gate_path=str(self.exe),eval_002_gpu_gate_sha256=self.pins['nvidia_smi_sha256'])
        samples=[]
        for i in range(3):
            if i:self.sleep(5)
            samples.append(self.sample())
        window=dict(sample_count=3,span_ns=10*10**9,samples=samples)
        result=resources.run_prelaunch_gate(self.root/'legacy',mode='training',attempt_id='CPU',authorization_sha256='b'*64,
            collector=lambda:window,resource_pins=pins)
        self.assertEqual(result['status'],'PASS')
        window['samples'][1]['gpu']['utilization_percent']=6
        with self.assertRaisesRegex(RuntimeError,'5 percent'):
            resources.run_prelaunch_gate(self.root/'legacy-fail',mode='training',attempt_id='CPU',authorization_sha256='b'*64,
                collector=lambda:window,resource_pins=pins)


if __name__=='__main__':unittest.main()
