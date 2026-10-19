"""Fake executable only; actual subprocess capture/cleanup, no real GPU query."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tools.r6_recovery import Journal,RecoveryError,file_pin
from tools.r6_resource_queue import begin,finish
from tools.r6_resource_sampler import prepare,consume,CAP
from tools.r6_cpu_analysis import environment
from rlflow.jobs import owns

ROOT=Path(__file__).resolve().parents[1]
FAKE='''#!{python}
import os,sys,time
{body}
if '--query-gpu=' in sys.argv[1]:print('0, NVIDIA GeForce RTX 5070, 580.173.02, 12227, 560, 3')
else:print('3950, /usr/libexec/gnome-remote-desktop-daemon, 164')
'''


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.j=Journal(self.root/'batch');budget=json.loads((ROOT/'tools/fixtures/r6_budget.json').read_text())
        self.spec=self.root/'spec.json';self.spec.write_text(json.dumps(dict(id='CPU-SAMPLE',backend='runner_cpu')))
        with self.j.writer():
            self.j.initialize(budget);self.key=begin(self.j,file_pin(self.spec),ordinal=0,free_disk_bytes=500*1024**3)
        self.exe=self.root/'fake-smi';self.proc=None;self.addCleanup(self.cleanup_worker)

    def cleanup_worker(self):
        if self.proc is not None and self.proc.poll() is None:
            (self.j.directory/'STOP').touch();self.proc.terminate();self.proc.wait(timeout=20)

    def prepare(self,body=''):
        self.exe.write_text(FAKE.format(python=sys.executable,body=body));self.exe.chmod(0o700)
        with self.j.writer():return prepare(self.j,self.key,0,file_pin(self.exe))

    def launch(self):
        self.proc=subprocess.Popen([sys.executable,'-B','-m','tools.r6_resource_sampler',str(self.j.directory),self.key,'0'],
            cwd=ROOT,env=environment(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return self.proc

    @property
    def folder(self):return self.j.directory/'resource-queue'/self.key/'sample-0'

    def run_worker(self):
        proc=self.launch();stdout,stderr=proc.communicate(timeout=25)
        return proc.returncode,stdout,stderr

    def consume(self):
        with self.j.writer():return consume(self.j,self.key,0)

    def test_exact_pair_sample_receipt_and_idempotent_consumption(self):
        self.prepare();code,_,err=self.run_worker();self.assertEqual(code,0,err)
        sample=self.consume();self.assertEqual(sample['gpu']['utilization_percent'],3)
        self.assertEqual(self.consume(),sample);self.assertEqual(len(self.j.read()['events'][self.key]['samples']),1)
        self.assertLess(sum(p.stat().st_size for p in self.folder.iterdir()),CAP)
        receipt=json.loads((self.folder/'receipt.json').read_text());self.assertEqual(len(receipt['queries']),2)
        self.assertFalse(receipt['launch_admission']);self.assertFalse(any(owns(r['child_identity']) for r in receipt['queries']))

    def test_same_sample_directory_never_runs_twice(self):
        self.prepare();self.assertEqual(self.run_worker()[0],0)
        before=(self.folder/'receipt.json').read_bytes();self.assertNotEqual(self.run_worker()[0],0)
        self.assertEqual((self.folder/'receipt.json').read_bytes(),before)

    def test_capture_limit_retains_prefix_before_refusal(self):
        self.prepare("print('x'*100000,flush=True)");self.assertEqual(self.run_worker()[0],1)
        r=json.loads((self.folder/'receipt.json').read_text());self.assertIn('OUTPUT_CAP_BEFORE_WRITE',r['queries'][0]['error'])
        self.assertLessEqual((self.folder/'0.stdout').stat().st_size,16*1024)
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.consume()
        self.assertEqual(self.j.read()['events'][self.key]['samples'],[])

    def test_nonzero_query_preserves_native_code_and_no_second_query(self):
        self.prepare("print('query failure evidence',flush=True);sys.exit(7)");self.assertEqual(self.run_worker()[0],1)
        r=json.loads((self.folder/'receipt.json').read_text());self.assertEqual(r['queries'][0]['exit_code'],7)
        self.assertEqual(len(r['queries']),1);self.assertIn('query failure evidence',(self.folder/'0.stdout').read_text())

    def test_stop_cleans_sleeping_query(self):
        self.prepare('time.sleep(30)');self.launch()
        deadline=time.monotonic()+5
        while not (self.folder/'0.child.json').exists():
            if time.monotonic()>deadline:raise AssertionError('child did not start')
            time.sleep(.01)
        (self.j.directory/'STOP').touch();self.proc.communicate(timeout=20)
        r=json.loads((self.folder/'receipt.json').read_text());self.assertIn('STOP_DURING',r['queries'][0]['error'])
        self.assertFalse(owns(r['queries'][0]['child_identity']))
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.consume()
        with self.j.writer():self.assertEqual(finish(self.j,self.key)['status'],'STOPPED')

    def test_timeout_cleans_query_and_keeps_failure(self):
        self.prepare('time.sleep(30)');code,_,_=self.run_worker();self.assertEqual(code,1)
        r=json.loads((self.folder/'receipt.json').read_text());self.assertIn('QUERY_TIMEOUT',r['queries'][0]['error'])
        self.assertFalse(owns(r['queries'][0]['child_identity']))

    def test_adopted_orphan_is_cleaned(self):
        body="if os.fork()==0:\n os.setsid();time.sleep(30);os._exit(0)"
        self.prepare(body);self.assertEqual(self.run_worker()[0],1) # inherited pipe stays open until query timeout
        r=json.loads((self.folder/'receipt.json').read_text())
        self.assertGreaterEqual(r['queries'][0]['cleanup']['signalled_process_count'],1)
        self.assertFalse(any(owns(i) for i in r['queries'][0]['cleanup']['observed_identities']))

    def test_receipt_parse_drift_refused(self):
        self.prepare();self.assertEqual(self.run_worker()[0],0)
        p=self.folder/'receipt.json';r=json.loads(p.read_text());r['sample']['gpu']['utilization_percent']=0;p.write_text(json.dumps(r))
        with self.assertRaisesRegex(RecoveryError,'PARSE_DRIFT'):self.consume()

    def test_claim_without_receipt_not_retried_or_finished_as_pass(self):
        original=self.prepare()
        with self.j.writer():
            self.assertEqual(prepare(self.j,self.key,0,file_pin(self.exe)),original)
            with self.assertRaisesRegex(RecoveryError,'CLAIM_UNRESOLVED'):finish(self.j,self.key)
            with self.assertRaisesRegex(RecoveryError,'CLAIM_UNRESOLVED'):finish(self.j,self.key,interrupted=True)
        with self.assertRaisesRegex(RecoveryError,'MISSING_NO_RETRY'):self.consume()

    def test_crash_after_append_before_receipt_binding_reuses_sample(self):
        self.prepare();self.assertEqual(self.run_worker()[0],0)
        original=self.j.save;calls=[]
        def fail_second(state):
            calls.append(1)
            if len(calls)==2:raise RuntimeError('CPU crash')
            original(state)
        with self.j.writer(),patch.object(self.j,'save',side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError,'CPU crash'):consume(self.j,self.key,0)
        self.assertEqual(len(self.j.read()['events'][self.key]['samples']),1)
        self.consume();self.assertEqual(len(self.j.read()['events'][self.key]['samples']),1)


    def test_changed_worker_identity_receipt_rejected(self):
        self.prepare();self.assertEqual(self.run_worker()[0],0)
        p=self.folder/'worker.json';value=json.loads(p.read_text());value['ordinal']=1;p.write_text(json.dumps(value))
        with self.assertRaisesRegex(RecoveryError,'WORKER_BINDING'):self.consume()

    def test_cleanup_uncertainty_cannot_close_window(self):
        self.prepare();self.assertEqual(self.run_worker()[0],0)
        p=self.folder/'receipt.json';r=json.loads(p.read_text());r['queries'][0]['cleanup']['status']='UNKNOWN';p.write_text(json.dumps(r))
        with self.assertRaisesRegex(RecoveryError,'NOT_CLEAN'):self.consume()
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'CLAIM_UNRESOLVED'):finish(self.j,self.key,interrupted=True)


    def test_non_utf8_failure_output_remains_consumable_as_failed_evidence(self):
        self.prepare("os.write(1,bytes([255,254]));sys.exit(7)")
        self.assertEqual(self.run_worker()[0],1)
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.consume()
        self.assertEqual((self.folder/'0.stdout').read_bytes(),bytes([255,254]))
        with self.j.writer():self.assertEqual(finish(self.j,self.key,interrupted=True)['status'],'INTERRUPTED')


if __name__=='__main__':unittest.main()
