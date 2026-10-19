"""Isolated CPU worker tests. No nvidia-smi, simulator or model invocation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from tools.test_r6_analysis_binding import BindingTests
from tools.r6_recovery import Journal,RecoveryError,file_pin
from tools.r6_cpu_analysis import register_analysis,handoff,validate,environment,RESERVATION
from tools.r6_postrun import cancel_bounded
from rlflow.jobs import identity,owns
from rlflow.store import sha

ROOT=Path(__file__).resolve().parents[1]
SCRIPT='''import json,os,sys,time
from pathlib import Path
root=Path(__file__).parent
with (root/'calls').open('a') as f:f.write('call\\n');f.flush();os.fsync(f.fileno())
{body}
out=Path(sys.argv[3]);out.mkdir()
r=json.loads((root/'template.json').read_text())
(out/'report.json').write_text(json.dumps(r))
(out/'summary.json').write_text(json.dumps(r['summary']))
'''


class CPUAnalysisTests(unittest.TestCase):
    write=BindingTests.write
    def setUp(self):
        BindingTests.setUp(self)
        self.spec['backend']='runner_cpu';self.spec_pin=self.write('spec.json',self.spec)
        self.jid='J-'+sha({'id':'CPU','spec':self.spec})[:16]
        self.job.update(id=self.jid,spec=self.spec)
        self.store.replay=lambda:{'jobs':{self.jid:self.job}}
        self.closure=self.write('close.json',dict(event_key=self.key,job_id=self.jid,spec_sha256=sha(self.spec),native_status='FAILED',native_exit_code=-9))
        self.inventory.update(job_id=self.jid,spec=self.spec_pin,closure=self.closure)
        self.inventory_pin=self.write('inventory.json',self.inventory)
        p=json.loads(Path(self.postrun['path']).read_text());p.update(job_id=self.jid,closure=self.closure,inventory=self.inventory_pin)
        self.postrun=self.write('postrun.json',p)
        self.entry.update(identity={'job_id':self.jid},request={'spec':self.spec_pin},closure=self.closure,
            inventory=self.inventory_pin,postrun=self.postrun,kind='job',purpose='development')
        self.j=Journal(self.root/'batch');budget=json.loads((ROOT/'tools/fixtures/r6_budget.json').read_text())
        with self.j.writer():
            self.j.initialize(budget);s=self.j.read();s['events'][self.key]=self.entry;s['status']='JOB_BOUND_EXCEEDED';self.j.save(s)
        self.write('template.json',self.report);self.program=self.root/'fake_analysis.py';self.proc=None
        self.addCleanup(self.cleanup_worker)

    def cleanup_worker(self):
        if self.proc is not None and self.proc.poll() is None:
            with self.j.writer():cancel_bounded(self.j,self.cpu_key,'TEST_CLEANUP',self.proc,directory_kind='cpu-analysis',status_prefix='CPU_ANALYSIS')
            self.proc.wait(timeout=5)

    def register(self,body='time.sleep(.03)',timeout=10):
        self.program.write_text(SCRIPT.format(body=body))
        with self.j.writer():
            self.cpu_key=register_analysis(self.j,self.key,self.store,free_disk_bytes=500*1024**3,
                python=file_pin(sys.executable),program=file_pin(self.program),timeout_seconds=timeout)
        return self.cpu_key

    def complete(self):
        with self.j.writer():return handoff(self.j,self.cpu_key,self.store,poll_seconds=.01)

    def count(self):
        p=self.root/'calls';return len(p.read_text().splitlines()) if p.exists() else 0

    def claim(self):
        with self.j.writer():
            s=self.j.read();s['events'][self.cpu_key]['dispatch_claim']=dict(controller=identity(os.getpid()),
                worker=file_pin(ROOT/'tools/r6_cpu_analysis.py'),deadline_monotonic=time.monotonic()+40,deadline_wall=time.time()+40)
            self.j.save(s)

    def launch_detached(self):
        self.claim()
        self.proc=subprocess.Popen([sys.executable,'-B','-m','tools.r6_cpu_analysis',str(self.j.directory),self.cpu_key],
            cwd=ROOT,env=environment(),start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        deadline=time.monotonic()+5
        while self.count()!=1:
            if self.proc.poll() is not None or time.monotonic()>deadline:raise AssertionError('fixture worker did not start')
            time.sleep(.01)

    def test_complete_bind_once_restarts_preserve_native_failure(self):
        self.register();result=self.complete();self.assertEqual(self.count(),1)
        self.assertEqual(result['native_exit_code'],-9);self.assertFalse(result['scientific_acceptance'])
        self.j=Journal(self.j.directory);self.assertEqual(self.complete(),result);self.assertEqual(self.count(),1)
        s=self.j.read();self.assertEqual(s['status'],'JOB_BOUND_EXCEEDED')
        self.assertEqual(len(s['budget']['reservations']),1)
        self.assertEqual(s['budget']['reservations'][0]['resources']['output_actual_plus_unreleased_reservations_bytes'],RESERVATION)
        self.assertEqual(sum(e['kind']=='model' for e in s['events'].values()),0)

    def finalize_analysis(self,timeout=10):
        from tools.r6_finalize_job import analyze
        with self.j.writer():
            return analyze(self.j,self.key,self.store,free_disk_bytes=500*1024**3,
                python=file_pin(sys.executable),program=file_pin(self.program),timeout_seconds=timeout)

    def test_finalizer_registers_analysis_once_and_replay_preserves_state(self):
        self.program.write_text(SCRIPT.format(body='pass'))
        first=self.finalize_analysis();self.assertEqual(self.count(),1)
        with self.j.writer():
            state=self.j.read();state['status']='MODEL_DECISION_RECORDED';self.j.save(state)
        before=self.j.read()
        self.assertEqual(self.finalize_analysis(),first)
        self.assertEqual(self.count(),1);self.assertEqual(self.j.read(),before)

    def test_finalizer_rejects_changed_analysis_deadline_before_dispatch(self):
        self.register(timeout=10)
        with self.assertRaisesRegex(RecoveryError,'FINALIZE_ANALYSIS_BINDING'):
            self.finalize_analysis(timeout=11)
        self.assertEqual(self.count(),0)

    def test_finalizer_recovers_same_live_analysis_worker(self):
        self.register(body='time.sleep(.5)');self.launch_detached()
        self.j=Journal(self.j.directory)
        result=self.finalize_analysis();self.proc.wait(timeout=5)
        self.assertEqual(result['native_status'],'FAILED');self.assertEqual(self.count(),1)

    def test_finalizer_does_not_retry_failed_analysis(self):
        self.program.write_text(SCRIPT.format(body='raise SystemExit(9)'))
        for _ in range(2):
            with self.assertRaisesRegex(RecoveryError,'CPU_ANALYSIS_FAILED_PRESERVED'):
                self.finalize_analysis()
        self.assertEqual(self.count(),1)

    def test_restart_rebinds_same_live_worker_without_second_call(self):
        self.register(body='time.sleep(.5)');self.launch_detached()
        self.j=Journal(self.j.directory)
        result=self.complete();self.proc.wait(timeout=5)
        self.assertEqual(result['native_status'],'FAILED');self.assertEqual(self.count(),1)

    def test_claim_without_handle_is_not_retried(self):
        self.register();self.claim()
        with self.assertRaisesRegex(RecoveryError,'AMBIGUOUS_NO_RETRY'):self.complete()
        self.assertEqual(self.count(),0)

    def test_partial_receipt_is_not_retried(self):
        self.register();self.claim();d=self.j.directory/'cpu-analysis'/self.cpu_key;d.mkdir(parents=True)
        (d/'receipt.json').write_text('{')
        with self.assertRaisesRegex(RecoveryError,'AMBIGUOUS_NO_RETRY'):self.complete()
        self.assertEqual(self.count(),0);self.assertEqual((d/'receipt.json').read_text(),'{')

    def test_stop_prevents_dispatch(self):
        self.register();(self.j.directory/'STOP').touch()
        with self.assertRaisesRegex(RecoveryError,'STOP'):self.complete()
        self.assertEqual(self.count(),0)

    def test_worker_timeout_preserves_receipt_and_no_retry(self):
        self.register(body='time.sleep(30)',timeout=1)
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.complete()
        e=self.j.read()['events'][self.cpu_key];r=json.loads(Path(e['receipt']['path']).read_text())
        self.assertIn('CPU_ANALYSIS_TIMEOUT',r['error']);self.assertFalse(owns(r['child_identity']))
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.complete()
        self.assertEqual(self.count(),1)

    def test_log_limit_is_enforced_before_write(self):
        self.register(body="print('x'*(2*1024**2),flush=True)")
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.complete()
        d=self.j.directory/'cpu-analysis'/self.cpu_key
        self.assertLess((d/'stdout').stat().st_size,1024**2)
        self.assertIn('LOG_CAP_BEFORE_WRITE',json.loads((d/'receipt.json').read_text())['error'])

    def test_changed_analysis_request_cannot_create_second_event(self):
        self.register()
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'ALREADY_BOUND'):
            register_analysis(self.j,self.key,self.store,free_disk_bytes=500*1024**3,
                python=file_pin(sys.executable),program=file_pin(self.program),timeout_seconds=20)

    def test_source_drift_is_rejected(self):
        self.register()
        with self.j.writer():
            s=self.j.read();e=s['events'][self.cpu_key];e['request']['sources'][0]['sha256']='0'*64;self.j.save(s)
        with self.assertRaisesRegex(RecoveryError,'REQUEST_IDENTITY'):self.complete()
        self.assertEqual(self.count(),0)

    def test_closed_job_binding_checked_before_dispatch(self):
        self.register();self.job['status']='RUNNING'
        with self.assertRaisesRegex(RecoveryError,'TERMINAL_JOB'):self.complete()
        self.assertEqual(self.count(),0)

    def test_controller_deadline_kills_sigstopped_worker_and_owned_child(self):
        self.register(body='time.sleep(30)');self.launch_detached()
        os.kill(self.proc.pid,signal.SIGSTOP)
        with self.j.writer():
            s=self.j.read();s['events'][self.cpu_key]['dispatch_claim']['deadline_monotonic']=time.monotonic()-1;self.j.save(s)
        with self.assertRaisesRegex(RecoveryError,'CANCELLED_NO_RETRY'):self.complete()
        self.proc.wait(timeout=5);self.assertEqual(self.count(),1)
        s=self.j.read();c=s['events'][self.cpu_key]['controller_cancel']
        self.assertEqual(c['status'],'NO_KNOWN_OWNED_SURVIVORS')
        self.assertFalse(any(owns(i) for i in c['known']))

    def test_worker_nonzero_preserves_original_output_and_receipt(self):
        self.register(body="print('CPU failure evidence',flush=True);sys.exit(7)")
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.complete()
        d=self.j.directory/'cpu-analysis'/self.cpu_key;r=json.loads((d/'receipt.json').read_text())
        self.assertEqual(r['exit_code'],7);self.assertIn('CPU failure evidence',(d/'stdout').read_text())


    def test_actual_analyzer_missing_raw_binds_incomplete_summary(self):
        self.program.write_text((ROOT/'tools/analyze_r6_evaluation.py').read_text())
        with self.j.writer():
            self.cpu_key=register_analysis(self.j,self.key,self.store,free_disk_bytes=500*1024**3,
                python=file_pin('/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11'),
                program=file_pin(self.program),timeout_seconds=30)
        result=self.complete()
        self.assertIn('MOTION_EVIDENCE_INCOMPLETE_OR_MISMATCHED',result['issues'])
        self.assertIn('MEASUREMENT_REPLAY_INCOMPLETE',result['issues'])
        self.assertFalse(result['scientific_acceptance'])
        report=json.loads(Path(result['report']['path']).read_text())
        self.assertEqual(len(report['blocks']),16)
        self.assertTrue(all(not row['replay_complete'] for row in report['blocks']))

    def test_summary_drift_is_not_bound(self):
        self.register(body="time.sleep(.03)")
        # Preserve an actual terminal receipt, but alter its content with fresh
        # pins to ensure the consumer checks semantics as well as raw hashes.
        self.launch_detached();self.proc.wait(timeout=5)
        d=self.j.directory/'cpu-analysis'/self.cpu_key
        (d/'result/summary.json').write_text('{}')
        r=json.loads((d/'receipt.json').read_text());r['outputs']['summary.json']=file_pin(d/'result/summary.json')
        (d/'receipt.json').write_text(json.dumps(r))
        with self.assertRaisesRegex(RecoveryError,'SUMMARY_DRIFT'):self.complete()
        self.assertNotIn('analysis',self.j.read()['events'][self.key])


    def test_unresolved_cpu_dispatch_blocks_new_model_worker(self):
        from tools.r6_batch import model_handoff
        self.register();self.claim()
        with self.j.writer():
            state=self.j.read();state['events']['model-fixture']={'kind':'model','receipt':None}
            self.j.save(state)
            with self.assertRaisesRegex(RecoveryError,'CPU_WORKER_HANDOFF_UNRESOLVED'):
                model_handoff(self.j,'model-fixture')
        self.assertFalse((self.j.directory/'calls').exists())

    def test_stop_cancels_owned_analysis(self):
        self.register(body='time.sleep(30)');self.launch_detached()
        (self.j.directory/'STOP').touch()
        with self.assertRaisesRegex(RecoveryError,'CANCELLED_NO_RETRY|FAILED_PRESERVED|STOP_NEW_ACTIONS'):
            self.complete()
        self.proc.wait(timeout=5)
        self.assertEqual(self.count(),1)
        self.assertNotIn('analysis',self.j.read()['events'][self.key])


    def test_unresolved_resource_sample_blocks_analysis_dispatch(self):
        self.register()
        with self.j.writer():
            state=self.j.read();state['events']['resource-fixture']={'kind':'cpu',
                'identity':{'job_id':'OTHER','role':'resource-window'},'sample_claims':{'0':{'receipt':None}}}
            self.j.save(state)
        with self.assertRaisesRegex(RecoveryError,'OTHER_EVENT_UNRESOLVED'):self.complete()
        self.assertEqual(self.count(),0)


if __name__=='__main__':unittest.main()
