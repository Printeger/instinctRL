"""CPU subprocess/Store substitutes; no live permission, GPU query or model API."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from rlflow.jobs import identity
from rlflow.store import atomic, canonical, sha
from tools.r6_recovery import Journal, RecoveryError, file_pin
from tools.r6_budget import totals
from tools.r6_job_handoff import close_terminal_job, job_handoff

ROOT = Path(__file__).resolve().parents[1]


class StoreFixture:
    def __init__(self, root):
        self.root = root
        self.state = {'paused': False, 'jobs': {}}
    def replay(self):
        return copy.deepcopy(self.state)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.base = Path(self.tmp.name)
        self.j = Journal(self.base/'batch')
        budget = json.loads((ROOT/'tools/fixtures/r6_budget.json').read_text())
        with self.j.writer():self.j.initialize(budget)
        self.store = StoreFixture(self.base/'store')
        self.spec = {'id':'CPU-JOB-CLOSE-01','backend':'runner_cpu','resources':{'wall_seconds':10}}
        specpath = self.base/'spec.json';specpath.write_text(json.dumps(self.spec))
        self.jid = 'J-'+sha({'id':self.spec['id'],'spec':self.spec})[:16]
        self.request = dict(spec=file_pin(specpath),wait_deadline_unix=time.time()+10)
        event = dict(goal_id=budget['goal_id'],event_id='CPU-CLOSE-01',job_id=self.jid,
                     input_sha256=sha(self.request),role='job')
        with self.j.writer():
            self.key,_ = self.j.intent(event,kind='job',request=self.request,
                resources={'live_starts':1,'live_wall_seconds':190,'allocated_gpu_seconds':190},
                purpose='development',free_disk_bytes=500*1024**3)
        self.process = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        self.owner = identity(self.process.pid)
        self.job = dict(id=self.jid,spec=self.spec,status='RUNNING',identity=self.owner)
        self.store.state['jobs'][self.jid] = self.job
        self.directory = self.store.root/'jobs'/self.jid;self.directory.mkdir(parents=True)
        self.submit_patch = patch('tools.r6_job_handoff.experiments.submit',side_effect=AssertionError('NO SUBMIT'))
        self.submit = self.submit_patch.start()
        self.reconcile_patch = patch('tools.r6_job_handoff.experiments.reconcile',side_effect=lambda s,j:s.replay()['jobs'][j])
        self.reconcile = self.reconcile_patch.start()
        self.cancel_patch = patch('tools.r6_job_handoff.experiments.cancel',side_effect=self.cancel)
        self.cancels = self.cancel_patch.start()

    def tearDown(self):
        if self.process.poll() is None:self.process.kill();self.process.wait(timeout=5)
        self.submit_patch.stop();self.reconcile_patch.stop();self.cancel_patch.stop()
        self.tmp.cleanup()

    def save(self,name,value):atomic(self.directory/name,canonical(value))

    def finish(self,status='FAILED',elapsed=2):
        if self.process.poll() is None:self.process.terminate();self.process.wait(timeout=5)
        receipt=dict(job_id=self.jid,spec_sha256=sha(self.spec),status=status,
            exit_code=-9 if status=='FAILED' else 0,elapsed_seconds=elapsed,allocated_seconds=1.)
        clean=dict(quiescent=True,timed_out=False,survivors=[],stable_scans=2,required_stable_scans=2,discovered=[])
        for name,value in {'receipt.json':receipt,'descendant-cleanup.json':clean,
                'lifeguard-ack.json':dict(owner=self.owner,no_survivors=True,descendant_cleanup=clean),
                'lifeguard-ready.json':dict(identity=self.owner),'process-identity.json':self.owner,
                'descendant-ledger.json':dict(owner=self.owner,descendants=[])}.items():self.save(name,value)
        self.job.update(status=status,receipt=receipt)

    def cancel(self,store,jid):
        self.assertEqual(jid,self.jid)
        atomic(self.directory/'cancel',b'cancel')
        self.finish('STOPPED')

    def handoff(self):
        with self.j.writer():return job_handoff(self.j,self.key,self.store,poll_seconds=.01)

    def test_native_failure_closed_without_refund_or_scientific_acceptance(self):
        self.finish();before=copy.deepcopy(self.j.read()['budget'])
        result=self.handoff()
        self.assertEqual((result['native_status'],result['native_exit_code']),('FAILED',-9))
        self.assertFalse(result['scientific_acceptance']);self.assertFalse(result['postrun_complete'])
        self.assertEqual(totals(before),totals(self.j.read()['budget']));self.assertEqual(self.j.read()['status'],'CPU_ANALYZE')
        self.submit.assert_not_called()

    def test_real_live_handle_waits_locally_then_exact_terminal(self):
        with patch('tools.r6_job_handoff.time.sleep',side_effect=lambda _:self.finish()):result=self.handoff()
        self.assertEqual(result['job_id'],self.jid)
        row=self.j.read()['events'][self.key]['last_local_wait']
        self.assertEqual(row['process'],self.owner);self.assertFalse(row['model_workers_present'])
        self.assertEqual(row['model_events'],0)
        self.submit.assert_not_called();self.cancels.assert_not_called()

    def test_restart_reuses_close_and_clears_current_job(self):
        self.finish();first=self.handoff();pin=self.j.read()['events'][self.key]['closure']
        self.j=Journal(self.j.directory)
        self.assertEqual(first,self.handoff())
        self.assertEqual(pin,self.j.read()['events'][self.key]['closure'])
        self.assertIsNone(self.j.read()['current_job']);self.assertEqual(len(self.j.read()['budget']['reservations']),1)

    def test_crash_after_close_file_before_state_recovers(self):
        self.finish()
        with self.j.writer(),patch.object(self.j,'save',side_effect=RuntimeError('CPU_CRASH')):
            with self.assertRaisesRegex(RuntimeError,'CPU_CRASH'):close_terminal_job(self.j,self.key,self.store)
        self.assertTrue((self.j.directory/'jobs'/self.key/'close.json').exists())
        self.assertNotIn('closure',self.j.read()['events'][self.key])
        self.handoff();self.assertIn('closure',self.j.read()['events'][self.key])

    def test_stop_cancels_exact_job_once_then_preserves_stop(self):
        (self.j.directory/'STOP').touch()
        self.handoff();self.handoff()
        self.assertEqual(self.cancels.call_count,1)
        self.assertEqual(self.j.read()['status'],'STOPPED_AFTER_OWNED_CLEANUP')
        self.assertTrue((self.j.directory/'STOP').exists())
        self.submit.assert_not_called()

    def test_missing_job_does_not_submit(self):
        self.store.state['jobs'].clear()
        with self.assertRaisesRegex(RecoveryError,'MISSING_NO_RESUBMISSION'):self.handoff()
        self.submit.assert_not_called()

    def test_dead_handle_without_receipt_does_not_restart(self):
        self.process.kill();self.process.wait(timeout=5)
        with self.assertRaisesRegex(RecoveryError,'HANDLE_MISSING_NO_RETRY'):self.handoff()
        self.reconcile.assert_called_once();self.submit.assert_not_called()

    def test_wrong_spec_rejected_before_cancel(self):
        self.job['spec']=dict(self.spec,seed=99);(self.j.directory/'STOP').touch()
        with self.assertRaisesRegex(RecoveryError,'JOB_SPEC_DRIFT'):self.handoff()
        self.cancels.assert_not_called()

    def test_incomplete_ack_and_live_ledger_member_refuse_close(self):
        self.finish();ack=json.loads((self.directory/'lifeguard-ack.json').read_text())
        ack['no_survivors']=False;self.save('lifeguard-ack.json',ack)
        with self.assertRaisesRegex(RecoveryError,'NORMAL_ACK_REQUIRED'):self.handoff()
        ack['no_survivors']=True;self.save('lifeguard-ack.json',ack)
        self.save('descendant-ledger.json',dict(owner=self.owner,descendants=[identity(__import__('os').getpid())]))
        with self.assertRaisesRegex(RecoveryError,'OWNED_PROCESS_STILL_ALIVE'):self.handoff()

    def test_partial_receipt_preserved_no_new_side_effect(self):
        path=self.directory/'receipt.json';path.write_text('{')
        self.reconcile.side_effect=json.JSONDecodeError('partial','{',0)
        with self.assertRaises(json.JSONDecodeError):self.handoff()
        self.assertEqual(path.read_text(),'{');self.submit.assert_not_called();self.cancels.assert_not_called()

    def test_overrun_saved_without_refund_then_blocks_next_phase(self):
        self.finish(elapsed=191)
        result=self.handoff()
        self.assertIn('RESERVATION_EXCEEDED_live_wall_seconds',result['budget_violations'])
        self.assertEqual(self.j.read()['status'],'JOB_BOUND_EXCEEDED')
        self.assertEqual(self.j.read()['budget']['reservations'][0]['resources']['live_wall_seconds'],190)

    def test_closed_evidence_drift_is_rejected(self):
        self.finish();self.handoff()
        (self.j.directory/'jobs'/self.key/'close.json').write_text('{}')
        with self.assertRaisesRegex(RecoveryError,'PIN_DRIFT'):self.handoff()

    def test_model_dispatch_ambiguity_blocks_job_wait(self):
        with self.j.writer():
            state=self.j.read();state['events']['CPU-AMBIGUOUS']=dict(kind='model',process=None,dispatch_claim={},receipt=None)
            state['events']['CPU-AMBIGUOUS']['dispatch_claim']={'claimed':True};self.j.save(state)
        with self.assertRaisesRegex(RecoveryError,'MODEL_DISPATCH_UNRESOLVED'):self.handoff()

    def test_deadline_cancels_without_extending_on_restart(self):
        with patch('tools.r6_job_handoff.time.time',return_value=self.request['wait_deadline_unix']+1):
            self.handoff();self.handoff()
        self.assertEqual(self.cancels.call_count,1)
        self.assertEqual(self.j.read()['status'],'JOB_BOUND_EXCEEDED')

    def test_goal_level_stop_is_observed(self):
        (self.j.directory.parent/'STOP').touch()
        self.handoff()
        self.assertEqual(self.cancels.call_count,1)
        self.assertEqual(self.j.read()['status'],'STOPPED_AFTER_OWNED_CLEANUP')

    def test_receipt_spec_mismatch_not_accepted(self):
        self.finish();receipt=self.job['receipt'];receipt['spec_sha256']='0'*64
        self.save('receipt.json',receipt)
        with self.assertRaisesRegex(RecoveryError,'NATIVE_RECEIPT_BINDING'):self.handoff()

    def test_live_receipt_child_rejected_even_without_child_file(self):
        receipt=self.base/'model-receipt.json';receipt.write_text(json.dumps(dict(
            child_identity=self.owner,cleanup={'status':'VERIFIED_NO_OWNED_DESCENDANTS'})))
        with self.j.writer():
            state=self.j.read();state['events']['old-model']=dict(kind='model',process=None,
                receipt=file_pin(receipt));self.j.save(state)
        with self.assertRaisesRegex(RecoveryError,'MODEL_RECEIPT_PROCESS_ALIVE'):self.handoff()

    def test_legacy_model_cleanup_absence_is_not_silently_accepted(self):
        receipt=self.base/'legacy-model.json';receipt.write_text('{}')
        with self.j.writer():
            state=self.j.read();state['events']['legacy']=dict(kind='model',process=None,
                receipt=file_pin(receipt));self.j.save(state)
        with self.assertRaisesRegex(RecoveryError,'MODEL_CLEANUP_UNCONFIRMED'):self.handoff()

    def test_cancel_claim_without_marker_never_repeats_signal(self):
        with self.j.writer():
            state=self.j.read();state['events'][self.key]['cancel_claim']={'job_id':self.jid};self.j.save(state)
        (self.j.directory/'STOP').touch()
        with self.assertRaisesRegex(RecoveryError,'CANCEL_AMBIGUOUS_NO_RETRY'):self.handoff()
        self.cancels.assert_not_called()


if __name__=='__main__':unittest.main()
