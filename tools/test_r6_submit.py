"""Actual journal/queue plus inert Store submission; never starts a job or GPU."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from rlflow.store import sha
from tools.r6_recovery import Journal,RecoveryError,file_pin
from tools.r6_budget import BudgetError
from tools import r6_resource_queue as q
from tools.r6_submit import submit_once
from training_workloads.scale_003.resources import GPU_IDENTITY

ROOT=Path(__file__).resolve().parents[1]


class StoreFixture:
    def __init__(self,root):self.root=root;self.state={'jobs':{}}
    def replay(self):return copy.deepcopy(self.state)


class SubmitTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.j=Journal(self.root/'batch');self.store=StoreFixture(self.root/'store')
        budget=json.loads((ROOT/'tools/fixtures/r6_budget.json').read_text());budget['capacity_pass']=True
        with self.j.writer():self.j.initialize(budget)
        self.spec=dict(id='CPU-SUBMIT',backend='runner_cpu',resources=dict(starts=1,wall_seconds=10,gpu_seconds=0,transitions=0))
        self.path=self.root/'spec.json';self.path.write_text(json.dumps(self.spec))
        self.clock=dict(boot_id='CPU',monotonic_ns=10**12)
        timer=patch.object(q,'now',side_effect=lambda:dict(self.clock));timer.start();self.addCleanup(timer.stop)
        with self.j.writer():
            self.queue_key=q.begin(self.j,file_pin(self.path),ordinal=0,free_disk_bytes=500*1024**3)
            for i in range(13):
                if i:self.clock['monotonic_ns']+=5*10**9
                q.append_sample(self.j,self.queue_key,i,dict(monotonic_ns=self.clock['monotonic_ns'],
                    gpu=dict(GPU_IDENTITY,memory_used_mib=560,free_memory_mib=GPU_IDENTITY['memory_total_mib']-560,utilization_percent=3),compute_apps=[]))
            q.finish(self.j,self.queue_key)
            request=dict(spec=file_pin(self.path),wait_deadline_unix=time.time()+10)
            self.jid='J-'+sha({'id':self.spec['id'],'spec':self.spec})[:16]
            event=dict(goal_id=budget['goal_id'],event_id='CPU-SUBMIT',job_id=self.jid,input_sha256=sha(request),role='job')
            self.key,_=self.j.intent(event,kind='job',request=request,resources={'live_starts':1},purpose='development',free_disk_bytes=500*1024**3)
        self.patches=[patch('tools.r6_submit.experiments.validate',return_value=self.spec),
            patch('tools.r6_submit.experiments.authorized',return_value={'approval_id':'CPU-APPROVAL'}),
            patch('tools.r6_submit.experiments.submit',side_effect=self.launch)]
        self.mocks=[p.start() for p in self.patches]
        for p in self.patches:self.addCleanup(p.stop)
        self.submit=self.mocks[-1]

    def launch(self,store,spec):
        self.assertIn('submission_claim',self.j.read()['events'][self.key])
        job=dict(id=self.jid,spec=spec,approval_id='CPU-APPROVAL',status='RUNNING',identity=None)
        store.state['jobs'][self.jid]=job;return job

    def invoke(self):
        with self.j.writer():return submit_once(self.j,self.key,self.store,queue_key=self.queue_key)

    def change(self,callback):
        with self.j.writer():s=self.j.read();callback(s);self.j.save(s)

    def test_first_submit_and_replay_use_one_call_without_budget_change(self):
        budget=self.j.read()['budget'];first=self.invoke();self.assertEqual(self.invoke(),first)
        self.assertEqual(self.submit.call_count,1);self.assertEqual(self.j.read()['budget'],budget)
        self.assertEqual(self.j.read()['current_job'],self.jid)

    def test_crash_after_store_reservation_recovers_same_job(self):
        def partial(store,spec):self.launch(store,spec);raise RuntimeError('CPU crash after Store')
        self.submit.side_effect=partial
        with self.assertRaisesRegex(RuntimeError,'CPU crash'):self.invoke()
        self.assertEqual(self.invoke()['job_id'],self.jid);self.assertEqual(self.submit.call_count,1)

    def test_claim_without_store_job_never_retries(self):
        self.submit.side_effect=RuntimeError('CPU before Store')
        with self.assertRaises(RuntimeError):self.invoke()
        with self.assertRaisesRegex(RecoveryError,'AMBIGUOUS_NO_RETRY'):self.invoke()
        self.assertEqual(self.submit.call_count,1)

    def test_missing_capacity_blocks_before_claim(self):
        self.change(lambda s:s['budget'].update(capacity_pass=False))
        with self.assertRaisesRegex(RecoveryError,'CAPACITY_NOT_ACCEPTED'):self.invoke()
        self.assertNotIn('submission_claim',self.j.read()['events'][self.key]);self.submit.assert_not_called()

    def test_stale_and_other_boot_queue_block_before_claim(self):
        for clock in [dict(self.clock,monotonic_ns=self.clock['monotonic_ns']+9*10**9),dict(self.clock,boot_id='OTHER')]:
            self.clock=clock
            with self.assertRaisesRegex(RecoveryError,'QUEUE_STALE'):self.invoke()
        self.submit.assert_not_called()

    def test_stop_blocks_new_submit_but_allows_receipt_recovery(self):
        stop=self.j.directory/'STOP';stop.touch()
        with self.assertRaises(RecoveryError):self.invoke()
        self.submit.assert_not_called();stop.unlink()
        first=self.invoke();stop.touch();self.assertEqual(self.invoke(),first);self.assertEqual(self.submit.call_count,1)

    def test_active_model_blocks_submit(self):
        self.change(lambda s:s.update(current_model_event='CPU_ACTIVE'))
        with self.assertRaisesRegex(RecoveryError,'MODEL_HANDOFF_STILL_ACTIVE'):self.invoke()
        self.submit.assert_not_called()

    def test_external_existing_job_is_not_adopted_without_claim(self):
        self.store.state['jobs'][self.jid]={'id':self.jid,'spec':self.spec}
        with self.assertRaisesRegex(RecoveryError,'EXISTING_JOB_WITHOUT_CLAIM'):self.invoke()
        self.submit.assert_not_called()

    def test_unclaimed_job_artifacts_block_new_submit(self):
        (self.j.directory/'jobs'/self.key).mkdir(parents=True)
        with self.assertRaisesRegex(RecoveryError,'UNCLAIMED_SIDE_EFFECT'):self.invoke()
        self.submit.assert_not_called()

    def test_exact_store_and_implementation_required_on_recovery(self):
        self.invoke();old_root=self.store.root;self.store.root=self.root/'another-store'
        with self.assertRaisesRegex(RecoveryError,'CLAIM_DRIFT'):self.invoke()
        self.store.root=old_root
        self.change(lambda s:s['events'][self.key]['submission_claim']['implementation'].update(sha256='0'*64))
        with self.assertRaisesRegex(RecoveryError,'IMPLEMENTATION_DRIFT'):self.invoke()
        self.assertEqual(self.submit.call_count,1)

    def test_recovery_after_receipt_before_state_commit_does_not_resubmit(self):
        save=self.j.save
        def crash(state):
            if state['events'][self.key].get('submission'):raise RuntimeError('CPU receipt crash')
            return save(state)
        with patch.object(self.j,'save',side_effect=crash):
            with self.assertRaisesRegex(RuntimeError,'receipt crash'):self.invoke()
        self.assertEqual(self.invoke()['job_id'],self.jid);self.assertEqual(self.submit.call_count,1)

    def test_budget_drift_blocks_previously_reserved_event(self):
        self.change(lambda s:s['budget']['used'].update(live_starts=64))
        with self.assertRaises(BudgetError):self.invoke()
        self.submit.assert_not_called()

    def test_native_failed_store_job_is_preserved_on_recovery(self):
        def partial(store,spec):
            self.launch(store,spec);store.state['jobs'][self.jid].update(status='FAILED',exit_code=-9)
            raise RuntimeError('CPU late failure')
        self.submit.side_effect=partial
        with self.assertRaises(RuntimeError):self.invoke()
        self.invoke();self.assertEqual(self.store.state['jobs'][self.jid]['exit_code'],-9)
        self.assertEqual(self.submit.call_count,1)

    def test_completed_handoff_is_not_resurrected_by_replay(self):
        self.invoke();self.change(lambda s:s.update(current_job=None,status='CPU_ANALYZE'))
        self.invoke();self.assertIsNone(self.j.read()['current_job']);self.assertEqual(self.j.read()['status'],'CPU_ANALYZE')


if __name__=='__main__':unittest.main()
