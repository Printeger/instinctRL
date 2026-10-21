"""Synthetic timestamp/window tests: no sleeps, GPU queries or live launches."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.r6_recovery import Journal,RecoveryError,file_pin
from tools.r6_budget import BudgetError,totals
from tools import r6_resource_queue as q
from training_workloads.scale_003.resources import GPU_IDENTITY

ROOT=Path(__file__).resolve().parents[1]


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.j=Journal(self.root/'batch');budget=json.loads((ROOT/'tools/fixtures/r6_budget.json').read_text())
        with self.j.writer():self.j.initialize(budget)
        self.spec=self.root/'spec.json';self.spec.write_text(json.dumps(dict(id='CPU-QUEUE',backend='runner_cpu')))
        self.clock=dict(boot_id='CPU_BOOT',monotonic_ns=10**12)
        patcher=patch.object(q,'now',side_effect=lambda:dict(self.clock));patcher.start();self.addCleanup(patcher.stop)

    def advance(self,seconds):self.clock['monotonic_ns']+=int(seconds*1e9)

    def begin(self,ordinal=0):
        with self.j.writer():return q.begin(self.j,file_pin(self.spec),ordinal=ordinal,free_disk_bytes=500*1024**3)

    def sample(self,u=3):
        return dict(monotonic_ns=self.clock['monotonic_ns'],gpu=dict(GPU_IDENTITY,memory_used_mib=560,
            free_memory_mib=GPU_IDENTITY['memory_total_mib']-560,utilization_percent=u),
            compute_apps=[dict(pid=3950,name='/usr/libexec/gnome-remote-desktop-daemon',used_memory_mib=164)])

    def window(self,key,values=None):
        for i,u in enumerate(values or [3]*13):
            if i:self.advance(5)
            with self.j.writer():q.append_sample(self.j,key,i,self.sample(u))

    def finish(self,key,**kw):
        with self.j.writer():return q.finish(self.j,key,**kw)

    def test_pass_is_complete_v2_observation_not_launch_permission(self):
        key=self.begin();self.window(key,[3]*6+[12]+[3]*6);r=self.finish(key)
        self.assertEqual(r['status'],'OBSERVATION_PASS');self.assertFalse(r['launch_admission'])
        self.assertEqual(r['window']['sample_count'],13);self.assertEqual(r['window']['samples'][6]['gpu']['utilization_percent'],12)
        usage=totals(self.j.read()['budget']);self.assertEqual(usage['queue_seconds'],180)
        for n in ('live_starts','simulator_initializations','automatic_model_sessions'):self.assertEqual(usage[n],0)

    def test_busy_full_window_local_backoff_and_next_slot(self):
        key=self.begin();self.window(key,[6]*13);r=self.finish(key);self.assertEqual(r['status'],'BUSY')
        with self.assertRaisesRegex(RecoveryError,'BACKOFF'):self.begin(1)
        self.advance(60);second=self.begin(1);self.assertNotEqual(second,key)
        self.assertEqual(totals(self.j.read()['budget'])['queue_seconds'],360)

    def test_restart_preserves_clock_samples_and_reservation(self):
        key=self.begin();first=self.sample()
        with self.j.writer():q.append_sample(self.j,key,0,first)
        clock=copy.deepcopy(self.j.read()['events'][key]['queue_clock']);self.advance(5);self.j=Journal(self.j.directory)
        self.assertEqual(self.begin(),key)
        with self.j.writer():q.append_sample(self.j,key,0,first);q.append_sample(self.j,key,1,self.sample())
        self.assertEqual(self.j.read()['events'][key]['queue_clock'],clock)
        self.assertEqual(len(self.j.read()['budget']['reservations']),1)

    def test_expired_partial_window_not_spliced_into_new_window(self):
        key=self.begin()
        with self.j.writer():q.append_sample(self.j,key,0,self.sample())
        self.advance(180)
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'EXPIRED'):q.append_sample(self.j,key,1,self.sample())
        self.assertEqual(self.finish(key)['status'],'EXPIRED');self.advance(60)
        second=self.begin(1);self.assertEqual(self.j.read()['events'][second]['samples'],[])

    def test_stop_preserves_partial_window(self):
        key=self.begin()
        with self.j.writer():q.append_sample(self.j,key,0,self.sample())
        (self.root/'STOP').touch()
        with self.assertRaisesRegex(RecoveryError,'STOP'):self.begin(1)
        result=self.finish(key);self.assertEqual(result['status'],'STOPPED');self.assertEqual(len(result['window']['samples']),1)
        self.advance(1000)
        with self.j.writer():self.assertEqual(q.account_wait(self.j,key),180)

    def test_interrupted_is_not_busy_and_requires_new_full_window(self):
        key=self.begin();r=self.finish(key,interrupted=True);self.assertEqual(r['status'],'INTERRUPTED')
        self.advance(60);second=self.begin(1);self.window(second);self.assertEqual(self.finish(second)['status'],'OBSERVATION_PASS')

    def test_unknown_gpu_is_invalid_not_permanent_busy_loop(self):
        key=self.begin();self.window(key)
        with self.j.writer():
            s=self.j.read();s['events'][key]['samples'][4]['gpu']['name']='other';self.j.save(s)
        self.assertEqual(self.finish(key)['status'],'INVALID_WINDOW');self.advance(60)
        with self.assertRaisesRegex(RecoveryError,'REQUIRES_DECISION'):self.begin(1)

    def test_clock_reboot_requires_reconciliation(self):
        key=self.begin();self.clock['boot_id']='NEW_BOOT'
        with self.assertRaisesRegex(RecoveryError,'CLOCK_REQUIRES_RECONCILIATION'):self.finish(key)
        self.assertEqual(self.begin(),key)
        self.assertEqual(totals(self.j.read()['budget'])['queue_seconds'],180)

    def test_stale_future_and_duplicate_drift_rejected(self):
        key=self.begin();sample=self.sample()
        with self.j.writer():q.append_sample(self.j,key,0,sample)
        changed=copy.deepcopy(sample);changed['gpu']['utilization_percent']=0
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'REPLAY_DRIFT'):q.append_sample(self.j,key,0,changed)
        self.advance(20)
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'STALE_OR_FUTURE'):q.append_sample(self.j,key,1,sample)
        future=self.sample();future['monotonic_ns']+=1
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'STALE_OR_FUTURE'):q.append_sample(self.j,key,1,future)

    def test_cumulative_queue_cap_blocks_next_window_without_live_start(self):
        with self.j.writer():
            s=self.j.read();s['budget']['used']['queue_seconds']=21600-180;self.j.save(s)
        key=self.begin();self.window(key,[6]*13);self.finish(key);self.advance(60)
        with self.assertRaises(BudgetError):self.begin(1)
        self.assertEqual(totals(self.j.read()['budget'])['queue_seconds'],21600)
        self.assertEqual(totals(self.j.read()['budget'])['live_starts'],0)

    def test_overrun_recorded_and_never_refunded(self):
        key=self.begin();self.advance(220);r=self.finish(key)
        self.assertEqual(r['status'],'EXPIRED');self.assertEqual(totals(self.j.read()['budget'])['queue_seconds'],220)
        self.assertEqual(self.finish(key),r)
        self.assertEqual(len(self.j.read()['budget']['reservations'][0]['observations']),1)

    def test_crash_after_record_before_state_reuses_same_evidence(self):
        key=self.begin();self.window(key)
        with self.j.writer(),patch.object(self.j,'save',side_effect=RuntimeError('CPU crash')):
            with self.assertRaisesRegex(RuntimeError,'CPU crash'):q.finish(self.j,key)
        path=self.j.directory/'resource-queue'/key/'window.json';raw=path.read_bytes();self.advance(1)
        result=self.finish(key);self.assertEqual(path.read_bytes(),raw);self.assertEqual(result['observed_elapsed_seconds'],60)

    def test_two_windows_cannot_collect_simultaneously(self):
        self.begin()
        with self.assertRaisesRegex(RecoveryError,'STILL_OPEN'):self.begin(1)

    def test_real_queue_requires_capacity_and_exact_review(self):
        self.spec.write_text(json.dumps(dict(id='R6-REAL',backend='real')))
        with self.assertRaisesRegex(RecoveryError,'CAPACITY_NOT_ACCEPTED'):self.begin()
        with self.j.writer():
            s=self.j.read();s['budget']['capacity_pass']=True;self.j.save(s)
        with self.assertRaisesRegex(RecoveryError,'REVIEW_REQUIRED'):self.begin()
        self.assertEqual(len(self.j.read()['budget']['reservations']),0)


    def test_restart_after_long_backoff_charges_downtime_before_next_slot(self):
        key=self.begin();self.window(key,[6]*13);self.finish(key)
        self.advance(1000);self.j=Journal(self.j.directory);self.begin(1)
        self.assertEqual(totals(self.j.read()['budget'])['queue_seconds'],1060+180)
        self.assertEqual(self.j.read()['events'][key]['wait_accounted_seconds'],1060)

    def test_long_queue_downtime_over_cap_is_retained_and_blocks_new_slot(self):
        key=self.begin();self.window(key,[6]*13);self.finish(key);self.advance(21600)
        with self.assertRaises(BudgetError):self.begin(1)
        self.assertEqual(totals(self.j.read()['budget'],enforce_caps=False)['queue_seconds'],21660)
        self.assertEqual(len(self.j.read()['budget']['reservations']),1)

    def test_stop_wait_accounting_does_not_query_or_launch(self):
        key=self.begin();self.window(key,[6]*13);self.finish(key);self.advance(200)
        (self.root/'STOP').touch()
        with self.j.writer():self.assertEqual(q.account_wait(self.j,key),260)
        self.assertEqual(totals(self.j.read()['budget'])['queue_seconds'],260)
        self.advance(100)
        with self.j.writer():self.assertEqual(q.account_wait(self.j,key),260)
        self.assertEqual(len(self.j.read()['events'][key]['wait_records']),1)
        self.assertEqual(self.j.read()['events'][key]['queue_status'],'STOPPED')


if __name__=='__main__':unittest.main()


class StalePassedWindowTests(unittest.TestCase):
    setUp=QueueTests.setUp
    advance=QueueTests.advance
    begin=QueueTests.begin
    sample=QueueTests.sample
    window=QueueTests.window
    finish=QueueTests.finish
    def test_stale_pass_collects_new_window_without_rewriting_old_evidence(self):
        from tools import r6_resource_wait as wait
        key=self.begin();self.window(key);first=self.finish(key)
        old=self.j.read()['events'][key]['window_record']
        original=Path(old['path']).read_bytes();self.advance(9)
        sleeps=[];calls=[]
        def sleep(seconds):sleeps.append(seconds);self.advance(seconds)
        def collect(journal,next_key,executable,**kwargs):
            calls.append(next_key)
            for i in range(13):
                if i:self.advance(5)
                q.append_sample(journal,next_key,i,self.sample())
            return q.finish(journal,next_key)
        with self.j.writer(),patch.object(wait,'collect_window',side_effect=collect),patch.object(wait.time,'sleep',side_effect=sleep):
            second=wait.wait_for_observation(self.j,file_pin(self.spec),{},free_disk_bytes=500*1024**3)
        self.assertEqual(len(calls),1);self.assertNotEqual(second['event_key'],key)
        self.assertEqual(sum(sleeps),51)
        self.assertEqual(Path(old['path']).read_bytes(),original)
        self.assertEqual(self.j.read()['events'][key]['queue_status'],'OBSERVATION_PASS')
        self.assertEqual(totals(self.j.read()['budget'])['queue_seconds'],360)
        with self.j.writer(),patch.object(wait,'collect_window',side_effect=AssertionError('fresh replay must not query')):
            self.assertEqual(wait.wait_for_observation(self.j,file_pin(self.spec),{},free_disk_bytes=500*1024**3),second)
        for name in ('live_starts','simulator_initializations','automatic_model_sessions'):
            self.assertEqual(totals(self.j.read()['budget'])[name],0)

    def test_old_pass_downtime_is_charged_and_fresh_pass_cannot_be_replaced(self):
        key=self.begin();self.window(key);self.finish(key)
        with self.assertRaisesRegex(RecoveryError,'PRIOR_REQUIRES_DECISION'):self.begin(1)
        self.advance(240);self.begin(1)
        self.assertEqual(totals(self.j.read()['budget'])['queue_seconds'],480)

    def test_stop_during_stale_pass_recovery_spends_no_new_slot(self):
        from tools import r6_resource_wait as wait
        key=self.begin();self.window(key);self.finish(key);self.advance(9)
        (self.j.directory/'STOP').touch()
        with self.j.writer(),patch.object(wait,'collect_window',side_effect=AssertionError('STOP must not query')):
            with self.assertRaisesRegex(RecoveryError,'STOP_DURING_RESOURCE_BACKOFF'):
                wait.wait_for_observation(self.j,file_pin(self.spec),{},free_disk_bytes=500*1024**3)
        self.assertEqual(len(self.j.read()['events']),1)
        self.assertEqual(totals(self.j.read()['budget'])['queue_seconds'],180)
