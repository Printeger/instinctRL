"""Real local worker handoff with a fake query program; no real GPU/model."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from tools.test_r6_resource_sampler import SamplerTests
from tools.r6_resource_wait import sample_handoff,cancel,collect_window
from tools.r6_recovery import Journal,RecoveryError,file_pin
from rlflow.jobs import owns,identity


class WaitTests(unittest.TestCase):
    setUp=SamplerTests.setUp
    cleanup_worker=SamplerTests.cleanup_worker
    prepare=SamplerTests.prepare
    launch=SamplerTests.launch
    folder=SamplerTests.folder

    def handoff(self):
        with self.j.writer():return sample_handoff(self.j,self.key,0,file_pin(self.exe),poll_seconds=.01)

    def unclaimed(self,body=''):
        self.prepare(body)
        with self.j.writer():
            s=self.j.read();s['events'][self.key]['sample_claims']={};self.j.save(s)

    def test_handoff_launches_once_and_replays_same_receipt(self):
        self.unclaimed();sample=self.handoff();receipt=(self.folder/'receipt.json').read_bytes()
        self.j=Journal(self.j.directory);self.assertEqual(self.handoff(),sample)
        self.assertEqual((self.folder/'receipt.json').read_bytes(),receipt)

    def test_reattach_unbound_live_worker_without_second_start(self):
        self.prepare('time.sleep(.3)');self.launch();deadline=time.monotonic()+5
        while not (self.folder/'0.child.json').exists():
            if time.monotonic()>deadline:raise AssertionError('no child')
            time.sleep(.01)
        first=json.loads((self.folder/'worker.json').read_text())['identity']
        self.j=Journal(self.j.directory);sample=self.handoff();self.proc.communicate(timeout=5)
        self.assertEqual(self.j.read()['events'][self.key]['sample_claims']['0']['process'],first)
        self.assertEqual(sample['gpu']['utilization_percent'],3)

    def test_claim_without_worker_is_not_relaunched(self):
        self.prepare()
        with self.assertRaisesRegex(RecoveryError,'AMBIGUOUS_NO_RETRY'):self.handoff()
        self.assertFalse(self.folder.exists())

    def test_partial_receipt_does_not_trigger_retry(self):
        self.prepare();self.folder.mkdir(parents=True);(self.folder/'receipt.json').write_text('{')
        with self.assertRaisesRegex(RecoveryError,'INVALID_RECEIPT_NO_RETRY'):self.handoff()
        self.assertEqual((self.folder/'receipt.json').read_text(),'{')

    def test_sigstop_deadline_cleanup_does_not_kill_unrelated_process(self):
        self.prepare('time.sleep(30)');self.launch();deadline=time.monotonic()+5
        while not (self.folder/'0.child.json').exists():
            if time.monotonic()>deadline:raise AssertionError('no child')
            time.sleep(.01)
        unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
        try:
            os.kill(self.proc.pid,signal.SIGSTOP)
            with self.j.writer():
                s=self.j.read();s['events'][self.key]['sample_claims']['0']['deadline_monotonic']=time.monotonic()-1;self.j.save(s)
            with self.assertRaisesRegex(RecoveryError,'CANCELLED_NO_RETRY'):self.handoff()
            self.proc.communicate(timeout=5);self.assertIsNone(unrelated.poll())
            c=self.j.read()['events'][self.key]['sample_claims']['0']['cancel']
            self.assertEqual(c['status'],'NO_KNOWN_OWNED_SURVIVORS');self.assertFalse(any(owns(i) for i in c['known']))
        finally:unrelated.terminate();unrelated.wait(timeout=5)

    def test_stop_terminates_worker_without_waiting_for_its_query_timeout(self):
        self.prepare('time.sleep(30)');self.launch();deadline=time.monotonic()+5
        while not (self.folder/'0.child.json').exists():
            if time.monotonic()>deadline:raise AssertionError('no child')
            time.sleep(.01)
        (self.j.directory/'STOP').touch();start=time.monotonic()
        with self.assertRaisesRegex(RecoveryError,'CANCELLED_NO_RETRY|FAILED_PRESERVED'):self.handoff()
        self.proc.communicate(timeout=5);self.assertLess(time.monotonic()-start,6)

    def test_fresh_dispatch_rechecks_budget_after_queue_reservation(self):
        self.unclaimed()
        with self.j.writer():
            s=self.j.read();s['budget']['used']['queue_seconds']=21600;self.j.save(s)
        with self.assertRaises(ValueError):self.handoff()
        self.assertFalse(self.folder.exists())

    def test_recovered_gap_ends_partial_window_without_new_sample(self):
        self.unclaimed();self.handoff()
        with self.j.writer():
            s=self.j.read();s['events'][self.key]['samples'][0]['monotonic_ns']-=9*10**9;self.j.save(s)
            result=collect_window(self.j,self.key,file_pin(self.exe))
        self.assertEqual(result['status'],'INTERRUPTED');self.assertEqual(result['window']['sample_count'],1)
        self.assertFalse((self.folder.parent/'sample-1').exists())

    def test_late_receipt_cannot_become_current_resource_sample(self):
        self.prepare();self.launch();self.proc.communicate(timeout=5)
        with self.j.writer():
            s=self.j.read();s['events'][self.key]['sample_claims']['0']['deadline_monotonic']=time.monotonic()-10;self.j.save(s)
        with self.assertRaisesRegex(RecoveryError,'LATE_OR_CANCELLED'):self.handoff()
        self.assertEqual(self.j.read()['events'][self.key]['samples'],[])


    def test_stale_unconsumed_receipt_ends_window_without_requery(self):
        self.prepare();self.launch();self.proc.communicate(timeout=5)
        from tools import r6_resource_queue as queue
        clock=queue.now();clock['monotonic_ns']+=9*10**9
        with self.j.writer(),patch.object(queue,'now',return_value=clock):
            result=collect_window(self.j,self.key,file_pin(self.exe))
        self.assertEqual(result['status'],'INTERRUPTED')
        self.assertEqual(result['window']['sample_count'],0)
        self.assertTrue(self.j.read()['events'][self.key]['sample_claims']['0']['sample_stale'])

    def test_full_thirteen_sample_local_window_has_zero_live_or_model_use(self):
        self.unclaimed()
        with self.j.writer():result=collect_window(self.j,self.key,file_pin(self.exe))
        self.assertEqual(result['status'],'OBSERVATION_PASS');self.assertFalse(result['launch_admission'])
        self.assertEqual(result['window']['sample_count'],13)
        self.assertGreaterEqual(result['window']['span_ns'],60*10**9)
        from tools.r6_budget import totals
        usage=totals(self.j.read()['budget'])
        self.assertEqual(usage['queue_seconds'],180)
        self.assertEqual(usage['live_starts'],0);self.assertEqual(usage['automatic_model_sessions'],0)
        entry=self.j.read()['events'][self.key]
        self.assertEqual(len(entry['sample_claims']),13)
        self.assertTrue(all(c['receipt'] for c in entry['sample_claims'].values()))


    def test_busy_window_backoff_then_fresh_complete_window(self):
        from tools import r6_resource_wait as wait
        from tools import r6_resource_queue as queue
        from tools.r6_budget import totals
        self.unclaimed()
        # Isolated synthetic clock for the loop; the full actual subprocess
        # window is independently exercised by the preceding integration test.
        clock=queue.now();calls=[];sleeps=[]
        with self.j.writer():
            state=self.j.read();state['events'][self.key]['queue_clock']=dict(clock);self.j.save(state)
        def sleep(seconds):
            sleeps.append(seconds);clock['monotonic_ns']+=int(seconds*1e9)
        def collect(journal,key,executable,**kwargs):
            number=len(calls);calls.append(key)
            from training_workloads.scale_003.resources import GPU_IDENTITY
            for i in range(13):
                if i:clock['monotonic_ns']+=5*10**9
                sample=dict(monotonic_ns=clock['monotonic_ns'],gpu=dict(GPU_IDENTITY,
                    memory_used_mib=560,free_memory_mib=11667,utilization_percent=6 if number==0 else 3),compute_apps=[])
                queue.append_sample(journal,key,i,sample)
            return queue.finish(journal,key)
        with self.j.writer(),patch.object(queue,'now',side_effect=lambda:dict(clock)),patch.object(wait,'collect_window',side_effect=collect),patch.object(wait.time,'sleep',side_effect=sleep):
            result=wait.wait_for_observation(self.j,file_pin(self.spec),file_pin(self.exe),free_disk_bytes=500*1024**3)
        self.assertEqual(result['status'],'OBSERVATION_PASS');self.assertEqual(len(set(calls)),2)
        self.assertEqual(sum(sleeps),60);self.assertTrue(all(n<=15 for n in sleeps))
        self.assertEqual(totals(self.j.read()['budget'])['queue_seconds'],360)


    def test_new_sample_refused_while_model_handoff_active(self):
        self.unclaimed()
        with self.j.writer():
            state=self.j.read();state['current_model_event']='CPU-MODEL-IN-FLIGHT';self.j.save(state)
        with self.assertRaisesRegex(RecoveryError,'NOT_IDLE'):self.handoff()
        self.assertFalse(self.folder.exists())


if __name__=='__main__':unittest.main()
