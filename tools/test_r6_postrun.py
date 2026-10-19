"""CPU fake-query receipts and restart tests; never execute nvidia-smi or Codex."""
import json
import copy
import signal
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from tools import test_r6_job_inventory as fixture
from tools.r6_recovery import RecoveryError,file_pin,Journal
from tools.r6_postrun import register_postruns,complete_postruns,postrun_handoff,CAP
from tools.r6_job_inventory import record_inventory
from tools.r6_batch import worker_environment
from rlflow.jobs import owns,identity
from rlflow.store import sha
from tools.r6_budget import observe,BudgetError

ROOT = fixture.fixture.ROOT
FAKE = '''#!{python}
import sys,time,os
from pathlib import Path
p=Path(__file__).with_suffix('.calls')
with p.open('a') as f:f.write(sys.argv[1]+'\\n');f.flush();os.fsync(f.fileno())
{body}
print('CPU FIXTURE ONLY')
'''


class PostrunTests(unittest.TestCase):
    save = fixture.InventoryTests.save
    finish = fixture.InventoryTests.finish
    cancel = fixture.InventoryTests.cancel
    handoff = fixture.InventoryTests.handoff
    record = fixture.InventoryTests.record

    def setUp(self):
        fixture.InventoryTests.setUp(self)
        self.record()
        self.job_key = self.key
        self.exe = self.base/'fake-query'
        self.controller = None

    def tearDown(self):
        if self.controller is not None and self.controller.poll() is None:
            self.controller.terminate();self.controller.wait(timeout=10)
        # Query worker is bounded and normally joined by the handoff; retain
        # temporary paths until an intentionally interrupted test is reconciled.
        fixture.InventoryTests.tearDown(self)

    def register(self,body='time.sleep(.03)'):
        self.exe.write_text(FAKE.format(python=sys.executable,body=body));self.exe.chmod(0o700)
        with self.j.writer():
            self.keys = register_postruns(self.j,self.job_key,file_pin(self.exe),free_disk_bytes=500*1024**3)
        return self.keys

    def complete(self):
        with self.j.writer():return complete_postruns(self.j,self.job_key,self.keys,poll_seconds=.01)

    def count(self):
        path = self.exe.with_suffix('.calls')
        return len(path.read_text().splitlines()) if path.exists() else 0

    def test_two_fixed_queries_run_once_and_repeat_only_validates(self):
        self.register();result = self.complete();self.assertEqual(self.count(),2)
        self.assertTrue(result['postrun_complete']);self.assertFalse(result['scientific_acceptance'])
        self.assertEqual(self.complete(),result);self.assertEqual(self.count(),2)
        self.assertEqual(len(self.j.read()['budget']['reservations']),3)
        self.assertEqual(self.j.read()['status'],'POSTRUN_COMPLETE_PENDING_ANALYSIS')

    def test_postrun_success_does_not_clear_missing_output_state(self):
        # Synthetic incomplete inventory; inventory generation itself is tested
        # in test_r6_job_inventory with an actually missing attempt directory.
        with self.j.writer():
            state=self.j.read();job=state['events'][self.job_key]
            path=Path(job['inventory']['path']);value=json.loads(path.read_text())
            value['output_inventory_complete']=False;value['missing_roots']=['attempt']
            path.write_text(json.dumps(value));job['inventory']=file_pin(path)
            state['status']='JOB_EVIDENCE_INCOMPLETE';self.j.save(state)
        self.register();result=self.complete()
        self.assertTrue(result['postrun_complete'])
        self.assertEqual(self.j.read()['status'],'JOB_EVIDENCE_INCOMPLETE')

    def test_restart_after_first_query_does_not_repeat_it(self):
        self.register()
        with self.j.writer():postrun_handoff(self.j,self.keys[0],poll_seconds=.01)
        self.assertEqual(self.count(),1);self.j=Journal(self.j.directory)
        self.complete();self.assertEqual(self.count(),2)

    def test_claim_without_worker_is_ambiguous_and_never_retried(self):
        self.register()
        with self.j.writer():
            state=self.j.read();state['events'][self.keys[0]]['dispatch_claim']={'claimed':True};self.j.save(state)
            with self.assertRaisesRegex(RecoveryError,'AMBIGUOUS_NO_RETRY'):
                postrun_handoff(self.j,self.keys[0],poll_seconds=.01)
        self.assertEqual(self.count(),0)

    def test_stop_before_query_keeps_unexecuted_events(self):
        self.register();(self.j.directory/'STOP').touch()
        with self.assertRaisesRegex(RecoveryError,'STOP'):self.complete()
        self.assertEqual(self.count(),0)
        self.assertNotIn('dispatch_claim',self.j.read()['events'][self.keys[0]])

    def test_nonzero_query_receipt_preserved_no_second_query(self):
        self.register('sys.exit(17)')
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.complete()
        entry=self.j.read()['events'][self.keys[0]];receipt=json.loads(Path(entry['receipt']['path']).read_text())
        self.assertEqual(receipt['exit_code'],17);self.assertEqual(self.count(),1)
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.complete()
        self.assertEqual(self.count(),1)

    def test_large_output_bounded_before_write(self):
        self.register("sys.stdout.write('x'*2000000);sys.stdout.flush()")
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.complete()
        call=self.j.directory/'postrun'/self.keys[0]
        self.assertLessEqual(sum(p.stat().st_size for p in call.iterdir()),CAP)
        receipt=json.loads((call/'receipt.json').read_text())
        self.assertIn('OUTPUT_CAP_BEFORE_WRITE',receipt['error']);self.assertEqual(self.count(),1)

    def test_query_artifact_drift_rejected(self):
        self.register();self.complete()
        (self.j.directory/'postrun'/self.keys[0]/'stdout').write_text('changed')
        with self.assertRaisesRegex(RecoveryError,'PIN_DRIFT'):self.complete()
        self.assertEqual(self.count(),2)

    def test_changed_executable_cannot_rebind_same_ordinal(self):
        self.register();self.complete()
        self.exe.write_text(self.exe.read_text()+'\n# changed\n')
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'ORDINAL_ALREADY_BOUND'):
            register_postruns(self.j,self.job_key,file_pin(self.exe),free_disk_bytes=500*1024**3)
        self.assertEqual(self.count(),2)

    def test_summary_before_state_crash_recovers(self):
        self.register();original=self.j.save
        def saved(state):
            if state['events'][self.job_key].get('postrun'):raise RuntimeError('CPU_CRASH')
            return original(state)
        with patch.object(self.j,'save',side_effect=saved):
            with self.assertRaisesRegex(RuntimeError,'CPU_CRASH'):self.complete()
        path=self.j.directory/'jobs'/self.job_key/'postrun-summary.json';pin=file_pin(path)
        self.complete();self.assertEqual(pin,self.j.read()['events'][self.job_key]['postrun'])
        self.assertEqual(self.count(),2)

    def test_controller_death_recovers_exact_live_query_worker(self):
        self.register('time.sleep(.7)')
        script='''from tools.r6_recovery import Journal
from tools.r6_postrun import postrun_handoff
import sys
j=Journal(sys.argv[1])
with j.writer():postrun_handoff(j,sys.argv[2],poll_seconds=.01)
'''
        self.controller=subprocess.Popen([sys.executable,'-B','-c',script,str(self.j.directory),self.keys[0]],
            cwd=ROOT,env=worker_environment(),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        child=self.j.directory/'postrun'/self.keys[0]/'child.json';deadline=time.monotonic()+5
        while not child.exists() and time.monotonic()<deadline:time.sleep(.01)
        self.assertTrue(child.exists());worker=self.j.read()['events'][self.keys[0]]['process']
        self.assertTrue(owns(worker));self.controller.kill();self.controller.wait(timeout=5)
        self.complete();self.assertEqual(self.count(),2)
        self.assertEqual(worker,self.j.read()['events'][self.keys[0]]['process'])

    def test_timeout_stops_owned_query_and_keeps_receipt(self):
        self.register('time.sleep(30)')
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.complete()
        entry=self.j.read()['events'][self.keys[0]]
        receipt=json.loads(Path(entry['receipt']['path']).read_text())
        self.assertIn('POSTRUN_TIMEOUT',receipt['error'])
        self.assertFalse(owns(receipt['child_identity']))
        self.assertEqual(receipt['cleanup']['status'],'VERIFIED_NO_OWNED_DESCENDANTS')
        self.assertEqual(self.count(),1)

    def test_orphan_descendant_is_reaped_when_query_leader_exits(self):
        body="""import subprocess
from rlflow.jobs import identity
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],
 start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
Path(__file__).with_suffix('.descendant').write_text(__import__('json').dumps(identity(child.pid)))
"""
        self.register(body)
        self.complete()
        descendant=json.loads(self.exe.with_suffix('.descendant').read_text())
        self.assertFalse(owns(descendant))
        for key in self.keys:
            receipt=json.loads(Path(self.j.read()['events'][key]['receipt']['path']).read_text())
            self.assertGreaterEqual(receipt['cleanup']['signalled_process_count'],1)
            self.assertFalse(any(owns(i) for i in receipt['cleanup']['observed_identities']))

    def test_existing_job_aliases_cannot_register_or_dispatch_queries(self):
        self.register()
        with self.j.writer():
            state=self.j.read();alias=copy.deepcopy(state['events'][self.job_key])
            alias['identity']['event_id']='CPU-ALIAS'
            alias_key=sha(alias['identity']);state['events'][alias_key]=alias;self.j.save(state)
            with self.assertRaisesRegex(RecoveryError,'JOB_EVENT_ALIAS'):
                register_postruns(self.j,alias_key,file_pin(self.exe),free_disk_bytes=500*1024**3)
            with self.assertRaisesRegex(RecoveryError,'JOB_EVENT_ALIAS'):
                postrun_handoff(self.j,self.keys[0],poll_seconds=.01)
        self.assertEqual(self.count(),0)

    def test_budget_excess_between_intent_and_dispatch_blocks_query(self):
        self.register()
        with self.j.writer():
            state=self.j.read();entry=state['events'][self.keys[0]]
            state['budget']=observe(state['budget'],event_key=entry['identity'],
                resources={'live_starts':65},evidence={'path':'/CPU_ONLY','sha256':'a'*64})
            self.j.save(state)
            with self.assertRaises(BudgetError):postrun_handoff(self.j,self.keys[0],poll_seconds=.01)
        self.assertEqual(self.count(),0)
        self.assertNotIn('dispatch_claim',self.j.read()['events'][self.keys[0]])

    def stuck_worker(self, *, expired):
        self.register()
        self.controller=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        worker=identity(self.controller.pid);os.kill(self.controller.pid,signal.SIGSTOP)
        deadline=time.monotonic()-1 if expired else time.monotonic()+45
        with self.j.writer():
            self.j.bind_process(self.keys[0],worker)
            state=self.j.read();state['events'][self.keys[0]]['dispatch_claim']=dict(
                controller=identity(os.getpid()),worker=file_pin(ROOT/'tools/r6_postrun.py'),
                deadline_monotonic=deadline,deadline_wall=time.time()+45)
            self.j.save(state)
        return worker,deadline

    def test_restart_past_original_deadline_kills_stopped_worker_not_unrelated(self):
        worker,deadline=self.stuck_worker(expired=True)
        other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        unrelated=identity(other.pid)
        try:
            self.j=Journal(self.j.directory);start=time.monotonic()
            with self.assertRaisesRegex(RecoveryError,'CANCELLED_NO_RETRY'):self.complete()
            self.assertLess(time.monotonic()-start,7)
            self.assertFalse(owns(worker));self.assertTrue(owns(unrelated))
            entry=self.j.read()['events'][self.keys[0]]
            self.assertEqual(entry['dispatch_claim']['deadline_monotonic'],deadline)
            self.assertEqual(entry['controller_cancel']['status'],'NO_KNOWN_OWNED_SURVIVORS')
            self.assertFalse((self.j.directory/'postrun'/self.keys[0]/'receipt.json').exists())
            self.assertEqual(self.count(),0)
        finally:other.terminate();other.wait(timeout=5)

    def test_stop_cancels_stopped_worker_without_waiting_for_its_own_timer(self):
        worker,_=self.stuck_worker(expired=False)
        (self.j.directory/'STOP').touch();start=time.monotonic()
        with self.assertRaisesRegex(RecoveryError,'CANCELLED_NO_RETRY'):self.complete()
        self.assertLess(time.monotonic()-start,7);self.assertFalse(owns(worker));self.assertEqual(self.count(),0)

    def test_partial_receipt_does_not_launch_query(self):
        self.register();folder=self.j.directory/'postrun'/self.keys[0];folder.mkdir(parents=True)
        (folder/'receipt.json').write_text('{')
        with self.j.writer():
            state=self.j.read();state['events'][self.keys[0]]['dispatch_claim']={'claimed':True};self.j.save(state)
        with self.assertRaisesRegex(RecoveryError,'AMBIGUOUS_NO_RETRY'):self.complete()
        self.assertEqual(self.count(),0);self.assertEqual((folder/'receipt.json').read_text(),'{')


if __name__=='__main__':unittest.main()
