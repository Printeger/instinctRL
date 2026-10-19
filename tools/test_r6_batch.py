"""R6 handoff restart tests; fake CLI only, no API or simulator."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

from rlflow.jobs import owns
from rlflow.store import sha
from tools.r6_batch import model_handoff,worker_environment
from tools.r6_recovery import RecoveryError,file_pin
from tools.r6_budget import observe,BudgetError
from tools import test_r6_model_worker as fixture

ROOT=fixture.ROOT


class BatchTests(unittest.TestCase):
    setUp=fixture.WorkerTests.setUp
    tearDown=fixture.WorkerTests.tearDown

    def register(self,prompt='CPU fixture'):
        p=Path(self.request['prompt']['path']);p.write_text(prompt);self.request['prompt']=file_pin(p)
        event=dict(goal_id=self.j.read()['goal_id'],event_id='CPU-BATCH-REVIEW',job_id='CPU-NO-JOB',
                   input_sha256=sha(self.request),role='review')
        with self.j.writer():
            self.key,_=self.j.intent(event,kind='model',request=self.request,
                resources={'automatic_model_sessions':1},purpose='development',free_disk_bytes=500*1024**3)
        self.call=self.j.directory/'calls'/self.key

    def start_controller(self):
        script='''import sys
from tools.r6_recovery import Journal
from tools.r6_batch import model_handoff
j=Journal(sys.argv[1])
with j.writer():model_handoff(j,sys.argv[2],poll_seconds=.02)
'''
        self.proc=subprocess.Popen([sys.executable,'-B','-c',script,str(self.j.directory),self.key],
            cwd=ROOT,env=worker_environment(),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            start_new_session=True)

    def test_consumed_review_is_not_relaunched(self):
        self.register()
        with self.j.writer():result=model_handoff(self.j,self.key,poll_seconds=.02)
        self.assertEqual(result,{'decision':'CPU_FIXTURE_ONLY'})
        receipt=file_pin(self.call/'receipt.json')
        with self.j.writer():self.assertEqual(result,model_handoff(self.j,self.key,poll_seconds=.02))
        self.assertEqual(receipt,file_pin(self.call/'receipt.json'))
        self.assertEqual(len(self.j.read()['budget']['reservations']),1)

    def test_budget_excess_after_intent_blocks_first_model_dispatch(self):
        self.register()
        with self.j.writer():
            state=self.j.read();event=state['events'][self.key]
            state['budget']=observe(state['budget'],event_key=event['identity'],
                resources={'automatic_model_sessions':33},evidence={'path':'/CPU_ONLY','sha256':'a'*64})
            self.j.save(state)
            with self.assertRaises(BudgetError):model_handoff(self.j,self.key,poll_seconds=.01)
        self.assertFalse(self.call.exists())
        self.assertNotIn('dispatch_claim',self.j.read()['events'][self.key])

    def test_controller_restart_reattaches_exact_live_worker(self):
        self.restart_reattaches(clear_binding=False)

    def test_restart_after_popen_before_binding_recovers_worker_identity(self):
        self.restart_reattaches(clear_binding=True)

    def restart_reattaches(self,clear_binding):
        # Enough time for restart; unlike 'wait', this finishes within the cap.
        exe=Path(self.request['executable']['path'])
        exe.write_text(exe.read_text().replace("if 'wait' in prompt:time.sleep(10)","time.sleep(.6)"))
        self.request['executable']=file_pin(exe)
        self.register();self.start_controller()
        deadline=time.monotonic()+5
        while not (self.call/'child.json').exists() and time.monotonic()<deadline:time.sleep(.01)
        self.assertTrue((self.call/'child.json').exists())
        original=self.j.read()['events'][self.key]['process'];self.assertTrue(owns(original))
        self.proc.kill();self.proc.wait(timeout=5)
        with self.j.writer():
            if clear_binding:
                state=self.j.read();entry=state['events'][self.key]
                entry.update(process=None,status='INTENT');self.j.save(state)
            result=model_handoff(self.j,self.key,poll_seconds=.02)
        self.assertEqual(result['decision'],'CPU_FIXTURE_ONLY')
        self.assertEqual(original,self.j.read()['events'][self.key]['process'])
        self.assertEqual(len(self.j.read()['budget']['reservations']),1)

    def test_dispatch_claim_without_handle_never_launches(self):
        self.register()
        with self.j.writer():
            state=self.j.read();state['events'][self.key]['dispatch_claim']={'time':0}
            self.j.save(state)
            with self.assertRaisesRegex(RecoveryError,'AMBIGUOUS_NO_RETRY'):
                model_handoff(self.j,self.key,poll_seconds=.02)
        self.assertFalse(self.call.exists())

    def test_stop_before_dispatch_preserves_unstarted_event(self):
        self.register();(self.j.directory/'STOP').touch()
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'STOP_NEW_ACTIONS'):
            model_handoff(self.j,self.key,poll_seconds=.02)
        self.assertFalse(self.call.exists())
        self.assertNotIn('dispatch_claim',self.j.read()['events'][self.key])

    def test_failed_schema_recorded_but_not_accepted(self):
        self.register('malformed')
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'MODEL_FAILED_PRESERVE_RECEIPT'):
            model_handoff(self.j,self.key,poll_seconds=.02)
        entry=self.j.read()['events'][self.key]
        self.assertEqual(entry['status'],'RECEIPT_RECORDED')
        self.assertTrue(Path(entry['receipt']['path']).exists())

    def test_completed_artifact_drift_refuses_replay(self):
        self.register()
        with self.j.writer():model_handoff(self.j,self.key,poll_seconds=.02)
        (self.call/'last-message.json').write_text('{}')
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'PIN_DRIFT'):
            model_handoff(self.j,self.key,poll_seconds=.02)


if __name__=='__main__':unittest.main()
