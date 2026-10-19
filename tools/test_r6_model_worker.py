"""CPU executable fixtures, never the Codex CLI or an API call."""
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from rlflow.jobs import identity
from rlflow.store import sha
from tools.r6_recovery import Journal,file_pin

ROOT=Path(__file__).resolve().parents[1]
FAKE='''#!{python}
import json,sys,time,subprocess
from pathlib import Path
prompt=sys.stdin.read()
if 'descendant' in prompt:
 from rlflow.jobs import identity
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],start_new_session=True,
     stdout=subprocess.DEVNULL if 'closed' in prompt else None,stderr=subprocess.DEVNULL)
 Path(sys.argv[0]).with_suffix('.child').write_text(json.dumps(identity(child.pid)))
if 'wait' in prompt:time.sleep(10)
text='{{}}' if 'malformed' in prompt else json.dumps({{'decision':'CPU_FIXTURE_ONLY'}})
if 'oversize' in prompt:text='x'*300000
print(json.dumps({{'type':'item.completed','item':{{'type':'agent_message','text':text}}}}))
if 'manyusage' in prompt:
 for i in range(1000):print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':7,'output_tokens':3,'cached_input_tokens':0}}}}))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':7,'output_tokens':3,'cached_input_tokens':0}}}}))
'''


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        exe=self.base/'fake-codex';exe.write_text(FAKE.format(python=sys.executable));exe.chmod(0o700)
        prompt=self.base/'prompt.txt';prompt.write_text('CPU fixture')
        schema=self.base/'schema.json';schema.write_text(json.dumps({'type':'object','properties':{'decision':{'const':'CPU_FIXTURE_ONLY'}},'required':['decision'],'additionalProperties':False}))
        self.request=dict(prompt=file_pin(prompt),schema=file_pin(schema),executable=file_pin(exe),cwd=str(ROOT),
                          sandbox='read-only',timeout_seconds=5,output_cap_bytes=1024**2,stop_paths=[])
        self.j=Journal(self.base/'batch')
        budget=json.loads((ROOT/'tools/fixtures/r6_budget.json').read_text())
        with self.j.writer():self.j.initialize(budget)
        self.proc=None

    def tearDown(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate();self.proc.wait(timeout=20)
        self.tmp.cleanup()

    def launch(self,prompt=None,timeout=None):
        if prompt is not None:
            p=Path(self.request['prompt']['path']);p.write_text(prompt);self.request['prompt']=file_pin(p)
        if timeout is not None:self.request['timeout_seconds']=timeout
        event=dict(goal_id=self.j.read()['goal_id'],event_id='CPU-REVIEW',job_id='CPU-NO-JOB',
                   input_sha256=sha(self.request),role='review')
        with self.j.writer():
            self.key,fresh=self.j.intent(event,kind='model',request=self.request,
                resources={'automatic_model_sessions':1},purpose='development',free_disk_bytes=500*1024**3)
            self.assertTrue(fresh)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.:tools/rlflow',CUDA_VISIBLE_DEVICES='')
            self.proc=subprocess.Popen([sys.executable,'-B','-m','tools.r6_model_worker',str(self.j.directory),self.key],
                cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            self.j.bind_process(self.key,identity(self.proc.pid))
        self.call=self.j.directory/'calls'/self.key

    def result(self):
        self.proc.wait(timeout=20)
        self.assertTrue((self.call/'receipt.json').exists(),
                        f'Worker exited {self.proc.returncode} without receipt; run with the existing orchestrator Python (jsonschema required).')
        return json.loads((self.call/'receipt.json').read_text())

    def test_schema_usage_receipt_and_recovery_consume_once(self):
        self.launch();r=self.result()
        self.assertTrue(r['schema_valid']);self.assertIsNone(r['error']);self.assertEqual(r['exit_code'],0)
        self.assertEqual(r['usage']['totals']['input_tokens'],7)
        recovered=Journal(self.j.directory)
        with recovered.writer():
            self.assertTrue(recovered.consume(self.key,self.call/'receipt.json'))
            self.assertFalse(recovered.consume(self.key,self.call/'receipt.json'))
        self.assertEqual(len(recovered.read()['budget']['reservations']),1)
        again=subprocess.run([sys.executable,'-B','-m','tools.r6_model_worker',str(self.j.directory),self.key],
            cwd=ROOT,env=dict(os.environ,PYTHONPATH='.:tools/rlflow',PYTHONDONTWRITEBYTECODE='1'),capture_output=True)
        self.assertNotEqual(again.returncode,0)

    def test_invalid_schema_is_not_accepted_as_decision(self):
        self.launch('malformed');r=self.result()
        self.assertFalse(r['schema_valid']);self.assertIsNotNone(r['error'])
        self.assertEqual(r['exit_code'],0) # Native CLI exit preserved separately.
        self.assertEqual(self.proc.returncode,1)

    def test_timeout_cleans_child_and_keeps_session_charge(self):
        self.launch('wait',timeout=.3);r=self.result()
        self.assertIn('MODEL_TIMEOUT',r['error']);self.assertFalse(r['schema_valid'])
        from rlflow.jobs import owns
        self.assertFalse(owns(r['child_identity']))
        self.assertEqual(len(self.j.read()['budget']['reservations']),1)

    def test_stop_during_worker_no_retry_or_stop_removal(self):
        self.launch('wait')
        deadline=time.monotonic()+10
        while not (self.call/'child.json').exists() and time.monotonic()<deadline:time.sleep(.02)
        self.assertTrue((self.call/'child.json').exists())
        (self.j.directory/'STOP').touch()
        r=self.result();self.assertIn('STOP_DURING_MODEL',r['error'])
        self.assertTrue((self.j.directory/'STOP').exists())
        self.assertFalse(r['automatic_retry'])

    def test_worker_death_preserves_ambiguous_event_without_retry(self):
        self.launch('wait')
        deadline=time.monotonic()+10
        while not (self.call/'child.json').exists() and time.monotonic()<deadline:time.sleep(.02)
        self.assertTrue((self.call/'child.json').exists())
        child=json.loads((self.call/'child.json').read_text())['identity']
        from rlflow.jobs import owns
        try:
            self.proc.kill();self.proc.wait(timeout=5)
            self.assertFalse((self.call/'receipt.json').exists())
            self.assertEqual(self.j.observe(self.key)['action'],'RECONCILE')
            self.assertEqual(len(self.j.read()['budget']['reservations']),1)
            self.assertTrue((self.call/'ONCE').exists())
        finally:
            if owns(child):os.killpg(child['pgid'],signal.SIGKILL)

    def test_cpu_job_completion_event_and_registered_next_step(self):
        from rlflow import jobs
        root=self.base/'cpu-jobs';spec={'backend':'fake','duration':.01}
        first=jobs.submit(root,'CPU-FIRST',spec)
        deadline=time.monotonic()+5
        while jobs.status(root,'CPU-FIRST')['state'] not in jobs.TERMINAL and time.monotonic()<deadline:time.sleep(.02)
        self.assertEqual(jobs.status(root,'CPU-FIRST')['state'],'SUCCEEDED')
        self.launch();r=self.result();self.assertTrue(r['schema_valid'])
        with self.j.writer():self.j.consume(self.key,self.call/'receipt.json')
        next_job=jobs.submit(root,'CPU-REGISTERED-NEXT',spec)
        replay=jobs.submit(root,'CPU-REGISTERED-NEXT',spec)
        self.assertEqual(next_job['nonce'],replay['nonce'])
        deadline=time.monotonic()+5
        while jobs.status(root,'CPU-REGISTERED-NEXT')['state'] not in jobs.TERMINAL and time.monotonic()<deadline:time.sleep(.02)
        self.assertEqual(jobs.status(root,'CPU-REGISTERED-NEXT')['state'],'SUCCEEDED')
        self.assertEqual(jobs.status(root,'CPU-REGISTERED-NEXT')['start_attempts'],1)
        for proc in jobs._CHILDREN:
            proc.wait(timeout=5)

    def test_cli_exit_with_detached_child_closed_pipes_is_cleaned(self):
        self.launch('descendant closed');r=self.result()
        from rlflow.jobs import owns
        child=json.loads((self.base/'fake-codex.child').read_text())
        self.assertFalse(owns(child))
        self.assertEqual(r['cleanup']['status'],'VERIFIED_NO_OWNED_DESCENDANTS')
        self.assertGreaterEqual(r['cleanup']['signalled_process_count'],1)

    def test_cli_exit_with_child_holding_pipes_is_cleaned_on_timeout(self):
        self.launch('descendant',timeout=.3);r=self.result()
        from rlflow.jobs import owns
        child=json.loads((self.base/'fake-codex.child').read_text())
        self.assertFalse(owns(child));self.assertIn('MODEL_TIMEOUT',r['error'])
        self.assertEqual(r['exit_code'],0)

    def test_final_message_and_usage_summary_stay_bounded(self):
        self.launch('oversize');r=self.result()
        self.assertIn('FINAL_MESSAGE_CAP',r['error'])
        self.assertIsNone(r['last_message'])
        self.assertLess(sum(p.stat().st_size for p in self.call.iterdir()),self.request['output_cap_bytes'])

    def test_many_usage_records_do_not_expand_receipt(self):
        self.launch('manyusage');r=self.result()
        self.assertEqual(r['usage']['reported_turn_count'],1001)
        self.assertEqual(r['usage']['totals']['input_tokens'],7007)
        self.assertLess((self.call/'receipt.json').stat().st_size,4096)

    def test_minimum_cap_rejects_stream_before_overflow(self):
        self.request['output_cap_bytes']=65536
        self.launch('manyusage');r=self.result()
        self.assertIn('MODEL_LOG_CAP_BEFORE_WRITE',r['error'])
        self.assertLessEqual(sum(p.stat().st_size for p in self.call.iterdir()),65536)
        self.assertFalse(r['schema_valid'])


if __name__=='__main__':unittest.main()
