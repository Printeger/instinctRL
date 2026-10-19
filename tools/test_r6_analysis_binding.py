"""Binding unit tests; process cleanup itself is covered by job-handoff tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from rlflow.store import sha
from tools.r6_recovery import file_pin, RecoveryError, Journal
from tools.r6_analysis_binding import bind_analysis, record_analysis
from training_workloads.unified_h1 import robustness_analysis as analysis, robustness


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.attempt=self.root/'attempts/CPU';self.output=self.attempt/'results/research-r6-evaluation'
        self.output.mkdir(parents=True);self.key='CPU-EVENT'
        protocol=json.loads((Path(__file__).resolve().parents[1]/'docs/experiments/r6-robustness-auto/TEST_PROTOCOL.json').read_text())
        protocol['frozen']=True;self.protocol=self.write('protocol.json',protocol)
        self.request=dict(protocol=self.protocol,block_ids=[b['id'] for b in protocol['blocks']],run_id='CPU',seed=17,split='development')
        self.auth=self.write('attempts/CPU/commands/AUTHORIZATION.json',dict(mode='research-r6-evaluation',parameters={'r6':self.request}))
        self.spec=dict(id='CPU',h1=dict(attempt=str(self.attempt),mode='research-r6-evaluation'),pins={'authorization':self.auth},
            argv=['python','--authorization',self.auth['path'],'--attempt',str(self.attempt),'--mode','research-r6-evaluation'])
        self.spec_pin=self.write('spec.json',self.spec);self.jid='J-'+sha({'id':'CPU','spec':self.spec})[:16]
        self.job=dict(id=self.jid,spec=self.spec,status='FAILED')
        self.store=SimpleNamespace(root=self.root,replay=lambda:{'jobs':{self.jid:self.job}})
        self.closure=self.write('close.json',dict(event_key=self.key,job_id=self.jid,spec_sha256=sha(self.spec),native_status='FAILED',native_exit_code=-9))
        self.raw=self.write('attempts/CPU/results/research-r6-evaluation/raw.json',{'fixture':True})
        self.inventory=dict(event_key=self.key,job_id=self.jid,spec=self.spec_pin,closure=self.closure,roots={'attempt':str(self.attempt)},
            files=[dict(**p,bytes=Path(p['path']).stat().st_size) for p in (self.auth,self.raw)],output_inventory_complete=True)
        self.inventory_pin=self.write('inventory.json',self.inventory)
        receipts=[self.write('receipt'+str(i)+'.json',{}) for i in range(2)]
        self.postrun=self.write('postrun.json',dict(job_event_key=self.key,job_id=self.jid,closure=self.closure,inventory=self.inventory_pin,
            postrun_complete=True,postrun_queries_used=2,receipts=receipts))
        self.entry=dict(identity={'job_id':self.jid},request={'spec':self.spec_pin},closure=self.closure,inventory=self.inventory_pin,postrun=self.postrun)
        self.report=dict(schema_version='r6-motion-replay/1',scientific_acceptance=False,run_id='CPU',seed=17,split='development',
            blocks=[dict(block_id=b['id'],replay_complete=False,saved_result_equal=False) for b in protocol['blocks']],
            analyzer=analysis.pin(analysis.__file__),scorer=analysis.pin(robustness.__file__),
            inputs=[analysis.pin(self.protocol['path']),analysis.pin(self.raw['path'])],summary={'original':{'replay_complete':False}})
        root=Path(__file__).resolve().parents[1]
        self.report['measurement_replayer']=analysis.pin(root/'training_workloads/unified_h1/robustness_measurement_replay.py')
        self.report['delivery_implementation']=analysis.pin(root/'isaaclab-training/instinctlab/observations/robustness.py')
        cleanup=patch('tools.r6_analysis_binding.cleanup_evidence',return_value=({},dict(status='FAILED',exit_code=-9)))
        cleanup.start();self.addCleanup(cleanup.stop)

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(analysis.encoded(value));return file_pin(path)

    def bind(self):
        return bind_analysis(self.entry,self.key,self.write('report.json',self.report),self.store)

    def test_missing_evidence_remains_incomplete_native_failure_retained(self):
        result=self.bind()
        self.assertEqual(result['native_exit_code'],-9)
        self.assertEqual(result['issues'],['MODEL_BINDING_MISSING','MEASUREMENT_REPLAY_INCOMPLETE','MOTION_EVIDENCE_INCOMPLETE_OR_MISMATCHED'])
        self.assertFalse(result['scientific_acceptance'])

    def test_other_job_closure(self):
        value=json.loads(Path(self.closure['path']).read_text());value['job_id']='OTHER'
        self.entry['closure']=self.write('wrong-close.json',value)
        with self.assertRaisesRegex(RecoveryError,'JOB_CHAIN_BINDING'):self.bind()

    def test_other_attempt_input(self):
        other=self.write('another-attempt/raw.json',{})
        self.report['inputs'].append(analysis.pin(other['path']))
        with self.assertRaisesRegex(RecoveryError,'OTHER_ATTEMPT'):self.bind()

    def test_raw_changed_after_inventory(self):
        Path(self.raw['path']).write_text('{}')
        with self.assertRaises(RecoveryError):self.bind()

    def test_wrong_seed(self):
        self.report['seed']=18
        with self.assertRaisesRegex(RecoveryError,'RUN_IDENTITY'):self.bind()

    def test_uninventoried_original(self):
        extra=self.write('attempts/CPU/results/research-r6-evaluation/unlisted.json',{})
        self.report['inputs'].append(analysis.pin(extra['path']))
        with self.assertRaisesRegex(RecoveryError,'EXACT_INVENTORY'):self.bind()

    def test_duplicate_input(self):
        self.report['inputs'].append(self.report['inputs'][0])
        with self.assertRaisesRegex(RecoveryError,'DUPLICATE_INPUT'):self.bind()

    def test_omitted_protocol(self):
        self.report['inputs'].pop(0)
        with self.assertRaisesRegex(RecoveryError,'PROTOCOL_INPUT_MISSING'):self.bind()

    def test_complete_claim_without_originals(self):
        self.report['blocks'][0]['replay_complete']=True
        with self.assertRaisesRegex(RecoveryError,'COMPLETE_BLOCK_WITHOUT_INPUTS'):self.bind()
        self.report['blocks'][0]['replay_complete']=False;self.report['summary']['original']['replay_complete']=True
        with self.assertRaisesRegex(RecoveryError,'COMPLETE_ORIGINAL_WITHOUT_INPUTS'):self.bind()

    def test_changed_analyzer(self):
        self.report['analyzer']['sha256']='0'*64
        with self.assertRaisesRegex(RecoveryError,'IMPLEMENTATION_DRIFT'):self.bind()

    def test_not_terminal(self):
        self.job['status']='RUNNING'
        with self.assertRaisesRegex(RecoveryError,'TERMINAL_JOB'):self.bind()

    def test_missing_block(self):
        self.report['blocks'].pop()
        with self.assertRaisesRegex(RecoveryError,'BLOCK_COVERAGE'):self.bind()

    def journal(self):
        journal=Journal(self.root/'batch')
        with journal.writer():
            journal.save(dict(events={self.key:dict(self.entry,kind='job')},status='JOB_BOUND_EXCEEDED',current_job=None))
        return journal

    def test_record_idempotent_and_preserves_control_failure(self):
        journal=self.journal();report=self.write('report.json',self.report)
        with journal.writer():first=record_analysis(journal,self.key,report,self.store)
        restarted=Journal(journal.directory)
        with restarted.writer():self.assertEqual(first,record_analysis(restarted,self.key,report,self.store))
        self.assertEqual(restarted.read()['status'],'JOB_BOUND_EXCEEDED')
        self.assertIn('analysis',restarted.read()['events'][self.key])

    def test_crash_after_artifact_before_state_reuses_same_result(self):
        journal=self.journal();report=self.write('report.json',self.report)
        with journal.writer(),patch.object(journal,'save',side_effect=RuntimeError('CPU crash')):
            with self.assertRaisesRegex(RuntimeError,'CPU crash'):record_analysis(journal,self.key,report,self.store)
        self.assertTrue((journal.directory/'jobs'/self.key/'analysis.json').exists())
        with journal.writer():result=record_analysis(journal,self.key,report,self.store)
        self.assertEqual(result['native_status'],'FAILED')

    def test_stop_refuses_new_record(self):
        journal=self.journal();(journal.directory/'STOP').touch()
        with journal.writer(),self.assertRaisesRegex(RecoveryError,'STOP_NEW_ACTIONS'):
            record_analysis(journal,self.key,self.write('report.json',self.report),self.store)

    def test_changed_report_not_overwritten(self):
        journal=self.journal();report=self.write('report.json',self.report)
        with journal.writer():record_analysis(journal,self.key,report,self.store)
        self.report['extra']='new analysis'
        changed=self.write('report-new.json',self.report)
        with journal.writer(),self.assertRaisesRegex(RecoveryError,'ANALYSIS_REPLAY_DRIFT'):
            record_analysis(journal,self.key,changed,self.store)


if __name__=='__main__':unittest.main()
