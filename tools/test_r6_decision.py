"""CPU substitute decisions after bound analysis. Never call an actual model."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tools.test_r6_cpu_analysis import CPUAnalysisTests
from tools.r6_decision_packet import prepare,CAP
from tools.r6_decision import register,handoff,record
from tools.r6_recovery import RecoveryError,file_pin,Journal
from tools.r6_analysis_binding import record_analysis

ROOT=Path(__file__).resolve().parents[1]
FAKE='''#!{python}
import json,sys
from pathlib import Path
text=sys.stdin.read();packet=json.loads(text.splitlines()[-1])
path=text.split('Input packet path: ',1)[1].splitlines()[0]
digest=text.split('Input packet SHA256: ',1)[1].splitlines()[0]
count=Path(__file__).with_suffix('.count')
with count.open('a') as f:f.write('call\\n')
value={{'decision':'NEEDS_REPAIR','packet_sha256':digest,'summary':'CPU fixture: original evidence incomplete',
'evidence_pins':[{{'path':path,'sha256':digest}},packet['report']], 'changed_files':[],
'must_fix':['Obtain or repair missing evidence without reclassifying missing lanes as success'],
'remaining_budget':{{'automatic_sessions_remaining':packet['budget_before_packet_reservation']['remaining']['automatic_model_sessions']-1}},
'next_plan':{{'hypothesis':'Missing evidence prevents comparison','comparison':'Preserve frozen u640 comparison',
'steps':['Audit exact missing originals and propose a bounded repair'],'independent_review_required':True}}}}
print(json.dumps({{'type':'item.completed','item':{{'type':'agent_message','text':json.dumps(value)}}}}))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':1,'output_tokens':1}}}}))
'''


class DecisionTests(unittest.TestCase):
    write=CPUAnalysisTests.write
    def setUp(self):
        CPUAnalysisTests.setUp(self)
        with self.j.writer():
            state=self.j.read();state['events'][self.key]['identity']['role']='job';self.j.save(state)
        self.report_pin=self.write('report.json',self.report)
        with self.j.writer():record_analysis(self.j,self.key,self.report_pin,self.store)
        self.exe=self.root/'fake_codex';self.exe.write_text(FAKE.format(python=sys.executable));self.exe.chmod(0o700)

    cleanup_worker=CPUAnalysisTests.cleanup_worker

    def prepare(self):
        with self.j.writer():return prepare(self.j,self.key,self.store,free_disk_bytes=500*1024**3)

    def register(self):
        with self.j.writer():return register(self.j,self.key,self.store,executable=file_pin(self.exe),free_disk_bytes=500*1024**3)

    def complete(self,key):
        with self.j.writer():return handoff(self.j,key,self.store,poll_seconds=.01)

    def test_packet_and_prompt_are_bounded_and_include_failure_budget(self):
        p=self.prepare();packet=json.loads(Path(p['packet']['path']).read_text())
        self.assertEqual(packet['native_exit_code'],-9);self.assertTrue(packet['issues'])
        self.assertLessEqual(Path(p['prompt']['path']).stat().st_size,CAP)
        self.assertEqual(packet['budget_before_packet_reservation']['remaining']['automatic_model_sessions'],32)
        self.assertFalse(packet['capacity_pass']);self.assertFalse(p['dispatch_authorized'])

    def test_packet_restart_has_identical_bytes_and_single_reservation(self):
        first=self.prepare();self.j=Journal(self.j.directory);second=self.prepare()
        self.assertEqual(first,second);self.assertEqual(len(self.j.read()['budget']['reservations']),1)
        self.assertEqual(self.j.read()['status'],'JOB_BOUND_EXCEEDED')

    def test_crash_before_artifacts_reconstructs_reserved_original_packet(self):
        with self.j.writer(),patch('tools.r6_decision_packet.persist_exact',side_effect=RuntimeError('crash')):
            with self.assertRaisesRegex(RuntimeError,'crash'):prepare(self.j,self.key,self.store,free_disk_bytes=500*1024**3)
        self.assertEqual(len(self.j.read()['budget']['reservations']),1)
        result=self.prepare();self.assertTrue(Path(result['packet']['path']).exists())
        self.assertEqual(len(self.j.read()['budget']['reservations']),1)

    def test_fake_model_event_records_proposal_once_without_dispatch(self):
        key=self.register();result=self.complete(key)
        self.assertEqual(result['next_step'],'REPAIR_AND_REVIEW');self.assertFalse(result['dispatch_authorized'])
        self.j=Journal(self.j.directory);self.assertEqual(self.register(),key)
        self.assertEqual(self.complete(key),result)
        self.assertEqual(self.exe.with_suffix('.count').read_text(),'call\n')
        self.assertEqual(len(self.j.read()['budget']['reservations']),2)

    def finalize_decision(self,sandbox='read-only'):
        from tools.r6_finalize_job import decide
        with self.j.writer():
            return decide(self.j,self.key,self.store,file_pin(self.exe),
                free_disk_bytes=500*1024**3,sandbox=sandbox,poll_seconds=.01)

    def test_finalizer_decision_runs_once_and_preserves_replay_state(self):
        first=self.finalize_decision()
        before=self.j.read()
        self.assertEqual(self.finalize_decision(),first)
        self.assertEqual(self.j.read(),before)
        self.assertEqual(self.exe.with_suffix('.count').read_text(),'call\n')
        self.assertFalse(first['dispatch_authorized'])

    def test_finalizer_rejects_changed_sandbox_before_model_dispatch(self):
        self.register()
        with self.assertRaisesRegex(RecoveryError,'FINALIZE_DECISION_BINDING'):
            self.finalize_decision(sandbox='danger-full-access')
        self.assertFalse(self.exe.with_suffix('.count').exists())

    def test_finalizer_repairs_missing_input_binding_before_dispatch(self):
        key=self.register()
        with self.j.writer():
            state=self.j.read();state['events'][key].pop('decision_input');self.j.save(state)
        self.assertEqual(self.finalize_decision()['next_step'],'REPAIR_AND_REVIEW')
        self.assertEqual(self.exe.with_suffix('.count').read_text(),'call\n')
        self.assertEqual(len(self.j.read()['budget']['reservations']),2)

    def test_finalizer_does_not_reconstruct_input_after_ambiguous_dispatch(self):
        key=self.register()
        with self.j.writer():
            state=self.j.read();entry=state['events'][key]
            entry.pop('decision_input');entry['dispatch_claim']={'fixture':'unknown'};self.j.save(state)
        with self.assertRaisesRegex(RecoveryError,'FINALIZE_DECISION_INPUT_AMBIGUOUS'):
            self.finalize_decision()
        self.assertFalse(self.exe.with_suffix('.count').exists())

    def test_finalizer_rechecks_raw_analysis_before_first_model_dispatch(self):
        self.register();Path(self.raw['path']).write_text('changed raw')
        with self.assertRaises(RecoveryError):self.finalize_decision()
        self.assertFalse(self.exe.with_suffix('.count').exists())

    def test_registered_decision_recovery_skips_completed_earlier_phases(self):
        from tools.r6_finalize_job import finish_and_decide
        self.register()
        with patch('tools.r6_finalize_job.finalize',side_effect=AssertionError('repeat postclose')), \
                patch('tools.r6_finalize_job.analyze',side_effect=AssertionError('repeat analysis')):
            with self.j.writer():
                first=finish_and_decide(self.j,self.key,self.store,file_pin(self.exe),file_pin(self.exe),
                    free_disk_bytes=500*1024**3,poll_seconds=.01)
            before=self.j.read()
            with self.j.writer():
                second=finish_and_decide(self.j,self.key,self.store,file_pin(self.exe),file_pin(self.exe),
                    free_disk_bytes=500*1024**3,poll_seconds=.01)
        self.assertEqual(first,second);self.assertEqual(self.j.read(),before)
        self.assertEqual(self.exe.with_suffix('.count').read_text(),'call\n')

    def test_finalizer_repairs_consumed_receipt_with_stale_model_marker(self):
        from tools.r6_finalize_job import finish_and_decide
        key=self.register();first=self.complete(key)
        with self.j.writer():
            state=self.j.read();state.update(current_model_event=key,status='MODEL_REVIEW');self.j.save(state)
            result=finish_and_decide(self.j,self.key,self.store,file_pin(self.exe),file_pin(self.exe),
                free_disk_bytes=500*1024**3,poll_seconds=.01)
        self.assertEqual(result['decision'],first)
        self.assertIsNone(self.j.read()['current_model_event'])
        self.assertEqual(self.exe.with_suffix('.count').read_text(),'call\n')

    def mutate_response(self,key,change):
        s=self.j.read();entry=s['events'][key];r=json.loads(Path(entry['receipt']['path']).read_text())
        path=Path(r['last_message']['path']);value=json.loads(path.read_text());change(value)
        path.write_text(json.dumps(value));r['last_message']=file_pin(path)
        Path(entry['receipt']['path']).write_text(json.dumps(r));entry['receipt']=file_pin(entry['receipt']['path'])
        with self.j.writer():self.j.save(s)

    def test_evidence_gaps_cannot_become_candidate_proposal(self):
        key=self.register();self.complete(key)
        self.mutate_response(key,lambda v:v.update(decision='PROPOSE_CANDIDATE',must_fix=[]))
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'UNRESOLVED_EVIDENCE'):record(self.j,key,self.store)

    def test_model_cannot_invent_budget(self):
        key=self.register();self.complete(key)
        self.mutate_response(key,lambda v:v['remaining_budget'].update(automatic_sessions_remaining=32))
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'BUDGET_MISMATCH'):record(self.j,key,self.store)

    def test_wrong_packet_hash_rejected(self):
        key=self.register();self.complete(key)
        self.mutate_response(key,lambda v:v.update(packet_sha256='0'*64))
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'PACKET_IDENTITY'):record(self.j,key,self.store)

    def test_report_citation_required(self):
        key=self.register();self.complete(key)
        self.mutate_response(key,lambda v:v['evidence_pins'].__setitem__(1,file_pin(self.exe)))
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'EVIDENCE_REQUIRED'):record(self.j,key,self.store)

    def test_stop_does_not_create_packet(self):
        (self.j.directory/'STOP').touch()
        with self.assertRaisesRegex(RecoveryError,'STOP'):self.prepare()
        self.assertFalse((self.j.directory/'decisions').exists())

    def test_all_workspace_stops_block_before_packet_and_reservation(self):
        for relative in ('STOP','coordination/STOP','coordination/commander/STOP',
                         'coordination/executor/STOP'):
            with self.subTest(relative=relative),patch('tools.r6_postrun.ROOT',self.root):
                stop=self.root/relative;stop.parent.mkdir(parents=True,exist_ok=True);stop.touch()
                before=self.j.read()
                try:
                    with self.assertRaisesRegex(RecoveryError,'DECISION_STOP'):self.register()
                    self.assertEqual(self.j.read(),before)
                    self.assertFalse((self.j.directory/'decisions').exists())
                finally:stop.unlink()

    def test_workspace_stop_after_registration_blocks_real_fake_cli_launch(self):
        with patch('tools.r6_postrun.ROOT',self.root):
            key=self.register()
            stop=self.root/'coordination/executor/STOP'
            stop.parent.mkdir(parents=True,exist_ok=True);stop.touch()
            request=self.j.read()['events'][key]['request']
            self.assertIn(str(stop),request['stop_paths'])
            with self.assertRaises(RecoveryError):self.complete(key)
            self.assertFalse(self.exe.with_suffix('.count').exists())
            receipt=self.j.read()['events'][key]['receipt']
            self.assertIsNotNone(receipt)
            value=json.loads(Path(receipt['path']).read_text())
            self.assertIn('STOP_BEFORE_MODEL_START',value['error'])
            self.assertIsNone(value['child_identity'])
            # An already consumed stopped event cannot start the CLI on replay.
            with self.assertRaises(RecoveryError):self.complete(key)
            self.assertFalse(self.exe.with_suffix('.count').exists())

    def test_finalize_replay_uses_same_stop_contract_without_another_model(self):
        from tools.r6_finalize_job import finish_and_decide
        with patch('tools.r6_postrun.ROOT',self.root):
            key=self.register();original=self.complete(key)
            with self.j.writer():
                result=finish_and_decide(self.j,self.key,self.store,
                    file_pin(self.exe),file_pin(self.exe),free_disk_bytes=500*1024**3,
                    poll_seconds=.01)
            self.assertEqual(result['decision'],original)
            self.assertEqual(self.exe.with_suffix('.count').read_text(),'call\n')

    def test_new_model_request_cannot_alias_same_decision(self):
        self.register();self.exe.write_text(self.exe.read_text()+'\n# different identity\n')
        with self.assertRaisesRegex(RecoveryError,'ALREADY_BOUND'):self.register()

    def test_raw_change_refuses_packet(self):
        Path(self.raw['path']).write_text('changed')
        with self.assertRaises(RecoveryError):self.prepare()
        self.assertFalse((self.j.directory/'decisions').exists())

    def test_oversized_summary_is_refused_before_reservation(self):
        self.report['summary']['large']='x'*(48*1024)
        self.report_pin=self.write('new_report.json',self.report)
        # Build a new valid binding for this fixture without overwriting a
        # previously accepted analysis; normal production rejects such drift.
        with self.j.writer():
            s=self.j.read();s['events'][self.key].pop('analysis');self.j.save(s)
        (self.j.directory/'jobs'/self.key/'analysis.json').unlink()
        with self.j.writer():record_analysis(self.j,self.key,self.report_pin,self.store)
        with self.assertRaisesRegex(RecoveryError,'SUMMARY_BOUND'):self.prepare()
        self.assertEqual(len(self.j.read()['budget']['reservations']),0)


    def test_changed_budget_blocks_first_dispatch(self):
        key=self.register()
        with self.j.writer():
            state=self.j.read();state['budget']['used']['automatic_model_sessions']+=1;self.j.save(state)
        with self.assertRaisesRegex(RecoveryError,'BUDGET_SNAPSHOT_STALE'):self.complete(key)
        self.assertFalse(self.exe.with_suffix('.count').exists())

    def test_crash_after_decision_artifact_recovers_without_model_call(self):
        key=self.register()
        from tools.r6_batch import model_handoff
        with self.j.writer():model_handoff(self.j,key,poll_seconds=.01)
        with self.j.writer(),patch.object(self.j,'save',side_effect=RuntimeError('CPU crash')):
            with self.assertRaisesRegex(RuntimeError,'CPU crash'):record(self.j,key,self.store)
        self.assertEqual(self.complete(key)['next_step'],'REPAIR_AND_REVIEW')
        self.assertEqual(self.exe.with_suffix('.count').read_text(),'call\n')


    def test_cpu_analysis_to_decision_to_next_step_survives_replay(self):
        # The isolated job already has exact close/inventory/postrun fixtures.
        # Remove only this test's manual binding, then exercise actual local
        # subprocess handoffs for both CPU analysis and the substitute model.
        with self.j.writer():
            state=self.j.read();state['events'][self.key].pop('analysis');self.j.save(state)
        (self.j.directory/'jobs'/self.key/'analysis.json').unlink()
        CPUAnalysisTests.register(self)
        cpu=CPUAnalysisTests.complete(self)
        key=self.register();result=self.complete(key)
        self.assertEqual(result['analysis'],self.j.read()['events'][self.key]['analysis'])
        self.assertEqual(result['next_step'],'REPAIR_AND_REVIEW')
        self.assertEqual(CPUAnalysisTests.complete(self),cpu)
        self.assertEqual(self.complete(key),result)
        self.assertEqual((self.root/'calls').read_text(),'call\n')
        self.assertEqual(self.exe.with_suffix('.count').read_text(),'call\n')
        self.assertEqual(len(self.j.read()['budget']['reservations']),3)


if __name__=='__main__':unittest.main()
