"""Actual local fake CLI workers, synthetic analysis, no model or live dispatch."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import test_r6_decision as fixtures
from tools import r6_decision as decision,r6_decision_repair as repair
from tools.r6_recovery import RecoveryError,file_pin,Journal


class DecisionRepairTests(unittest.TestCase):
    setUp=fixtures.DecisionTests.setUp
    write=fixtures.DecisionTests.write
    cleanup_worker=fixtures.DecisionTests.cleanup_worker
    register=fixtures.DecisionTests.register
    complete=fixtures.DecisionTests.complete
    finalize_decision=fixtures.DecisionTests.finalize_decision

    def fail(self,schema_error=False):
        script=fixtures.FAKE.format(python=sys.executable).replace("'decision':'NEEDS_REPAIR'","'decision':'INVALID'") if schema_error else '#!'+sys.executable+'\nimport sys,time\nsys.stdin.read();time.sleep(.05);sys.exit(7)\n'
        self.exe.write_text(script)
        self.oldkey=self.register()
        with self.assertRaises(RecoveryError):self.complete(self.oldkey)
        self.old=copy.deepcopy(self.j.read()['events'][self.oldkey])
        self.exe=self.root/'repaired_fake_codex'
        self.exe.write_text(fixtures.FAKE.format(python=sys.executable));self.exe.chmod(0o700)

    def proposal(self,reason='Fix local fixture CLI exit failure'):
        return self.write('repair-'+str(len(reason))+'.json',dict(job_event_key=self.key,
            predecessor=self.oldkey,predecessor_receipt=self.j.read()['events'][self.oldkey]['receipt'],
            reason=reason,evidence_pins=[self.report_pin]))

    def replace(self,pin):
        with self.j.writer():
            return decision.register(self.j,self.key,self.store,executable=file_pin(self.exe),
                free_disk_bytes=500*1024**3,repair_pin=pin)

    def test_failed_call_replaced_and_controller_uses_new_proposal(self):
        from tools.r6_controller import parent_for_job
        self.fail();pin=self.proposal();key=self.replace(pin);budget=self.j.read()['budget']
        self.assertEqual(self.replace(pin),key);self.assertEqual(self.j.read()['budget'],budget)
        self.j=Journal(self.j.directory);result=self.finalize_decision()
        self.assertEqual(result['decision']['decision'],'NEEDS_REPAIR')
        self.assertFalse(result['dispatch_authorized'])
        self.assertEqual(result['decision']['remaining_budget']['automatic_sessions_remaining'],30)
        self.assertEqual(self.finalize_decision(),result)
        with self.j.writer():self.assertEqual(parent_for_job(self.j,self.key,self.store),(key,result))
        from tools import r6_source_preparation as source,r6_profile_preparation as profile
        with self.j.writer(),patch.object(source,'prepare_source') as prepare_source,patch.object(profile,'prepare_profile') as prepare_profile:
            source.after_job(self.j,self.key,self.store,executable=file_pin(self.exe),free_disk_bytes=500*1024**3)
            profile.after_job(self.j,self.key,self.report_pin,self.store,executable=file_pin(self.exe),free_disk_bytes=500*1024**3)
            self.assertEqual(prepare_source.call_args.args[1],key)
            self.assertEqual(prepare_profile.call_args.args[1],key)
        self.assertEqual(self.exe.with_suffix('.count').read_text(),'call\n')
        self.assertEqual(self.j.read()['events'][self.oldkey],self.old)
        self.assertEqual(self.j.read()['budget'],budget)
        self.assertEqual(len(budget['reservations']),4)
        self.assertEqual([r['purpose'] for r in budget['reservations'][-2:]],['repair','repair'])
        with self.assertRaisesRegex(RecoveryError,'NOT_ACTIVE'):self.complete(self.oldkey)

    def test_crash_before_packet_artifacts_recovers_same_reservations(self):
        self.fail();pin=self.proposal()
        with patch('tools.r6_decision_packet.persist_exact',side_effect=RuntimeError('packet write crash')):
            with self.assertRaisesRegex(RuntimeError,'packet write crash'):self.replace(pin)
        self.assertEqual(len(self.j.read()['budget']['reservations']),3)
        key=self.replace(pin);self.assertEqual(len(self.j.read()['budget']['reservations']),4)
        self.assertEqual(self.finalize_decision()['model_event_key'],key)

    def test_missing_input_binding_recovers_replacement_not_root(self):
        self.fail();key=self.replace(self.proposal());budget=self.j.read()['budget']
        with self.j.writer():
            state=self.j.read();state['events'][key].pop('decision_input');self.j.save(state)
        self.assertEqual(self.finalize_decision()['model_event_key'],key)
        self.assertEqual(self.j.read()['budget'],budget)

    def test_valid_needs_repair_is_not_replaced_as_model_failure(self):
        self.oldkey=self.register();self.complete(self.oldkey);before=self.j.read()
        with self.assertRaisesRegex(RecoveryError,'PREDECESSOR'):self.replace(self.proposal())
        self.assertEqual(self.j.read(),before)

    def test_zero_exit_invalid_schema_can_be_replaced(self):
        self.fail(schema_error=True)
        receipt=json.loads(Path(self.old['receipt']['path']).read_text())
        self.assertEqual(receipt['exit_code'],0);self.assertFalse(receipt['schema_valid'])
        key=self.replace(self.proposal('Fix invalid structured response schema'))
        self.assertEqual(self.finalize_decision()['model_event_key'],key)

    def test_new_budget_charge_blocks_dispatch_of_stale_replacement_packet(self):
        self.fail();self.replace(self.proposal())
        with self.j.writer():
            state=self.j.read();state['budget']['used']['automatic_model_sessions']+=1;self.j.save(state)
        with self.assertRaisesRegex(RecoveryError,'BUDGET_SNAPSHOT_STALE'):self.finalize_decision()
        self.assertFalse(self.exe.with_suffix('.count').exists())

    def test_no_fork_from_obsolete_failure(self):
        self.fail();self.replace(self.proposal());before=self.j.read()
        with self.assertRaisesRegex(RecoveryError,'NOT_ACTIVE_TIP'):
            self.replace(self.proposal('Another repair from the old failed call'))
        self.assertEqual(self.j.read(),before)

    def test_missing_cleanup_or_receipt_blocks_without_charge(self):
        self.fail();receipt=json.loads(Path(self.old['receipt']['path']).read_text())
        for damage in ('cleanup','receipt','exit'):
            with self.subTest(damage=damage):
                r=copy.deepcopy(receipt)
                if damage=='cleanup':r['cleanup']['stable_empty_scans']=0
                if damage=='exit':r.pop('exit_code')
                with self.j.writer():
                    state=self.j.read();state['events'][self.oldkey]['receipt']=None if damage=='receipt' else self.write('bad-'+damage+'.json',r);self.j.save(state)
                before=self.j.read()
                with self.assertRaises(RecoveryError):self.replace(self.proposal())
                self.assertEqual(self.j.read(),before)

    def test_stop_blocks_new_packet_and_model(self):
        self.fail();(self.j.directory/'STOP').touch();before=self.j.read()
        with self.assertRaisesRegex(RecoveryError,'STOP'):self.replace(self.proposal())
        self.assertEqual(self.j.read(),before)
