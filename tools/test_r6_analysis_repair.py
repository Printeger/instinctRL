"""Real CPU subprocess replacement/recovery; synthetic science and Store fixture."""
import copy
import json
import os
import sys
from pathlib import Path
import unittest
from rlflow.jobs import identity
from tools import test_r6_cpu_analysis as fixtures
from tools import r6_analysis_repair as repair
from tools.r6_recovery import RecoveryError,file_pin,Journal


class AnalysisRepairTests(unittest.TestCase):
    setUp=fixtures.CPUAnalysisTests.setUp
    write=fixtures.CPUAnalysisTests.write
    cleanup_worker=fixtures.CPUAnalysisTests.cleanup_worker
    register=fixtures.CPUAnalysisTests.register
    complete=fixtures.CPUAnalysisTests.complete
    count=fixtures.CPUAnalysisTests.count
    finalize_analysis=fixtures.CPUAnalysisTests.finalize_analysis

    def fail(self):
        self.register(body="print('original failure');raise SystemExit(7)")
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.complete()
        self.oldkey=self.cpu_key;self.old=copy.deepcopy(self.j.read()['events'][self.oldkey])

    def proposal(self,reason='Replace deliberately failing CPU fixture'):
        return self.write('repair-'+str(len(reason))+'.json',dict(job_event_key=self.key,
            predecessor=self.oldkey,predecessor_receipt=self.j.read()['events'][self.oldkey]['receipt'],
            reason=reason,evidence_pins=[self.spec_pin]))

    def replace(self,pin):
        with self.j.writer():
            return repair.register(self.j,self.key,self.store,pin,free_disk_bytes=500*1024**3,
                python=file_pin(sys.executable),program=file_pin(self.program),timeout_seconds=10)

    def test_failed_actual_worker_replaced_and_finalizer_recovers_once(self):
        self.fail();pin=self.proposal()
        self.program=self.root/'repaired_analysis.py';self.program.write_text(fixtures.SCRIPT.format(body='pass'))
        self.cpu_key=self.replace(pin);budget=self.j.read()['budget']
        self.assertNotEqual(self.cpu_key,self.oldkey)
        self.assertEqual(self.replace(pin),self.cpu_key);self.assertEqual(self.j.read()['budget'],budget)
        with self.j.writer(),self.assertRaisesRegex(RecoveryError,'NOT_ACTIVE'):
            fixtures.validate(self.j,self.oldkey)
        self.j=Journal(self.j.directory);result=self.finalize_analysis()
        self.assertEqual(result['native_status'],'FAILED');self.assertFalse(result['scientific_acceptance'])
        self.assertEqual(self.finalize_analysis(),result);self.assertEqual(self.count(),2)
        self.assertEqual(self.j.read()['events'][self.oldkey],self.old)
        self.assertEqual(self.j.read()['budget'],budget)
        self.assertEqual(len(budget['reservations']),2)
        self.assertEqual(budget['reservations'][-1]['purpose'],'repair')
        with self.assertRaisesRegex(RecoveryError,'ALREADY_USED'):self.replace(pin)

    def test_unchanged_input_cannot_blind_retry(self):
        self.fail();pin=self.proposal();before=self.j.read()
        with self.assertRaisesRegex(RecoveryError,'UNCHANGED_INPUT'):self.replace(pin)
        self.assertEqual(self.j.read(),before);self.assertEqual(self.count(),1)

    def test_second_failed_repair_remains_in_chain(self):
        self.fail();pin=self.proposal()
        self.program=self.root/'second_analysis.py'
        self.program.write_text(fixtures.SCRIPT.format(body='raise SystemExit(8)'))
        self.cpu_key=self.replace(pin)
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.complete()
        self.oldkey=self.cpu_key
        pin=self.proposal('Repair second distinct CPU failure')
        self.program=self.root/'third_analysis.py';self.program.write_text(fixtures.SCRIPT.format(body='pass'))
        self.cpu_key=self.replace(pin)
        self.assertEqual(repair.active(self.j,self.key),self.cpu_key)
        self.finalize_analysis();self.assertEqual(self.count(),3)
        self.assertEqual(len(self.j.read()['budget']['reservations']),3)

    def test_budget_rejection_does_not_register_replacement(self):
        from tools.r6_budget import BudgetError,totals
        from tools.r6_cpu_analysis import RESERVATION
        self.fail();pin=self.proposal();self.program.write_text(fixtures.SCRIPT.format(body='pass'))
        with self.j.writer():
            state=self.j.read();resource='output_actual_plus_unreleased_reservations_bytes'
            # Keep protected capacity valid but consume the remaining envelope.
            state['budget']['caps'][resource]=totals(state['budget'])[resource]+RESERVATION-1
            state['budget']['protected_confirmation_delivery_repair'][resource]=0
            self.j.save(state)
        before=self.j.read()
        with self.assertRaises(BudgetError):self.replace(pin)
        self.assertEqual(self.j.read(),before);self.assertEqual(self.count(),1)

    def test_no_fork_from_obsolete_failure(self):
        self.fail();pin=self.proposal();self.program.write_text(fixtures.SCRIPT.format(body='pass'))
        self.replace(pin);before=self.j.read()
        with self.assertRaisesRegex(RecoveryError,'NOT_ACTIVE_TIP'):
            self.replace(self.proposal('Different explanation for the same old failure'))
        self.assertEqual(self.j.read(),before)

    def test_missing_cleanup_or_live_child_rejects_before_reservation(self):
        self.fail();self.program.write_text(fixtures.SCRIPT.format(body='pass'))
        receipt=json.loads(Path(self.old['receipt']['path']).read_text())
        for damage in ('cleanup','live','exit','receipt'):
            with self.subTest(damage=damage):
                r=copy.deepcopy(receipt)
                if damage=='cleanup':r['cleanup']['stable_empty_scans']=0
                if damage=='live':r['child_identity']=identity(os.getpid())
                if damage=='exit':r.pop('exit_code')
                with self.j.writer():
                    s=self.j.read();s['events'][self.oldkey]['receipt']=None if damage=='receipt' else self.write('bad-'+damage+'.json',r);self.j.save(s)
                before=self.j.read()
                with self.assertRaises(RecoveryError):self.replace(self.proposal())
                self.assertEqual(self.j.read(),before)
        self.assertEqual(self.count(),1)

    def test_stop_rejects_replacement_without_reservation(self):
        self.fail();pin=self.proposal();self.program.write_text(fixtures.SCRIPT.format(body='pass'))
        (self.j.directory/'STOP').touch();before=self.j.read()
        with self.assertRaisesRegex(RecoveryError,'STOP'):self.replace(pin)
        self.assertEqual(self.j.read(),before)

    def test_missing_replacement_reservation_blocks_worker(self):
        self.fail();pin=self.proposal();self.program.write_text(fixtures.SCRIPT.format(body='pass'))
        self.cpu_key=self.replace(pin)
        with self.j.writer():
            state=self.j.read();state['budget']['reservations'].pop();self.j.save(state)
        with self.assertRaisesRegex(RecoveryError,'RESERVATION'):self.complete()
        self.assertEqual(self.count(),1)

    def test_accepted_analysis_cannot_be_replaced(self):
        self.register();self.complete();self.oldkey=self.cpu_key
        pin=self.proposal();self.program.write_text(fixtures.SCRIPT.format(body='time.sleep(.1)'))
        with self.assertRaisesRegex(RecoveryError,'ALREADY_USED'):self.replace(pin)
