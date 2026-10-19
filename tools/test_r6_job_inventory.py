"""Inventory and conservative settlement CPU tests on isolated job substitutes."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from rlflow.store import sha
from tools import test_r6_job_handoff as fixture
from tools.r6_recovery import RecoveryError, file_pin
from tools.r6_budget import totals
from tools.r6_job_inventory import inventory, record_inventory, DISK


class InventoryTests(unittest.TestCase):
    tearDown = fixture.JobTests.tearDown
    save = fixture.JobTests.save
    finish = fixture.JobTests.finish
    cancel = fixture.JobTests.cancel
    handoff = fixture.JobTests.handoff

    def setUp(self):
        fixture.JobTests.setUp(self)
        self.attempt = self.store.root/'attempts'/'CPU-ONLY-ATTEMPT'
        self.attempt.mkdir(parents=True)
        (self.attempt/'failed-output.bin').write_bytes(b'original negative evidence')
        self.spec['h1'] = {'attempt':str(self.attempt)}
        specpath = Path(self.request['spec']['path']);specpath.write_text(json.dumps(self.spec))
        self.request['spec'] = file_pin(specpath)
        self.jid = 'J-'+sha({'id':self.spec['id'],'spec':self.spec})[:16]
        self.job.update(id=self.jid,spec=self.spec)
        self.store.state['jobs'] = {self.jid:self.job}
        self.directory.rmdir();self.directory = self.store.root/'jobs'/self.jid;self.directory.mkdir()
        with self.j.writer():
            state = self.j.read();state['events'] = {};state['budget']['reservations'] = [];self.j.save(state)
            event = dict(goal_id=state['goal_id'],event_id='CPU-INVENTORY',job_id=self.jid,
                input_sha256=sha(self.request),role='job')
            self.key,_ = self.j.intent(event,kind='job',request=self.request,
                resources={'live_starts':1,'live_wall_seconds':190,'allocated_gpu_seconds':190,DISK:100*1024**2},
                purpose='development',free_disk_bytes=500*1024**3)
        self.finish()

    def record(self):
        with self.j.writer():return record_inventory(self.j,self.key,self.store)

    def test_all_originals_hashed_no_copies_or_refunds(self):
        before = totals(self.j.read()['budget'])
        result = self.record()
        paths = {r['path'] for r in result['files']}
        self.assertIn(str(self.attempt/'failed-output.bin'),paths)
        self.assertIn(str(self.directory/'receipt.json'),paths)
        self.assertEqual(result['total_original_bytes'],sum(Path(p).stat().st_size for p in paths))
        self.assertEqual(result['total_archive_copy_bytes'],0)
        self.assertEqual(result['native_exit_code'],-9);self.assertFalse(result['scientific_acceptance'])
        self.assertTrue(result['output_inventory_complete']);self.assertEqual(before,totals(self.j.read()['budget']))
        self.assertEqual(self.j.read()['status'],'INVENTORIED_PENDING_POSTRUN_AND_ANALYSIS')

    def test_restart_has_same_inventory_and_single_observation_per_evidence(self):
        first = self.record();pin = self.j.read()['events'][self.key]['inventory']
        self.assertEqual(first,self.record());self.assertEqual(pin,self.j.read()['events'][self.key]['inventory'])
        self.assertEqual(len(self.j.read()['budget']['reservations'][0]['observations']),2)

    def test_original_content_drift_refuses_replay(self):
        self.record();(self.attempt/'failed-output.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(RecoveryError,'INVENTORY_REPLAY_DRIFT'):self.record()

    def test_inventory_replay_preserves_later_analysis_state(self):
        first=self.record()
        with self.j.writer():
            state=self.j.read();state['status']='MODEL_DECISION_RECORDED';self.j.save(state)
        before=self.j.read()
        self.assertEqual(self.record(),first)
        self.assertEqual(self.j.read(),before)

    def test_failed_inventory_replay_does_not_rewind_analysis(self):
        self.record()
        with self.j.writer():
            state=self.j.read();state['status']='MODEL_DECISION_RECORDED';self.j.save(state)
        before=self.j.read()
        (self.attempt/'failed-output.bin').write_bytes(b'drift')
        with self.assertRaisesRegex(RecoveryError,'INVENTORY_REPLAY_DRIFT'):self.record()
        self.assertEqual(self.j.read(),before)

    def test_missing_attempt_is_explicit_and_not_complete(self):
        (self.attempt/'failed-output.bin').unlink();self.attempt.rmdir()
        result = self.record()
        self.assertEqual(result['missing_roots'],['attempt']);self.assertFalse(result['output_inventory_complete'])
        self.assertEqual(self.j.read()['status'],'JOB_EVIDENCE_INCOMPLETE')

    def test_symlink_and_special_files_are_rejected(self):
        link = self.attempt/'alias';link.symlink_to(self.attempt/'failed-output.bin')
        with self.assertRaisesRegex(RecoveryError,'SYMLINK_OR_SPECIAL_FILE'):self.record()
        link.unlink();os.mkfifo(link)
        with self.assertRaisesRegex(RecoveryError,'SYMLINK_OR_SPECIAL_FILE'):self.record()

    def test_overlapping_roots_rejected(self):
        with self.assertRaisesRegex(RecoveryError,'ROOT_OVERLAP'):
            inventory({'a':self.store.root,'b':self.attempt})

    def test_hardlinks_are_counted_conservatively_per_path(self):
        os.link(self.attempt/'failed-output.bin',self.attempt/'another-name.bin')
        result = inventory({'attempt':self.attempt})
        self.assertEqual(result['total_original_bytes'],2*(self.attempt/'failed-output.bin').stat().st_size)

    def test_earlier_file_mutated_during_later_file_hash_is_detected(self):
        a = self.attempt/'a.bin';b = self.attempt/'b.bin';a.write_bytes(b'a');b.write_bytes(b'b')
        original = os.open
        def opened(path,*args,**kwargs):
            if Path(path) == b:a.write_bytes(b'changed after a was hashed')
            return original(path,*args,**kwargs)
        with patch('tools.r6_job_inventory.os.open',side_effect=opened):
            with self.assertRaisesRegex(RecoveryError,'INVENTORY_FILE_CHANGED'):
                inventory({'attempt':self.attempt})

    def test_crash_after_inventory_before_journal_recovers(self):
        self.handoff();original = self.j.save
        def saved(state):
            if state['events'][self.key].get('inventory'):raise RuntimeError('CPU_CRASH')
            return original(state)
        with patch.object(self.j,'save',side_effect=saved):
            with self.assertRaisesRegex(RuntimeError,'CPU_CRASH'):self.record()
        path = self.j.directory/'jobs'/self.key/'inventory.json';pin = file_pin(path)
        self.assertNotIn('inventory',self.j.read()['events'][self.key])
        self.record();self.assertEqual(pin,self.j.read()['events'][self.key]['inventory'])
        self.assertEqual(len(self.j.read()['budget']['reservations'][0]['observations']),2)

    def test_stop_does_not_discard_inventory_or_resume(self):
        (self.j.directory/'STOP').touch();self.handoff();self.record()
        self.assertEqual(self.j.read()['status'],'STOPPED_AFTER_OWNED_CLEANUP')

    def test_budget_overrun_observation_survives_replay(self):
        with self.j.writer():
            state = self.j.read();state['events'][self.key]['resources'][DISK] = 1
            state['budget']['reservations'][0]['resources'][DISK] = 1;self.j.save(state)
        result = self.record();self.assertEqual(self.j.read()['status'],'JOB_BOUND_EXCEEDED')
        self.assertEqual(totals(self.j.read()['budget'])[DISK],result['total_original_bytes'])
        self.record();self.assertEqual(totals(self.j.read()['budget'])[DISK],result['total_original_bytes'])


if __name__=='__main__':unittest.main()
