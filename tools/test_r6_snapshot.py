"""Tiny real Git snapshot fixtures; config verifier substituted, no live calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from tools import prepare_r6_snapshot as s
from tools.r6_recovery import Journal,RecoveryError,file_pin

ROOT=Path(__file__).resolve().parents[1]


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.clean)
        self.root=Path(self.tmp.name);self.repo=self.root/'repo';self.repo.mkdir()
        self.target=self.root/'snapshot'
        cfg=self.repo/'training_workloads/unified_h1/candidate_config.json';cfg.parent.mkdir(parents=True);cfg.write_text('{}')
        subprocess.run(['git','init','-q',str(self.repo)],check=True)
        s.git(self.repo,'add','.');s.git(self.repo,'-c','user.name=CPU fixture','-c','user.email=cpu@example.invalid','commit','-qm','fixture')
        self.commit=s.git(self.repo,'rev-parse','HEAD').decode().strip()
        self.tree=s.git(self.repo,'rev-parse','HEAD^{tree}').decode().strip()
        self.config=dict(product=dict(source_sha256={}),workload=dict(source_sha256={}),execution=dict(python_path=str(Path(sys.executable).resolve()),python_sha256=file_pin(sys.executable)['sha256']))
        self.path=self.root/'input.json';self.path.write_text(json.dumps(self.config))
        self.journal=Journal(self.root/'batch')
        budget=json.loads((ROOT/'docs/experiments/r6-robustness-auto/goal_budget.json').read_text())
        budget['reservations']=[];budget['used']={k:0 for k in budget['used']}
        with self.journal.writer():self.journal.initialize(budget)
        self.runs=0;self.realrun=subprocess.run

    def clean(self):
        for p in [Path(self.tmp.name),*Path(self.tmp.name).rglob('*')]:
            if not p.is_symlink():p.chmod(p.stat().st_mode|0o700)
        self.tmp.cleanup()

    def command(self,argv,**kwargs):
        if argv[0]==self.config['execution']['python_path']:
            self.runs+=1
            value={a+'_'+b:v for a in ('product','workload') for b,v in [('commit',self.commit),('tree',self.tree)]}|{'config_sha256':file_pin(self.path)['sha256']}
            return subprocess.CompletedProcess(argv,0,json.dumps(value),'')
        return self.realrun(argv,**kwargs)

    def prepare(self,*,purpose="development"):
        with patch.object(s,'ROOT',self.repo),patch.object(s,'preflight',return_value=(self.config,self.target,self.commit)),patch.object(s.subprocess,'run',side_effect=self.command):
            with self.journal.writer():return s.prepare(self.journal,self.path,self.commit,purpose=purpose)

    def test_real_clone_freeze_and_replay_no_second_run_or_reservation(self):
        result=self.prepare();before=self.journal.read()['budget']
        self.assertEqual(result['status'],'CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY')
        self.assertFalse(self.target.stat().st_mode&0o222)
        self.assertEqual(self.prepare(),result);self.assertEqual(self.runs,1)
        self.assertEqual(self.journal.read()['budget'],before)

    def test_final_snapshot_uses_protected_purpose_and_cannot_change_it(self):
        result=self.prepare(purpose='confirmation_delivery')
        self.assertEqual(self.prepare(purpose='confirmation_delivery'),result)
        state=self.journal.read();reservation=state['budget']['reservations'][0]
        self.assertEqual(reservation['purpose'],'confirmation_delivery')
        self.assertEqual(state['events'][result['event_key']]['request']['purpose'],'confirmation_delivery')
        with self.assertRaisesRegex(RecoveryError,'TARGET_ALREADY_BOUND'):self.prepare()
        self.assertEqual(self.runs,1)

    def test_replay_rejects_thawed_snapshot(self):
        self.prepare();self.target.chmod(self.target.stat().st_mode|0o200)
        with self.assertRaisesRegex(RecoveryError,'NOT_READ_ONLY'):self.prepare()

    def test_failed_validation_keeps_reserve_and_blocks_retry(self):
        self.command=lambda argv,**kw:subprocess.CompletedProcess(argv,1,'','CPU injected failure') if argv[0]==self.config['execution']['python_path'] else self.realrun(argv,**kw)
        with self.assertRaisesRegex(RecoveryError,'FAILED_PRESERVED'):self.prepare()
        state=self.journal.read();self.assertEqual(len(state['budget']['reservations']),1)
        self.assertEqual(next(iter(state['events'].values()))['status'],'FAILED_PRESERVED_NO_RETRY')
        with self.assertRaisesRegex(RecoveryError,'PRIOR_FAILURE_NO_RETRY'):self.prepare()

    def test_readonly_untracked_python_is_rejected(self):
        self.prepare();self.target.chmod(self.target.stat().st_mode|0o200)
        extra=self.target/'sitecustomize.py';extra.write_text('raise RuntimeError("must never execute")');extra.chmod(0o444)
        self.target.chmod(self.target.stat().st_mode&~0o222)
        with self.assertRaisesRegex(RecoveryError,'INVENTORY_DRIFT'):self.prepare()

    def test_readonly_ignored_bytecode_is_rejected(self):
        self.prepare();self.target.chmod(self.target.stat().st_mode|0o200)
        extra=self.target/'unexpected.pyc';extra.write_bytes(b'not real bytecode');extra.chmod(0o444)
        self.target.chmod(self.target.stat().st_mode&~0o222)
        with self.assertRaisesRegex(RecoveryError,'INVENTORY_DRIFT'):self.prepare()

    def test_symlink_addition_is_rejected(self):
        self.prepare();self.target.chmod(self.target.stat().st_mode|0o200)
        (self.target/'external.py').symlink_to(self.path)
        self.target.chmod(self.target.stat().st_mode&~0o222)
        with self.assertRaisesRegex(RecoveryError,'SYMLINK_ENTRY'):self.prepare()

    def test_stop_prevents_reservation_or_clone(self):
        (self.journal.directory/'STOP').touch()
        with self.assertRaisesRegex(RecoveryError,'STOP'):self.prepare()
        self.assertFalse(self.target.exists());self.assertEqual(self.journal.read()['budget']['reservations'],[])


if __name__=='__main__':unittest.main()
