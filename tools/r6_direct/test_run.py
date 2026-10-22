"""Real CPU child counterexamples; no GPU or simulator."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from tools.r6_direct.run import supervise,reserve


class RunnerTests(unittest.TestCase):
    def test_child_exit_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=supervise([sys.executable,'-c','import sys; assert sys.stdin.buffer.read(1)==b"1"; print("CPU child")'],
                cwd=tmp,env=os.environ.copy(),directory=tmp,deadline_seconds=3,stop=lambda:False)
            self.assertEqual(result['exit_code'],0);self.assertEqual(result['remaining'],[])
            self.assertIn('CPU child',(Path(tmp)/'stdout.log').read_text())
            original=(Path(tmp)/'receipt.json').read_bytes()
            with self.assertRaises(FileExistsError):
                supervise([sys.executable,'-c','raise RuntimeError("must not start")'],
                    cwd=tmp,env=os.environ.copy(),directory=tmp,deadline_seconds=3,stop=lambda:False)
            self.assertEqual((Path(tmp)/'receipt.json').read_bytes(),original)

    def test_timeout_kills_own_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=supervise([sys.executable,'-c','import sys,time;sys.stdin.buffer.read(1);time.sleep(60)'],
                cwd=tmp,env=os.environ.copy(),directory=tmp,deadline_seconds=.2,stop=lambda:False)
            self.assertIn('TIMEOUT',result['error']);self.assertEqual(result['remaining'],[])
            self.assertLess(result['wall_seconds'],10)

    def test_stop_before_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=supervise([sys.executable,'-c','import sys;sys.stdin.buffer.read(1);print("should not run")'],
                cwd=tmp,env=os.environ.copy(),directory=tmp,deadline_seconds=3,stop=lambda:True)
            self.assertIn('STOP',result['error']);self.assertEqual(result['remaining'],[])
            self.assertEqual((Path(tmp)/'stdout.log').read_bytes(),b'')

    def test_adopted_detached_descendant_is_cleaned(self):
        script='import os,sys,time;sys.stdin.buffer.read(1);pid=os.fork();os.setsid() if pid==0 else None;time.sleep(60) if pid==0 else None'
        with tempfile.TemporaryDirectory() as tmp:
            result=supervise([sys.executable,'-c',script],cwd=tmp,env=os.environ.copy(),
                directory=tmp,deadline_seconds=.3,stop=lambda:False)
            self.assertIn('TIMEOUT',result['error']);self.assertEqual(result['remaining'],[])
            self.assertGreaterEqual(len(result['owned_processes']),2)

    def test_protected_budget_and_duplicate(self):
        ledger=dict(caps={'x':100},historical_charged={'x':10},new_used={'x':2},
            protected_confirmation_delivery_repair={'x':50},new_reservations=[])
        with self.assertRaises(ValueError):reserve(copy.deepcopy(ledger),'r',{'x':39})
        value=reserve(ledger,'r',{'x':38})
        with self.assertRaises(ValueError):reserve(value,'r',{'x':0})


if __name__=='__main__':unittest.main()
