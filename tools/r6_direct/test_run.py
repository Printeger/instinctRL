"""Real CPU child counterexamples; no GPU or simulator."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
from tools.r6_direct.run import supervise,reserve


class RunnerTests(unittest.TestCase):
    def test_scientific_preflight_before_pure_system_accounting(self):
        from tools.r6_direct.run import preflight_resources
        from tools.r6_direct.import_legacy import pin
        contract=json.loads(Path('docs/experiments/r6-direct-training/continuation-20261101/confirmation-128/seed-01-contract.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            inputs=Path(tmp)/'inputs.json';inputs.write_text('{}')
            cp=Path(tmp)/'contract.json';cp.write_text(json.dumps(contract))
            ip=pin(inputs);bound=pin(cp)
            def call(code):
                return preflight_resources([sys.executable,'-c',code],tmp,os.environ.copy(),inputs,ip,bound,contract)
            # CPU worker stub: process exit and exact pins are real. Scientific
            # contract validation itself is covered by the scientific tests.
            value=call('pass')
            self.assertEqual(value['training_transitions'],65536)
            self.assertNotIn('major_scientific_candidates',value)
            with self.assertRaises(subprocess.CalledProcessError):call('raise SystemExit(2)')
            with self.assertRaisesRegex(ValueError,'PREFLIGHT_INPUT_DRIFT'):
                call("from pathlib import Path;Path('contract.json').write_text('{}')")
            cp.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError,'PREFLIGHT_INPUT_DRIFT'):
                call("from pathlib import Path;Path('inputs.json').write_text('changed')")

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
