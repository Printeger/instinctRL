"""Real CPU fork/exec/pipe tests for R6 native output bounds."""
import json
import hashlib
import os
from unittest.mock import patch
from training_workloads.unified_h1 import local_dispatch as dispatch
from pathlib import Path
import sys
import tempfile
import unittest
from training_workloads.unified_h1.local_dispatch import dispatch_held,DispatchError


class NativeOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.coordination=self.root/'coordination';self.coordination.mkdir()

    def run_child(self,code,cap=4096,timeout=5):
        return dispatch_held([sys.executable,'-B','-c',code],cwd=self.root,identity_path=self.root/'held.json',coordination_root=self.coordination,completion_timeout_s=timeout,native_output_cap_bytes=cap)

    def receipt(self):return json.loads((self.root/'native-output.json').read_text())

    def test_both_streams_preserved_byte_exact_with_native_nonzero_exit(self):
        self.assertEqual(self.run_child("import os;os.write(1,b'first\\x00\\xff');os.write(2,b'second');os._exit(7)"),7)
        self.assertEqual((self.root/'native-output.log').read_bytes(),b'first\x00\xffsecond')
        self.assertEqual(self.receipt()['native_exit_code'],7);self.assertEqual(self.receipt()['status'],'COMPLETE')

    def test_output_flood_preserves_exact_cap_and_fails(self):
        with self.assertRaisesRegex(DispatchError,'cap exceeded'):
            self.run_child("import os;\nwhile True:os.write(1,b'x'*8192)")
        self.assertEqual((self.root/'native-output.log').read_bytes(),b'x'*4096)
        self.assertEqual(self.receipt()['written_bytes'],4096)
        self.assertEqual(self.receipt()['sha256'],hashlib.sha256(b'x'*4096).hexdigest())
        self.assertEqual(self.receipt()['status'],'FAILED_PREFIX_PRESERVED')

    def test_exact_boundary_succeeds_and_existing_log_not_overwritten(self):
        self.assertEqual(self.run_child("import os;os.write(1,b'x'*4096)"),0)
        with self.assertRaisesRegex(DispatchError,'receipt already exists'):self.run_child('pass')
        self.assertEqual((self.root/'native-output.log').stat().st_size,4096)

    def test_timeout_preserves_output_and_failure(self):
        with self.assertRaisesRegex(DispatchError,'completion timeout'):
            self.run_child("import os,time;os.write(1,b'prefix');time.sleep(30)",timeout=.15)
        self.assertEqual((self.root/'native-output.log').read_bytes(),b'prefix')
        self.assertIn('completion timeout',self.receipt()['error'])

    def test_stop_during_run_preserves_output(self):
        code="import os,time;from pathlib import Path;os.write(1,b'before-stop');Path('coordination/STOP').touch();time.sleep(30)"
        with self.assertRaisesRegex(DispatchError,'STOP'):self.run_child(code)
        self.assertEqual((self.root/'native-output.log').read_bytes(),b'before-stop')
        self.assertIn('STOP',self.receipt()['error'])

    def test_stop_before_spawn_creates_no_output(self):
        (self.coordination/'STOP').touch()
        with self.assertRaisesRegex(DispatchError,'STOP'):self.run_child('pass')
        self.assertFalse((self.root/'native-output.log').exists())

    def test_timeout_retains_sigterm_diagnostics_and_exit_seven(self):
        code="import os,time,signal;signal.signal(signal.SIGTERM,lambda *a:(os.write(2,b'term-tail'),os._exit(7)));os.write(1,b'prefix');time.sleep(30)"
        with self.assertRaisesRegex(DispatchError,'completion timeout'):self.run_child(code,timeout=.15)
        self.assertEqual((self.root/'native-output.log').read_bytes(),b'prefixterm-tail')
        self.assertEqual(self.receipt()['native_exit_code'],7)

    def test_already_exited_child_is_reaped_without_signalling_reused_group(self):
        class API:
            def waitpid(self,*args):return 4242,7<<8
            def getpgid(self,*args):raise AssertionError('must not inspect reused PID')
            def killpg(self,*args):raise AssertionError('must not signal reused group')
        self.assertEqual(dispatch._reclaim(4242,4242,process_api=API()),7)

    def test_fork_failure_closes_descriptors_and_saves_failure(self):
        before=set(os.listdir('/proc/self/fd'))
        with patch.object(os,'fork',side_effect=OSError('CPU fork EAGAIN')):
            with self.assertRaisesRegex(OSError,'EAGAIN'):self.run_child('pass')
        self.assertEqual(set(os.listdir('/proc/self/fd')),before)
        self.assertIn('EAGAIN',self.receipt()['error'])
        self.assertEqual(self.receipt()['status'],'FAILED_PREFIX_PRESERVED')

    def test_selector_setup_failure_reaps_held_child_and_closes_descriptors(self):
        before=set(os.listdir('/proc/self/fd'));children=[];fork=os.fork
        def spawn():
            pid=fork()
            if pid:children.append(pid)
            return pid
        class Broken:
            def register(self,*args):raise RuntimeError('CPU register failure')
            def close(self):pass
        with patch.object(os,'fork',side_effect=spawn),patch.object(dispatch.selectors,'DefaultSelector',return_value=Broken()):
            with self.assertRaisesRegex(RuntimeError,'register failure'):self.run_child('pass')
        self.assertEqual(set(os.listdir('/proc/self/fd')),before)
        for pid in children:
            with self.assertRaises(ChildProcessError):os.waitpid(pid,os.WNOHANG)
        self.assertIn('register failure',self.receipt()['error'])

    def test_native_fsync_failure_cannot_return_success(self):
        original=os.fsync
        def sync(fd):
            if os.readlink('/proc/self/fd/'+str(fd)).endswith('native-output.log'):
                raise OSError('CPU native fsync failure')
            return original(fd)
        with patch.object(os,'fsync',side_effect=sync):
            with self.assertRaisesRegex(DispatchError,'cleanup/persistence failed'):self.run_child('pass')
        self.assertEqual(self.receipt()['status'],'FAILED_PREFIX_PRESERVED')
        self.assertEqual(self.receipt()['native_exit_code'],0)

    def test_invalid_limit_rejected_before_spawn(self):
        for cap in [0,-1,True,256*1024**2+1]:
            with self.subTest(cap=cap):
                with self.assertRaises(ValueError):self.run_child('pass',cap=cap)
        self.assertFalse((self.root/'native-output.log').exists())


if __name__=='__main__':unittest.main()
