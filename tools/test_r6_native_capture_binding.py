"""Real CPU pipe/descriptor checks; no Carb import, GPU or simulator."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from training_workloads.unified_h1.local_dispatch import dispatch_held, DispatchError, filtered_environment

ROOT=Path(__file__).resolve().parents[1]
IMPORT='import os,json\nfrom training_workloads.scale_003.resources import _r6_capture_binding\n'


class CaptureBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.control=self.root/'control';self.control.mkdir()

    def child(self,code,cap=4096):
        return dispatch_held([sys.executable,'-B','-c',IMPORT+code],cwd=self.root,
            identity_path=self.root/'held.json',coordination_root=self.control,
            completion_timeout_s=5,native_output_cap_bytes=cap,
            environment={'PYTHONPATH':str(ROOT),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})

    def test_actual_pipe_owner_binding_and_file_path_share_same_capped_stream(self):
        code="b=_r6_capture_binding();assert b['cap_bytes']==4096\nos.write(1,b'console:')\nwith open('/proc/self/fd/1','wb',buffering=0) as f:f.write(b'file-log')\nos.write(2,b':stderr')"
        self.assertEqual(self.child(code),0)
        self.assertEqual((self.root/'native-output.log').read_bytes(),b'console:file-log:stderr')
        binding=json.loads((self.root/'held.json').read_text())['native_capture']
        self.assertEqual(binding['path'],str(self.root/'native-output.log'))
        self.assertEqual(json.loads((self.root/'native-output.json').read_text())['status'],'COMPLETE')

    def test_file_path_flood_is_subject_to_same_total_before_write_limit(self):
        code="_r6_capture_binding()\nf=open('/proc/self/fd/1','wb',buffering=0)\nwhile True:f.write(b'x'*8192)"
        with self.assertRaisesRegex(DispatchError,'cap exceeded'):self.child(code)
        self.assertEqual((self.root/'native-output.log').read_bytes(),b'x'*4096)
        self.assertEqual(json.loads((self.root/'native-output.json').read_text())['status'],'FAILED_PREFIX_PRESERVED')

    def test_missing_binding_rejected(self):
        code="del os.environ['INSTINCTRL_R6_NATIVE_CAPTURE']\ntry:_r6_capture_binding()\nexcept RuntimeError:os.write(1,b'rejected')\nelse:raise AssertionError('missing binding passed')"
        self.assertEqual(self.child(code),0)
        self.assertEqual((self.root/'native-output.log').read_bytes(),b'rejected')

    def test_wrong_owner_start_boot_pipe_and_limit_rejected(self):
        code="original=json.loads(os.environ['INSTINCTRL_R6_NATIVE_CAPTURE'])\nfor field,value in [('owner_pid',0),('owner_starttime','0'),('boot_id','wrong'),('pipe_inode',0),('cap_bytes',True),('cap_bytes',268435457)]:\n b=dict(original);b[field]=value;os.environ['INSTINCTRL_R6_NATIVE_CAPTURE']=json.dumps(b)\n try:_r6_capture_binding()\n except RuntimeError:pass\n else:raise AssertionError(field)\nos.write(1,b'all rejected')"
        self.assertEqual(self.child(code),0)
        self.assertEqual((self.root/'native-output.log').read_bytes(),b'all rejected')

    def test_replaced_stdout_descriptor_rejected(self):
        code="f=open('unbounded-file','wb');os.dup2(f.fileno(),1)\ntry:_r6_capture_binding()\nexcept RuntimeError:os.write(2,b'rejected')\nelse:raise AssertionError('regular file passed')"
        self.assertEqual(self.child(code),0)
        self.assertEqual((self.root/'native-output.log').read_bytes(),b'rejected')
        self.assertEqual((self.root/'unbounded-file').stat().st_size,0)

    def test_parent_environment_cannot_supply_capture_authority(self):
        self.assertNotIn('INSTINCTRL_R6_NATIVE_CAPTURE',filtered_environment({'INSTINCTRL_R6_NATIVE_CAPTURE':'spoof'}))


if __name__=='__main__':unittest.main()
