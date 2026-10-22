"""Actual CPU POSIX semaphore under private mount + production write boundary."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class PrivateIPCTests(unittest.TestCase):
    def test_named_semaphore_and_shared_memory_stay_in_bounded_native_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);native=base/'native';native.mkdir();ipc=native/'ipc';ipc.mkdir()
            protected=base/'protected';protected.write_text('unchanged')
            script='''
import ctypes,os,sys,errno
from pathlib import Path
from training_workloads.unified_h1.write_boundary import prepare,enforce
native=Path(sys.argv[1]);protected=Path(sys.argv[2])
assert Path('/dev/shm').stat().st_ino==(native/'ipc').stat().st_ino
enforce(prepare([str(native),'/dev/shm'],['/dev/null','/dev/zero']))
lib=ctypes.CDLL(None,use_errno=True);lib.sem_open.restype=ctypes.c_void_p
ptr=lib.sem_open(b'/carbonite-sharedmemory',os.O_CREAT|os.O_EXCL,0o666,1)
assert ptr not in (None,ctypes.c_void_p(-1).value),ctypes.get_errno()
assert (native/'ipc/sem.carbonite-sharedmemory').is_file()
lib.sem_close.argtypes=[ctypes.c_void_p];assert lib.sem_close(ptr)==0
Path('/dev/shm/owned-cpu-test').write_bytes(b'bounded')
assert (native/'ipc/owned-cpu-test').read_bytes()==b'bounded'
try:protected.write_text('wrong')
except OSError as exc:assert exc.errno in (errno.EROFS,errno.EPERM,errno.EACCES)
else:raise AssertionError('outside write allowed')
print('PASS_PRIVATE_IPC_AND_READ_ONLY_OUTSIDE')
'''
            env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
            result=subprocess.run(['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/',
                '--bind',str(native),str(native),'--bind',str(ipc),'/dev/shm','--',
                '/usr/bin/python3','-B','-c',script,str(native),str(protected)],env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('PASS_PRIVATE_IPC',result.stdout)
            self.assertEqual(protected.read_text(),'unchanged')
            self.assertEqual((ipc/'owned-cpu-test').read_bytes(),b'bounded')


if __name__=='__main__':unittest.main()
