"""Bounded CPU-only LLVM probe using the already installed Warp compiler."""
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import resource
import shutil

ROOT=Path('/home/mint/.local/share/ov/data/exts/v2/omni.warp.core-1.8.2+lx64/warp')
LIB_SHA='3c5d656d3b45fa0ac3db5237fe8f496137d9f5606dcf9b2a1c8e2e3b9f774ac5'
TARGET=Path('/run/lock/instinctrl-r6-cpu-jit-01')
OUTPUT=Path(__file__).resolve().parents[1]/'docs/experiments/r6-robustness-auto/CPU_JIT_NOEXEC_01'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    lib=ROOT/'bin/warp-clang.so'
    assert sha(lib)==LIB_SHA
    mounts=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1);f=left.split()
        if f[4]=='/run/lock':mounts.append(dict(line=line,options=f[5].split(',')))
    assert len(mounts)==1 and 'noexec' in mounts[0]['options']
    TARGET.mkdir(mode=0o700);OUTPUT.mkdir()
    report=dict(status='STARTED',mount=mounts[0],library=dict(path=str(lib),sha256=LIB_SHA),
                tool_sha256=sha(Path(__file__)),gpu_initialization=False,simulator_import=False)
    try:
        source=TARGET/'probe.cpp';obj=TARGET/'probe.o'
        source.write_text('extern "C" int r6_probe(int a, int b) { return a*b; }\n')
        llvm=ctypes.CDLL(str(lib))
        llvm.compile_cpp.argtypes=[ctypes.c_char_p]*4+[ctypes.c_bool]*3
        llvm.compile_cpp.restype=ctypes.c_int
        assert llvm.compile_cpp(source.read_bytes(),os.fsencode(source),os.fsencode(ROOT/'native'),os.fsencode(obj),False,False,True)==0
        # A real file-backed executable mapping must be rejected on this mount.
        with obj.open('rb') as stream:
            try:
                mapping=mmap.mmap(stream.fileno(),0,flags=mmap.MAP_PRIVATE,prot=mmap.PROT_READ|mmap.PROT_EXEC)
            except OSError as error:report['file_exec_mapping_errno']=error.errno
            else:
                mapping.close();raise AssertionError('noexec mapping unexpectedly allowed')
        assert report['file_exec_mapping_errno'] in (1,13)
        llvm.load_obj.argtypes=[ctypes.c_char_p,ctypes.c_char_p];llvm.load_obj.restype=ctypes.c_int
        assert llvm.load_obj(os.fsencode(obj),b'r6_cpu_noexec_probe')==0
        llvm.lookup.argtypes=[ctypes.c_char_p,ctypes.c_char_p];llvm.lookup.restype=ctypes.c_uint64
        address=llvm.lookup(b'r6_cpu_noexec_probe',b'r6_probe');assert address
        call=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_int,ctypes.c_int)(address)
        report['result']=call(6,7);assert report['result']==42
        report['status']='PASS_CPU_LLVM_ON_NOEXEC'
        report['limitations']=['tmpfs noexec, not production ext4 quota volume','CPU compiler/object loader only; no CUDA, Kit dependency resolution or total native write claim']
    except BaseException as error:
        report['status']='FAIL';report['error']=repr(error);raise
    finally:
        report['originals']=[]
        for p in sorted(TARGET.iterdir()):
            if p.is_file():
                shutil.copyfile(p,OUTPUT/p.name)
                report['originals'].append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
        (OUTPUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))


if __name__=='__main__':main()
