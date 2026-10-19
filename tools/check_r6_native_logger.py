"""Standalone Carbonite logging CPU check; no Kit/framework startup or GPU.

The parent uses the existing bounded held-child dispatcher. Native libraries are
read from the frozen installation, never installed or modified. The child runs
without Python site initialization and imports only the explicit Carb binding.
"""
import argparse
import ctypes
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
KIT=Path('/home/mint/miniconda3/envs/navrl_blackwell_test/lib/python3.11/site-packages/isaacsim/kit')
PYTHON=Path('/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11')
PYTHON_LIBRARY=PYTHON.parent.parent/'lib/libpython3.11.so.1.0'


def child():
    ctypes.CDLL(str(PYTHON_LIBRARY),mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(KIT/'libcarb.so'),mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0,str(KIT/'kernel/py'))
    import carb
    logging=carb.logging.acquire_logging()
    # Inspect the installed binding before selecting a file-logger API. Never
    # start the framework or load plugins to make an unavailable API appear.
    print(json.dumps({'schema_version':'r6-native-logger-api/1',
        'logging_members':{n:str(getattr(logging,n).__doc__) for n in dir(logging) if not n.startswith('_')},
        'logging_module_members':[n for n in dir(carb.logging) if not n.startswith('_')],
        'loaded_plugins':[str(p.name) for p in carb.get_framework().get_plugins()],
        'mapped_files':sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line}),
        'framework_startup_called':False,'kit_started':False},sort_keys=True),flush=True)


def run(output):
    from training_workloads.unified_h1.local_dispatch import dispatch_held
    from tools.r6_recovery import file_pin
    output=output.resolve();output.mkdir(exist_ok=False)
    inputs=[Path(__file__),ROOT/'training_workloads/unified_h1/local_dispatch.py',PYTHON,PYTHON_LIBRARY,
            KIT/'libcarb.so',KIT/'kernel/py/carb/__init__.py',
            KIT/'kernel/py/carb/_carb.cpython-311-x86_64-linux-gnu.so']
    pins=[file_pin(p) for p in inputs]
    (output/'inputs.json').write_text(json.dumps(pins,indent=2)+'\n')
    code=dispatch_held([str(PYTHON),'-I','-S','-B',str(Path(__file__).resolve()),'--child'],
        cwd=output,identity_path=output/'held.json',coordination_root=ROOT/'coordination/research',
        completion_timeout_s=15,native_output_cap_bytes=256*1024,
        environment={'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
    if [file_pin(p) for p in inputs]!=pins:raise RuntimeError('CPU native input drift')
    print(json.dumps({'native_exit_code':code,'output':str(output),'inputs_unchanged':True}))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.child:child()
    elif args.output is not None:sys.exit(run(args.output))
    else:parser.error('--output required')
