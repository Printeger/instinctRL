"""Run migrated workload modules in isolated CPU processes with existing interpreters."""
from __future__ import annotations
import argparse, ast, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CONTROL = '/home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python'
PRODUCT = '/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11'
def interpreter(path):
    tree = ast.parse(path.read_text())
    uses_torch = any(isinstance(n, (ast.Import, ast.ImportFrom)) and
        (getattr(n, 'module', '') == 'torch' or any(a.name == 'torch' for a in getattr(n, 'names', [])))
        for n in ast.walk(tree))
    return PRODUCT if uses_torch or path.name in ('test_r4_evaluation_start.py', 'test_paired_pilot_unittest.py') else CONTROL

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('modules',nargs='*');args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    modules=[ROOT/p for p in args.modules] if args.modules else sorted((ROOT/'tests/training_workloads').glob('test_*.py'))
    results=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=f'{ROOT}:{ROOT}/isaaclab-training:{ROOT}/tools/rlflow:{ROOT}/tools/rlflow/tests')
    for path in modules:
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():raise ValueError('test path escapes repository')
        relative=str(path.relative_to(ROOT));name=path.stem
        argv=[interpreter(path),'-m','pytest','-q',relative,'--junitxml='+str(output/(name+'.xml'))]
        start=time.monotonic()
        with (output/(name+'.stdout')).open('w') as stdout,(output/(name+'.stderr')).open('w') as stderr:
            try:completed=subprocess.run(argv,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=600);exit_code=completed.returncode
            except subprocess.TimeoutExpired:exit_code=124
        result={'module':relative,'argv':argv,'exit_code':exit_code,'seconds':time.monotonic()-start};results.append(result)
        print(relative,exit_code,flush=True)
        (output/'COMMANDS.json').write_text(json.dumps({'env_overrides':{k:env[k] for k in ('PYTHONDONTWRITEBYTECODE','PYTEST_DISABLE_PLUGIN_AUTOLOAD','CUDA_VISIBLE_DEVICES','PYTHONPATH')},'results':results},indent=2)+'\n')
    return int(any(row['exit_code'] for row in results))
if __name__=='__main__':raise SystemExit(main())
