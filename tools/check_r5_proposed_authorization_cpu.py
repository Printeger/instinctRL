"""Call the actual frozen AUTH loader on an explicitly temporary attempt fixture.

Only the attempt directory is rebound, because the real future attempt must not
exist before Commander issuance. All scientific/source/input/order/budget bytes
are checked by the unmodified frozen loader. No Store or runtime is constructed.
"""
import argparse
import json
from pathlib import Path
import subprocess


PROGRAM = r'''
import json,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
from training_workloads.unified_h1.authorization import load_r3_authorization
original=Path(sys.argv[2]);value=json.loads(original.read_text());mode=sys.argv[3]
with tempfile.TemporaryDirectory(prefix='r5-auth-cpu-fixture-') as temporary:
    attempt=Path(temporary)/'attempt';attempt.mkdir()
    value['attempt']=str(attempt)
    fixture=Path(temporary)/'AUTH.json';fixture.write_text(json.dumps(value))
    try:
        checked=load_r3_authorization(fixture,attempt,mode)
    except PermissionError as error:
        if not sys.argv[4] or sys.argv[4] not in str(error):raise
        result={'status':'EXPECTED_REJECTION','error':str(error)}
    else:
        if sys.argv[4]:raise AssertionError('Expected rejection did not occur')
        result={'status':'PASS_FULL_FROZEN_LOADER_TEMP_ATTEMPT_ONLY',
                'order_id':checked['order_id'],'mode':checked['mode'],
                'budgets':checked['budgets']}
    result.update(CPU_fixture_only=True,actual_authorization_issued=False,
                  only_rebound_field='attempt',GPU_queries=0,live_starts=0)
    print(json.dumps(result))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--authorization',type=Path,required=True)
    parser.add_argument('--mode',required=True)
    parser.add_argument('--expect-rejection',default='')
    args=parser.parse_args()
    config=json.loads((args.snapshot/'training_workloads/unified_h1/candidate_config.json').read_text())
    result=subprocess.run([config['execution']['python_path'],'-I','-B','-c',PROGRAM,
                           str(args.snapshot.resolve()),str(args.authorization.resolve()),
                           args.mode,args.expect_rejection],cwd=args.snapshot,
                          env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'',
                               'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=60)
    print(result.stdout,end='')
    if result.returncode:
        print(result.stderr,end='')
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
