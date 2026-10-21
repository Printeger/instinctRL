"""CPU-only proposed evaluation or training authority from exact frozen inputs.

Uses the snapshot's own config, parameter, compatibility and budget validators.
Prints a reviewable proposal; does not write an authority file or mutate Store.
"""
import argparse
import json
from pathlib import Path
import subprocess
from tools.r6_recovery import checked,RecoveryError
from tools.prepare_r6_snapshot import verify_frozen
from tools.r6_job_resources import MODES

PROGRAM='''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from training_workloads.unified_h1.config import load_config
c=load_config();request=json.load(sys.stdin)
if 'final_selection' in request['parameters']:
 from training_workloads.unified_h1.config import R6_FINAL_CONFIG_SCHEMA
 from training_workloads.unified_h1.robustness_final_contract import MODE,validate_parameters,required_budget,validate_compatibility
 if c['schema_version']!=R6_FINAL_CONFIG_SCHEMA:raise PermissionError('final config schema required')
 p=validate_parameters(MODE,request['parameters'])
 validate_compatibility(p['checkpoint_compatibility'],p['final_selection'],c)
elif 'evaluation_selection' in request['parameters']:
 from training_workloads.unified_h1.robustness_candidate_contract import MODE,validate_parameters,required_budget,validate_compatibility
 p=validate_parameters(MODE,request['parameters'])
 validate_compatibility(p['checkpoint_compatibility'],p['evaluation_selection'],c)
elif 'r6_training_plan' in request['parameters']:
 from training_workloads.unified_h1.robustness_training_contract import MODE,validate_parameters,required_budget,validate_compatibility
 p=validate_parameters(MODE,request['parameters'])
 validate_compatibility(p['checkpoint_compatibility'],p['r6_training_plan'],c)
else:
 from training_workloads.unified_h1.robustness_contract import MODE,validate_parameters,required_budget,validate_compatibility,baseline
 p=validate_parameters(MODE,request['parameters'])
 validate_compatibility(p['checkpoint_compatibility'],baseline(p)['source_identity'],c)
a=dict(schema_version='unified-h1-r3-authorization.v2',status='AUTHORIZED',
 order_id=p['experiment_id']+'-'+p['group']+'-'+MODE.upper(),method_sha256=c['method']['proposal_sha256'],
 **c['resolved_identity'],attempt=request['attempt'],mode=MODE,
 budgets=required_budget(MODE,p),parameters=p)
print(json.dumps(a,sort_keys=True))
'''


def prepare(snapshot_receipt,parameters_pin,attempt):
    receipt=json.loads(checked(snapshot_receipt).read_text())
    if receipt.get('status')!='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY':
        raise RecoveryError('PREPARE_AUTH_FROZEN_SNAPSHOT_REQUIRED')
    config=json.loads(checked(receipt['config']).read_text())
    verify_frozen(receipt,config)
    params=json.loads(checked(parameters_pin).read_text())
    if sum(k in params for k in ('evaluation_selection','r6_training_plan','final_selection'))>1:
        raise RecoveryError('PREPARE_AUTH_AMBIGUOUS_STAGE')
    mode=MODES[3] if 'final_selection' in params else MODES[2] if 'evaluation_selection' in params else MODES[1] if 'r6_training_plan' in params else MODES[0]
    attempt=Path(attempt)
    if (not attempt.is_absolute() or attempt.resolve()!=attempt or attempt.is_symlink()
            or not attempt.is_dir() or not (attempt/'results').is_dir()):
        raise RecoveryError('PREPARE_AUTH_ATTEMPT_REQUIRED')
    if any((attempt/'results'/m).exists() for m in MODES) or list((attempt/'results').glob('*/process_start.json')):
        raise RecoveryError('PREPARE_AUTH_ATTEMPT_CONSUMED')
    executable=checked(dict(path=config['execution']['python_path'],sha256=config['execution']['python_sha256']))
    stdin=json.dumps(dict(parameters=params,attempt=str(attempt)),sort_keys=True)
    if len(stdin.encode())>65536:raise RecoveryError('PREPARE_AUTH_INPUT_BOUND')
    result=subprocess.run([str(executable),'-I','-B','-c',PROGRAM,receipt['snapshot']],
        input=stdin,cwd=receipt['snapshot'],env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'',
        'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=60)
    if result.returncode:raise RecoveryError('PREPARE_AUTH_VALIDATION_FAILED: '+result.stderr[-3000:])
    if len(result.stdout.encode())>65536:raise RecoveryError('PREPARE_AUTH_OUTPUT_BOUND')
    auth=json.loads(result.stdout)
    if (any(auth.get(k)!=v for k,v in receipt['source_identity'].items())
        or auth.get('parameters')!=params or auth.get('attempt')!=str(attempt)
        or auth.get('mode')!=mode):
        raise RecoveryError('PREPARE_AUTH_IDENTITY_DRIFT')
    return dict(status='PROPOSED_NOT_ISSUED',snapshot_receipt=snapshot_receipt,
        parameters=parameters_pin,authorization_proposal=auth,
        store_permission_issued=False,job_submitted=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('request',type=Path);p.add_argument('--request-sha256',required=True)
    args=p.parse_args();q=json.loads(checked(dict(path=str(args.request.resolve()),sha256=args.request_sha256)).read_text())
    if set(q)!={'snapshot_receipt','parameters_pin','attempt'}:raise RecoveryError('PREPARE_AUTH_FIELDS')
    print(json.dumps(prepare(**q),sort_keys=True))


if __name__=='__main__':main()
