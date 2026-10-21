"""Build three confirmation plans in exact frozen source; no reservation or launch."""
import json
import subprocess
from pathlib import Path
from tools.r6_recovery import checked,RecoveryError
from tools.r6_prepare_authorization import verify_frozen

FIELDS={'freeze_pin','recipe_review','baseline_bindings','experiment_ids','wall_seconds','storage'}
PROGRAM='''import json,sys
sys.path.insert(0,sys.argv[1])
from training_workloads.unified_h1.config import load_config,R6_TRAINING_CONFIG_SCHEMA
from training_workloads.unified_h1.robustness_contract import pinned_json
from training_workloads.unified_h1.robustness_confirmation import build_plans
c=load_config();q=json.load(sys.stdin)
if c['schema_version']!=R6_TRAINING_CONFIG_SCHEMA:raise PermissionError('confirmation training config required')
freeze=pinned_json(q['freeze_pin']);recipe=pinned_json(freeze['recipe'])
if recipe['source_identity']!=c['resolved_identity']:raise PermissionError('confirmation execution identity')
bundle=build_plans(**q)
print(json.dumps(dict(source_identity=c['resolved_identity'],bundle=bundle),sort_keys=True))
'''


def prepare(snapshot_pin,request):
    if not isinstance(request,dict) or set(request)!=FIELDS:raise RecoveryError('CONFIRMATION_PLAN_FIELDS')
    for key in ('freeze_pin','recipe_review','baseline_bindings'):checked(request[key])
    snapshot=json.loads(checked(snapshot_pin).read_text())
    if snapshot.get('status')!='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY':raise RecoveryError('CONFIRMATION_PLAN_SNAPSHOT')
    config=json.loads(checked(snapshot['config']).read_text());verify_frozen(snapshot,config)
    exe=checked(dict(path=config['execution']['python_path'],sha256=config['execution']['python_sha256']))
    data=json.dumps(request,sort_keys=True)
    if len(data.encode())>65536:raise RecoveryError('CONFIRMATION_PLAN_INPUT_CAP')
    # Three complete provenance validations, including original file hashes.
    result=subprocess.run([str(exe),'-I','-B','-c',PROGRAM,snapshot['snapshot']],input=data,
        cwd=snapshot['snapshot'],env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},
        capture_output=True,text=True,timeout=180)
    if result.returncode:raise RecoveryError('CONFIRMATION_PLAN_FROZEN: '+result.stderr[-2000:])
    if len(result.stdout.encode())>65536:raise RecoveryError('CONFIRMATION_PLAN_OUTPUT_CAP')
    value=json.loads(result.stdout);bundle=value.get('bundle',{})
    if (set(value)!={'source_identity','bundle'} or value['source_identity']!=snapshot['source_identity']
            or bundle.get('schema_version')!='r6-three-confirmation-plans/1'
            or any(bundle.get(k)is not False for k in ('scientific_acceptance','dispatch_authorized','holdout_released'))
            or bundle.get('freeze')!=request['freeze_pin'] or bundle.get('baseline_bindings')!=request['baseline_bindings']
            or len(bundle.get('plans',[]))!=3
            or [p['experiment_id'] for p in bundle['plans']]!=request['experiment_ids']):
        raise RecoveryError('CONFIRMATION_PLAN_IDENTITY')
    return dict(status='THREE_CONFIRMATION_PLANS_PREPARED_NOT_RESERVED',snapshot=snapshot_pin,request=request,
        **value,dispatch_authorized=False)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',type=Path);parser.add_argument('--inputs-sha256',required=True)
    args=parser.parse_args();q=json.loads(checked(dict(path=str(args.inputs.absolute()),sha256=args.inputs_sha256)).read_text())
    if set(q)!={'snapshot','request'}:raise RecoveryError('CONFIRMATION_PLAN_CLI_FIELDS')
    print(json.dumps(prepare(q['snapshot'],q['request']),sort_keys=True))


if __name__=='__main__':main()
