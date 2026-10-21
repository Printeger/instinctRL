"""Build exact prospective review payloads in frozen training source. No issuance."""
import json
import subprocess
from pathlib import Path
from tools.r6_recovery import checked,RecoveryError
from tools.r6_prepare_authorization import verify_frozen

PROGRAM='''import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from training_workloads.unified_h1.config import load_config,R6_TRAINING_CONFIG_SCHEMA
from training_workloads.unified_h1.robustness_training_plan import validate_recipe,pinned_json
from training_workloads.unified_h1.robustness_training_contract import capacity_payload,compatibility_payload
c=load_config();q=json.load(sys.stdin)
if c['schema_version']!=R6_TRAINING_CONFIG_SCHEMA:raise PermissionError('training config schema required')
if q['kind']=='recipe':
 r=validate_recipe(pinned_json(q['recipe']))
 if r['source_identity']!=c['resolved_identity']:raise PermissionError('recipe execution identity')
 payloads={'recipe':r}
elif q['kind']=='freeze':
 from training_workloads.unified_h1.robustness_confirmation import freeze_payload
 if pinned_json(q['recipe'])['source_identity']!=c['resolved_identity']:raise PermissionError('freeze execution identity')
 payloads={'freeze':freeze_payload(q['recipe'],q['recipe_review'],q['selection_evidence'])}
else:
 p=pinned_json(q['plan']);config=Path(sys.argv[1])/'training_workloads/unified_h1/candidate_config.json'
 cp=dict(path=str(config),sha256=hashlib.sha256(config.read_bytes()).hexdigest())
 payloads=dict(capacity=capacity_payload(p),compatibility=compatibility_payload(q['origin_config'],cp,p,c))
print(json.dumps(dict(source_identity=c['resolved_identity'],payloads=payloads),sort_keys=True))
'''


def prepare(snapshot_pin,request):
    if not isinstance(request,dict) or request.get('kind') not in ('recipe','stage','freeze'):
        raise RecoveryError('TRAINING_REVIEW_KIND')
    expected={'kind','recipe'} if request['kind']=='recipe' else {'kind','recipe','recipe_review','selection_evidence'} if request['kind']=='freeze' else {'kind','plan','origin_config'}
    if set(request)!=expected:raise RecoveryError('TRAINING_REVIEW_FIELDS')
    for name in expected-{'kind'}:checked(request[name])
    snapshot=json.loads(checked(snapshot_pin).read_text())
    if snapshot.get('status')!='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY':
        raise RecoveryError('TRAINING_REVIEW_SNAPSHOT')
    config=json.loads(checked(snapshot['config']).read_text());verify_frozen(snapshot,config)
    exe=checked(dict(path=config['execution']['python_path'],sha256=config['execution']['python_sha256']))
    data=json.dumps(request,sort_keys=True)
    if len(data.encode())>65536:raise RecoveryError('TRAINING_REVIEW_INPUT_CAP')
    result=subprocess.run([str(exe),'-I','-B','-c',PROGRAM,snapshot['snapshot']],
        input=data,cwd=snapshot['snapshot'],env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'',
        'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=60)
    if result.returncode:raise RecoveryError('TRAINING_REVIEW_FROZEN: '+result.stderr[-2000:])
    if len(result.stdout.encode())>65536:raise RecoveryError('TRAINING_REVIEW_OUTPUT_CAP')
    value=json.loads(result.stdout)
    names={request['kind']} if request['kind'] in ('recipe','freeze') else {'capacity','compatibility'}
    if (set(value)!={'source_identity','payloads'} or value['source_identity']!=snapshot['source_identity']
            or not isinstance(value['payloads'],dict) or set(value['payloads'])!=names
            or any(not isinstance(p,dict) or 'independent_review' in p for p in value['payloads'].values())):
        raise RecoveryError('TRAINING_REVIEW_IDENTITY')
    return dict(status='REVIEW_PAYLOADS_PREPARED_NOT_ACCEPTED',snapshot=snapshot_pin,request=request,
        **value,dispatch_authorized=False)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',type=Path);parser.add_argument('--inputs-sha256',required=True)
    args=parser.parse_args()
    value=json.loads(checked(dict(path=str(args.inputs.resolve()),sha256=args.inputs_sha256)).read_text())
    if set(value)!={'snapshot','request'}:raise RecoveryError('TRAINING_REVIEW_CLI_FIELDS')
    print(json.dumps(prepare(value['snapshot'],value['request']),sort_keys=True))


if __name__=='__main__':main()
