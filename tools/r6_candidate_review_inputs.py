"""Prospective selected-model source compatibility, frozen CPU validation only."""
import json
import subprocess
from pathlib import Path
from tools.r6_recovery import checked, RecoveryError
from tools.r6_prepare_authorization import verify_frozen

PROGRAM='''import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from training_workloads.unified_h1.config import load_config,R6_CANDIDATE_CONFIG_SCHEMA
from training_workloads.unified_h1.robustness_contract import pinned_json
from training_workloads.unified_h1.robustness_candidate_contract import compatibility_payload
from training_workloads.unified_h1.robustness_evaluation_selection import validate_draft
c=load_config();q=json.load(sys.stdin)
if c['schema_version']!=R6_CANDIDATE_CONFIG_SCHEMA:raise PermissionError('candidate config schema required')
p=Path(sys.argv[1])/'training_workloads/unified_h1/candidate_config.json'
cp=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
if q.get('kind')=='selection':
 payload=pinned_json(q['selection']);validate_draft(payload)
else:
 payload=compatibility_payload(q['origin_config'],cp,pinned_json(q['selection']),c)
print(json.dumps(dict(source_identity=c['resolved_identity'],payload=payload),sort_keys=True))
'''


def prepare(snapshot_pin,request):
    if not isinstance(request,dict):raise RecoveryError('CANDIDATE_REVIEW_FIELDS')
    kind=request.get('kind','compatibility')
    fields={'kind','selection'} if kind=='selection' else {'kind','origin_config','selection'}
    if kind not in ('selection','compatibility') or set(request) not in (fields,fields-{'kind'} if kind=='compatibility' else fields):
        raise RecoveryError('CANDIDATE_REVIEW_FIELDS')
    for name,pin in request.items():
        if name!='kind':checked(pin)
    snapshot=json.loads(checked(snapshot_pin).read_text())
    if snapshot.get('status')!='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY':raise RecoveryError('CANDIDATE_REVIEW_SNAPSHOT')
    config=json.loads(checked(snapshot['config']).read_text());verify_frozen(snapshot,config)
    exe=checked(dict(path=config['execution']['python_path'],sha256=config['execution']['python_sha256']))
    data=json.dumps(request,sort_keys=True)
    if len(data.encode())>65536:raise RecoveryError('CANDIDATE_REVIEW_INPUT_CAP')
    result=subprocess.run([str(exe),'-I','-B','-c',PROGRAM,snapshot['snapshot']],input=data,
        cwd=snapshot['snapshot'],env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},
        capture_output=True,text=True,timeout=60)
    if result.returncode:raise RecoveryError('CANDIDATE_REVIEW_FROZEN: '+result.stderr[-2000:])
    if len(result.stdout.encode())>65536:raise RecoveryError('CANDIDATE_REVIEW_OUTPUT_CAP')
    value=json.loads(result.stdout)
    if (set(value)!={'source_identity','payload'} or value['source_identity']!=snapshot['source_identity']
            or not isinstance(value['payload'],dict) or 'independent_review' in value['payload']):
        raise RecoveryError('CANDIDATE_REVIEW_IDENTITY')
    return dict(status='REVIEW_PAYLOAD_PREPARED_NOT_ACCEPTED',snapshot=snapshot_pin,request=request,
        **value,dispatch_authorized=False)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',type=Path);parser.add_argument('--inputs-sha256',required=True)
    args=parser.parse_args();q=json.loads(checked(dict(path=str(args.inputs.absolute()),sha256=args.inputs_sha256)).read_text())
    if set(q)!={'snapshot','request'}:raise RecoveryError('CANDIDATE_REVIEW_CLI_FIELDS')
    print(json.dumps(prepare(q['snapshot'],q['request']),sort_keys=True))


if __name__=='__main__':main()
