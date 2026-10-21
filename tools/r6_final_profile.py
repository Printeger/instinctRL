"""Validate a prospective final profile in its exact frozen source, CPU only."""
import json
import subprocess
from tools.r6_recovery import checked, RecoveryError
from tools.r6_prepare_authorization import verify_frozen

PROGRAM='''import json,sys
sys.path.insert(0,sys.argv[1])
from training_workloads.unified_h1.config import load_config,R6_FINAL_CONFIG_SCHEMA
from training_workloads.unified_h1.robustness_final_contract import MODE,validate_draft,validate_compatibility
c=load_config();q=json.load(sys.stdin);p=validate_draft(MODE,q['profile'],q['payload_bytes'])
if c['schema_version']!=R6_FINAL_CONFIG_SCHEMA:raise PermissionError('final config schema required')
validate_compatibility(p['checkpoint_compatibility'],p['final_selection'],c)
print(json.dumps(dict(source_identity=c['resolved_identity'],payload_bytes=q['payload_bytes']),sort_keys=True))
'''


def validate(snapshot_pin,profile,payload_bytes):
    snapshot=json.loads(checked(snapshot_pin).read_text())
    if snapshot.get('status')!='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY':
        raise RecoveryError('FINAL_PROFILE_SNAPSHOT')
    config=json.loads(checked(snapshot['config']).read_text());verify_frozen(snapshot,config)
    executable=checked(dict(path=config['execution']['python_path'],sha256=config['execution']['python_sha256']))
    data=json.dumps(dict(profile=profile,payload_bytes=payload_bytes),sort_keys=True)
    if len(data.encode())>65536:raise RecoveryError('FINAL_PROFILE_INPUT_CAP')
    result=subprocess.run([str(executable),'-I','-B','-c',PROGRAM,snapshot['snapshot']],
        input=data,cwd=snapshot['snapshot'],env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'',
        'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=60)
    if result.returncode:raise RecoveryError('FINAL_PROFILE_FROZEN_VALIDATION: '+result.stderr[-2000:])
    if len(result.stdout.encode())>65536:raise RecoveryError('FINAL_PROFILE_OUTPUT_CAP')
    value=json.loads(result.stdout)
    if (set(value)!={'source_identity','payload_bytes'} or value['source_identity']!=snapshot['source_identity']
            or type(value['payload_bytes']) is not int or value['payload_bytes']<=0 or value['payload_bytes']!=payload_bytes):
        raise RecoveryError('FINAL_PROFILE_IDENTITY')
    return value['payload_bytes']
