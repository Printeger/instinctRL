"""Validate a prospective training profile in its exact frozen source, CPU only."""
import json
import subprocess
from tools.r6_recovery import checked, RecoveryError
from tools.r6_prepare_authorization import verify_frozen

PROGRAM='''import json,sys
sys.path.insert(0,sys.argv[1])
from training_workloads.unified_h1.config import load_config,R6_TRAINING_CONFIG_SCHEMA
from training_workloads.unified_h1.robustness_training_contract import MODE,validate_draft,validate_compatibility
c=load_config();p=validate_draft(MODE,json.load(sys.stdin))
if c['schema_version']!=R6_TRAINING_CONFIG_SCHEMA:raise PermissionError('training config schema required')
validate_compatibility(p['checkpoint_compatibility'],p['r6_training_plan'],c)
print(json.dumps(dict(source_identity=c['resolved_identity'],payload_bytes=p['r6_training_plan']['storage']['payload_bytes']),sort_keys=True))
'''


def validate(snapshot_pin,profile):
    snapshot=json.loads(checked(snapshot_pin).read_text())
    if snapshot.get('status')!='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY':
        raise RecoveryError('TRAINING_PROFILE_SNAPSHOT')
    config=json.loads(checked(snapshot['config']).read_text());verify_frozen(snapshot,config)
    executable=checked(dict(path=config['execution']['python_path'],sha256=config['execution']['python_sha256']))
    data=json.dumps(profile,sort_keys=True)
    if len(data.encode())>65536:raise RecoveryError('TRAINING_PROFILE_INPUT_CAP')
    result=subprocess.run([str(executable),'-I','-B','-c',PROGRAM,snapshot['snapshot']],
        input=data,cwd=snapshot['snapshot'],env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'',
        'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=60)
    if result.returncode:raise RecoveryError('TRAINING_PROFILE_FROZEN_VALIDATION: '+result.stderr[-2000:])
    if len(result.stdout.encode())>65536:raise RecoveryError('TRAINING_PROFILE_OUTPUT_CAP')
    value=json.loads(result.stdout)
    if (set(value)!={'source_identity','payload_bytes'} or value['source_identity']!=snapshot['source_identity']
            or type(value['payload_bytes']) is not int or value['payload_bytes']<=0):
        raise RecoveryError('TRAINING_PROFILE_IDENTITY')
    return value['payload_bytes']
