"""Build the complete frozen final-source compatibility batch, CPU only."""
import json,subprocess
from tools.r6_recovery import checked,RecoveryError
from tools.r6_prepare_authorization import verify_frozen

PROGRAM='''import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from training_workloads.unified_h1.config import load_config,R6_FINAL_CONFIG_SCHEMA
from training_workloads.unified_h1.robustness_contract import pinned_json
from training_workloads.unified_h1.robustness_final_contract import compatibility_payload
c=load_config();q=json.load(sys.stdin)
if c['schema_version']!=R6_FINAL_CONFIG_SCHEMA:raise PermissionError('final config schema required')
p=Path(sys.argv[1])/'training_workloads/unified_h1/candidate_config.json'
cp=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
batch=pinned_json(q['batch']);origins=q['origin_configs']
if not isinstance(origins,list) or len(origins)!=len(batch['assignments']) or len(origins) not in (3,4):
 raise PermissionError('complete final origin configs required')
payloads={}
for i,origin in enumerate(origins):
 selection=dict(schema_version='r6-final-selection/1',batch=q['batch'],assignment_index=i)
 payloads[str(i)]=compatibility_payload(origin,cp,selection,c)
print(json.dumps(dict(source_identity=c['resolved_identity'],payloads=payloads),sort_keys=True))
'''


def prepare(snapshot_pin,request):
    if (not isinstance(request,dict) or set(request)!={'batch','origin_configs'}
            or not isinstance(request['origin_configs'],list) or len(request['origin_configs']) not in (3,4)):
        raise RecoveryError('FINAL_REVIEW_FIELDS')
    checked(request['batch'])
    for p in request['origin_configs']:checked(p)
    snapshot=json.loads(checked(snapshot_pin).read_text())
    if snapshot.get('status')!='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY':raise RecoveryError('FINAL_REVIEW_SNAPSHOT')
    config=json.loads(checked(snapshot['config']).read_text());verify_frozen(snapshot,config)
    exe=checked(dict(path=config['execution']['python_path'],sha256=config['execution']['python_sha256']))
    data=json.dumps(request,sort_keys=True)
    if len(data.encode())>65536:raise RecoveryError('FINAL_REVIEW_INPUT_CAP')
    result=subprocess.run([str(exe),'-I','-B','-c',PROGRAM,snapshot['snapshot']],input=data,cwd=snapshot['snapshot'],
        env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=180)
    if result.returncode:raise RecoveryError('FINAL_REVIEW_FROZEN: '+result.stderr[-2000:])
    if len(result.stdout.encode())>262144:raise RecoveryError('FINAL_REVIEW_OUTPUT_CAP')
    value=json.loads(result.stdout)
    if (set(value)!={'source_identity','payloads'} or value['source_identity']!=snapshot['source_identity']
            or not isinstance(value['payloads'],dict) or set(value['payloads'])!={str(i) for i in range(len(request['origin_configs']))}
            or any(not isinstance(p,dict) or 'independent_review' in p for p in value['payloads'].values())):
        raise RecoveryError('FINAL_REVIEW_IDENTITY')
    return dict(status='REVIEW_PAYLOAD_PREPARED_NOT_ACCEPTED',snapshot=snapshot_pin,request=request,**value,dispatch_authorized=False)
