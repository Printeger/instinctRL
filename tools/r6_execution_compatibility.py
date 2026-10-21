"""Review exact baseline-compatible source changes, then prepare a new context.

No Store approval, volume creation or submission. Training/method changes must
use their own phase contract, not masquerade as baseline compatibility.
"""
import json
from pathlib import Path
import subprocess

from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_budget import totals
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_decision import CLI,CLI_SHA,MODEL_CAP,model_stop_paths,record as parent_decision
from tools.r6_decision_packet import persist_exact
from tools.r6_prepare_authorization import verify_frozen
from tools import r6_profile_preparation as profile

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/source-compatibility.schema.json'
ROLE='source-compatibility-review'
CAP=256*1024
PROGRAM='''import json,sys
sys.path.insert(0,sys.argv[1])
from training_workloads.unified_h1.config import load_config
from training_workloads.unified_h1.robustness_contract import validate_compatibility
v=json.load(sys.stdin);c=load_config()
validate_compatibility(v['record'],v['origin'],c)
print(json.dumps(c['resolved_identity'],sort_keys=True))
'''


def require(value,reason):
    if not value:raise RecoveryError('SOURCE_COMPATIBILITY_'+reason)


def payload(snapshot_pin,reference_pin):
    snapshot=json.loads(checked(snapshot_pin).read_text())
    require(snapshot.get('status')=='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY','SNAPSHOT')
    config=json.loads(checked(snapshot['config']).read_text());verify_frozen(snapshot,config)
    reference=json.loads(checked(reference_pin).read_text())
    previous=json.loads(checked(reference['checkpoint_compatibility']).read_text())
    old=json.loads(checked(previous['origin_config']).read_text())
    for name in ('versions','actor','training','execution'):
        require(old[name]==config[name],'PHASE_CONTRACT_REQUIRED_'+name)
    for name in ('version','proposal_sha256','hashes'):
        require(old['method'][name]==config['method'][name],'PHASE_CONTRACT_REQUIRED_method')
    metric='training_workloads/unified_h1/evaluation.py'
    require(old['workload']['source_sha256'].get(metric) is not None
        and old['workload']['source_sha256'][metric]==config['workload']['source_sha256'].get(metric),'ORIGINAL_METRIC')
    for section in ('product','workload'):
        for relative,digest in old[section]['source_sha256'].items():
            require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'SOURCE_PATH')
            checked(dict(path=str(Path(old[section]['path'])/relative),sha256=digest))
    a=old['product']['source_sha256'];b=config['product']['source_sha256']
    result=dict(schema_version='r6-baseline-compatibility/1',
        origin_source_identity=previous['origin_source_identity'],execution_source_identity=snapshot['source_identity'],
        origin_config=previous['origin_config'],
        execution_config=file_pin(Path(snapshot['snapshot'])/'training_workloads/unified_h1/candidate_config.json'),
        changed_product_sources={k:dict(origin=a.get(k),execution=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)})
    require(result['execution_config']['sha256']==snapshot['source_identity']['config_sha256']
        and result['origin_config']['sha256']==result['origin_source_identity']['config_sha256'],'CONFIG_IDENTITY')
    return result


def frozen_validate(snapshot_pin,record_pin,origin):
    snapshot=json.loads(checked(snapshot_pin).read_text())
    config=json.loads(checked(snapshot['config']).read_text());verify_frozen(snapshot,config)
    exe=checked(dict(path=config['execution']['python_path'],sha256=config['execution']['python_sha256']))
    result=subprocess.run([str(exe),'-I','-B','-c',PROGRAM,snapshot['snapshot']],
        cwd=snapshot['snapshot'],input=json.dumps(dict(record=record_pin,origin=origin)),
        env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},
        capture_output=True,text=True,timeout=60)
    require(result.returncode==0,'FROZEN_VALIDATION_FAILED: '+result.stderr[-2000:])
    require(len(result.stdout)<65536 and json.loads(result.stdout)==snapshot['source_identity'],'FROZEN_IDENTITY')


def prepare(journal,parent_key,snapshot_pin,context_pin,store,*,executable,free_disk_bytes):
    journal.require_writer();state=journal.read()
    # Reconnect an existing model before applying new-action idle/STOP guards.
    name='SOURCE_COMPATIBILITY:'+parent_key+':'+snapshot_pin['sha256']+':'+context_pin['sha256']
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if not prior:
        profile.guard(journal);parent_decision(journal,parent_key,store);state=journal.read()
        context=json.loads(checked(context_pin).read_text());require(set(context)==profile.CONTEXT,'CONTEXT')
        data=payload(snapshot_pin,context['baseline_profile'])
        packet=dict(parent=state['events'][parent_key]['decision_record'],context=context_pin,
            snapshot=snapshot_pin,payload=data,implementation=file_pin(__file__),store_sha256=sha(store.replay()),
            budget=totals(state['budget']),model_cap=state['budget']['caps']['automatic_model_sessions'])
        request=dict(packet=packet)
        event=dict(goal_id=state['goal_id'],event_id=name,job_id=state['events'][parent_key]['identity']['job_id'],
            input_sha256=sha(request),role='source-compatibility-input')
        key,_=journal.intent(event,kind='cpu',request=request,
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose=context['runner']['purpose'],free_disk_bytes=free_disk_bytes)
    else:key,entry=prior[0];packet=entry['request']['packet']
    checked(packet['implementation']);checked(packet['context']);checked(packet['snapshot']);checked(packet['parent'])
    context=json.loads(checked(context_pin).read_text());purpose=context['runner']['purpose']
    require(packet['context']==context_pin and packet['snapshot']==snapshot_pin,'PACKET_DRIFT')
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},purpose)
    folder=journal.directory/'source-compatibility'/key
    def save(name,value):
        data=canonical(value)+b'\n'
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)<CAP,'OUTPUT_CAP');persist_exact(folder/name,data);return file_pin(folder/name)
    payload_pin=save('payload.json',packet['payload'])
    model_event='SOURCE_COMPATIBILITY_REVIEW:'+key
    models=[(k,e) for k,e in journal.read()['events'].items() if e['identity'].get('event_id')==model_event]
    require(len(models)<=1,'MODEL_ALIAS')
    prompt=('Independently review the exact R6 source/config compatibility payload, read only, then exit. '
        'Read old/new source bytes and pinned snapshot inventory. Check baseline u640 loading, actor '
        'information boundary, original evaluator, method/config invariants, actual changed product '
        'sources and newly manifested dependencies. This is baseline compatibility only: reject '
        'training or method changes requiring another execution contract. Do not edit files, mutate '
        'Store/Journal, query GPU, simulate, create volumes, launch workers or jobs. Do not inherit '
        'acceptance from the previous execution snapshot. PASS_R6_EXECUTION_COMPATIBILITY requires '
        'no must_fix, changed_files=[], reviewer=r6-independent-source-compatibility, exact canonical '
        'payload_sha256 and payload/parent/context/snapshot pins in evidence_pins. remaining sessions '
        'equals model_cap minus budget.automatic_model_sessions minus one.\n'+json.dumps(dict(packet,
            payload_pin=payload_pin,payload_sha256=sha(packet['payload'])),sort_keys=True)+'\n').encode()
    require(len(prompt)<=65536,'PROMPT_CAP')
    prompt_pin=dict(path=str((folder/'prompt.txt').resolve()),sha256=__import__('hashlib').sha256(prompt).hexdigest())
    request=dict(prompt=prompt_pin,schema=file_pin(SCHEMA),executable=executable,cwd=str(ROOT),sandbox='read-only',
        timeout_seconds=3600,output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    event=dict(goal_id=state['goal_id'],event_id=model_event,job_id=state['events'][parent_key]['identity']['job_id'],
        input_sha256=sha(request),role=ROLE)
    if models:
        model_key,entry=models[0];require(entry['identity']==event and entry['request']==request,'MODEL_DRIFT')
    else:
        profile.guard(journal);exe=checked(executable)
        if state['goal_id']=='R6-CPU-TEST-FIXTURE':require(exe.is_relative_to(journal.directory.parent.resolve()),'FIXTURE_EXE')
        else:require(executable==dict(path=str(CLI),sha256=CLI_SHA),'CLI')
        model_key,_=journal.intent(event,kind='model',request=request,
            resources={'automatic_model_sessions':1,'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},
            purpose=purpose,free_disk_bytes=free_disk_bytes)
    entry=journal.read()['events'][model_key]
    profile.reservation(journal.read(),model_key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},purpose)
    if not entry.get('dispatch_claim') and not entry.get('receipt'):
        expected=dict(packet['budget']);expected['automatic_model_sessions']+=1
        expected['output_actual_plus_unreleased_reservations_bytes']+=CAP+MODEL_CAP+65536
        require(totals(journal.read()['budget'])==expected,'STALE_BUDGET')
        profile.guard(journal);persist_exact(folder/'prompt.txt',prompt)
    model_handoff(journal,model_key)
    profile.guard(journal)
    parent_decision(journal,parent_key,store)
    require(journal.read()['events'][parent_key]['decision_record']==packet['parent']
        and sha(store.replay())==packet['store_sha256'],'PARENT_OR_STORE_DRIFT')
    result=verified_decision(journal,model_key)
    require(result['decision']=='PASS_R6_EXECUTION_COMPATIBILITY' and not result['must_fix']
        and result['changed_files']==[] and result['payload_sha256']==sha(packet['payload']),'REVIEW_REJECTED')
    require(result['remaining_budget']['automatic_sessions_remaining']==packet['model_cap']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    require(all(p in result['evidence_pins'] for p in (payload_pin,packet['parent'],context_pin,snapshot_pin)),'REVIEW_EVIDENCE')
    for pin in result['evidence_pins']:checked(pin)
    require(payload(snapshot_pin,context['baseline_profile'])==packet['payload'],'PAYLOAD_DRIFT')
    receipt=json.loads(checked(journal.read()['events'][model_key]['receipt']).read_text())
    require(receipt['event_key']==model_key and receipt['request_sha256']==sha(request),'RECEIPT_BINDING')
    accepted=save('accepted.json',dict(packet['payload'],independent_review=receipt['last_message']))
    frozen_validate(snapshot_pin,accepted,packet['payload']['origin_source_identity'])
    profile.guard(journal)
    reference=json.loads(checked(context['baseline_profile']).read_text());old=reference['checkpoint_compatibility']
    reference['checkpoint_compatibility']=accepted
    new_context=dict(context,snapshot_receipt=snapshot_pin,baseline_profile=save('profile-reference.json',reference),
        conditions=[c for c in context['conditions'] if c['sha256']!=old['sha256']])
    # New compatibility is an exact parameter pin checked by frozen validator
    # and next-job prelive; old snapshot's Store condition is not inherited.
    return dict(status='EXACT_COMPATIBILITY_ACCEPTED_CONTEXT_READY',context=save('context.json',new_context),
        accepted=accepted,review_event_key=model_key,dispatch_authorized=False)
