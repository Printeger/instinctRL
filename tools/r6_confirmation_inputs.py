"""One bounded read-only decision creates confirmation data, never new source."""
import json
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact,pin_bytes
from tools.r6_decision import CLI,CLI_SHA,MODEL_CAP,model_stop_paths
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_budget import totals
from tools import r6_profile_preparation as profile
from tools import r6_confirmation_preparation as preparation

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/confirmation-inputs.schema.json'
ROLE='confirmation-inputs'
CAP=256*1024


def require(value,reason):
    if not value:raise RecoveryError('CONFIRMATION_INPUTS_'+reason)


def prepare(journal,parent_key,context_pin,store,*,executable,free_disk_bytes):
    journal.require_writer();state=journal.read()
    name='CONFIRMATION_INPUTS:'+parent_key
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if prior:
        key,entry=prior[0];packet=entry['request']['packet']
        require(packet['context']==context_pin and packet['executable']==executable,'INPUT_DRIFT')
    else:
        profile.guard(journal);parent=profile.parent_decision(journal,parent_key,store)
        require(parent['decision']['decision']=='PROPOSE_CONFIRMATION','PARENT')
        checked(context_pin);exe=checked(executable);state=journal.read()
        if state['goal_id']=='R6-CPU-TEST-FIXTURE':require(exe.is_relative_to(journal.directory.parent.resolve()),'FIXTURE_EXE')
        else:require(executable==dict(path=str(CLI),sha256=CLI_SHA),'CLI')
        packet=dict(parent=state['events'][parent_key]['decision_record'],context=context_pin,executable=executable,
            journal_directory=str(journal.directory),store_sha256=sha(store.replay()),
            implementation=file_pin(__file__),preparation=file_pin(preparation.__file__),
            budget=totals(state['budget']),caps=state['budget']['caps'],
            protected=state['budget']['protected_confirmation_delivery_repair'])
        event=dict(goal_id=state['goal_id'],event_id=name,job_id=state['events'][parent_key]['identity']['job_id'],
            input_sha256=sha(dict(packet=packet)),role=ROLE)
        key,_=journal.intent(event,kind='cpu',request=dict(packet=packet),
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    for field in ('parent','context','executable','implementation','preparation'):checked(packet[field])
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    folder=journal.directory/'confirmation-inputs'/key
    def save(name,value):
        data=canonical(value)+b'\n';path=folder/name
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)<=CAP,'OUTPUT_CAP')
        if not path.exists():profile.guard(journal)
        persist_exact(path,data);return file_pin(path)
    packet_pin=pin_bytes(folder/'packet.json',canonical(packet)+b'\n')
    prompt=('Prepare confirmation INPUT DATA only, read-only, then exit. Never change files, source, '
        'Journal, Store, dependencies, runtime, scoring or seeds; never launch jobs or spawn agents. '
        'Read the parent decision and original paired development evidence. Read the Journal to locate '
        'the existing frozen TRAINING snapshot for the chosen recipe; do not create a new snapshot '
        'or substitute the current candidate-evaluation snapshot. Return inputs_json encoding exactly '
        'snapshot, recipe, recipe_review, selection_evidence, origin_config, experiment_ids, wall_seconds, storage. '
        'All pin fields refer to existing exact files. selection_evidence is an embedded r6-recipe-selection-evidence/1 '
        'OBJECT, with protocol, split_manifest, protocol_review, paired_evaluations (baseline_control and '
        'trained_candidate bound analysis pins), checkpoint_selection (terminal_checkpoint, common_start_update=640, '
        'target_update=640+recipe.branch_updates, select_using_holdout=false), and schema_version. '
        'Use three distinct unused experiment IDs in the preregistered confirmation seed order. '
        'wall_seconds and storage must cover each COMPLETE frozen recipe from common u640, all checkpoint nodes '
        'and raw outputs; preserve capacity for final evaluation, delivery and repairs. No acceptance or '
        'learning gain follows from preparing data. Exact source/evidence/budget and independent reviews '
        'are checked subsequently. If necessary evidence is missing return MUST_FIX with nonempty findings '
        'and inputs_json empty. Otherwise INPUTS_READY with no findings. Cite packet, parent and context '
        'pins. changed_files=[]; remaining sessions=caps minus packet budget minus one.\n'
        +json.dumps(dict(packet,packet_pin=packet_pin),sort_keys=True)+'\n').encode()
    require(len(prompt)<=65536,'PROMPT_CAP')
    request=dict(prompt=pin_bytes(folder/'prompt.txt',prompt),schema=file_pin(SCHEMA),executable=executable,
        cwd=str(ROOT),sandbox='read-only',timeout_seconds=3600,output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    event=dict(goal_id=state['goal_id'],event_id='CONFIRMATION_INPUT_MODEL:'+key,
        job_id=state['events'][parent_key]['identity']['job_id'],input_sha256=sha(request),role='confirmation-input-model')
    models=[(k,e) for k,e in journal.read()['events'].items() if e['identity'].get('event_id')==event['event_id']]
    require(len(models)<=1,'MODEL_ALIAS')
    if models:
        model_key,entry=models[0];require(entry['identity']==event and entry['request']==request,'MODEL_DRIFT')
    else:
        profile.guard(journal)
        model_key,_=journal.intent(event,kind='model',request=request,
            resources={'automatic_model_sessions':1,'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},
            purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),model_key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},'confirmation_delivery')
    entry=journal.read()['events'][model_key]
    if not entry.get('dispatch_claim') and not entry.get('receipt'):
        expected=dict(packet['budget']);expected['automatic_model_sessions']+=1
        expected['output_actual_plus_unreleased_reservations_bytes']+=CAP+MODEL_CAP+65536
        require(totals(journal.read()['budget'])==expected,'STALE_BUDGET')
        profile.guard(journal);save('packet.json',packet);persist_exact(folder/'prompt.txt',prompt)
    model_handoff(journal,model_key);profile.guard(journal)
    parent=profile.parent_decision(journal,parent_key,store)
    require(parent['decision']['decision']=='PROPOSE_CONFIRMATION'
        and journal.read()['events'][parent_key]['decision_record']==packet['parent']
        and sha(store.replay())==packet['store_sha256'],'PARENT_OR_STORE_DRIFT')
    result=verified_decision(journal,model_key)
    require(result['decision']=='INPUTS_READY' and not result['must_fix'] and result['changed_files']==[],'NOT_READY')
    require(result['packet_sha256']==packet_pin['sha256'] and all(p in result['evidence_pins']
        for p in (packet_pin,packet['parent'],context_pin)),'EVIDENCE')
    for pin in result['evidence_pins']:checked(pin)
    require(result['remaining_budget']['automatic_sessions_remaining']==
        packet['caps']['automatic_model_sessions']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    data=json.loads(result['inputs_json']);require(set(data)==preparation.FIELDS-{'context'},'FIELDS')
    require(isinstance(data['selection_evidence'],dict),'SELECTION_OBJECT')
    data['selection_evidence']=save('selection-evidence.json',data['selection_evidence'])
    context=json.loads(checked(context_pin).read_text());require(set(context)==profile.CONTEXT,'CONTEXT')
    context['runner']=dict(context['runner'],purpose='confirmation_delivery')
    data['context']=save('context.json',context)
    for field in ('snapshot','recipe','recipe_review','origin_config'):checked(data[field])
    return dict(status='CONFIRMATION_INPUTS_PREPARED_NOT_REVIEWED',inputs=save('inputs.json',data),
        event_key=model_key,dispatch_authorized=False)
