"""Derive final model identities from the controller's three closed confirmations.

No model selection discretion, holdout entropy, scientific acceptance or launch.
The terminal checkpoint rule was fixed before these trajectories were run.
"""
import json
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile
from tools import r6_confirmation_preparation as confirmation
from tools.r6_prepare_authorization import verify_frozen
from training_workloads.unified_h1 import robustness_contract as base
from training_workloads.unified_h1 import robustness_training_plan as training

ROLE='final-model-preparation'
CAP=256*1024


def require(value,reason):
    if not value:raise RecoveryError('FINAL_MODELS_'+reason)


def describe(journal,controller_key,store):
    journal.require_writer();state=journal.read();controller=state['events'][controller_key]
    require(controller['identity']['role']=='r6-controller'
        and controller['request']['store_root']==str(store.root.resolve()),'CONTROLLER_STORE')
    profile.reservation(state,controller_key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':256*1024},'development')
    sequence=controller['cursor']['confirmation'];prepared=sequence['preparation']
    require(prepared['status']=='THREE_CONFIRMATION_CONTEXTS_READY_NOT_RESERVED'
        and prepared['contexts']==sequence['contexts'] and len(sequence['results'])==3,'THREE_COMPLETED_STREAMS')
    preparation_pin=file_pin(journal.directory/'confirmation-preparation'/prepared['event_key']/'result.json')
    require(json.loads(checked(preparation_pin).read_text())==prepared,'PREPARATION_RECORD')
    profile.reservation(state,prepared['event_key'],'cpu',
        {'output_actual_plus_unreleased_reservations_bytes':confirmation.CAP},'confirmation_delivery')
    entry=state['events'][prepared['event_key']]
    require(entry['identity']['role']==confirmation.ROLE,'PREPARATION_ROLE')
    q=json.loads(checked(entry['request']['inputs']).read_text())
    snapshot=json.loads(checked(prepared['snapshot']).read_text())
    require(prepared['snapshot']==q['snapshot'],'SNAPSHOT_BINDING')
    verify_frozen(snapshot,json.loads(checked(snapshot['config']).read_text()))
    frozen=json.loads(checked(prepared['freeze']).read_text())
    recipe=training.recipe(frozen['recipe'],q['recipe_review'])
    require(recipe['source_identity']==snapshot['source_identity'] and q['recipe']==frozen['recipe'],'RECIPE_SOURCE')
    selection=json.loads(checked(frozen['selection_evidence']).read_text())
    rule=selection['checkpoint_selection']
    require(rule==dict(kind='terminal_checkpoint',common_start_update=640,
        target_update=640+recipe['branch_updates'],select_using_holdout=False),'TERMINAL_RULE')
    bundle=json.loads(checked(prepared['plans']).read_text())['bundle']
    require(bundle['freeze']==prepared['freeze'] and len(bundle['plans'])==3,'PLAN_BUNDLE')
    baseline=base.baseline({'baseline_bindings':bundle['baseline_bindings']})
    models={'baseline':baseline['selected_model']};streams=[];jobs=set();experiments=set()
    for index,(context_pin,record) in enumerate(zip(sequence['contexts'],sequence['results'])):
        context=json.loads(checked(context_pin).read_text())
        draft=json.loads(checked(context['baseline_profile']).read_text());plan=training.validate(draft['r6_training_plan'])
        require(plan==bundle['plans'][index] and context['snapshot_receipt']==prepared['snapshot']
            and draft['baseline_bindings']==bundle['baseline_bindings'],'EXACT_STREAM_PLAN')
        require(plan['phase']=='confirmation' and plan['recipe']==frozen['recipe']
            and plan['selection_freeze']==prepared['freeze'] and plan['training_seed']==recipe['confirmation_seeds'][index]
            and plan['start_update']==640 and plan['target_update']==rule['target_update']
            and plan['initialization']['kind']=='COMMON_U640_NEW_STREAM'
            and plan['initialization']['checkpoint']==models['baseline']
            and plan['initialization']['source_identity']==baseline['source_identity']
            and plan['initialization']['method_hashes']==baseline['method_hashes'],'COMMON_ORIGIN_AND_SEED')
        require(record['job_event_key'] not in jobs and plan['experiment_id'] not in experiments,'DISTINCT_JOBS')
        jobs.add(record['job_event_key']);experiments.add(plan['experiment_id'])
        require(record==confirmation.collect(journal,record['job_event_key'],context_pin,store),'COLLECTION_DRIFT')
        bound=json.loads(checked(record['analysis']).read_text());report=json.loads(checked(bound['report']).read_text())
        rows=[r for r in report['summary']['checkpoints'] if r['update']==rule['target_update']]
        require(len(rows)==1,'EXACT_TERMINAL_NODE');checkpoint=rows[0]['pin'];path=checked(checkpoint)
        auth=json.loads(checked(bound['authorization']).read_text())
        require(auth['mode']=='research-r6-training' and auth['parameters']['r6_training_plan']==plan
            and path==Path(auth['attempt'])/'results/research-r6-training/checkpoints'/f"u{plan['target_update']}.pt",'CHECKPOINT_ATTEMPT')
        inventory=json.loads(checked(bound['inventory']).read_text())
        matches=[r for r in inventory['files'] if {k:r[k] for k in ('path','sha256')}==checkpoint]
        require(len(matches)==1 and matches[0]['bytes']==path.stat().st_size,'CHECKPOINT_INVENTORY')
        require(checkpoint['sha256']!=models['baseline']['sha256'],'UNTRAINED_CHECKPOINT')
        models['seed'+str(index+1)]=checkpoint
        streams.append(dict(record,checkpoint=checkpoint,native_status=bound['native_status'],native_exit_code=bound['native_exit_code']))
    return dict(schema_version='r6-final-freeze/2',branch='THREE_CONTINUATIONS',protocol=selection['protocol'],
        recipe=frozen['recipe'],selection_rule=rule,models=models,
        evidence=dict(controller_key=controller_key,store_root=str(store.root.resolve()),preparation=preparation_pin,
            recipe_freeze=prepared['freeze'],baseline_bindings=bundle['baseline_bindings'],streams=streams))


def validate(journal,freeze,store=None):
    require(freeze.get('schema_version')=='r6-final-freeze/2' and freeze.get('branch')=='THREE_CONTINUATIONS',
        'SUPPORTED_EVIDENCE_SCHEMA')
    evidence=freeze['evidence']
    if store is None:
        from rlflow.store import Store
        root=Path(evidence['store_root']).resolve(strict=True)
        require((root/'runtime/research.db').is_file(),'STORE_DATABASE')
        store=Store(root)
    require(freeze==describe(journal,evidence['controller_key'],store),'FREEZE_EVIDENCE_DRIFT')
    return freeze


def prepare(journal,controller_key,store,*,free_disk_bytes):
    profile.guard(journal);payload=describe(journal,controller_key,store)
    request=dict(controller_key=controller_key,payload=payload,implementation=file_pin(__file__),
        collector=file_pin(confirmation.__file__))
    state=journal.read();event=dict(goal_id=state['goal_id'],event_id='FINAL_MODELS:'+controller_key,
        job_id='FINAL-BATCH',input_sha256=sha(request),role=ROLE)
    prior=[e for e in state['events'].values() if e['identity'].get('event_id')==event['event_id']]
    require(all(e['identity']==event and e['request']==request for e in prior),'INPUT_DRIFT')
    key,_=journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
        purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    folder=journal.directory/'final-models'/key;data=canonical(payload)+b'\n'
    require(len(data)<CAP//2,'OUTPUT_CAP');profile.guard(journal)
    persist_exact(folder/'freeze.json',data)
    result=dict(status='FINAL_MODELS_BOUND_REVIEW_REQUIRED',freeze=file_pin(folder/'freeze.json'),
        event_key=key,scientific_acceptance=False,holdout_released=False,dispatch_authorized=False)
    persist_exact(folder/'result.json',canonical(result)+b'\n')
    return result
