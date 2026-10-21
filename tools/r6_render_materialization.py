"""Materialize reviewed replay parameters and reuse the exact Store issuer."""
from pathlib import Path
from rlflow.store import canonical, sha
from tools.r6_recovery import checked, file_pin
from tools.r6_decision_packet import persist_exact
from tools import r6_render_preparation as preparation, r6_render_review as review
from tools import r6_render_contract as contract, r6_build_spec as builder
from tools import r6_profile_preparation as profile, r6_storage_allocation as storage
from tools import r6_issue_next as issuer

ROLE='render-input-materialization'
CAP=256*1024
read=preparation.read
require=preparation.require


def materialize(journal,preparation_pin,review_key,review_pin,store,*,free_disk_bytes):
    profile.guard(journal)
    prepared=preparation.describe_prepared(journal,preparation_pin,store,free_disk_bytes=free_disk_bytes)
    review.verify(journal,review_key,review_pin,preparation_pin,store)
    request=dict(preparation=preparation_pin,review_key=review_key,review=review_pin,store_root=str(store.root.resolve()),
        implementation=[file_pin(m.__file__) for m in (preparation,review,contract,builder,storage)]+[file_pin(__file__)])
    name='RENDER_MATERIALIZE:'+prepared['event_key'];state=journal.read()
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'MATERIALIZE_ALIAS')
    event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-RENDER',input_sha256=sha(request),role=ROLE)
    if prior:require(prior[0][1]['identity']==event,'MATERIALIZE_DRIFT')
    key,_=journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
        purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    directory=journal.directory/'render-materialized'/key
    def save(name,value):
        profile.guard(journal);data=canonical(value)+b'\n'
        size=sum(p.stat().st_size for p in directory.iterdir() if p.name!=name) if directory.exists() else 0
        require(size+len(data)<=CAP,'MATERIALIZE_CAP');path=directory/name;persist_exact(path,data);return file_pin(path)
    p=dict(read(prepared['parameters']),independent_review=review_pin)
    auth=dict(schema_version='r6-video-replay-authorization/1',mode=contract.MODE,attempt=prepared['attempt'],
        parameters=p,budgets=contract.budget(p['wall_seconds']),config=prepared['config'])
    before=sha(store.replay());authorization=save('authorization.json',auth)
    spec=builder.build(prepared['config'],authorization,prepared['attempt'],prepared['stage_id'],prepared['conditions'],store.root)
    spec_pin=save('spec.json',spec)
    storage.for_spec(journal,spec_pin,purpose='confirmation_delivery',preview=True)
    plan=save('plan.json',dict(prepared['runner'],spec_pin=spec_pin))
    require(sha(store.replay())==before,'MATERIALIZE_STORE_MUTATION')
    output=dict(status='RENDER_EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED',event_key=key,
        preparation=preparation_pin,review_key=review_key,review=review_pin,storage_key=prepared['storage_key'],
        job_id='J-'+sha(dict(id=spec['id'],spec=spec))[:16],authorization=authorization,spec=spec_pin,plan=plan,
        scientific_acceptance=False,dispatch_authorized=False)
    pin=save('result.json',output);state=journal.read();state['events'][key]['materialization']=pin;journal.save(state)
    return output


def issue(journal,materialization_pin,store,*,free_disk_bytes):
    profile.guard(journal);result=read(materialization_pin)
    preparation.final_plans.registered(journal,result['event_key'],ROLE,'materialization',result,CAP)
    entry=journal.read()['events'][result['event_key']]
    require(entry['request']['store_root']==str(store.root.resolve())
            and all(entry['request'][k]==result[k] for k in ('preparation','review_key','review')),'MATERIALIZE_RECEIPT')
    for pin in entry['request']['implementation']:checked(pin)
    require(materialize(journal,result['preparation'],result['review_key'],result['review'],store,
        free_disk_bytes=free_disk_bytes)==result,'MATERIALIZE_REPLAY')
    plan=read(result['plan']);spec=read(result['spec']);auth=read(result['authorization'])
    return issuer._issue_reviewed(journal,'RENDER:'+result['event_key'],result['plan'],result['review_key'],
        result['review'],auth['parameters']['master_authority'],store,plan,spec,result['job_id'],
        free_disk_bytes=free_disk_bytes,role='render-authorization')
