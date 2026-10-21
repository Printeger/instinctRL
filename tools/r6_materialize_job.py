"""Resolve prepaid volumes into immutable R6 evaluation/training inputs and an exact spec.

Pure preparation: no provisioning, accepted-condition import, approval, binding
or submission. The existing frozen validator and real Store spec validator own
the execution contract. New stages still require their exact frozen validators.
"""
import argparse
import json
from pathlib import Path
import shutil

from rlflow.store import canonical, sha
from tools.r6_recovery import Journal, RecoveryError, checked, file_pin
from tools.r6_decision_packet import persist_exact
from tools.r6_job_handoff import no_model_workers
from tools.r6_postrun import stop_paths
from tools import r6_storage_allocation as storage
from tools import r6_check_prepared_storage as volumes
from tools import r6_prepare_authorization as authority
from tools import r6_build_spec as builder
from tools import r6_job_resources as job_resources

ROLE='job-input-materialization'
CAP=256*1024
FIELDS={'snapshot_receipt','profile_draft','storage_key','stage_id','conditions','runner'}
MANAGED={'experiment_card','native_volume','result_volume','output_cap_bytes'}


def require(value,reason):
    if not value:raise RecoveryError('MATERIALIZE_'+reason)


def implementation():
    return [file_pin(m.__file__) for m in (storage,volumes,authority,builder,job_resources)]+[file_pin(__file__)]


def guards(journal):
    journal.require_writer();journal.guard();require(not stop_paths(journal),'STOP')
    no_model_workers(journal)
    require(journal.read().get('current_job') is None,'ACTIVE_JOB')
    journal.require_dispatch_budget()


def materialize(journal,input_pin,store,*,free_disk_bytes):
    guards(journal)
    inputs=json.loads(checked(input_pin).read_text())
    require(isinstance(inputs,dict) and set(inputs)==FIELDS,'INPUT_FIELDS')
    profile=json.loads(checked(inputs['profile_draft']).read_text())
    require(isinstance(profile,dict) and not MANAGED.intersection(profile),'MANAGED_FIELDS')
    require(profile.get('group')=='R1','GROUP')
    runner=inputs['runner']
    require(isinstance(runner,dict) and set(runner)=={'query_executable','decision_executable',
        'queue_review','sampler_review','purpose','sandbox'}
        and runner['purpose'] in {'development','repair','confirmation_delivery'}
        and runner['sandbox'] in {'read-only','danger-full-access'},'RUNNER_FIELDS')
    for field in ('query_executable','decision_executable','queue_review','sampler_review'):checked(runner[field])
    allocation,native,result,cap=storage.validate(journal,inputs['storage_key'])
    require(profile.get('experiment_id')==allocation['request']['experiment_id']
        and allocation['purpose']==runner['purpose'],'STORAGE_BINDING')
    attempt=Path(result['target']).parent
    require(Path(result['target']).name=='results','RESULT_TARGET')
    snapshot=json.loads(checked(inputs['snapshot_receipt']).read_text())
    require(snapshot.get('status')=='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY','SNAPSHOT')
    request=dict(inputs=input_pin,implementation=implementation(),store_root=str(store.root.resolve()))
    state=journal.read();event=dict(goal_id=state['goal_id'],
        event_id='MATERIALIZE:'+profile['experiment_id']+':'+sha(request),
        job_id='MATERIALIZE:'+profile['experiment_id'],input_sha256=sha(request),role=ROLE)
    key=sha(event);directory=journal.directory/'materialized'/key
    require(not directory.is_symlink() and directory.resolve()==directory.absolute(),'OUTPUT_PATH')
    if key not in state['events']:require(not directory.exists(),'UNCLAIMED_OUTPUT')
    key,_=journal.intent(event,kind='cpu',request=request,
        resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
        purpose=runner['purpose'],free_disk_bytes=free_disk_bytes)
    rows=[r for r in journal.read()['budget']['reservations'] if r['identity']==event]
    require(len(rows)==1 and rows[0]['resources']=={'output_actual_plus_unreleased_reservations_bytes':CAP}
        and rows[0]['purpose']==runner['purpose']
        and rows[0]['event_key']==json.dumps(tuple(event[k] for k in sorted(event)),separators=(',',':')),
        'RESERVATION')
    observed=volumes.inspect(journal,inputs['storage_key'])
    if observed['status']=='WAITING_FOR_ADMIN_PROVISION':
        state=journal.read();state['events'][key]['status']='WAITING_FOR_ADMIN_PROVISION';journal.save(state)
        return dict(status='WAITING_FOR_ADMIN_PROVISION',event_key=key,missing=observed['missing'],
                    dispatch_authorized=False)
    require(observed['status']=='VOLUMES_VERIFIED_NOT_ACCEPTED','VOLUME_STATUS')
    entry=journal.read()['events'][key]
    if entry.get('materialization'):
        artifact=json.loads(checked(entry['materialization']).read_text())
        require(artifact['inputs']==input_pin and artifact['event_key']==key,'RECEIPT_BINDING')
        for pin in artifact['artifacts'].values():checked(pin)
        authority.verify_frozen(snapshot,json.loads(checked(snapshot['config']).read_text()))
        storage.for_spec(journal,artifact['artifacts']['spec'],purpose=runner['purpose'],preview=True)
        return artifact
    before=sha(store.replay())
    params=dict(profile,native_volume=observed['manifests']['native_volume'],
        result_volume=observed['manifests']['result_volume'],output_cap_bytes=cap)
    def save(name,value):
        guards(journal)
        data=canonical(value)+b'\n'
        prior=sum(p.stat().st_size for p in directory.iterdir() if p.is_file() and p.name!=name) if directory.exists() else 0
        require(prior+len(data)<=CAP,'OUTPUT_CAP')
        path=directory/name;persist_exact(path,data);return file_pin(path)
    card=save('experiment-card.json',dict(experiment_id=params['experiment_id'],profiles={'R1':params}))
    params['experiment_card']=card
    parameters=save('parameters.json',params)
    proposal=authority.prepare(inputs['snapshot_receipt'],parameters,str(attempt))
    require(proposal['status']=='PROPOSED_NOT_ISSUED' and not proposal['store_permission_issued']
        and not proposal['job_submitted'],'PROPOSAL_STATUS')
    proposal_pin=save('authorization-proposal.json',proposal)
    auth=save('authorization.json',proposal['authorization_proposal'])
    config=file_pin(Path(snapshot['snapshot'])/'training_workloads/unified_h1/candidate_config.json')
    spec=builder.build(config,auth,str(attempt),inputs['stage_id'],inputs['conditions'],store.root)
    spec_pin=save('spec.json',spec)
    storage.for_spec(journal,spec_pin,purpose=runner['purpose'],preview=True)
    plan_pin=save('plan.json',dict(runner,spec_pin=spec_pin))
    require(sha(store.replay())==before,'STORE_MUTATED')
    artifact=dict(status='EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED',event_key=key,inputs=input_pin,
        storage_key=inputs['storage_key'],job_id='J-'+sha(dict(id=spec['id'],spec=spec))[:16],
        artifacts=dict(card=card,parameters=parameters,authorization_proposal=proposal_pin,
                       authorization=auth,spec=spec_pin,plan=plan_pin),
        scientific_acceptance=False,dispatch_authorized=False)
    receipt=save('materialization.json',artifact)
    state=journal.read();state['events'][key].update(materialization=receipt,status='INPUTS_MATERIALIZED')
    journal.save(state)
    return artifact


def main():
    from rlflow.store import Store
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('inputs',type=Path)
    parser.add_argument('--inputs-sha256',required=True);parser.add_argument('--store',required=True,type=Path)
    args=parser.parse_args();root=args.store.resolve(strict=True)
    require((root/'runtime/research.db').is_file(),'STORE_REQUIRED')
    journal=Journal(args.directory.resolve())
    with journal.writer():
        result=materialize(journal,dict(path=str(args.inputs.resolve()),sha256=args.inputs_sha256),
            Store(root),free_disk_bytes=shutil.disk_usage(journal.directory).free)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
