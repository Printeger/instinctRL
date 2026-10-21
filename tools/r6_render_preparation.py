"""Reuse final's exact frozen checkout and prepay a distinct render volume pair.

No new snapshot, model call, mount, approval or submission. Missing administrator
volumes remain a durable preparation boundary, without repeated reservations.
"""
import json
from pathlib import Path
import subprocess

from rlflow.store import canonical, sha
from tools.r6_recovery import checked, file_pin, RecoveryError
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile, prepare_r6_snapshot as snapshot
from tools import r6_render_contract as contract, r6_render_job as provenance
from tools import r6_storage_allocation as storage, r6_check_prepared_storage as volumes
from tools import r6_final_inputs as final_inputs, r6_final_plans as final_plans

ROOT=Path(__file__).resolve().parents[1]
ROLE='render-input-preparation'
CAP=512*1024
WALL_SECONDS=600


def require(value, reason):
    if not value:raise RecoveryError('RENDER_PREPARATION_'+reason)


def read(pin):return json.loads(checked(pin).read_text())


def describe(journal, controller_key, store):
    profile.guard(journal);state=journal.read();controller=state['events'][controller_key]
    require(controller['identity']['role']=='r6-controller'
            and controller['request']['store_root']==str(store.root.resolve()),'CONTROLLER')
    cursor=controller['cursor'];video=cursor['video_trajectories'];f=cursor['final_inputs']
    provenance.validate(journal,dict(extraction=video['result']),'confirmation_delivery')
    event=state['events'][f['event_key']]
    final_plans.registered(journal,f['event_key'],final_inputs.ROLE,'result',f,final_inputs.CAP)
    require(event['request']['controller_key']==controller_key,'FINAL_INPUT_OWNER')
    for pin in event['request']['implementation']:checked(pin)
    frozen=read(f['snapshot']);snapshot_event=state['events'][frozen['event_key']]
    require(snapshot_event['identity']['role']=='source-snapshot' and snapshot_event['receipt']==f['snapshot']
            and frozen['status']=='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY','SNAPSHOT_EVENT')
    profile.reservation(state,frozen['event_key'],'cpu',
        {'output_actual_plus_unreleased_reservations_bytes':snapshot.RESERVATION},'confirmation_delivery')
    snapshot.verify_frozen(frozen,read(frozen['config']))
    target=Path(frozen['snapshot']);config=file_pin(target/'training_workloads/unified_h1/candidate_config.json')
    require(config['sha256']==frozen['config']['sha256'],'CONFIG')
    implementation={}
    for name,pin in contract.sources().items():
        original=checked(pin)
        require(original.is_relative_to(ROOT),'CONTROL_SOURCE_ROOT')
        path=target/original.relative_to(ROOT)
        candidate=file_pin(path)
        require(candidate['sha256']==pin['sha256'] and not path.stat().st_mode & 0o222,'FROZEN_RENDER_SOURCE')
        implementation[name]=candidate
    request=read(controller['request']['inputs']);initial=read(request['initial_request'])
    plan=read(initial['plan_pin']);spec=read(plan['spec_pin']);auth=read(spec['pins']['authorization'])
    context=read(request['profile_context'])
    runner=dict(context['runner'],purpose='confirmation_delivery')
    for field in ('query_executable','decision_executable','queue_review','sampler_review'):checked(runner[field])
    return dict(controller_key=controller_key,extraction=video['result'],snapshot=f['snapshot'],config=config,
        source_snapshot=str(target),implementation=implementation,master_authority=initial['master_pin'],
        extension_inputs=auth['parameters']['extension_inputs'],stage_id=context['stage_id'],
        conditions=context['conditions'],runner=runner,store_root=str(store.root.resolve()))


PROGRAM='''import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(sys.argv[1]);sys.path[:0]=[str(root),str(root/'tools/rlflow')]
from training_workloads.unified_h1.config import load_config
from tools.r6_render_contract import parameters,budget
request=json.load(sys.stdin);config=load_config()
p=parameters(request['parameters'],attempt=Path(request['attempt']),require_review=False)
print(json.dumps({'parameters':p,'budgets':budget(p['wall_seconds']),'source_identity':config['resolved_identity']},sort_keys=True))
'''


def validate_frozen(description, parameters, attempt):
    frozen=read(description['snapshot']);config=read(description['config'])
    snapshot.verify_frozen(frozen,config)
    execution=config['execution'];python=checked(dict(path=execution['python_path'],sha256=execution['python_sha256']))
    request=dict(parameters=parameters,attempt=str(attempt))
    run=subprocess.run([str(python),'-I','-B','-c',PROGRAM,description['source_snapshot']],
        input=json.dumps(request),text=True,capture_output=True,timeout=120,cwd=description['source_snapshot'],
        env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
    require(run.returncode==0,'FROZEN_VALIDATION: '+run.stderr[-2000:])
    value=json.loads(run.stdout)
    require(value==dict(parameters=parameters,budgets=contract.budget(parameters['wall_seconds']),
                        source_identity=frozen['source_identity']),'FROZEN_RESULT')
    return value


def prepare(journal,controller_key,store,*,free_disk_bytes):
    description=describe(journal,controller_key,store)
    implementation=[file_pin(m.__file__) for m in (profile,snapshot,contract,provenance,storage,volumes)]+[file_pin(__file__)]
    request=dict(description=description,implementation=implementation)
    name='RENDER_INPUTS:'+controller_key
    state=journal.read();prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    identity=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-RENDER',input_sha256=sha(request),role=ROLE)
    if prior:
        key,entry=prior[0];require(entry['identity']==identity and entry['request']==request,'INPUT_DRIFT')
    else:
        key,_=journal.intent(identity,kind='cpu',request=request,
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    directory=journal.directory/'render-inputs'/key
    def save(name,value):
        profile.guard(journal);data=canonical(value)+b'\n'
        size=sum(p.stat().st_size for p in directory.iterdir() if p.name!=name) if directory.exists() else 0
        require(size+len(data)<=CAP,'OUTPUT_CAP');path=directory/name;persist_exact(path,data);return file_pin(path)
    experiment_id='R6-REPLAY-'+key[:20].upper();volume_id='J-'+key[:16]
    attempt=storage.result_tool.ATTEMPTS/experiment_id
    require(attempt.parent==store.root.resolve()/'attempts','STORE_ATTEMPT_ROOT')
    native=storage.native_tool.plan(volume_id)
    result=storage.result_tool.plan(volume_id,attempt/'results',contract.PAYLOAD+96*1024**2,contract.PAYLOAD,512)
    np=save('native-plan.json',native);rp=save('result-plan.json',result)
    allocation=storage.register(journal,experiment_id,np,rp,purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    observed=volumes.inspect(journal,allocation)
    if observed['status']=='WAITING_FOR_ADMIN_PROVISION':
        return dict(status='WAITING_FOR_ADMIN_PROVISION',event_key=key,storage_key=allocation,
                    native_plan=np,result_plan=rp,missing=observed['missing'],dispatch_authorized=False)
    require(observed['status']=='VOLUMES_VERIFIED_NOT_ACCEPTED','VOLUME_STATUS')
    p={name:description[name] for name in ('master_authority','extraction','source_snapshot','implementation','extension_inputs')}
    p.update(experiment_id=experiment_id,native_volume=observed['manifests']['native_volume'],
        result_volume=observed['manifests']['result_volume'],output_cap_bytes=observed['output_cap_bytes'],
        wall_seconds=WALL_SECONDS,independent_review=None)
    before=sha(store.replay());validation=validate_frozen(description,p,attempt)
    require(sha(store.replay())==before,'STORE_MUTATED')
    parameters_pin=save('parameters.json',p)
    prepared=dict(status='RENDER_INPUTS_READY_NOT_REVIEWED',event_key=key,controller_key=controller_key,
        snapshot=description['snapshot'],config=description['config'],parameters=parameters_pin,
        attempt=str(attempt),storage_key=allocation,validation=save('frozen-validation.json',validation),
        stage_id=description['stage_id'],conditions=description['conditions'],runner=description['runner'],
        scientific_acceptance=False,dispatch_authorized=False)
    pin=save('result.json',prepared);state=journal.read();state['events'][key]['preparation']=pin;journal.save(state)
    return prepared


def describe_prepared(journal, preparation_pin, store, *, free_disk_bytes):
    """Revalidate original preparation and dry-check the prepaid live allowance."""
    profile.guard(journal);prepared=read(preparation_pin);key=prepared['event_key']
    final_plans.registered(journal,key,ROLE,'preparation',prepared,CAP)
    require(prepared['status']=='RENDER_INPUTS_READY_NOT_REVIEWED'
            and prepared['scientific_acceptance'] is False and prepared['dispatch_authorized'] is False,'RESULT')
    entry=journal.read()['events'][key];description=describe(journal,prepared['controller_key'],store)
    require(description==entry['request']['description'],'DESCRIPTION_DRIFT')
    for pin in entry['request']['implementation']:checked(pin)
    p=read(prepared['parameters']);allocation,native,result,cap=storage.validate(journal,prepared['storage_key'])
    observed=volumes.inspect(journal,prepared['storage_key'])
    require(observed['status']=='VOLUMES_VERIFIED_NOT_ACCEPTED'
            and allocation['purpose']=='confirmation_delivery'
            and allocation['request']['experiment_id']==p['experiment_id']
            and p['experiment_id']=='R6-REPLAY-'+key[:20].upper()
            and native['volume_id']==result['volume_id']=='J-'+key[:16]
            and result['target']==str(Path(prepared['attempt'])/'results')
            and result['minimum_free_bytes']==contract.PAYLOAD
            and result['minimum_free_inodes']==512
            and p['output_cap_bytes']==cap==observed['output_cap_bytes']
            and all(p[k]==observed['manifests'][k] for k in ('native_volume','result_volume')),'STORAGE')
    require(prepared['snapshot']==description['snapshot'] and prepared['config']==description['config']
            and prepared['stage_id']==description['stage_id'] and prepared['conditions']==description['conditions']
            and prepared['runner']==description['runner'] and p['wall_seconds']==WALL_SECONDS
            and p['independent_review'] is None
            and all(p[k]==description[k] for k in ('master_authority','extraction','source_snapshot','implementation','extension_inputs')),'INPUT_BINDING')
    require(validate_frozen(description,p,Path(prepared['attempt']))==read(prepared['validation']),'VALIDATION_DRIFT')
    from tools.r6_budget import reserve
    auth=dict(mode=contract.MODE,parameters=p,budgets=contract.budget(p['wall_seconds']))
    spec=dict(backend='real',h1={'mode':contract.MODE},resources=dict(starts=1,
        wall_seconds=p['wall_seconds']+10,gpu_seconds=p['wall_seconds']+10,transitions=0))
    resources=contract.resources(spec,auth,prepaid=True)
    identity=dict(goal_id=journal.read()['goal_id'],event_id='CHECK_RENDER:'+key,
        job_id='FINAL-RENDER',input_sha256=preparation_pin['sha256'],role='budget-check-only')
    reserve(journal.read()['budget'],event_key=identity,resources=resources,
        purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    return prepared


def payload_sha(preparation_pin):
    p=read(read(preparation_pin)['parameters'])
    return sha({k:v for k,v in p.items() if k!='independent_review'})
