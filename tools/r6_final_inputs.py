"""Deterministic final config/input preparation from the frozen model batch.

Only configuration manifests are committed. No method edit or model decision.
The existing frozen compatibility review remains mandatory before issuance.
"""
import copy
import hashlib
import json
from pathlib import Path
from rlflow.store import canonical, sha
from tools.r6_recovery import checked, file_pin, RecoveryError
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile
from tools import r6_holdout_publication as publication
from tools import prepare_r6_snapshot as snapshot
from training_workloads.unified_h1.config import R6_FINAL_CONFIG_SCHEMA, required_source_sets

ROOT=Path(__file__).resolve().parents[1]
ROLE='final-input-preparation'
CAP=512*1024


def require(value,reason):
    if not value:raise RecoveryError('FINAL_INPUTS_'+reason)


def read(pin):return json.loads(checked(pin).read_text())


def git(*args):return snapshot.git(ROOT,*args).decode().strip()


def describe(journal,controller_key,batch_pin,store):
    profile.guard(journal);batch=publication.validate_batch(journal,batch_pin)
    freeze=read(batch['freeze']);state=journal.read();controller=state['events'][controller_key]
    require(controller['identity']['role']=='r6-controller' and freeze['evidence']['controller_key']==controller_key
        and controller['request']['store_root']==str(store.root.resolve()),'CONTROLLER')
    request=read(controller['request']['inputs']);context=read(request['profile_context'])
    require(set(context)==profile.CONTEXT,'CONTEXT')
    reference=read(context['baseline_profile'])
    baseline=read(freeze['evidence']['baseline_bindings'])
    require(reference['baseline_bindings']==freeze['evidence']['baseline_bindings']
        and reference['r6']['protocol']==batch['protocol'],'BASELINE_PROTOCOL')
    origin=file_pin(Path(baseline['snapshot']['snapshot'])/'training_workloads/unified_h1/candidate_config.json')
    require(origin['sha256']==baseline['source_identity']['config_sha256'],'ORIGIN_CONFIG')
    initial=read(request['initial_request']);plan=read(initial['plan_pin']);spec=read(plan['spec_pin'])
    auth=read(spec['pins']['authorization']);params=auth['parameters']
    require(auth['mode']=='research-r6-evaluation' and params['baseline_bindings']==reference['baseline_bindings']
        and params['r6']['protocol']==batch['protocol'],'BASELINE_CAPACITY_SOURCE')
    protocol=read(batch['protocol'])
    require(params['r6']['block_ids']==[b['id'] for b in protocol['blocks']],'FULL_BASELINE_CAPACITY')
    result_volume=read(params['result_volume'])
    # Every final assignment runs the same full suite with the same bounded logs.
    payload=result_volume['minimum_free_bytes'];wall=params['wall_seconds']
    origins=[origin]*len(batch['assignments'])
    if freeze['branch']=='THREE_CONTINUATIONS':
        prepared=read(freeze['evidence']['preparation']);selected_snapshot=read(prepared['snapshot'])
        config_pin=selected_snapshot['config']
        origins=[origin]+[config_pin]*3
    else:
        require(freeze['branch']=='BASELINE_SUFFICIENT','BRANCH')
        selected_snapshot=read(context['snapshot_receipt']);config_pin=selected_snapshot['config']
    template=read(config_pin);snapshot.verify_frozen(selected_snapshot,template)
    return dict(batch=batch_pin,context=context,reference=reference,template_config=config_pin,
        origin_configs=origins,wall_seconds=wall,payload_bytes=payload)


def configuration(template,target,commit):
    value=copy.deepcopy(template);value.pop('resolved_identity',None)
    value['schema_version']=R6_FINAL_CONFIG_SCHEMA
    value['future_entry']['modes']=[m for m in value['future_entry']['modes']
        if m not in ('research-r6-training','research-r6-candidate-evaluation','research-r6-final-evaluation')]
    value['future_entry']['modes'].append('research-r6-final-evaluation')
    for section,names in zip(('product','workload'),required_source_sets(R6_FINAL_CONFIG_SCHEMA)):
        value[section].update(path=str(target),candidate_commit='SELF',candidate_tree='SELF',branch='',
            source_sha256={name:hashlib.sha256(snapshot.git(ROOT,'show',commit+':'+name)).hexdigest() for name in sorted(names)})
    return value


def prepare(journal,controller_key,batch_pin,store,*,free_disk_bytes):
    profile.guard(journal);description=describe(journal,controller_key,batch_pin,store)
    state=journal.read();name='FINAL_INPUTS:'+batch_pin['sha256']
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    implementation=[file_pin(m.__file__) for m in (profile,publication,snapshot)]+[file_pin(__file__)]
    if prior:
        key,entry=prior[0];request=entry['request']
        require(request['description']==description and request['implementation']==implementation,'INPUT_DRIFT')
    else:
        require(git('symbolic-ref','--short','HEAD')=='main','MAIN')
        commit=git('rev-parse','HEAD')
        # Inputs are committed below; all executable code must already be committed.
        for p in implementation:
            require(snapshot.git(ROOT,'show',commit+':'+str(Path(p['path']).relative_to(ROOT)))==checked(p).read_bytes(),'UNCOMMITTED_TOOL')
        request=dict(description=description,implementation=implementation,base_commit=commit,controller_key=controller_key)
        event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-BATCH',input_sha256=sha(request),role=ROLE)
        key,_=journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    folder=journal.directory/'final-inputs'/key
    def save(name,value):
        profile.guard(journal);data=canonical(value)+b'\n'
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)<=CAP,'CAP');path=folder/name;persist_exact(path,data);return file_pin(path)
    target=ROOT/'coordination/research/workspaces'/('R6-final-'+key[:16])
    config=configuration(read(description['template_config']),target,request['base_commit'])
    config_pin=save('config.json',config);relative=str(Path(config_pin['path']).relative_to(ROOT))
    commits=git('log','-2','--format=%H','HEAD','--',relative).splitlines();require(len(commits)<=1,'CONFIG_HISTORY')
    if commits:commit=commits[0]
    else:
        require(git('rev-parse','HEAD')==request['base_commit'],'HEAD_DRIFT')
        profile.guard(journal);git('add','--',relative)
        profile.guard(journal);git('commit','--only','-m','R6 final config '+key,'--',relative)
        commit=git('rev-parse','HEAD')
    require(git('show','-s','--format=%P',commit)==request['base_commit']
        and git('show','-s','--format=%s',commit)=='R6 final config '+key
        and git('diff-tree','--no-commit-id','--name-only','-r',commit)==relative
        and snapshot.git(ROOT,'show',commit+':'+relative)==checked(config_pin).read_bytes(),'CONFIG_COMMIT')
    tag='r6-final-inputs-'+key[:16]
    tags=git('tag','--list',tag)
    if tags:require(git('rev-parse',tag)==commit,'TAG_DRIFT')
    else:profile.guard(journal);git('tag',tag,commit)
    # Normal push only; recovery repeats the same refs without changing history.
    profile.guard(journal);git('push','origin','main','refs/tags/'+tag)
    frozen=snapshot.prepare(journal,checked(config_pin),commit,purpose='confirmation_delivery')
    require(frozen['status']=='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY','SNAPSHOT')
    snapshot_pin=journal.read()['events'][frozen['event_key']]['receipt']
    context=description['context'];context=dict(context,runner=dict(context['runner'],purpose='confirmation_delivery'))
    inputs=dict(batch=batch_pin,context=save('context.json',context),origin_configs=description['origin_configs'],
        experiment_ids=['R6-FINAL-'+key[:12].upper()+'-'+str(i) for i in range(len(description['origin_configs']))],
        wall_seconds=description['wall_seconds'],payload_bytes=description['payload_bytes'])
    result=dict(status='FINAL_INPUTS_AND_SNAPSHOT_READY_NOT_REVIEWED',event_key=key,snapshot=snapshot_pin,
        inputs=save('inputs.json',inputs),source_commit=commit,tag=tag,scientific_acceptance=False,dispatch_authorized=False)
    pin=save('result.json',result);state=journal.read();state['events'][key]['result']=pin;journal.save(state)
    return result
