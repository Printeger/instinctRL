"""Prepare immutable R6 delivery evidence without declaring scientific completion.

All attempts and native failures remain visible. Missing worker, video, platform
or independent-audit evidence is explicit; an empty list is never a success.
"""
import json
from rlflow.store import canonical, sha
from tools.r6_recovery import checked, file_pin, RecoveryError
from tools.r6_decision_packet import persist_exact
from tools.r6_budget import admissible, charged, reserve
from tools import r6_profile_preparation as profile
from tools import r6_final_collection as collection

ROLE='delivery-evidence-preparation'
CAP=4*1024**2


def require(value,reason):
    if not value:raise RecoveryError('DELIVERY_'+reason)


def read(pin):return json.loads(checked(pin).read_text())


def resource_audit(state):
    budget=state['budget'];usage=admissible(budget)
    return dict(schema_version='r6-resource-audit/1',caps=budget['caps'],charged=usage,
        remaining={k:budget['caps'][k]-v for k,v in usage.items()},historical_used=budget['used'],
        reservations=[dict(event_key=r['event_key'],identity=r['identity'],purpose=r['purpose'],
            reserved=r['resources'],charged=charged(r,budget['caps']),observations=r.get('observations',[])) for r in budget['reservations']],
        protected=budget['protected_confirmation_delivery_repair'],protected_models=budget['protected_model_sessions'],
        disk_free_floor_bytes=budget['disk_free_floor_bytes'],budget_amendments=budget.get('budget_amendments',[]),
        accounting='MAX_RESERVED_OR_OBSERVED_NO_IMPLICIT_REFUNDS',actual_total_disk_bytes='UNKNOWN',
        scope='SNAPSHOT_BEFORE_VIDEO_AND_FINAL_REVIEW_NOT_FINAL_ACCOUNTING')


def model_audit(state):
    calls=[];waits=[];unknown=[];reported={};tokens=('input_tokens','output_tokens','cached_input_tokens','cache_write_input_tokens','reasoning_output_tokens')
    for key,entry in sorted(state['events'].items()):
        if entry['kind']=='model':
            receipt=read(entry['receipt']) if entry.get('receipt') else None
            if receipt:
                require(receipt['event_key']==key and receipt['request_sha256']==sha(entry['request']),'MODEL_RECEIPT')
            usage=receipt.get('usage',{'availability':'UNKNOWN'}) if receipt else {'availability':'UNKNOWN'}
            totals=usage.get('totals')
            if totals is None and isinstance(usage.get('reported_turns'),list):
                totals={name:sum(row.get(name,0) for row in usage['reported_turns'] if type(row.get(name))is int) for name in tokens}
            for name,value in (totals or {}).items():
                if name in tokens:
                    require(type(value)is int and value>=0,'TOKEN_AMOUNT');reported[name]=reported.get(name,0)+value
            if usage.get('availability')!='REPORTED':unknown.append(key)
            calls.append(dict(event_key=key,role=entry['identity']['role'],purpose=entry['purpose'],
                status=entry['status'],receipt=entry.get('receipt'),usage=usage))
        elif entry['kind']=='job':
            audit=entry.get('local_wait_audit')
            verified=bool(audit and audit.get('schema_version')=='r6-local-wait-audit/1'
                and audit.get('event_key')==key and audit.get('job_id')==entry['identity']['job_id']
                and audit.get('scope')=='REGISTERED_LOCAL_MODEL_WORKERS_ONLY'
                and audit.get('complete') and audit.get('segments')
                and not audit.get('unobserved_restart_gap') and all(s['checks']>0 and
                s['registered_model_workers_present']is False and not s['interrupted'] for s in audit['segments']))
            waits.append(dict(event_key=key,job_id=entry['identity']['job_id'],audit=audit,
                registered_local_wait_verified=verified,platform_inactivity_proven=False,
                observed_local_wait_seconds=sum(s['last_checked_at']-s['started_at'] for s in audit['segments']) if audit else 'UNKNOWN'))
    return dict(schema_version='r6-model-usage-audit/1',calls=calls,registered_model_event_count=len(calls),
        automatic_sessions_charged=admissible(state['budget'])['automatic_model_sessions'],
        platform_wake_accounting=state.get('platform_automatic_wake_accounting'),
        reported_token_subtotals=reported,unknown_usage_events=unknown,platform_token_usage='UNKNOWN',
        waits=waits,all_registered_local_waits_verified=bool(waits) and all(w['registered_local_wait_verified'] for w in waits),
        pure_wait_model_calls='UNKNOWN',duplicate_event_rejections='UNKNOWN',platform_inactivity_proven=False,
        scope='REGISTERED_RECEIPTS_AND_LOCAL_WAIT_RECORDS_PLATFORM_AUDIT_STILL_REQUIRED')


def jobs_manifest(state):
    jobs=[]
    for key,entry in sorted(state['events'].items()):
        if entry['kind']!='job':continue
        spec=read(entry['request']['spec']);auth=read(spec['pins']['authorization'])
        evidence={name:entry[name] for name in ('closure','inventory','postrun','analysis') if entry.get(name)}
        for pin in evidence.values():checked(pin)
        closure=read(evidence['closure']) if 'closure' in evidence else {}
        analysis=read(evidence['analysis']) if 'analysis' in evidence else {}
        report=analysis.get('report')
        if report:checked(report)
        jobs.append(dict(event_key=key,job_id=entry['identity']['job_id'],status=entry['status'],purpose=entry['purpose'],
            mode=auth['mode'],spec=entry['request']['spec'],authorization=spec['pins']['authorization'],
            argv=spec['argv'],cwd=spec['cwd'],source_pins=spec['pins'],parameters=auth['parameters'],
            evidence=evidence,report=report,native_status=closure.get('native_status','UNKNOWN'),
            native_exit_code=closure.get('native_exit_code','UNKNOWN'),
            missing_evidence=[name for name in ('closure','inventory','postrun','analysis') if name not in evidence]))
    return dict(schema_version='r6-training-evaluation-manifest/1',jobs=jobs,
        actual_job_count=len(jobs),scientific_acceptance=False)


def markdown(final):
    lines=['# R6 最终测试证据草稿','',
        '状态：待独立审核及视频交付；尚未科学验收。',
        '原始 FAILED/退出码保留。压力域失败不移除；任一种子失败不以平均值掩盖。','',
        '| 模型/流 | 环境 seed | 原任务 | 目标域 | 原生状态 |',
        '| --- | ---: | --- | --- | --- |']
    for row in final['results']:
        a=row['assignment'];g=row['gates']
        lines.append(f"| {a['model_role']}/{row['assignment_index']} | {a['evaluation_seed']} | {g['original_pass']} | {g['target_pass']} | {row['native_status']} ({row['native_exit_code']}) |")
    lines += ['', '逐条件完整分母（每模型均保留全部块）：','', '| 流 | 条件 | 域 | 成功/总数 |', '| --- | --- | --- | --- |']
    for row in final['results']:
        for b in row['gates']['blocks']:
            lines.append(f"| {row['assignment_index']} | {b['block_id']} | {b['domain']} | {b['successes']}/{b['total']} |")
    lines += ['', '模型身份、逐 lane/方向/墙结果和统计原件见 JSON 清单与报告 pin；此表不替代原件。',
        '三条续训仅检验共同 u640 起点下的条件稳定性，不是从零独立训练。',
        '基线满足分支不声称训练收益；控制与学习归因仍需独审。','']
    return '\n'.join(lines)


def prepare(journal,controller_key,store,*,free_disk_bytes):
    profile.guard(journal);state=journal.read();controller=state['events'][controller_key]
    require(controller['identity']['role']=='r6-controller' and controller['request']['store_root']==str(store.root.resolve()),'CONTROLLER')
    outcome=controller['cursor']['final_collection'];require(outcome['status']=='FINAL_BATCH_COLLECTED_REQUIRES_INDEPENDENT_AUDIT','FINAL_SEQUENCE')
    final=outcome['collection'];require(final==collection.collect(journal,final['batch'],outcome['completed_jobs'],store),'COLLECTION_DRIFT')
    name='DELIVERY_EVIDENCE:'+controller_key+':'+sha(final)
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name];require(len(prior)<=1,'ALIAS')
    implementation=[file_pin(__file__),file_pin(collection.__file__)]
    if prior:
        key,entry=prior[0];request=entry['request']
        require(request['final']==final and request['implementation']==implementation,'INPUT_DRIFT')
    else:
        request=dict(final=final,implementation=implementation,controller_key=controller_key,
            budget=state['budget'],events=state['events'],platform_wake_accounting=state.get('platform_automatic_wake_accounting'))
        event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-BATCH',input_sha256=sha(request),role=ROLE)
        require(len(canonical(request))<CAP//2,'INPUT_CAP')
        key,_=journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    folder=journal.directory/'delivery'/key
    def save(name,value):
        data=(value.encode() if isinstance(value,str) else canonical(value)+b'\n')
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)+len(canonical(request))<=CAP,'OUTPUT_CAP');profile.guard(journal)
        path=folder/name;persist_exact(path,data);return file_pin(path)
    manifest=jobs_manifest(request);audit=model_audit(request)
    accounting=dict(request,budget=reserve(request['budget'],event_key=journal.read()['events'][key]['identity'],
        resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes))
    resources=resource_audit(accounting)
    freeze=read(read(final['batch'])['freeze'])
    models=freeze['models']
    for pin in models.values():checked(pin)
    decisions=[]
    for key0,e in sorted(request['events'].items()):
        if e.get('decision_record'):decisions.append(dict(event_key=key0,record=e['decision_record'],decision=read(e['decision_record'])))
    artifacts={}
    for name,value in [('FINAL_R6_RESULTS.json',final),('FINAL_R6_RESULTS.md',markdown(final)),
        ('TRAINING_AND_EVALUATION_MANIFEST.json',dict(manifest,models=models,branch=freeze['branch'])),
        ('RESOURCE_AUDIT.json',resources),('MODEL_USAGE_AUDIT.json',audit),('METHOD_DECISION_LOG.json',decisions)]:
        artifacts[name]=save(name,value)
    artifacts['EVIDENCE_INDEX.json']=save('EVIDENCE_INDEX.json',dict(artifacts=artifacts,freeze=read(final['batch'])['freeze'],batch=final['batch'],
        raw_inventories=[j['evidence'].get('inventory') for j in manifest['jobs']],
        note='All attempted jobs are retained, including native failures and missing evidence.'))
    pending=['INDEPENDENT_SCIENTIFIC_AUDIT','ACTUAL_SIMULATION_VIDEO_DELIVERY','PLATFORM_ZERO_MODEL_WAIT_AUDIT','FINAL_RESOURCE_AND_USAGE_ACCOUNTING','REQUIREMENTS_COMPLETION_AUDIT']
    if any(j['missing_evidence'] for j in manifest['jobs']):pending.append('ATTEMPT_EVIDENCE_GAPS')
    if not audit['all_registered_local_waits_verified']:pending.append('LOCAL_WAIT_EVIDENCE_GAPS')
    output=dict(status='DELIVERY_EVIDENCE_PREPARED_NOT_ACCEPTED',event_key=key,artifacts=artifacts,pending_requirements=pending,
        scientific_acceptance=False,goal_complete=False,dispatch_authorized=False)
    pin=save('result.json',output);state=journal.read();state['events'][key]['result']=pin;journal.save(state)
    return output
