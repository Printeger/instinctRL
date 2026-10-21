"""Prepare immutable reviewed delivery and post-review accounting; preserve history."""
import json
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact
from tools.r6_budget import reserve
from tools import r6_profile_preparation as profile, r6_delivery_evidence as evidence
from tools import r6_final_audit_packet as packet, r6_final_independent_review as review
from tools import prepare_r6_snapshot as snapshot

ROOT=Path(__file__).resolve().parents[1]
DESTINATION=Path('docs/experiments/r6-robustness-auto/deliveries')
ROLE='final-delivery-preparation'
CAP=16*1024**2
REQUIRED={'BASELINE_ROBUSTNESS.json','BASELINE_ROBUSTNESS.md','METHOD_DECISION_LOG.md',
 'TRAINING_AND_EVALUATION_MANIFEST.json','FINAL_R6_RESULTS.json','FINAL_R6_RESULTS.md',
 'RESOURCE_AUDIT.json','MODEL_USAGE_AUDIT.json','REQUIREMENTS_AUDIT.json','EVIDENCE_INDEX.md',
 'goal_budget.json','reviews/final-independent.json','videos/VIDEO_MANIFEST.json','videos/README.md'}
REQUIRED |= {'BASELINE_BINDINGS.json','TEST_PROTOCOL.json','SPLIT_MANIFEST.json',
 'GOAL_SCOPE_AND_MASTER_AUTHORITY.md','MASTER_AUTHORITY.json'}


def require(ok,reason):
    if not ok:raise RecoveryError('FINAL_DELIVERY_'+reason)


def read(pin):return json.loads(checked(pin).read_text())


def git(*args):return snapshot.git(ROOT,'-c','gc.auto=0',*args).decode().strip()


def describe(journal,controller_key,store):
    profile.guard(journal);state=journal.read();controller=state['events'][controller_key]
    require(controller['identity']['role']=='r6-controller' and controller['request']['store_root']==str(store.root.resolve()),'CONTROLLER')
    cursor=controller['cursor'];accepted=cursor['final_independent_audit'];audit_pin=cursor['final_audit_packet']
    require(accepted['status']=='FINAL_AUDIT_ACCEPTED_PENDING_PUBLICATION' and accepted['preparation']==audit_pin,'ACCEPTED_REVIEW')
    decision=review.verify(journal,accepted['event_key'],accepted['review'],audit_pin,store)
    require(decision['decision']=='PASS_R6_FINAL_AUDIT','MUST_FIX_CANNOT_PUBLISH')
    audited=packet.verify(journal,audit_pin,store);d=audited['description']
    require(d['controller_key']==controller_key,'AUDIT_OWNER')
    # No unreviewed extra job can be silently appended to the delivery.
    jobs=evidence.jobs_manifest(state);old=read(audited['artifacts']['TRAINING_AND_EVALUATION_MANIFEST.json'])
    for job in jobs['jobs']:
        if job['mode']=='research-r6-video-replay':
            job['missing_evidence']=[x for x in job['missing_evidence'] if x!='analysis']
            job['analysis_semantics']='VIDEO_VALIDATION_INSTEAD_OF_SCIENTIFIC_POLICY_ANALYSIS'
    require(jobs==old,'UNREVIEWED_JOB_DRIFT')
    batch=read(d['final']['batch']);frozen=read(batch['freeze']);models=frozen['models']
    for pin in models.values():checked(pin)
    baseline=[j for j in jobs['jobs'] if j['mode']=='research-r6-evaluation']
    require(bool(baseline) and any(j['report'] is not None for j in baseline),'BASELINE_REPORT_MISSING')
    protocol=read(batch['protocol']);split=protocol['split_manifest'];checked(split)
    bindings=frozen['evidence']['baseline_bindings'];checked(bindings)
    master=baseline[0]['parameters']['master_authority'];checked(master)
    scope=file_pin(ROOT/'docs/experiments/r6-robustness-auto/GOAL_SCOPE_AND_MASTER_AUTHORITY.md')
    decisions=[]
    for key,entry in sorted(state['events'].items()):
        if entry.get('decision_record'):decisions.append(dict(event_key=key,record=entry['decision_record'],decision=read(entry['decision_record'])))
    final_review=state['events'][accepted['event_key']]
    require(final_review.get('receipt') and final_review['status']=='RECEIPT_RECORDED','FINAL_REVIEW_RECEIPT')
    return dict(controller_key=controller_key,audit_packet=audit_pin,review_key=accepted['event_key'],review=accepted['review'],
        review_receipt=final_review['receipt'],scientific_status=decision['scientific_status'],final=d['final'],branch=d['branch'],
        models=models,jobs=jobs,baseline=baseline,decisions=decisions,video_manifest=d['video_manifest'],
        requirements=d['requirements'],requirements_audit=decision['requirements_audit'],
        protocol=batch['protocol'],split_manifest=split,baseline_bindings=bindings,master_authority=master,scope=scope)


def baseline_markdown(rows):
    lines=['# R6 冻结基线测量','', '每次基线 attempt 单列；原生失败保留。报告链接含全部 lane 和原评分证据。','']
    for job in rows:
        lines += [f"## {job['job_id']}",f"原生状态：{job['native_status']} / {job['native_exit_code']}",
            f"准确源码/config/argv：见 TRAINING_AND_EVALUATION_MANIFEST.json 中 {job['event_key']}。"]
        if job['report']:
            report=read(job['report']);lines += [f"报告：{job['report']['path']} (SHA256 {job['report']['sha256']})",'',
                '| 条件 | 域 | 成功/总数 |','| --- | --- | --- |']
            for b in report['blocks']:
                score=b['scoring'];lines.append(f"| {b['block_id']} | {b['domain']} | {score['successes']}/{score['total']} |")
            lines += ['', '原任务完整摘要：', '```json', json.dumps(report['summary']['original'],ensure_ascii=False,sort_keys=True), '```']
        else:lines += ['该 attempt 无有效分析报告；不作为成功或阴性证据。']
        lines += ['']
    return '\n'.join(lines)


def prepare(journal,controller_key,store,*,free_disk_bytes):
    from tools import r6_publish_final_delivery as publication, r6_terminal_audit as terminal
    d=describe(journal,controller_key,store);state=journal.read();name='FINAL_DELIVERY:'+d['review_receipt']['sha256']
    implementation=[file_pin(m.__file__) for m in (profile,evidence,packet,review,publication,terminal)]+[file_pin(__file__)]
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if prior:
        key,entry=prior[0];request=entry['request']
        require(request['description']==d and request['implementation']==implementation,'INPUT_DRIFT')
        require(request['model_events']=={k:e['receipt'] for k,e in state['events'].items() if e['kind']=='model' and e.get('receipt')}
            and request['platform_wakes']==state.get('platform_automatic_wake_accounting'),'NEW_MODEL_AFTER_ACCOUNTING')
    else:
        require(git('symbolic-ref','--short','HEAD')=='main','MAIN')
        base=git('rev-parse','HEAD')
        for p in implementation:
            require(snapshot.git(ROOT,'show',base+':'+str(Path(p['path']).relative_to(ROOT)))==checked(p).read_bytes(),'UNCOMMITTED_IMPLEMENTATION')
        model_audit=evidence.model_audit(state)
        require(any(c['event_key']==d['review_key'] and c['receipt']==d['review_receipt'] for c in model_audit['calls']),'FINAL_CALL_NOT_COUNTED')
        request=dict(description=d,implementation=implementation,base_commit=base,budget=state['budget'],model_audit=model_audit,
            model_events={k:e['receipt'] for k,e in state['events'].items() if e['kind']=='model' and e.get('receipt')},
            platform_wakes=state.get('platform_automatic_wake_accounting'),store_sha256=sha(store.replay()))
        require(len(canonical(request))<CAP//2,'INPUT_CAP')
        event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-DELIVERY',input_sha256=sha(request),role=ROLE)
        key,_=journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='final_review',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'final_review')
    folder=ROOT/DESTINATION/key;require(folder.resolve()==folder,'OUTPUT_ALIAS')
    def save(name,value):
        profile.guard(journal);path=folder/name;data=value.encode() if isinstance(value,str) else canonical(value)+b'\n'
        require(path.resolve()==path and not path.is_symlink(),'ARTIFACT_ALIAS')
        size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and p!=path) if folder.exists() else 0
        # Cover working-tree bytes and conservative Git object/pack duplication,
        # plus publication and completion metadata, inside the same reservation.
        require(3*(size+len(data)+256*1024)+len(canonical(request))<=CAP,'OUTPUT_CAP')
        persist_exact(path,data);return file_pin(path)
    budget=reserve(request['budget'],event_key=journal.read()['events'][key]['identity'],resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='final_review',free_disk_bytes=free_disk_bytes)
    require(journal.read()['budget']==budget,'BUDGET_CHANGED_AFTER_ACCOUNTING')
    resources=evidence.resource_audit(dict(budget=budget));resources.update(scope='POST_FINAL_REVIEW_WITH_DELIVERY_PUBLICATION_RESERVATION',final_review_receipt=d['review_receipt'],
        note='All reservations retained, including this final delivery. Inventory bytes are not global physical disk usage; UNKNOWN is not zero.')
    models=dict(request['model_audit'],scope='POST_FINAL_REVIEW_REGISTERED_RECEIPTS_AND_PLATFORM_ACCOUNTING',final_review_receipt=d['review_receipt'])
    require(models['automatic_sessions_charged']==resources['charged']['automatic_model_sessions'],'MODEL_ACCOUNTING')
    review_value=read(d['review']);final=dict(d['final'],scientific_status=d['scientific_status'],independent_review=d['review'],
        publication_status='REVIEWED_ARTIFACTS_PREPARED_COMMIT_AND_TERMINAL_AUDIT_PENDING',goal_complete=False)
    report=evidence.markdown(d['final']).replace('# R6 最终测试证据草稿','# R6 最终测试结果').replace(
        '状态：待独立审核及视频交付；尚未科学验收。',f"独审结论：{d['scientific_status']}。提交/终态核验尚待完成。")
    report+='\n## 独立审核结论与限制\n\n'+review_value['summary']+'\n'
    methods=['# R6 方法决策记录','', '以下保留原决策事件与证据；报告不新增方法或学习收益判断。','']
    for row in d['decisions']:methods += [f"## {row['event_key']}",json.dumps(row['decision'],ensure_ascii=False,sort_keys=True),'']
    methods += ['## 最终独审归因与限制',review_value['requirements_audit']['diagnosis_decisions']['reason'],
        review_value['requirements_audit']['attribution_limits']['reason'],'']
    media=read(d['video_manifest'])
    files={}
    for name,value in [('BASELINE_BINDINGS.json',read(d['baseline_bindings'])),('TEST_PROTOCOL.json',read(d['protocol'])),
        ('SPLIT_MANIFEST.json',read(d['split_manifest'])),('MASTER_AUTHORITY.json',read(d['master_authority'])),
        ('GOAL_SCOPE_AND_MASTER_AUTHORITY.md',checked(d['scope']).read_text()),
        ('BASELINE_ROBUSTNESS.json',dict(attempts=d['baseline'],independent_review=d['review'])),
        ('BASELINE_ROBUSTNESS.md',baseline_markdown(d['baseline'])),('METHOD_DECISION_LOG.md','\n'.join(methods)),
        ('TRAINING_AND_EVALUATION_MANIFEST.json',dict(d['jobs'],models=d['models'],branch=d['branch'])),
        ('FINAL_R6_RESULTS.json',final),('FINAL_R6_RESULTS.md',report),('RESOURCE_AUDIT.json',resources),('MODEL_USAGE_AUDIT.json',models),
        ('goal_budget.json',budget),('REQUIREMENTS_AUDIT.json',dict(requirements=d['requirements'],review=d['review'],checks=d['requirements_audit'],
            pending_controller_checks=['EXACT_GIT_PUBLICATION','NO_REMAINING_OWNED_PROCESSES','FINAL_STATE_AND_ARTIFACT_RECHECK'],goal_complete=False)),
        ('reviews/final-independent.json',review_value),('videos/VIDEO_MANIFEST.json',media),
        ('videos/README.md','# R6 视频观看\n\n原始视频和遥测保留在 manifest 的准确路径，不复制或重新编码。\n\n'
            +f"打开原观看页：[同步帧播放器]({media['viewer']['path']})\n\n"
            +'共同帧号对齐同条件/lane/随机流；控制与状态时刻相差一个 policy 周期。失败/reset和缺失类别保留。\n\n'
            +'抽帧独审不替代全部定量结果；原画面语义及视觉限制见 final-independent.json。\n')]:files[name]=save(name,value)
    lines=['# R6 交付证据索引','',f"独审结论：{d['scientific_status']}；本目录不可覆盖，旧报告保持原字节。",'']
    lines += [f"- [{name}]({name}) — SHA256 `{pin['sha256']}`" for name,pin in sorted(files.items())]
    lines += ['',f"原独审 receipt：{d['review_receipt']['path']} / {d['review_receipt']['sha256']}",
        f"原审查证据包：{d['audit_packet']['path']} / {d['audit_packet']['sha256']}",
        'MP4/模型/raw由清单链接至完整原件，未复制；所有失败和未验收尝试保留。','']
    files['EVIDENCE_INDEX.md']=save('EVIDENCE_INDEX.md','\n'.join(lines));require(set(files)==REQUIRED,'DELIVERABLE_SET')
    result=dict(status='FINAL_DELIVERY_READY_NOT_PUBLISHED',event_key=key,controller_key=controller_key,
        scientific_status=d['scientific_status'],review=d['review'],review_receipt=d['review_receipt'],folder=str(folder),files=files,
        goal_complete=False,dispatch_authorized=False)
    pin=save('DELIVERY_MANIFEST.json',result);state=journal.read();state['events'][key]['delivery']=pin;journal.save(state)
    return result


def verify(journal,delivery_pin,store):
    profile.guard(journal);result=read(delivery_pin);entry=journal.read()['events'][result['event_key']]
    profile.reservation(journal.read(),result['event_key'],'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'final_review')
    require(entry['identity']['role']==ROLE and entry['delivery']==delivery_pin and result['goal_complete']is False
        and result['status']=='FINAL_DELIVERY_READY_NOT_PUBLISHED','RESULT')
    require(entry['request']['description']==describe(journal,result['controller_key'],store)
        and entry['request']['store_sha256']==sha(store.replay()),'EVIDENCE_DRIFT')
    require(prepare(journal,result['controller_key'],store,free_disk_bytes=__import__('shutil').disk_usage(journal.directory).free)==result,'ARTIFACT_DRIFT')
    require(set(result['files'])==REQUIRED,'DELIVERABLE_SET')
    folder=ROOT/DESTINATION/result['event_key'];require(result['folder']==str(folder),'FOLDER')
    require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==REQUIRED|{'DELIVERY_MANIFEST.json'},'UNCLAIMED_OUTPUT')
    for name,pin in result['files'].items():require(checked(pin)==folder/name,'ARTIFACT_PATH')
    return result
