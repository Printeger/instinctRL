"""Immutable post-video evidence for one independent final review, never completion."""
import json
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile, r6_final_plans as registered
from tools import r6_delivery_evidence as delivery, r6_final_collection as collection
from tools import r6_video_validation as videos

ROOT=Path(__file__).resolve().parents[1]
ROLE='final-audit-evidence'
CAP=8*1024**2
REQUIREMENTS=ROOT/'docs/goals/InstinctRL_R6_Robustness_Autonomous_Goal_Requirements.md'
CHECKS={
 'scope':'Authorized low-speed horizontal motion/stop/static wall scope; no runtime, dependency, driver or algorithm-family expansion.',
 'actor_boundary':'Actor contains only approved measurement history/validity/timing/IMU/commands; no position, velocity truth/estimate, map or privileged state.',
 'baseline_identity':'Original R5 u640 exact SHA, original manifest/assets/method and clearance margin .15 retained; old raw and failures unchanged.',
 'authority_budget':'Current user authority and every budget amendment are genuine; all resource caps and protected final/repair allocations respected.',
 'frozen_inputs':'Exact committed/tagged inputs, hashes, readonly snapshots, source/config/asset/argv identity for all jobs; no inherited receipt across changed paths.',
 'protocol_split':'Frozen target/stress domains, all formulas/denominators/subgroups, original five tasks and split/realizations predate baseline results.',
 'measurement_physics':'Noise/latency/dropout timestamps and validity, actor/ICS measurement consistency, finite applied-force staging and reset/initialization counters match raw.',
 'baseline_measurement':'New baseline complete original32-lane regression and all registered development conditions; old videos are not new evidence.',
 'diagnosis_decisions':'Evidence-based weak-point diagnosis, registered candidates/selection and necessary comparisons; engineering/control gains not attributed to RL.',
 'curriculum':'Actual sampling/exposure/noise/q-positive/ICS/accidents and progression recorded; retained original/near-wall exposure and no unchecked degradation.',
 'training_restore':'Exact checkpoint/load/Adam/RNG/counters plus first-rollout probability gate before optimizer; fresh physical reset honestly identified.',
 'confirmation_training':'Three newly frozen-recipe continuations from common u640, actual independent random streams, all three terminal models retained; not from-scratch replication.',
 'baseline_sufficient':'If baseline branch: zero training, unchanged policy and three genuinely different environment random streams with complete validation.',
 'holdout':'Models/recipe frozen before single release; all final assignments and paired baseline use required realizations; no tuning on revealed holdout.',
 'complete_results':'All required models, original gates, target/stress blocks, lanes/directions/walls and confidence intervals reported; no omitted failures or averages hiding seeds.',
 'scientific_conclusion':'Positive result matches all frozen gates; negative conclusion requires all necessary registered experiments completed, not budget/engineering failure.',
 'attribution_limits':'Control/reward/optimization comparisons and learning attribution justified; q coverage/loss alone not learning; finite-seed/domain limitations explicit.',
 'resource_admission':'Fresh bounded admission, known GPU/driver/memory/ownership, one allowed reviewed scheduling revision only; no arbitrary relaxed threshold.',
 'lifecycle_failures':'Bounded cleanup, owned descendants only, exact postrun/settlement, native FAILED preserved; any usable failed run independently justified.',
 'idempotent_control':'Unique controller and durable event identities/receipts, restart/STOP/dedup/no duplicate submission; no blind retry or unknown ownership.',
 'zero_model_wait':'Actual run wait windows prove no task Codex worker/call or platform auto-reasoning; no missing interval treated as zero.',
 'model_usage':'All automatic wakes, CLI/child sessions, reasons, actual available token usage and unknowns recorded; protected review/repair limits respected.',
 'storage_quota':'Hard native/result quotas, bounded logs/checkpoints, all originals and retained reservations charged, free disk floor preserved; no hidden refund.',
 'media_identity':'Real Isaac Sim3D recorded-trajectory videos, exact raw/model/scene/time binding, complete selected clips, failures/resets; no new physics falsely claimed.',
 'media_visual':'Independently inspect every provided actual-frame sheet and relevant original video frames; visible scene/UAV/wall/trail/HUD correspond to recorded state; limitations explicit.',
 'media_pairing':'Same condition/lane/random realization comparisons use synchronized state times; unavailable success/failure categories disclosed; no fabricated outcomes.',
 'deliverables':'Baseline/results/method decision/training-evaluation/model/raw manifests, evidence index, videos/viewer, resource/model/requirements audits are substantively complete.',
 'preservation_publication':'Frozen R5/runtime inputs intact; historical failures and unaccepted attempts preserved; final publication must use exact reviewed bytes and normal main push.',
 'final_accounting':'Audit all completed job/CPU/model costs; this review and final publication are reserved and must be included in deterministic post-review accounting.',
}
BASELINE_NA={'curriculum','training_restore','confirmation_training'}


def require(ok,reason):
    if not ok:raise RecoveryError('FINAL_AUDIT_PACKET_'+reason)


def read(pin):return json.loads(checked(pin).read_text())


def describe(journal,controller_key,store):
    profile.guard(journal);s=journal.read();controller=s['events'][controller_key]
    require(controller['identity']['role']=='r6-controller' and controller['request']['store_root']==str(store.root.resolve()),'CONTROLLER')
    c=controller['cursor'];outcome=c['final_collection'];final=outcome['collection']
    require(outcome['status']=='FINAL_BATCH_COLLECTED_REQUIRES_INDEPENDENT_AUDIT'
        and final==collection.collect(journal,final['batch'],outcome['completed_jobs'],store),'FINAL_COLLECTION')
    d=c['delivery_evidence'];registered.registered(journal,d['event_key'],delivery.ROLE,'result',d,delivery.CAP)
    require(read(d['artifacts']['FINAL_R6_RESULTS.json'])==final,'DELIVERY_RESULTS')
    v=c['validated_videos'];key=c['video_validation_key'];entry=s['events'][key]
    require(v['event_key']==key and v['status']=='VIDEOS_CPU_VALIDATED_REQUIRES_VISUAL_AUDIT'
        and v['receipt']==entry['receipt'] and entry['request']['controller_key']==controller_key,'VIDEO_EVENT')
    require(videos.describe(journal,controller_key,store)==entry['request']['inputs'],'VIDEO_SOURCE')
    receipt=videos.verified_receipt(journal,key)
    require(receipt['outputs']['result.json']==v['result'],'VIDEO_RESULT')
    result=read(v['result']);media=read(result['manifest'])
    for row in media['visual_samples']:checked(row['image'])
    freeze=read(read(final['batch'])['freeze'])
    return dict(controller_key=controller_key,final=final,branch=freeze['branch'],delivery=d,
        video_result=v['result'],video_receipt=entry['receipt'],video_manifest=result['manifest'],
        requirements=file_pin(REQUIREMENTS),checks=CHECKS)


def prepare(journal,controller_key,store,*,free_disk_bytes):
    description=describe(journal,controller_key,store);state=journal.read()
    implementation=[file_pin(m.__file__) for m in (profile,delivery,collection,videos)]+[file_pin(__file__)]
    name='FINAL_AUDIT_EVIDENCE:'+controller_key+':'+sha(description)
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if prior:
        key,entry=prior[0];request=entry['request']
        require(request['description']==description and request['implementation']==implementation,'DRIFT')
    else:
        jobs=delivery.jobs_manifest(state)
        for job in jobs['jobs']:
            if job['mode']=='research-r6-video-replay':
                job['missing_evidence']=[x for x in job['missing_evidence'] if x!='analysis']
                job['analysis_semantics']='VIDEO_VALIDATION_INSTEAD_OF_SCIENTIFIC_POLICY_ANALYSIS'
        request=dict(description=description,implementation=implementation,jobs=jobs,
            resources=delivery.resource_audit(state),models=delivery.model_audit(state),store_sha256=sha(store.replay()))
        request['resources']['scope']='SNAPSHOT_AFTER_VIDEO_BEFORE_AUDIT_PACKET_AND_FINAL_REVIEW_NOT_FINAL_ACCOUNTING'
        require(len(canonical(request))<CAP//2,'INPUT_CAP')
        event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-AUDIT',input_sha256=sha(request),role=ROLE)
        key,_=journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='final_review',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'final_review')
    folder=journal.directory/'final-audit-evidence'/key
    def save(name,value):
        profile.guard(journal);data=canonical(value)+b'\n'
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)+len(canonical(request))<=CAP,'OUTPUT_CAP');path=folder/name;persist_exact(path,data);return file_pin(path)
    artifacts={name:save(name,value) for name,value in (
        ('TRAINING_AND_EVALUATION_MANIFEST.json',request['jobs']),('RESOURCE_AUDIT_PRE_REVIEW.json',request['resources']),
        ('MODEL_USAGE_AUDIT_PRE_REVIEW.json',request['models']),('REQUIREMENTS_CHECKLIST.json',dict(requirements=description['requirements'],checks=CHECKS)))}
    result=dict(status='FINAL_AUDIT_EVIDENCE_READY_NOT_ACCEPTED',event_key=key,description=description,artifacts=artifacts,
        scientific_acceptance=False,goal_complete=False)
    pin=save('result.json',result);state=journal.read();state['events'][key]['result']=pin;journal.save(state)
    return result


def verify(journal,pin,store):
    profile.guard(journal);result=read(pin);key=result['event_key'];state=journal.read();entry=state['events'][key]
    require(entry['identity']['role']==ROLE and read(entry['result'])==result,'EVENT')
    profile.reservation(state,key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'final_review')
    require(result['description']==describe(journal,result['description']['controller_key'],store)
        and entry['request']['description']==result['description'] and entry['request']['store_sha256']==sha(store.replay()),'SOURCE_DRIFT')
    for p in entry['request']['implementation']:checked(p)
    expected={'TRAINING_AND_EVALUATION_MANIFEST.json':entry['request']['jobs'],
        'RESOURCE_AUDIT_PRE_REVIEW.json':entry['request']['resources'],
        'MODEL_USAGE_AUDIT_PRE_REVIEW.json':entry['request']['models'],
        'REQUIREMENTS_CHECKLIST.json':dict(requirements=result['description']['requirements'],checks=CHECKS)}
    require(set(result['artifacts'])==set(expected),'ARTIFACT_SET')
    for name,value in expected.items():require(read(result['artifacts'][name])==value,'ARTIFACT_BINDING')
    require(result['status']=='FINAL_AUDIT_EVIDENCE_READY_NOT_ACCEPTED' and result['scientific_acceptance']is False and result['goal_complete']is False,'STATUS')
    return result
