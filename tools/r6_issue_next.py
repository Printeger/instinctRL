"""Issue one exact reviewed next-job approval under the user's R6 delegation.

No jobs are submitted. Journal intent precedes Store writes; existing exact
approval is reconciled after a crash, never replaced or silently unrevoked.
"""
import json
from pathlib import Path
from rlflow import experiments,policy
from rlflow.store import sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_next_job import reviewed_plan
from tools.r6_job_handoff import no_model_workers
from tools.r6_postrun import stop_paths
from tools.r6_budget import reserve

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/'docs/experiments/r6-robustness-auto/MASTER_AUTHORITY_V2.json'
MASTER_SHA='465c4e79bf64bad04a3dd4878eee307319966c58119e17506b1cfc9cbccf2f2c'
ROLE='next-job-authorization'
DISK='output_actual_plus_unreleased_reservations_bytes'
CAP=65536


def require(condition,reason):
    if not condition:raise RecoveryError('ISSUE_NEXT_'+reason)


def delegation(journal,master_pin,store):
    require(master_pin=={'path':str(MASTER),'sha256':MASTER_SHA},'MASTER_PIN')
    master=json.loads(checked(master_pin).read_text());s=journal.read()
    for name in ('user_authority','requirements','scope','budget_amendment','supersedes'):checked(master[name])
    require(master['status']=='REGISTERED_USER_DELEGATION' and master['id']==s['goal_id']
        and master['caps']==s['budget']['caps'],'MASTER_SCOPE')
    require(store.root.resolve()==ROOT/'coordination/research','STORE_ROOT')


def issue(journal,parent_key,plan_pin,review_key,review_pin,master_pin,store,*,free_disk_bytes):
    journal.require_writer();journal.guard();require(not stop_paths(journal),'STOP')
    no_model_workers(journal);delegation(journal,master_pin,store)
    s=journal.read();require(s['budget']['capacity_pass'] is True,'CAPACITY')
    require(s.get('current_job') is None,'ACTIVE_JOB')
    plan,spec,jid=reviewed_plan(journal,parent_key,plan_pin,review_key,review_pin,store)
    return _issue_reviewed(journal,parent_key,plan_pin,review_key,review_pin,master_pin,store,
        plan,spec,jid,free_disk_bytes=free_disk_bytes)


def _issue_reviewed(journal,parent_key,plan_pin,review_key,review_pin,master_pin,store,
                    plan,spec,jid,*,free_disk_bytes,role=ROLE):
    """Shared transaction after either exact first-job or next-job review checks."""
    journal.require_writer();journal.guard();require(not stop_paths(journal),'STOP')
    no_model_workers(journal);delegation(journal,master_pin,store)
    s=journal.read();require(s['budget']['capacity_pass'] is True,'CAPACITY')
    require(s.get('current_job') is None,'ACTIVE_JOB')
    experiments.validate(spec,store.root)
    require(spec['backend']=='real' and spec.get('h1',{}).get('mode') in ('research-r6-evaluation','research-r6-training','research-r6-candidate-evaluation','research-r6-final-evaluation','research-r6-video-replay'),'MODE')
    st=store.replay();require(jid not in st['jobs'],'CHILD_EXISTS')
    require(not st['paused'] and not policy.stopped(store.root),'STORE_STOP')
    require(st['stages'].get(spec['stage_id'],{}).get('status')!='REVOKED','STAGE_REVOKED')
    for c in spec['conditions']:
        evidence=st['evidence'].get(c['evidence_id'],{})
        require(evidence.get('accepted') and evidence.get('sha256')==c['sha256'],'CONDITION')
    request=dict(parent_key=parent_key,plan=plan_pin,review_key=review_key,review=review_pin,
        master=master_pin,store_root=str(store.root.resolve()))
    prior=[(k,e) for k,e in s['events'].items() if e['identity']['role']==role
        and e['request']['parent_key']==parent_key]
    require(len(prior)<=1,'PARENT_ALIAS')
    if prior:require(prior[0][1]['request']==request,'PARENT_ALREADY_BOUND')
    identity=dict(goal_id=s['goal_id'],event_id='AUTHORIZE_NEXT:'+parent_key,job_id=jid,
        input_sha256=sha(request),role=role)
    # Check the complete next-job allowance without consuming it here. Actual
    # reservation remains at existing resource admission, and rechecks budget.
    from tools.r6_storage_allocation import for_spec
    storage=for_spec(journal,plan['spec_pin'],purpose=plan['purpose'],preview=True)
    auth=json.loads(checked(spec['pins']['authorization']).read_text())
    if spec['h1']['mode']=='research-r6-video-replay':
        from tools.r6_render_job import validate
        validate(journal,auth['parameters'],plan['purpose'])
    if spec['h1']['mode']=='research-r6-final-evaluation':
        from tools.r6_final_job import validate
        validate(journal,auth['parameters'],plan['purpose'])
    from tools.r6_job_resources import resources as job_resources
    resources=job_resources(spec,auth,prepaid=storage is not None)
    # Include this control record in the hypothetical budget before mutations.
    hypothetical=reserve(s['budget'],event_key=identity,resources={DISK:CAP},
        purpose=plan['purpose'],free_disk_bytes=free_disk_bytes)
    dry=dict(identity,event_id='CHECK_ONLY:'+parent_key,role='budget-check-only')
    reserve(hypothetical,event_key=dry,resources=resources,purpose=plan['purpose'],free_disk_bytes=free_disk_bytes)
    key,_=journal.intent(identity,kind='cpu',request=request,resources={DISK:CAP},
        purpose=plan['purpose'],free_disk_bytes=free_disk_bytes)
    # Do not reset/re-show an existing revoked or rejected proposal on restart.
    existing=[p for p in store.replay()['proposals'].values()
        if p['content'].get('scope')=='EXPERIMENT' and p['content']['spec']['id']==spec['id']]
    require(len(existing)<=1,'PROPOSAL_ALIAS')
    if existing:
        proposal=existing[0];require(proposal['content']['spec']==spec and not proposal.get('revoked'),'PROPOSAL_DRIFT')
        pid=proposal['id']
    else:pid=experiments.propose(store,spec)
    proposal=store.replay()['proposals'][pid];digest=proposal['sha256']
    require(sha(proposal['content'])==digest,'PROPOSAL_HASH')
    approvals=[a for a in store.replay()['approvals'].values() if a['proposal_id']==pid]
    if not approvals:
        require(not stop_paths(journal),'STOP')
        policy.show_proposal(store,pid)
        require(not stop_paths(journal),'STOP')
        policy.decide(store,pid,digest,'APPROVED','R6 delegated exact next-job approval; journal event '+key)
    approval=experiments.authorized(store.replay(),spec,store.root)
    status=('EXACT_INITIAL_BASELINE_AUTHORIZED_NOT_SUBMITTED'
        if role=='initial-baseline-authorization' else 'EXACT_NEXT_JOB_AUTHORIZED_NOT_SUBMITTED')
    result=dict(status=status,event_key=key,
        proposal_id=pid,approval_id=approval['approval_id'],spec_sha256=sha(spec),job_id=jid)
    s=journal.read();old=s['events'][key].get('authorization_result')
    require(old is None or old==result,'RESULT_DRIFT')
    s['events'][key]['authorization_result']=result;journal.save(s)
    return result
