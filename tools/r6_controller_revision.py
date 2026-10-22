"""Reviewed control-code revisions preserve one controller identity and cursor.

This does not retry failed work, edit frozen inputs, issue permission or reset a
phase. A code repair must be committed, reviewed, then resumed by a new process.
"""
import json
import hashlib
from pathlib import Path
from rlflow.store import sha,canonical
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact,pin_bytes
from tools.r6_decision import CLI,CLI_SHA,MODEL_CAP,model_stop_paths
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_budget import totals
from tools import r6_profile_preparation as profile,prepare_r6_snapshot as snapshot

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/controller-revision.schema.json'
CAP=256*1024
ROLE='controller-revision-review'


def require(ok,reason):
    if not ok:raise RecoveryError('CONTROLLER_REVISION_'+reason)


def read(pin):return json.loads(checked(pin).read_text())


def packet_of(entry):
    packet=json.loads(checked(entry['request']['prompt']).read_text().splitlines()[-1])
    require(entry.get('revision_packet',packet)==packet,'PACKET_DRIFT')
    return packet


def committed_implementation(commit):
    """Recompute original control/schema identity from immutable Git blobs."""
    commit=snapshot.git(ROOT,'rev-parse',commit+'^{commit}').decode().strip()
    paths=snapshot.git(ROOT,'ls-tree','-r','--name-only',commit).decode().splitlines()
    source_schema='docs/experiments/r6-robustness-auto/prompts/source-preparation.schema.json'
    code=[p for p in paths if p.startswith('tools/') and p.endswith('.py')]
    code+=['training_workloads/unified_h1/config.py',source_schema]
    schemas=[p for p in paths if p.startswith('docs/experiments/r6-robustness-auto/prompts/')
             and '/' not in p.removeprefix('docs/experiments/r6-robustness-auto/prompts/') and p.endswith('.schema.json')]
    def digest(p):return hashlib.sha256(snapshot.git(ROOT,'show',commit+':'+p)).hexdigest()
    return dict(control_sha256=sha({p:digest(p) for p in code}),
        schemas_sha256=sha({p:digest(p) for p in schemas}))


def review_record(journal,key):
    state=journal.read();e=state['events'][key];r=e['request'];i=e['identity'];packet=packet_of(e)
    require(e['kind']=='model' and e['purpose']=='repair' and i['role']==ROLE
        and i['goal_id']==state['goal_id'] and sha(i)==key and i['input_sha256']==sha(r)
        and i['event_id']=='CONTROLLER_REVISION:'+sha(packet),'IDENTITY')
    profile.reservation(state,key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},'repair')
    cpu_name='CONTROLLER_REVISION_INPUT:'+packet['controller_key']+':'+packet['proposal']['sha256']
    cpu=[(k,x) for k,x in state['events'].items() if x['identity'].get('event_id')==cpu_name]
    require(len(cpu)==1 and cpu[0][1]['request']==dict(packet=packet),'CPU_PACKET')
    profile.reservation(state,cpu[0][0],'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'repair')
    require(r['schema']==file_pin(SCHEMA) and r['sandbox']=='read-only'
        and r['executable']==packet['executable'],'MODEL_CONTRACT')
    receipt=read(e['receipt']);require(receipt['event_key']==key and receipt['request_sha256']==sha(r),'RECEIPT')
    result=verified_decision(journal,key)
    require(result['decision']=='PASS_R6_CONTROLLER_REVISION' and result['must_fix']==[]
        and result['changed_files']==[] and result['reviewer']=='r6-independent-controller-revision'
        and result['packet_sha256']==sha(packet) and packet['proposal'] in result['evidence_pins'],'REJECTED')
    for pin in result['evidence_pins']:checked(pin)
    require(result['remaining_budget']['automatic_sessions_remaining']==
        packet['caps']['automatic_model_sessions']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    return packet


def effective(journal,key):
    """Validate the additive chain without rewriting the original hashed request."""
    e=journal.read()['events'][key];value=e['request']['implementation'];seen=set()
    for revision in e.get('implementation_revisions',[]):
        require(revision not in seen,'DUPLICATE');seen.add(revision)
        packet=review_record(journal,revision)
        require(packet['controller_key']==key and packet['before']==value,'CHAIN')
        value=packet['after']
    return value



def prompt_bytes(packet):
    return ('Read-only independent review of a committed R6 control-code repair. Read proposal, evidence and '
            'the exact Git diff from_commit..to_commit. Verify task scope, ownership/STOP/budget/actor/runtime protections, '
            'receipt and source pin compatibility of every preserved cursor stage. This revision preserves controller '
            'identity, cursor, all original requests/failed receipts and budgets; it grants no retry, new attempt, '
            'phase reset or inherited approval for changed frozen source. Reject incompatible old receipts requiring '
            'a separate repair transition. Do not edit, run GPU, mutate Store/Journal or spawn agents. '
            'PASS_R6_CONTROLLER_REVISION only if safe to resume this unchanged cursor under exact after implementation; '
            'otherwise MUST_FIX. changed_files=[]; reviewer=r6-independent-controller-revision. '
            'packet_sha256 is canonical SHA256 of packet; evidence_pins includes proposal. '
            'Remaining sessions=caps minus budget minus one.\n'+json.dumps(packet,sort_keys=True)+'\n').encode()


def prepare(journal,controller_key,proposal_pin,store,*,executable,free_disk_bytes):
    """Charge one independent review and atomically adopt only its exact revision."""
    from tools import r6_controller as controller
    profile.guard(journal);state=journal.read();e=state['events'][controller_key];proposal=read(proposal_pin)
    require(set(proposal)=={'from_commit','to_commit','reason','evidence_pins'} and bool(proposal['reason']),'PROPOSAL')
    for pin in proposal['evidence_pins']:checked(pin)
    require(e['identity']['role']==controller.ROLE and e['request']['store_root']==str(store.root.resolve())
        and sha(e['identity'])==controller_key and e['identity']['input_sha256']==sha(e['request']),'CONTROLLER')
    require(e['cursor']['phase']!='COMPLETE' and not state.get('goal_complete'),'ALREADY_COMPLETE')
    # A completed migration can be replayed, but a new source change needs a new proposal.
    matching=[(k,x) for k,x in state['events'].items() if x['identity']['role']==ROLE
              and packet_of(x).get('proposal')==proposal_pin]
    require(len(matching)<=1,'ALIAS')
    cpu_name='CONTROLLER_REVISION_INPUT:'+controller_key+':'+proposal_pin['sha256']
    cpu_prior=[(k,x) for k,x in state['events'].items() if x['identity'].get('event_id')==cpu_name]
    require(len(cpu_prior)<=1,'CPU_ALIAS')
    if matching:
        key,old=matching[0];packet=packet_of(old)
        require(packet['controller_key']==controller_key and packet['executable']==executable,'INPUT_DRIFT')
    elif cpu_prior:
        packet=cpu_prior[0][1]['request']['packet']
        require(packet['proposal']==proposal_pin and packet['executable']==executable,'CPU_INPUT_DRIFT')
    else:
        before=effective(journal,controller_key);after=controller.implementation()
        require(before!=after,'NO_CHANGE')
        require(committed_implementation(proposal['from_commit'])==before,'OLD_COMMIT')
        require(committed_implementation(proposal['to_commit'])==after,'UNCOMMITTED_SOURCE')
        require(snapshot.git(ROOT,'rev-parse','HEAD').decode().strip()==proposal['to_commit']
            and snapshot.git(ROOT,'symbolic-ref','--short','HEAD').decode().strip()=='main','TARGET_HEAD')
        require(snapshot.git(ROOT,'merge-base',proposal['from_commit'],proposal['to_commit']).decode().strip()==proposal['from_commit'],'HISTORY')
        exe=checked(executable)
        if state['goal_id']=='R6-CPU-TEST-FIXTURE':require(exe.is_relative_to(journal.directory.parent.resolve()),'FIXTURE_EXE')
        else:
            require(executable==dict(path=str(CLI),sha256=CLI_SHA),'CLI')
            from tools.r6_review_access import check
            check(executable,proposal_pin)
        packet=dict(controller_key=controller_key,proposal=proposal_pin,before=before,after=after,
            cursor=e['cursor'],cursor_sha256=sha(e['cursor']),journal_path=str(journal.path.resolve()),
            store_root=str(store.root.resolve()),store_sha256=sha(store.replay()),
            implementation=file_pin(__file__),executable=executable,budget=totals(state['budget']),caps=state['budget']['caps'])
        event=dict(goal_id=state['goal_id'],event_id=cpu_name,job_id='R6_CONTROLLER',input_sha256=sha(dict(packet=packet)),role='controller-revision-input')
        journal.intent(event,kind='cpu',request=dict(packet=packet),resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='repair',free_disk_bytes=free_disk_bytes)
    folder=journal.directory/'controller-revisions'/sha(packet)
    prompt=prompt_bytes(packet)
    require(len(prompt)<65536 and len(canonical(packet))<CAP//2,'CAP')
    cpu=[(k,x) for k,x in journal.read()['events'].items() if x['identity'].get('event_id')==cpu_name]
    require(len(cpu)==1 and cpu[0][1]['request']==dict(packet=packet),'CPU_PACKET')
    profile.reservation(journal.read(),cpu[0][0],'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'repair')
    profile.guard(journal);persist_exact(folder/'prompt.txt',prompt)
    request=dict(prompt=pin_bytes(folder/'prompt.txt',prompt),schema=file_pin(SCHEMA),executable=executable,
        cwd=str(ROOT),sandbox='read-only',timeout_seconds=3600,output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    event=dict(goal_id=state['goal_id'],event_id='CONTROLLER_REVISION:'+sha(packet),job_id='R6_CONTROLLER',input_sha256=sha(request),role=ROLE)
    if matching:
        require(matching[0][1]['request']==request and matching[0][1]['identity']==event,'MODEL_DRIFT')
    else:
        key,_=journal.intent(event,kind='model',request=request,resources={'automatic_model_sessions':1,
            'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},purpose='repair',free_disk_bytes=free_disk_bytes)
    require(read(proposal_pin)==proposal and controller.implementation()==packet['after'],'SOURCE_DRIFT')
    require(file_pin(__file__)==packet['implementation'],'VALIDATOR_DRIFT')
    state=journal.read();e=state['events'][controller_key]
    if key in e.get('implementation_revisions',[]):
        require(effective(journal,controller_key)==packet['after'],'APPLIED_DRIFT')
        return dict(status='CONTROLLER_REVISION_APPLIED',event_key=key,controller_key=controller_key)
    require(sha(e['cursor'])==packet['cursor_sha256'] and sha(store.replay())==packet['store_sha256'],'STATE_DRIFT')
    model=state['events'][key]
    if not model.get('dispatch_claim') and not model.get('receipt'):
        expected=dict(packet['budget']);expected['automatic_model_sessions']+=1
        expected['output_actual_plus_unreleased_reservations_bytes']+=CAP+MODEL_CAP+65536
        require(totals(state['budget'])==expected,'STALE_BUDGET')
    model_handoff(journal,key);profile.guard(journal);review_record(journal,key)
    state=journal.read();e=state['events'][controller_key]
    require(effective(journal,controller_key)==packet['before'] and sha(e['cursor'])==packet['cursor_sha256']
        and controller.implementation()==packet['after'] and sha(store.replay())==packet['store_sha256'],'POST_REVIEW_DRIFT')
    e.setdefault('implementation_revisions',[]).append(key);journal.save(state)
    return dict(status='CONTROLLER_REVISION_APPLIED',event_key=key,controller_key=controller_key)


def main():
    import argparse,shutil,subprocess,sys
    from rlflow.store import Store
    from tools.r6_recovery import Journal
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('controller_key')
    parser.add_argument('proposal',type=Path);parser.add_argument('--proposal-sha256',required=True)
    parser.add_argument('--store',required=True,type=Path)
    parser.add_argument('--review-only',action='store_true',help='Record acceptance without starting the reviewed resume process')
    args=parser.parse_args();root=args.store.resolve(strict=True)
    require((root/'runtime/research.db').is_file(),'STORE_REQUIRED')
    journal=Journal(args.directory.resolve())
    with journal.writer():
        result=prepare(journal,args.controller_key,dict(path=str(args.proposal.resolve()),sha256=args.proposal_sha256),
            Store(root),executable=dict(path=str(CLI),sha256=CLI_SHA),free_disk_bytes=shutil.disk_usage(journal.directory).free)
    print(json.dumps(result,sort_keys=True),flush=True)
    if not args.review_only:
        # The review process has released the sole writer lock. The child
        # imports reviewed code afresh and consumes one durable resume claim.
        completed=subprocess.run([sys.executable,'-B','-m','tools.r6_controller_resume',
            str(journal.directory),args.controller_key,result['event_key'],'--store',str(root)],
            cwd=ROOT,check=False)
        raise SystemExit(completed.returncode)


if __name__=='__main__':main()
