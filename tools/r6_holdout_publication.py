"""Commit the single holdout commitment before revealing final batch inputs.

Only the new commitment path is committed; unrelated work and the private nonce
remain outside that commit. Recovery discovers the exact existing commit.
"""
import json
import subprocess
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile
from tools import r6_holdout as holdout

ROOT=Path(__file__).resolve().parents[1]
ROLE='holdout-publication'
CAP=128*1024


def require(value,reason):
    if not value:raise RecoveryError('HOLDOUT_PUBLICATION_'+reason)


def git(*args):
    p=subprocess.run(['git','-C',str(ROOT),*args],capture_output=True,timeout=30)
    require(p.returncode==0,'GIT_FAILED:'+p.stderr.decode(errors='replace')[-2000:])
    require(len(p.stdout)<=65536,'GIT_OUTPUT_CAP')
    return p.stdout.decode().strip()


def assignments(freeze,realization):
    seeds=realization['realization_seeds']
    require(realization['split']=='final' and len(seeds)==3 and len(set(seeds))==3
        and all(type(s)is int and 0<=s<2**31 for s in seeds),'THREE_REALIZATIONS')
    if freeze['branch']=='BASELINE_SUFFICIENT':
        require(set(freeze['models'])=={'baseline'} and freeze['selection_rule']['final_realization_indices']==[0,1,2],
            'BASELINE_STREAM_RULE')
        pairs=[('baseline',i) for i in range(3)]
    else:
        require(freeze['branch']=='THREE_CONTINUATIONS'
            and set(freeze['models'])=={'baseline','seed1','seed2','seed3'},'THREE_MODEL_RULE')
        pairs=[(role,0) for role in ('baseline','seed1','seed2','seed3')]
    return [dict(model_role=role,checkpoint=freeze['models'][role],realization_index=i,
        evaluation_seed=seeds[i],split='final') for role,i in pairs]


def commit_publication(journal,key,pin):
    """Recover only a one-file, exact-parent commit made after publication intent."""
    profile.guard(journal);state=journal.read();entry=state['events'][key]
    request=entry['request'];relative=Path(pin['path']).relative_to(ROOT).as_posix()
    require(git('symbolic-ref','--short','HEAD')=='main','MAIN_REQUIRED')
    checked(pin)
    commits=git('log','-2','--format=%H','HEAD','--',relative).splitlines()
    require(len(commits)<=1,'PUBLICATION_HISTORY_DRIFT')
    if commits:
        commit=commits[0]
    else:
        require(git('rev-parse','HEAD')==request['base_commit'],'HEAD_CHANGED_BEFORE_PUBLICATION')
        # The intent already contains exact parent and artifact identities.
        profile.guard(journal);git('add','--',relative)
        profile.guard(journal)
        git('commit','--only','-m','R6 holdout commitment '+key,'--',relative)
        commit=git('rev-parse','HEAD')
    require(git('show','-s','--format=%P',commit)==request['base_commit'],'COMMIT_PARENT')
    require(git('show','-s','--format=%s',commit)=='R6 holdout commitment '+key,'COMMIT_IDENTITY')
    require(git('diff-tree','--no-commit-id','--name-only','-r',commit)==relative,'COMMIT_SCOPE')
    holdout.verify_publication(pin,commit)
    if entry.get('publication_commit'):
        require(entry['publication_commit']==commit and entry['public_pin']==pin,'COMMIT_DRIFT')
    else:
        state=journal.read();state['events'][key].update(publication_commit=commit,public_pin=pin,
            status='COMMITTED_LOCALLY_NOT_RELEASED');journal.save(state)
    return commit


def prepare(journal,request,*,free_disk_bytes):
    profile.guard(journal)
    require(set(request)=={'freeze','review'},'REQUEST_FIELDS')
    freeze=holdout.validate_freeze(journal,request)
    require(ROOT==holdout.ROOT and git('rev-parse','--show-toplevel')==str(ROOT)
        and git('symbolic-ref','--short','HEAD')=='main','REPOSITORY')
    state=journal.read();name='HOLDOUT_PUBLICATION'
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if prior:
        key,entry=prior[0];packet=entry['request']
        require(packet['freeze_review']==request,'INPUT_DRIFT')
    else:
        packet=dict(freeze_review=request,base_commit=git('rev-parse','HEAD'),
            implementation=file_pin(__file__),holdout_implementation=file_pin(holdout.__file__))
        event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-BATCH',input_sha256=sha(packet),role=ROLE)
        key,_=journal.intent(event,kind='cpu',request=packet,resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    checked(packet['implementation']);checked(packet['holdout_implementation'])
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    folder=ROOT/'docs/experiments/r6-robustness-auto/holdout'/key
    require(folder.resolve()==folder,'CANONICAL_OUTPUT')
    def save(name,value):
        data=canonical(value)+b'\n';path=folder/name
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)<=CAP,'OUTPUT_CAP');profile.guard(journal)
        persist_exact(path,data);return file_pin(path)
    holdout_key,commitment=holdout.prepare(journal,request,free_disk_bytes=free_disk_bytes)
    public=save('commitment.json',commitment)
    commit=commit_publication(journal,key,public)
    profile.guard(journal)
    realization=holdout.release(journal,holdout_key,public,publication_commit=commit)
    require(realization['commitment']['freeze_sha256']==request['freeze']['sha256']
        and realization['commitment']['protocol_sha256']==freeze['protocol']['sha256'],'REALIZATION_BINDING')
    batch=dict(schema_version='r6-final-batch-inputs/1',freeze=request['freeze'],freeze_review=request['review'],
        holdout_event_key=holdout_key,publication=public,publication_commit=commit,
        realization=save('realization.json',realization),assignments=assignments(freeze,realization),
        protocol=freeze['protocol'],scientific_acceptance=False,dispatch_authorized=False)
    result=dict(status='HOLDOUT_RELEASED_FINAL_BATCH_INPUTS_READY',batch=save('batch.json',batch),
        event_key=key,holdout_event_key=holdout_key,publication_commit=commit,
        dispatch_authorized=False,scientific_acceptance=False)
    save('result.json',result)
    return result


def validate_batch(journal,batch_pin):
    """Read back only the one committed, released batch; never generate entropy."""
    profile.guard(journal);batch=json.loads(checked(batch_pin).read_text());state=journal.read()
    require(set(batch)=={'schema_version','freeze','freeze_review','holdout_event_key','publication',
        'publication_commit','realization','assignments','protocol','scientific_acceptance','dispatch_authorized'}
        and batch['schema_version']=='r6-final-batch-inputs/1'
        and batch['scientific_acceptance']is False and batch['dispatch_authorized']is False,'BATCH_FIELDS')
    entries=[(k,e) for k,e in state['events'].items() if e['identity'].get('role')==ROLE]
    require(len(entries)==1,'PUBLICATION_EVENT')
    key,entry=entries[0]
    checked(entry['request']['implementation']);checked(entry['request']['holdout_implementation'])
    profile.reservation(state,key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    request=dict(freeze=batch['freeze'],review=batch['freeze_review'])
    require(entry['request']['freeze_review']==request and entry.get('publication_commit')==batch['publication_commit']
        and entry.get('public_pin')==batch['publication'],'PUBLISHED_BINDING')
    folder=ROOT/'docs/experiments/r6-robustness-auto/holdout'/key
    require(Path(batch_pin['path'])==folder/'batch.json' and Path(batch['realization']['path'])==folder/'realization.json',
        'BATCH_PATH')
    holdout.verify_publication(batch['publication'],batch['publication_commit'])
    event=holdout.validate_entry(state,batch['holdout_event_key'])
    require(event['request']==request and 'release' in event,'RELEASE_REQUIRED')
    public=json.loads(checked(batch['publication']).read_text())
    require(public==event['commitment'],'COMMITMENT')
    realization=json.loads(checked(batch['realization']).read_text())
    require(realization==holdout.reveal(public,realization['nonce'])
        and event['release']==dict(publication=batch['publication'],publication_commit=batch['publication_commit'],
            realization_sha256=sha(realization)),'REALIZATION')
    freeze=holdout.validate_freeze(journal,request)
    require(batch['protocol']==freeze['protocol'] and batch['assignments']==assignments(freeze,realization),'ASSIGNMENTS')
    return batch
