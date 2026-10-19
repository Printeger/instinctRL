"""New W Goal exact Commander; append-only authority, no inherited old budget."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import shutil
import subprocess
from tools.r5_formal_commander import previous_quiescence
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/experiments/r5-w-closure-auto'
CONTROL=ROOT/'coordination/research'
MASTER_ID='R5-W-CLOSURE-AUTO-20261019'
MASTER_EVIDENCE='E-'+MASTER_ID+'-MASTER'
HELPER=ROOT/'tools/r5_formal_commander.py'
FIELDS={'training_transitions':'training_transitions','nontraining_transitions':'nontraining_transitions','optimizer_updates':'optimizer_updates','training_checkpoint_writes':'checkpoint_writes'}

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def save(p,v,*,new=False):
    from rlflow.store import atomic
    p=Path(p)
    if new:assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True)
    atomic(p,(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def save_once_or_same(p,v):
    p=Path(p)
    if p.exists():
        assert read(p)==v,('Existing artifact differs',p)
    else:save(p,v,new=True)

def observed_counters(h1):
    counters=h1.get('counters') or {}
    fields={'training_transitions':'training_transitions','nontraining_transitions':'nontraining_transitions',
        'optimizer_updates':'real_optimizer_updates','training_checkpoint_writes':'real_checkpoint_writes',
        'actor_optimizer_steps':'actor_optimizer_steps','critic_optimizer_steps':'critic_optimizer_steps'}
    result={}
    for key,source in fields.items():
        value=counters.get(source)
        if value is not None and (type(value) is not int or value<0):
            raise ValueError('Invalid observed counter: '+source)
        result[key]=value
    if h1.get('counter_completeness')=='FINAL' and any(v is None for v in result.values()):
        raise ValueError('FINAL evidence with missing counter')
    return result

def checked(p):
    f=Path(p['path']);assert f.is_absolute() and f.resolve()==f and not f.is_symlink() and sha(f)==p['sha256'],f
    return f

def idle(store):
    from rlflow import policy
    s=store.replay();assert not policy.stopped(CONTROL) and not s['paused']
    assert not (BASE/'STOP').exists()
    previous_quiescence(CONTROL,s)
    return s

def review(path,required):
    r=read(path);assert r['verdict']=='PASS' and not r['must_fix']
    assert r['reviewer']!='root' and r['reviewer'], 'Independent reviewer required'
    for k,v in required.items():assert r['identities'][k]==v,(k,'review binding differs')
    return r

def accept(store,eid,path,scope):
    from rlflow.store import sha as object_sha
    value=dict(id=eid,**pin(path),accepted=True,producer='Supervisor-Commander',scope=scope)
    previous=store.replay()['evidence'].get(eid)
    assert previous is None or previous==value,'Evidence ID collision; never replace historical evidence'
    store.transact('W_GOAL_CONDITION_ACCEPTED',eid+':'+value['sha256'],lambda s:s['evidence'].update({eid:value}),object_id=eid,producer='Supervisor-Commander')
    return dict(evidence_id=eid,sha256=value['sha256'])

def authority(store,stagepath):
    masterpath=BASE/'MASTER_AUTHORITY.json';master=read(masterpath)
    assert master['id']==MASTER_ID and master['status']=='REGISTERED_USER_DELEGATION'
    checked(master['user_prompt']);checked(master['requirements'])
    evidence=store.replay()['evidence'][MASTER_EVIDENCE]
    assert evidence['accepted'] and evidence['path']==str(masterpath) and evidence['sha256']==sha(masterpath)
    stage=read(stagepath);assert stage['master_authority']==pin(masterpath) and stage['goal_id']==MASTER_ID
    for key,cap in stage['cumulative_caps'].items():assert type(cap) is int and 0<=cap<=master['master_caps'][key]
    assert set(stage['cumulative_caps'])==set(master['master_caps'])
    e=store.replay()['evidence'][stage['evidence_id']]
    assert e['accepted'] and e['path']==str(stagepath) and e['sha256']==sha(stagepath)
    return master,stage

def reservations(b,budget,output_cap):
    cost={k:budget[v] for k,v in FIELDS.items()}
    cost.update(actor_optimizer_steps=budget['optimizer_updates']*8,critic_optimizer_steps=budget['optimizer_updates']*8,
        allocated_gpu_seconds=budget['wall_seconds']+10,wall_seconds_including_cleanup=budget['wall_seconds']+10,
        live_starts=1,simulator_initialization_reservations=1)
    totals={k:b['used'][k]+sum(e['reserved'].get(k,0) for e in b['reservations']['live_entries'])+cost.get(k,0) for k in b['caps']}
    # Output reservations remain charged after settlement; observed bytes are
    # diagnostic and not charged a second time on top of the full reservation.
    totals['output_peak_and_reservations_bytes']=sum(e['output_cap_bytes'] for e in b['reservations']['live_entries'])+sum(s['reserved_bytes'] for s in b['reservations']['source_snapshots'])+output_cap+2*1024**3
    return cost,totals

def remaining_check(b,budget,output_cap,stage):
    cost,totals=reservations(b,budget,output_cap)
    for k,v in totals.items():assert v<=b['caps'][k] and v<=stage['cumulative_caps'][k],(k,v,'resource exhausted')
    return totals

def require_pushed(paths):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
    assert head==remote,'inputs not on remote main'
    for p in paths:
        p=Path(p)
        assert subprocess.check_output(['git','show',head+':'+str(p.relative_to(ROOT))],cwd=ROOT)==p.read_bytes(),p
    return head

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['register','issue','submit','settle','postrun'])
    ap.add_argument('experiment_id');ap.add_argument('group',choices=['R1']);ap.add_argument('mode',choices=['research-formal-training','research-formal-evaluation'])
    ap.add_argument('--review',type=Path);ap.add_argument('--checkpoint-acceptance',type=Path);ap.add_argument('--stage',type=Path,required=True)
    args=ap.parse_args();stagepath=args.stage.resolve();assert BASE/'stages' in stagepath.parents
    from rlflow.store import Store,sha as object_sha
    from rlflow import experiments,policy
    from rlflow.jobs import owns
    from rlflow.releases import verify,files
    store=Store(CONTROL);masterpath=BASE/'MASTER_AUTHORITY.json';b=read(BASE/'goal_budget.json')
    if args.action=='register':
        idle(store);assert args.review
        required=dict(master=pin(masterpath),commander=pin(Path(__file__)),stage=pin(stagepath),ownership_helper=pin(HELPER),snapshot_tool=pin(ROOT/'tools/prepare_r5_w_snapshot.py'))
        review(args.review,required);master=read(masterpath);stage=read(stagepath)
        assert b['master_authority']==pin(masterpath) and b['caps']==master['master_caps']
        assert master['id']==MASTER_ID and stage['goal_id']==MASTER_ID and stage['master_authority']==pin(masterpath)
        checked(master['user_prompt']);checked(master['requirements'])
        assert set(stage['cumulative_caps'])==set(master['master_caps'])
        assert all(type(v) is int and 0<=v<=master['master_caps'][k] for k,v in stage['cumulative_caps'].items())
        accept(store,MASTER_EVIDENCE,masterpath,'Explicit user new incremental W Goal delegation; not a live job approval')
        accept(store,stage['evidence_id'],stagepath,'Cumulative resource allocation under new W master')
        accept(store,stage['evidence_id']+'-REVIEW',args.review.resolve(),'Independent master/stage/adapter prereview')
        reg=BASE/'registrations'/(stage['id']+'.json')
        save(reg,dict(master=pin(masterpath),stage=pin(stagepath),review=pin(args.review),store_version=store.replay()['version'],scope='NEW_INCREMENTAL_NOT_HISTORICAL_REFUND'),new=True)
        b['stage_allocation_history'].append(pin(stagepath));save(BASE/'goal_budget.json',b);print(json.dumps(read(reg)));return
    master,stage=authority(store,stagepath)
    assert b['master_authority']==pin(masterpath) and b['caps']==master['master_caps']
    D=BASE/'experiments'/args.experiment_id;packetpath=D/'launch_inputs.proposed.json';packet=read(packetpath)
    assert packet['master_authority']==pin(masterpath) and packet['stage_allocation']==pin(stagepath)
    row=next(r for r in packet['entries'] if r['group']==args.group and r['mode']==args.mode)
    name=args.group+'-'+args.mode;specp=D/'inputs'/(name+'.authorized.json')
    if args.action=='issue':
        idle(store);assert args.review and args.checkpoint_acceptance
        review(args.review,dict(master=pin(masterpath),stage=pin(stagepath),packet=pin(packetpath),commander=pin(Path(__file__)),ownership_helper=pin(HELPER)))
        accepted=review(args.checkpoint_acceptance,dict(master=pin(masterpath)))
        auth=deepcopy(row['authorization_proposal']);spec=deepcopy(row['experiment_proposal'])
        p=auth['parameters'];ck=p['initialization_checkpoint'];checked({k:ck[k] for k in ('path','sha256')})
        required_cps=[ck]+p.get('checkpoints',[])
        assert all(c in accepted['accepted_checkpoints'] for c in required_cps),'unaccepted exact checkpoint'
        assert p['origin_method_hashes']==accepted['origin_method_hashes'] and p['checkpoint_source_identity']==accepted['origin_source_identity']
        remaining_check(b,auth['budgets'],p['output_cap_bytes'],stage)
        assert spec['stage_id']==MASTER_ID
        spec['conditions']=[dict(evidence_id=MASTER_EVIDENCE,sha256=sha(masterpath)),dict(evidence_id=stage['evidence_id'],sha256=sha(stagepath)),
            accept(store,'E-'+args.experiment_id+'-PRELIVE',args.review.resolve(),'Independent exact new W packet/source/method'),
            accept(store,'E-'+args.experiment_id+'-CHECKPOINT',args.checkpoint_acceptance.resolve(),'Actual checkpoint/source/method identity accepted')]
        actual=Path(spec['pins']['authorization']['path']);assert not actual.exists() and not actual.parent.parent.exists()
        actual.parent.mkdir(parents=True);(actual.parent.parent/'results').mkdir();save(actual,auth,new=True);actual.chmod(0o444)
        assert sha(actual)==row['expected_authorization']['sha256']
        spec['pins']['authorization']=pin(actual)
        experiments.validate(spec,CONTROL);pid=experiments.propose(store,spec);shown=policy.show_proposal(store,pid)
        st=policy.decide(store,pid,shown['sha256'],'APPROVED','Delegated Commander under explicit user W master '+sha(masterpath)+'; stage '+sha(stagepath)+'; independent packet review '+sha(args.review)+'. Not a new user click. Exact single release only after commit/tag/push; all original guards preserved.')
        approval=next(a for a in st['approvals'].values() if a['proposal_id']==pid)
        save(specp,spec,new=True);save(D/(name+'-approval.json'),dict(proposal_id=pid,proposal_sha256=shown['sha256'],approval=approval,actual_authorization=pin(actual),review=pin(args.review),checkpoint_acceptance=pin(args.checkpoint_acceptance),commander=pin(Path(__file__)),stage=pin(stagepath),ownership_helper=pin(HELPER)),new=True)
        print(json.dumps(dict(status='EXACT_DELEGATED_APPROVED_NOT_SUBMITTED',approval_id=approval['approval_id'])));return
    spec=read(specp);matches=[e for e in b['reservations']['live_entries'] if e['spec_id']==spec['id']]
    if args.action=='submit':
        assert not matches,'Already charged; never resubmit'
        idle(store);approval_record=read(D/(name+'-approval.json'))
        assert approval_record['commander']==pin(Path(__file__)) and approval_record['stage']==pin(stagepath) and approval_record['ownership_helper']==pin(HELPER)
        for k in ('review','checkpoint_acceptance','actual_authorization'):checked(approval_record[k])
        head=require_pushed([specp,masterpath,stagepath,packetpath,Path(__file__),HELPER,Path(approval_record['review']['path']),Path(approval_record['checkpoint_acceptance']['path'])])
        release=verify(Path(packet['baseline']['tool_release_path']))
        assert {n:sha(p) for n,p in files(ROOT/'tools/rlflow').items()}==release['files']
        auth=read(checked(spec['pins']['authorization']));budget=auth['budgets'];p=auth['parameters']
        remaining=remaining_check(b,budget,p['output_cap_bytes'],stage)
        assert shutil.disk_usage(ROOT).free>30*1024**3+p['output_cap_bytes']
        experiments.validate(spec,CONTROL);approval=experiments.authorized(store.replay(),spec,CONTROL)
        jid='J-'+object_sha({'id':spec['id'],'spec':spec})[:16];assert jid not in store.replay()['jobs']
        save(D/(name+'-admission.json'),dict(status='PASS',input_commit=head,job_id=jid,spec_id=spec['id'],approval_id=approval['approval_id'],STOP=[],old_jobs_terminal_noowned=True,disk_free_bytes=shutil.disk_usage(ROOT).free,remaining_check=remaining,prelaunch_GPU_queries=0),new=True)
        cost,_=reservations(b,budget,p['output_cap_bytes']);cost.pop('live_starts');cost.pop('simulator_initialization_reservations')
        entry=dict(spec_id=spec['id'],job_id=jid,experiment_id=args.experiment_id,group=args.group,mode=args.mode,status='RESERVED_BEFORE_SUBMIT',reserved=cost,output_cap_bytes=p['output_cap_bytes'],observed={},postrun_queries_used=0,settled=False)
        b['reservations']['live_entries'].append(entry);b['used']['live_starts']+=1;b['used']['simulator_initialization_reservations']+=1;save(BASE/'goal_budget.json',b)
        job=experiments.submit(store,spec);assert job['id']==jid;entry['status']=job['status'];save(BASE/'goal_budget.json',b);save(D/(name+'-started-job.json'),job,new=True)
        print(json.dumps(dict(job_id=jid,status=job['status'],identity=job['identity'])));return
    assert len(matches)==1;entry=matches[0];jd=CONTROL/'jobs'/entry['job_id'];receipt=read(jd/'receipt.json');assert receipt['status'] in ['FAILED','SUCCEEDED','STOPPED']
    job=store.replay()['jobs'][entry['job_id']];ack=read(jd/'lifeguard-ack.json');clean=ack['descendant_cleanup']
    assert ack['no_survivors'] and clean['quiescent'] and clean['stable_scans']>=2 and not clean['survivors']
    assert not owns(job['identity']) and not owns(read(jd/'lifeguard-ready.json')['identity']) and not owns(read(jd/'process-identity.json')) and not any(owns(i) for i in clean['discovered'])
    if args.action=='settle':
     if entry['settled']:
      save_once_or_same(D/(name+'-execution-summary.json'),entry)
      print(json.dumps(entry));return
     finalpath=D/(name+'-final-job.json')
     if finalpath.exists():
      job=read(finalpath);assert job['id']==entry['job_id'] and job['spec']==spec and job['receipt']==receipt
     else:
      job=experiments.reconcile(store,entry['job_id']);save(finalpath,job,new=True)
     attempt=Path(spec['h1']['attempt']);out=attempt/'results'/args.mode;raw=read(out/'status.json') if (out/'status.json').exists() else {'status':None,'reason':'ORIGINAL_STATUS_MISSING'};archive=D/'live'/name;objects=[];copies=0
     originalbytes=sum(p.stat().st_size for folder in [jd,attempt] for p in folder.rglob('*') if p.is_file());assert originalbytes<=entry['output_cap_bytes']
     for category,folder in [('job',jd),('attempt',attempt)]:
      for f in sorted(folder.rglob('*')):
       assert not f.is_symlink()
       if not f.is_file():continue
       r={'original':str(f),'category':category,'relative':str(f.relative_to(folder)),'bytes':f.stat().st_size,'sha256':sha(f),'archived':None}
       if r['bytes']<=2*1024**2:
        assert copies+r['bytes']<=128*1024**2 and originalbytes+copies+r['bytes']<=entry['output_cap_bytes'];dest=archive/category/f.relative_to(folder);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==r['sha256']
        else:shutil.copyfile(f,dest)
        assert sha(dest)==r['sha256'];r['archived']=str(dest);copies+=r['bytes']
       objects.append(r)
     inventory={'job_id':entry['job_id'],'roots':{'job':str(jd),'attempt':str(attempt)},'files':objects,'total_original_bytes':sum(r['bytes'] for r in objects),'total_archive_copy_bytes':copies,'coverage':'Every original job and entire attempt file; no history rewrite or replacement of big originals.'};assert inventory['total_original_bytes']+copies<=entry['output_cap_bytes'];save_once_or_same(archive/'INVENTORY.json',inventory)
     h1=job['h1_evidence'];obs=observed_counters(h1)
     for k,v in obs.items():
      if v is not None:b['observed'][k]=b['observed'].get(k,0)+v
     if h1['counter_completeness']=='FINAL':
      for k,v in obs.items():b['used'][k]+=v;entry['reserved'][k]=0
     for k,v in [('wall_seconds_including_cleanup',receipt['elapsed_seconds']),('allocated_gpu_seconds',receipt['allocated_seconds'])]:b['used'][k]+=v;entry['reserved'][k]=0
     outputbytes=inventory['total_original_bytes']+copies;b['used']['output_peak_and_reservations_bytes']+=outputbytes
     entry.update(status=job['status'],settled=True,observed=obs,native_exit_code=receipt['exit_code'],raw_status=raw.get('status'),raw_status_missing=not (out/'status.json').exists(),formal_counter_completeness=h1['counter_completeness'],formal_transitions=h1['transitions'],owned_cleanup='NORMAL_ACK_2_STABLE_EMPTY_NO_SURVIVORS',output_observed_bytes=outputbytes,inventory=str((archive/'INVENTORY.json').relative_to(BASE)))
     save(BASE/'goal_budget.json',b);save_once_or_same(D/(name+'-execution-summary.json'),entry)
     print(json.dumps(entry));return
    assert entry['settled'] and entry['postrun_queries_used']==0
    from training_workloads.unified_h1.config import resource_binding_pins
    pins=resource_binding_pins();exe=Path(pins['nvidia_smi_path']);assert sha(exe)==pins['nvidia_smi_sha256']
    for i,q in enumerate(['--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu','--query-compute-apps=pid,process_name,used_gpu_memory']):
     entry['postrun_queries_used']+=1;save(BASE/'goal_budget.json',b);argv=[str(exe),q,'--format=csv,noheader,nounits'];r=subprocess.run(argv,capture_output=True,text=True,timeout=15)
     save(D/(name+f'-postrun-{i}.json'),{'job_id':entry['job_id'],'after_normal_owned_ACK':True,'ordinal':i,'argv':argv,'executable_sha256':pins['nvidia_smi_sha256'],'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},new=True);assert r.returncode==0
    summary_path=D/(name+'-execution-summary.json');summary=read(summary_path);summary['postrun_queries_used']=entry['postrun_queries_used'];save(summary_path,summary)
    print(json.dumps({'job_id':entry['job_id'],'postrun_queries_used':2}))
if __name__=='__main__':main()
