"""Event-driven single-writer W batch: Python waits; models at decisions only."""
from pathlib import Path
import argparse,fcntl,json,os,signal,subprocess,sys,time
from tools.r5_w_commander import ROOT,BASE,CONTROL,read,save,pin,checked,sha,previous_quiescence,require_pushed
from rlflow.store import Store,sha as object_sha
from rlflow import experiments,policy
from rlflow.jobs import identity,owns

RUN=BASE/'batch'
ORIGINAL_RUN=BASE/'batch'
RESOURCE_RUN=BASE/'batch-resource-v2'
CONTROL_PY='/home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python'
TERMINAL={'FAILED','STOPPED','SUCCEEDED'}

def env():
    return dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.:tools/rlflow:isaaclab-training',CUDA_VISIBLE_DEVICES='',GIT_TERMINAL_PROMPT='0')

def stopped():return policy.stopped(CONTROL)+[str(p) for p in {BASE/'STOP',ORIGINAL_RUN/'STOP',RUN/'STOP'} if p.exists()]

def guard():
    assert not stopped() and not Store(CONTROL).replay()['paused'],'STOP_OR_PAUSE'

def log(event,**kw):
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'controller.log').open('a') as f:
        f.write(json.dumps(dict(event=event,time=time.time(),**kw))+'\n');f.flush();os.fsync(f.fileno())

def verify_resumption(p):
    if RUN == RESOURCE_RUN:
        assert p['run_directory']==str(RESOURCE_RUN)
        predecessor=read(checked(p['predecessor_state']))
        assert p['predecessor_state']['path']==str(ORIGINAL_RUN/'state.json')
        assert predecessor['status']=='STOPPED_EXCEPTION' and not owns(predecessor.get('controller_identity'))
        for call in predecessor['model_calls'].values():
            assert not owns(call.get('worker_identity'))
            receipt=read(checked(call['receipt']))
            assert not owns(receipt.get('identity'))
        supplement=checked(p['resource_user_authority'])
        evidence=Store(CONTROL).replay()['evidence']['E-R5-W-RESOURCE-ADMISSION-USER-20261019']
        assert evidence['accepted'] and evidence['path']==str(supplement) and evidence['sha256']==sha(supplement)
        require_pushed([supplement])

def verify_plan(planpath):
    p=read(planpath);r=read(checked(p['independent_review']))
    assert r['verdict']=='PASS' and not r['must_fix'] and r['reviewer']!='root'
    # Avoid a plan/review hash cycle: exact reviewed payload excludes only review pin.
    assert r['identities']['plan_payload_sha256']==object_sha({k:v for k,v in p.items() if k!='independent_review'})
    assert p['controller']==pin(Path(__file__)) and p['ownership_helper']==pin(ROOT/'tools/r5_formal_commander.py')
    for key in ('spec','commander','stage','after_prompt','schema','codex'):checked(p[key])
    assert p['commander']['path']==str(ROOT/'tools/r5_w_commander.py')
    spec=read(checked(p['spec']));auth=read(checked(spec['pins']['authorization']))
    expected=BASE/'experiments'/p['experiment_id']/'inputs'/('R1-'+spec['h1']['mode']+'.authorized.json')
    assert p['spec']['path']==str(expected) and auth['parameters']['experiment_id']==p['experiment_id'] and auth['parameters']['group']=='R1'
    assert auth['mode']==spec['h1']['mode'] and auth['attempt']==spec['h1']['attempt']
    assert p['poll_seconds']==15 and p['automatic_retries']==0
    assert p['model_timeout_seconds']==7200 and p['model_output_cap_bytes']==64*1024**2
    verify_resumption(p)
    require_pushed([planpath,Path(p['independent_review']['path']),Path(__file__),Path(p['ownership_helper']['path']),Path(p['after_prompt']['path']),Path(p['schema']['path'])])
    return p

def operation(state,label,argv):
    argv=list(argv)
    previous=state['operations'].get(label)
    if previous:
        assert previous.get('exit_code')==0,'AMBIGUOUS_OPERATION: inspect exact job, never retry'
        return
    state['operations'][label]=dict(argv=argv,intent=True,started_at=time.time());save(RUN/'state.json',state)
    out=RUN/(label+'.stdout');err=RUN/(label+'.stderr')
    with out.open('xb') as stdout,err.open('xb') as stderr:
        result=subprocess.run(argv,cwd=ROOT,env=env(),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=300)
    state['operations'][label].update(exit_code=result.returncode,finished_at=time.time());save(RUN/'state.json',state)
    assert result.returncode==0,'OPERATION_FAILED: '+label


def cancel_once(state,entry,store,jid):
    if entry.get('cancel_intent'):
        assert (CONTROL/'jobs'/jid/'cancel').exists(),'CANCEL_INTENT_UNCONFIRMED: preserve exact owned job'
        return
    entry['cancel_intent']=True;entry['stop_seen']=True;save(RUN/'state.json',state)
    experiments.cancel(store,jid)
    entry['cancel_requested']=True;save(RUN/'state.json',state)
    log('OWNED_CANCEL_REQUESTED',job_id=jid)


def model_worker(callpath):
    request=read(callpath/'request.json');checked(request['prompt']);checked(request['schema']);checked(request['codex'])
    once=callpath/'ONCE';fd=os.open(once,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    proc=None;ident=None;error=None;code=-1
    started=time.time()
    try:
        guard()
        with checked(request['prompt']).open('rb') as stdin,(callpath/'events.jsonl').open('xb') as stdout,(callpath/'stderr.log').open('xb') as stderr:
            proc=subprocess.Popen(request['argv'],cwd=ROOT,env=env(),stdin=stdin,stdout=stdout,stderr=stderr,start_new_session=True)
            ident=identity(proc.pid);save(callpath/'child.json',dict(identity=ident))
            while proc.poll() is None:
                if stopped() or Store(CONTROL).replay()['paused'] or time.time()-started>request['timeout_seconds'] or sum(p.stat().st_size for p in callpath.iterdir() if p.is_file())>request['output_cap_bytes']:
                    error='STOP_OR_MODEL_BOUND';break
                time.sleep(2)
    except BaseException as exc:error=repr(exc)
    finally:
        if proc is not None:
            if proc.poll() is None:
                if owns(ident):os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if owns(ident):os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait(timeout=15)
            code=proc.returncode
        save(callpath/'receipt.json',dict(exit_code=code,error=error,identity=ident,started_at=started,finished_at=time.time(),request=pin(callpath/'request.json'),last_message=pin(callpath/'last-message.json') if (callpath/'last-message.json').exists() else None))
    return 0 if code==0 and error is None else 1


def worker_argv(callpath):
    return [CONTROL_PY,str(Path(__file__).resolve()),'_worker',str(callpath),'--run-directory',str(RUN)]


def run(first_plan):
    RUN.mkdir(parents=True,exist_ok=True)
    ORIGINAL_RUN.mkdir(parents=True,exist_ok=True)
    with (ORIGINAL_RUN/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=RUN/'state.json'
        if path.exists():
            state=read(path);assert state['status'] not in ('STOPPED_EXCEPTION','COMPLETE'),'preserve terminal state'
        else:
            state=dict(status='RUNNING',phase='PRELAUNCH',plan=pin(first_plan),operations={},entries={},model_calls={},controller_identity=identity(os.getpid()))
            save(path,state)
        state['controller_identity']=identity(os.getpid());save(path,state)
        try:
            while True:
                planpath=checked(state['plan']);p=verify_plan(planpath)
                spec=read(checked(p['spec']));jid='J-'+object_sha(dict(id=spec['id'],spec=spec))[:16]
                key=spec['id'];entry=state['entries'].setdefault(key,dict(job_id=jid,done=False))
                assert entry['job_id']==jid
                if not entry['done']:
                    state['phase']='JOB:'+jid;save(path,state)
                    store=Store(CONTROL);job=store.replay()['jobs'].get(jid)
                    ledger=read(BASE/'goal_budget.json');charged=any(e['job_id']==jid for e in ledger['reservations']['live_entries'])
                    args=[CONTROL_PY,p['commander']['path'],None,p['experiment_id'],'R1',spec['h1']['mode'],'--stage',p['stage']['path']]
                    if job is None:
                        guard()
                        assert not charged and not entry.get('submit_intent'),'AMBIGUOUS_SUBMIT_NO_RETRY'
                        entry['submit_intent']=True;save(path,state);args[2]='submit';operation(state,key+'-submit',args)
                        job=store.replay()['jobs'].get(jid)
                    assert job and job['spec']==spec,'EXACT_JOB_MISSING_OR_DRIFT'
                    entry.setdefault('deadline',time.time()+spec['resources']['wall_seconds']+180);save(path,state)
                    stop_seen=entry.get('stop_seen',False)
                    while True:
                        stop_seen=stop_seen or bool(stopped()) or store.replay()['paused']
                        if stop_seen:cancel_once(state,entry,store,jid)
                        jd=CONTROL/'jobs'/jid
                        if (jd/'receipt.json').exists():
                            job=experiments.reconcile(store,jid)
                            if job['status'] in TERMINAL and (jd/'lifeguard-ack.json').exists():
                                previous_quiescence(CONTROL,store.replay());break
                        elif not owns(job.get('identity')):job=experiments.reconcile(store,jid)
                        assert time.time()<=entry['deadline'],'OWNERSHIP_RECEIPT_TIMEOUT'
                        log('LOCAL_WAIT',job_id=jid,status=job['status'],STOP=stop_seen);time.sleep(p['poll_seconds'])
                    for action in ('settle','postrun'):
                        args[2]=action;operation(state,key+'-'+action,list(args))
                    entry['done']=True;save(path,state);log('JOB_CLOSED',job_id=jid)
                    assert not stop_seen,'STOP_AFTER_OWNED_CLEANUP'
                # The completed raw evidence is inspected by the model once.
                # That model may prepare the next reviewed exact plan; it cannot
                # start another controller or submit while holding this handoff.
                guard();callpath=RUN/'calls'/key
                call=state['model_calls'].get(key)
                if call is None:
                    callpath.mkdir(parents=True,exist_ok=False)
                    argv=[p['codex']['path'],'-a','never','-s','danger-full-access','exec','--json','--color','never','-C',str(ROOT),'--output-schema',p['schema']['path'],'--output-last-message',str(callpath/'last-message.json'),'-']
                    request=dict(argv=argv,prompt=p['after_prompt'],schema=p['schema'],codex=p['codex'],timeout_seconds=p['model_timeout_seconds'],output_cap_bytes=p['model_output_cap_bytes'])
                    save(callpath/'request.json',request,new=True)
                    call=dict(intent=True,request=pin(callpath/'request.json'));state['model_calls'][key]=call;state['phase']='MODEL:'+key;save(path,state)
                    worker=subprocess.Popen(worker_argv(callpath),cwd=ROOT,env=env(),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
                    call['worker_identity']=identity(worker.pid);save(path,state)
                while not (callpath/'receipt.json').exists():
                    assert call.get('worker_identity') and owns(call['worker_identity']),'MODEL_WORKER_MISSING_NO_RETRY'
                    time.sleep(5)
                receipt=read(callpath/'receipt.json');assert receipt['exit_code']==0 and receipt['error'] is None,'MODEL_FAILED_NO_RETRY'
                decision=read(checked(receipt['last_message']));call['receipt']=pin(callpath/'receipt.json');save(path,state)
                if decision['status']=='CONTINUE':
                    nextpath=checked(decision['next_plan']);nextplan=verify_plan(nextpath)
                    nextspec=read(checked(nextplan['spec']));assert nextspec['id'] not in state['entries'],'CONTINUE_MUST_NAME_A_FRESH_JOB'
                    state['plan']=pin(nextpath);state['phase']='NEXT_PLAN';save(path,state);continue
                assert decision['status'] in ('COMPLETE','STOPPED_EXCEPTION','STOP_RESOURCE_EXHAUSTED')
                if decision['status']=='COMPLETE':
                    result=read(checked(decision['final_result']));finalreview=read(checked(decision['final_review']))
                    assert finalreview['verdict']=='PASS' and not finalreview['must_fix'] and finalreview['reviewer']!='root'
                    assert finalreview['identities']['final_result']==decision['final_result']
                    assert result['status'] in ('R5_ALL_ORIGINAL_GATES_PASS','SCIENTIFIC_NEGATIVE_WITH_COMPLETE_EVIDENCE')
                state['status']=decision['status'];state['phase']='FINAL_DECISION';state['final_decision']=receipt['last_message'];save(path,state);return
        except BaseException as exc:
            state.update(status='STOPPED_EXCEPTION',error=repr(exc));save(path,state);log('STOPPED_EXCEPTION',error=repr(exc));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','_worker']);ap.add_argument('path',type=Path)
    ap.add_argument('--run-directory',type=Path,default=ORIGINAL_RUN);a=ap.parse_args()
    RUN=a.run_directory.resolve();assert RUN in (ORIGINAL_RUN,RESOURCE_RUN)
    if a.action=='_worker':raise SystemExit(model_worker(a.path.resolve()))
    run(a.path.resolve())
