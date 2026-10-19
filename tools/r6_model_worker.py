"""One durable R6 noninteractive model event; derived from r5_w_batch worker.

The controller reserves the session and commits its exact request first. This
process never submits a job, changes a plan, retries an event or refunds a budget.
"""
from pathlib import Path
import argparse
import json
import os
import signal
import selectors
import subprocess
import time
import jsonschema
from rlflow.jobs import identity, owns
from rlflow.store import atomic, canonical, sha
from rlflow import policy
from tools.r6_recovery import checked, file_pin, Journal

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):atomic(path,canonical(value)+b'\n')


def usage(events):
    """Keep each reported turn; missing provider usage is UNKNOWN, never zero."""
    records=[]; malformed=0
    if events.exists():
        with events.open() as stream:
            for line in stream:
                try: row=json.loads(line)
                except ValueError: malformed+=1;continue
                if row.get('type')=='turn.completed':records.append(row.get('usage','UNKNOWN'))
    return {'reported_turns':records,'malformed_event_lines':malformed,
            'availability':'REPORTED' if records and all(isinstance(r,dict) for r in records) else 'UNKNOWN'}


def validate_request(request):
    required={'prompt','schema','executable','cwd','sandbox','timeout_seconds','output_cap_bytes','stop_paths'}
    if set(request)!=required:raise ValueError('REQUEST_FIELDS')
    prompt=checked(request['prompt']);schema=checked(request['schema']);exe=checked(request['executable'])
    if prompt.stat().st_size>65536:raise ValueError('MODEL_CONTEXT_CAP')
    if request['sandbox'] not in ('read-only','workspace-write','danger-full-access'):raise ValueError('SANDBOX')
    if type(request['timeout_seconds'])not in (int,float) or not 0<request['timeout_seconds']<=3600:raise ValueError('MODEL_TIMEOUT_CAP')
    if type(request['output_cap_bytes'])is not int or not 0<request['output_cap_bytes']<=64*1024**2:raise ValueError('MODEL_OUTPUT_CAP')
    if Path(request['cwd']).resolve()!=ROOT:raise ValueError('WORKSPACE_IDENTITY')
    if not isinstance(request['stop_paths'],list) or not all(isinstance(p,str) and Path(p).is_absolute() for p in request['stop_paths']):raise ValueError('STOP_PATHS')
    value=json.loads(schema.read_text());jsonschema.Draft202012Validator.check_schema(value)
    return prompt,schema,exe,value


def run(directory,key):
    journal=Journal(directory)
    state=journal.read();entry=state['events'][key];request=entry['request']
    if entry['kind']!='model' or entry['resources'].get('automatic_model_sessions')!=1:
        raise ValueError('MODEL_SESSION_NOT_RESERVED')
    if entry['identity']['input_sha256']!=sha(request) or sha(entry['identity'])!=key:raise ValueError('EVENT_IDENTITY')
    if entry['status'] not in ('INTENT','RUNNING') or entry['receipt'] is not None:raise ValueError('EVENT_ALREADY_CONSUMED')
    prompt,schema,executable,schema_value=validate_request(request)
    call=Path(directory)/'calls'/key;call.mkdir(parents=True,exist_ok=True)
    # O_EXCL persists the possible side effect before launching, independent of
    # controller lifetime. A missing receipt does not make this reusable.
    fd=os.open(call/'ONCE',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:os.write(fd,b'one model event\n');os.fsync(fd)
    finally:os.close(fd)
    fd=os.open(call,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    worker=identity(os.getpid());write(call/'worker.json',{'identity':worker,'event_key':key})
    argv=[str(executable),'-a','never','-s',request['sandbox'],'exec','--json','--color','never',
          '-C',str(ROOT),'--output-schema',str(schema),'--output-last-message',str(call/'last-message.json'),'-']
    write(call/'request.json',{'event_key':key,'request':request,'argv':argv})
    started=time.time();clock=time.monotonic();proc=None;child=None;error=None;code=-1;validated=False
    stops=[Path(p) for p in request['stop_paths']]
    def stopped():return journal.stopped()+policy.stopped(ROOT/'coordination/research')+[str(p) for p in stops if p.exists()]
    interrupted=[]
    def on_signal(signum,frame):interrupted.append(signum)
    previous={s:signal.signal(s,on_signal) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        if stopped():raise RuntimeError('STOP_BEFORE_MODEL_START')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',GIT_TERMINAL_PROMPT='0')
        with prompt.open('rb') as stdin,(call/'events.jsonl').open('xb') as stdout,(call/'stderr.log').open('xb') as stderr, selectors.DefaultSelector() as selector:
            proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=stdin,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            child=identity(proc.pid)
            if child is None:raise RuntimeError('CHILD_EXIT_BEFORE_IDENTITY')
            write(call/'child.json',{'identity':child,'event_key':key})
            selector.register(proc.stdout,selectors.EVENT_READ,stdout)
            selector.register(proc.stderr,selectors.EVENT_READ,stderr)
            log_bytes=0
            # Leave room for request/identity/receipt and final JSON. Read at
            # most the remaining logging allowance; never first write a huge
            # stdout chunk and discover the cap afterward.
            log_cap=max(0,request['output_cap_bytes']-min(1024**2,request['output_cap_bytes']//4))
            while selector.get_map() or proc.poll() is None:
                if stopped() or interrupted:raise RuntimeError('STOP_DURING_MODEL')
                if time.monotonic()-clock>request['timeout_seconds']:raise RuntimeError('MODEL_TIMEOUT')
                if sum(p.stat().st_size for p in call.iterdir() if p.is_file())>request['output_cap_bytes']:raise RuntimeError('MODEL_OUTPUT_CAP')
                for ready,_ in selector.select(.2):
                    data=os.read(ready.fileobj.fileno(),65536)
                    if not data:
                        selector.unregister(ready.fileobj);ready.fileobj.close();continue
                    if log_bytes+len(data)>log_cap:raise RuntimeError('MODEL_LOG_CAP_BEFORE_WRITE')
                    ready.data.write(data);ready.data.flush();log_bytes+=len(data)
            proc.wait()
        code=proc.returncode
        if code!=0:raise RuntimeError('MODEL_NONZERO_EXIT')
        if sum(p.stat().st_size for p in call.iterdir() if p.is_file())>request['output_cap_bytes']:raise RuntimeError('MODEL_OUTPUT_CAP')
        result=json.loads((call/'last-message.json').read_text())
        jsonschema.validate(result,schema_value)
        validated=True
    except BaseException as exc:error=repr(exc)
    finally:
        if proc is not None:
            if proc.poll() is None:
                if owns(child):os.killpg(child['pgid'],signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if owns(child):os.killpg(child['pgid'],signal.SIGKILL)
                    proc.wait(timeout=15)
            code=proc.returncode
        for s,handler in previous.items():signal.signal(s,handler)
    receipt={'event_key':key,'request_sha256':sha(request),'process':worker,'child_identity':child,
             'exit_code':code,'error':error,'started_at':started,'finished_at':time.time(),
             'elapsed_seconds':time.monotonic()-clock,'schema_valid':validated,'usage':usage(call/'events.jsonl'),
             'last_message':file_pin(call/'last-message.json') if (call/'last-message.json').exists() else None,
             'events':file_pin(call/'events.jsonl') if (call/'events.jsonl').exists() else None,
             'stderr':file_pin(call/'stderr.log') if (call/'stderr.log').exists() else None,
             'automatic_retry':False}
    write(call/'receipt.json',receipt)
    return 0 if code==0 and error is None and validated else 1


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('event_key')
    a=ap.parse_args();raise SystemExit(run(a.directory.resolve(),a.event_key))
