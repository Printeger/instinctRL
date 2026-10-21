"""Real local worker/receipt/Store chain; fake CLI, parent and runtime inspection.

No API call, GPU query or job submission. Only final runner boundary is replaced.
"""
import functools
import json
import sys
import subprocess
REAL_POPEN=subprocess.Popen
from pathlib import Path
from unittest.mock import Mock
import pytest
from rlflow.store import sha
from tools.test_r6_submit_real_contract import bound
from tools import r6_continue_job as continuation,r6_next_job,r6_issue_next
from tools.r6_batch import model_handoff
from tools.r6_recovery import Journal,file_pin,RecoveryError
from tools.r6_job_handoff import no_model_workers


@pytest.mark.parametrize('resume',[False,True,'missing_prompt','missing_prompt_stop','corrupt_prompt'])
@pytest.mark.parametrize('decision',['PASS_R6_NEXT_JOB','MUST_FIX'])
def test_local_worker_receipt_exact_approval_and_replay(bound,monkeypatch,tmp_path,decision,resume):
    b=bound;j=b.journal
    def worker_only(argv,*args,**kwargs):
        assert argv[1:4]==['-B','-m','tools.r6_model_worker']
        return REAL_POPEN(argv,*args,**kwargs)
    monkeypatch.setattr(subprocess,'Popen',worker_only)
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return file_pin(p)
    proposal={'decision':{'decision':'PROPOSE_CANDIDATE'}}
    parent_pin=write('parent-decision.json',proposal)
    req={'fixture':'already analyzed parent'}
    ident=dict(goal_id=j.read()['goal_id'],event_id='PARENT',job_id='J-parent',input_sha256=sha(req),role='post-analysis-decision')
    with j.writer():
        parent,_=j.intent(ident,kind='model',request=req,resources={'automatic_model_sessions':1},purpose='development',free_disk_bytes=500*1024**3)
        s=j.read();s['events'][parent]['decision_record']=parent_pin;j.save(s)
    spec=dict(b.spec,id='R6-LOCAL-WORKER-CPU')
    spec_pin=write('next-spec.json',spec)
    plan=dict(spec_pin=spec_pin,query_executable=file_pin(sys.executable),decision_executable=file_pin(sys.executable),
        queue_review=None,sampler_review=None,purpose='development',sandbox='read-only')
    plan_pin=write('next-plan.json',plan)
    result=dict(decision=decision,parent_decision_sha256=parent_pin['sha256'],plan_sha256=plan_pin['sha256'],
        summary='Unpaid CPU fake reviewer',evidence_pins=[parent_pin,plan_pin],changed_files=[],
        must_fix=[] if decision=='PASS_R6_NEXT_JOB' else ['CPU deliberate rejection'])
    launch_log=tmp_path/'fake-launches.txt';exe=tmp_path/'fake-cli'
    event={'type':'item.completed','item':{'type':'agent_message','text':json.dumps(result)}}
    exe.write_text('#!'+sys.executable+'\nimport sys,time\nfrom pathlib import Path\n'
        +'sys.stdin.read()\nwith Path('+repr(str(launch_log))+').open("a") as f:f.write("launch\\n")\n'
        +'time.sleep(.2)\nprint('+repr(json.dumps(event))+')\n')
    exe.chmod(0o700);exe_pin=file_pin(exe)
    monkeypatch.setattr(r6_next_job,'record_decision',Mock(return_value=proposal))
    monkeypatch.setattr(r6_issue_next,'delegation',Mock())
    monkeypatch.setattr(continuation,'model_handoff',functools.partial(model_handoff,poll_seconds=.05))
    runner=Mock(return_value={'status':'RESOURCE_OBSERVATION_REJECTED'})
    monkeypatch.setattr(r6_next_job,'run_job',runner)
    args=dict(parent_key=parent,plan_pin=plan_pin,master_pin=parent_pin,
        review_executable=exe_pin,store=b.store,free_disk_bytes=500*1024**3)
    approvals=len(b.store.replay()['approvals'])
    if resume:
        with j.writer():
            key=r6_next_job.register_review(j,parent,plan_pin,b.store,executable=exe_pin,free_disk_bytes=500*1024**3)
            prompt_path=Path(j.read()['events'][key]['request']['prompt']['path'])
            if resume in {'missing_prompt','missing_prompt_stop'}:
                prompt_path.unlink()
            elif resume=='corrupt_prompt':prompt_path.write_text('corrupt fixture prompt')
            else:model_handoff(j,key,poll_seconds=.05)
        j=Journal(j.directory)  # New controller resumes the already consumed real receipt.
    if resume in {'missing_prompt_stop','corrupt_prompt'}:
        budget=j.read()['budget']
        if resume=='missing_prompt_stop':(j.directory/'STOP').touch()
        with j.writer(),pytest.raises(RecoveryError,match='STOP' if resume=='missing_prompt_stop' else 'PACKET_DRIFT'):
            continuation.continue_job(j,**args)
        assert j.read()['budget']==budget and not launch_log.exists()
        assert len(b.store.replay()['approvals'])==approvals
        assert (not prompt_path.exists()) if resume=='missing_prompt_stop' else prompt_path.read_text()=='corrupt fixture prompt'
        return
    for iteration in range(2):
        with j.writer():
            if decision=='MUST_FIX':
                with pytest.raises(RecoveryError,match='REVIEW_BINDING'):
                    continuation.continue_job(j,**args)
            else:
                outcome=continuation.continue_job(j,**args)
                assert outcome['status']=='RESOURCE_OBSERVATION_REJECTED'
        if iteration==0:budget=j.read()['budget']
        else:assert j.read()['budget']==budget
    assert launch_log.read_text()=='launch\n'
    assert len(b.store.replay()['approvals'])==approvals+(decision=='PASS_R6_NEXT_JOB')
    assert runner.call_count==(2 if decision=='PASS_R6_NEXT_JOB' else 0)
    assert not b.store.replay()['jobs'];b.popen.assert_not_called()
    with j.writer():no_model_workers(j)
    reviews=[e for e in j.read()['events'].values() if e['identity']['role']=='next-job-review']
    assert len(reviews)==1 and reviews[0]['status']=='RECEIPT_RECORDED'
    assert j.read().get('current_model_event') is None
