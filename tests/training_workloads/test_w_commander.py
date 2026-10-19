from copy import deepcopy
import pytest
from tools.r5_w_commander import remaining_check,reservations

def fixture():
    caps=dict(training_transitions=1048576,nontraining_transitions=659200,optimizer_updates=2048,training_checkpoint_writes=40,
        actor_optimizer_steps=16384,critic_optimizer_steps=16384,live_starts=24,simulator_initialization_reservations=24,
        allocated_gpu_seconds=86400,wall_seconds_including_cleanup=86400,output_peak_and_reservations_bytes=128*1024**3)
    b=dict(caps=caps,used={k:0 for k in caps},reservations=dict(live_entries=[],source_snapshots=[dict(reserved_bytes=2*1024**3)]))
    budget=dict(training_transitions=0,nontraining_transitions=32960,optimizer_updates=0,checkpoint_writes=0,wall_seconds=1800)
    return b,budget,dict(cumulative_caps=deepcopy(caps))

def test_cleanup_and_metadata_always_reserved():
    b,u,s=fixture();r=remaining_check(b,u,8*1024**3,s)
    assert r['allocated_gpu_seconds']==r['wall_seconds_including_cleanup']==1810
    assert r['output_peak_and_reservations_bytes']==12*1024**3

@pytest.mark.parametrize('key',['live_starts','nontraining_transitions','allocated_gpu_seconds','wall_seconds_including_cleanup','output_peak_and_reservations_bytes'])
def test_stage_and_master_caps_cannot_be_exceeded(key):
    b,u,s=fixture();s['cumulative_caps'][key]=0
    with pytest.raises(AssertionError):remaining_check(b,u,8*1024**3,s)
    s['cumulative_caps']=deepcopy(b['caps']);b['caps'][key]=0
    with pytest.raises(AssertionError):remaining_check(b,u,8*1024**3,s)

def test_incomplete_reservations_and_output_survive_settlement():
    b,u,s=fixture();cost,_=reservations(b,u,8*1024**3)
    cost.pop('live_starts');cost.pop('simulator_initialization_reservations')
    b['reservations']['live_entries']=[dict(reserved=cost,output_cap_bytes=8*1024**3,settled=True)]
    b['used']['live_starts']=1;b['used']['simulator_initialization_reservations']=1
    b['used']['output_peak_and_reservations_bytes']=10**6
    r=remaining_check(b,u,8*1024**3,s)
    assert r['nontraining_transitions']==65920 and r['live_starts']==2
    assert r['output_peak_and_reservations_bytes']==20*1024**3


def test_missing_counters_remain_unknown():
    from tools.r5_w_commander import observed_counters
    assert all(v is None for v in observed_counters({'counter_completeness':'INCOMPLETE','counters':{}}).values())
    with pytest.raises(ValueError):observed_counters({'counter_completeness':'FINAL','counters':{}})


def test_missing_status_settle_recovers_after_archive_before_ledger_failure(tmp_path,monkeypatch):
    import json,sys
    from pathlib import Path
    from types import SimpleNamespace
    from tools import r5_w_commander as c
    import rlflow.store,rlflow.jobs
    base=tmp_path/'goal';control=tmp_path/'control';base.mkdir();control.mkdir()
    monkeypatch.setattr(c,'BASE',base);monkeypatch.setattr(c,'CONTROL',control)
    master=base/'MASTER_AUTHORITY.json';master.write_text('{}')
    stage=base/'stages/s.json';stage.parent.mkdir();stage.write_text('{}')
    b,u,st=fixture();b['master_authority']=c.pin(master);b['observed']={}
    cost,_=reservations(b,u,8*1024**3);cost.pop('live_starts');cost.pop('simulator_initialization_reservations')
    jid='J-fixture';eid='R5-W-FIXTURE';mode='research-formal-evaluation';name='R1-'+mode
    attempt=tmp_path/'attempt';attempt.mkdir();(attempt/'results').mkdir();(attempt/'results/large-original.bin').write_bytes(b'x'*(2*1024**2+1))
    spec={'id':'spec','h1':{'attempt':str(attempt)}}
    entry=dict(spec_id='spec',job_id=jid,experiment_id=eid,group='R1',mode=mode,reserved=cost,output_cap_bytes=8*1024**3,settled=False,postrun_queries_used=0)
    b['reservations']['live_entries']=[entry];b['used']['live_starts']=b['used']['simulator_initialization_reservations']=1
    c.save(base/'goal_budget.json',b)
    D=base/'experiments'/eid;(D/'inputs').mkdir(parents=True)
    c.save(D/'launch_inputs.proposed.json',dict(master_authority=c.pin(master),stage_allocation=c.pin(stage),entries=[dict(group='R1',mode=mode)]))
    c.save(D/'inputs'/(name+'.authorized.json'),spec)
    jd=control/'jobs'/jid;jd.mkdir(parents=True)
    receipt=dict(status='FAILED',exit_code=137,elapsed_seconds=50.,allocated_seconds=49.)
    for f,v in [('receipt.json',receipt),('lifeguard-ack.json',dict(no_survivors=True,descendant_cleanup=dict(quiescent=True,stable_scans=2,survivors=[],discovered=[]))),('lifeguard-ready.json',dict(identity={})),('process-identity.json',{})]:c.save(jd/f,v)
    job=dict(id=jid,spec=spec,receipt=receipt,identity={},h1_evidence=dict(counter_completeness='INCOMPLETE',counters={},transitions=None),status='FAILED')
    c.save(D/(name+'-final-job.json'),job)
    monkeypatch.setattr(rlflow.store,'Store',lambda p:SimpleNamespace(replay=lambda:{'jobs':{jid:job}}))
    monkeypatch.setattr(rlflow.jobs,'owns',lambda i:False)
    monkeypatch.setattr(c,'authority',lambda store,path:({'master_caps':b['caps']},st))
    monkeypatch.setattr(sys,'argv',['commander','settle',eid,'R1',mode,'--stage',str(stage)])
    original=c.save
    def fail_ledger(p,v,**kw):
        if Path(p)==base/'goal_budget.json':raise OSError('simulated before atomic ledger commit')
        return original(p,v,**kw)
    monkeypatch.setattr(c,'save',fail_ledger)
    with pytest.raises(OSError):c.main()
    assert not c.read(base/'goal_budget.json')['reservations']['live_entries'][0]['settled']
    monkeypatch.setattr(c,'save',original);c.main()
    result=c.read(base/'goal_budget.json');e=result['reservations']['live_entries'][0]
    assert e['settled'] and e['raw_status_missing'] and e['raw_status'] is None
    inventory=c.read(D/'live'/name/'INVENTORY.json')
    assert next(r for r in inventory['files'] if r['original'].endswith('large-original.bin'))['archived'] is None
    assert e['observed']['nontraining_transitions'] is None and e['reserved']['nontraining_transitions']==32960
    assert result['used']['wall_seconds_including_cleanup']==50.
    c.main() # settled replay does not double-charge
    assert c.read(base/'goal_budget.json')==result
