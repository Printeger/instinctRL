"""Real Journal/worker lifecycle with explicit structural report and Store cleanup fixtures."""
import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.test_r6_training_plan import case
from tools.r6_recovery import Journal,file_pin,RecoveryError
from tools.r6_analysis_binding import bind_analysis
from tools.r6_cpu_analysis import register_analysis,handoff,validate,program_for
from tools.r6_decision_packet import prepare
from tools import r6_training_analysis as a
from rlflow.store import sha


@pytest.fixture
def bound(case,tmp_path,monkeypatch):
    b=case;attempt=tmp_path/'store/attempts/CPU';b.output=attempt/'results/research-r6-training'
    b.output.mkdir(parents=True)
    (b.output/'status.json').write_text('{}')
    auth=dict(mode='research-r6-training',attempt=str(attempt),parameters={'r6_training_plan':b.plan},budgets=a.contract.required_budget(b.plan))
    b.auth=b.pin('authorization.json',auth)
    spec=dict(id='CPU',backend='runner_cpu',h1=dict(attempt=str(attempt),mode=auth['mode']),pins={'authorization':b.auth},
        argv=['python','--authorization',b.auth['path'],'--attempt',str(attempt),'--mode',auth['mode']])
    sp=b.pin('spec.json',spec);jid='J-'+sha({'id':'CPU','spec':spec})[:16];key='CPU-EVENT'
    job=dict(id=jid,spec=spec,status='FAILED')
    b.store=SimpleNamespace(root=tmp_path/'store',replay=lambda:{'jobs':{jid:job}})
    closure=b.pin('closure.json',dict(event_key=key,job_id=jid,spec_sha256=sha(spec),native_status='FAILED',native_exit_code=-9))
    summary=dict(schema_version='r6-training-summary/1',**{k:b.plan[k] for k in ('experiment_id','phase','training_seed','start_update','target_update')},issue_count=1,
        **{k:False for k in ('scientific_acceptance','learning_gain_established','physical_replay_performed','strict_checkpoint_restore_performed','rng_state_semantics_verified','probability_gate_recomputed')})
    report=dict(schema_version='r6-training-analysis/1',authorization=b.auth,plan_sha256=a.contract.object_sha(b.plan),
        analyzer=file_pin(a.__file__),codec=file_pin(a.codec.__file__),
        rng_auditor=file_pin(Path(a.__file__).with_name('r6_training_rng_audit.py')),
        checkpoint_auditor=file_pin(Path(a.__file__).with_name('r6_training_checkpoint_audit.py')),
        probability_replayer=file_pin(Path(a.__file__).with_name('r6_training_probability.py')),scientific_acceptance=False,summary=summary,
        issues=[{'reason':'explicit synthetic incomplete fixture'}],streams=[],
        inputs=[dict(file_pin(p),bytes=Path(p).stat().st_size) for p in (b.auth['path'],b.output/'status.json')])
    b.report=report
    inv=b.pin('inventory.json',dict(event_key=key,job_id=jid,spec=sp,closure=closure,roots={'attempt':str(attempt)},
        files=copy.deepcopy(report['inputs']),output_inventory_complete=True))
    post=b.pin('postrun.json',dict(job_event_key=key,job_id=jid,closure=closure,inventory=inv,postrun_complete=True,
        postrun_queries_used=2,receipts=[b.pin('receipt0.json',{}),b.pin('receipt1.json',{})]))
    b.entry=dict(identity={'job_id':jid,'role':'job'},request={'spec':sp},closure=closure,inventory=inv,postrun=post,kind='job',purpose='development')
    monkeypatch.setattr('tools.r6_analysis_binding.cleanup_evidence',lambda *args:({},dict(status='FAILED',exit_code=-9)))
    b.j=Journal(tmp_path/'batch');b.key=key
    with b.j.writer():
        b.j.initialize(json.loads(Path('tools/fixtures/r6_budget.json').read_text()));s=b.j.read();s['events'][key]=b.entry;b.j.save(s)
    b.program=tmp_path/'fake_analyzer.py'
    b.program.write_text("import json,sys,time\nfrom pathlib import Path\ntime.sleep(.05)\np=Path(sys.argv[3]);p.mkdir()\nr=json.loads((Path(__file__).parent/'report.json').read_text())\n(p/'report.json').write_text(json.dumps(r))\n(p/'summary.json').write_text(json.dumps(r['summary']))\n")
    b.pin('report.json',report)
    return b


def test_closed_training_analysis_worker_recovery_and_decision_packet(bound):
    b=bound
    with b.j.writer():
        key=register_analysis(b.j,b.key,b.store,free_disk_bytes=500*1024**3,python=file_pin(sys.executable),program=file_pin(b.program),timeout_seconds=10)
        _,argv=validate(b.j,key)
        assert argv[-2:]==['--authorization-sha256',b.auth['sha256']]
        result=handoff(b.j,key,b.store,poll_seconds=.01)
        assert handoff(b.j,key,b.store,poll_seconds=.01)==result
        packet=prepare(b.j,b.key,b.store,free_disk_bytes=500*1024**3)
    assert result['native_exit_code']==-9 and not result['scientific_acceptance']
    assert result['protocol'] is None and result['training_plan_sha256']==a.contract.object_sha(b.plan)
    assert result['pending_requirements']==['TRAINING_SEMANTIC_AUDIT_AND_EVALUATION_REQUIRED']
    assert result['issues']==['TRAINING_EVIDENCE_INCOMPLETE_OR_MISMATCHED']
    value=json.loads(Path(packet['packet']['path']).read_text())
    assert value['summary']['schema_version']=='r6-training-summary/1'
    assert value['pending_requirements']==result['pending_requirements']
    assert not any(e['kind']=='model' for e in b.j.read()['events'].values())
    assert program_for('research-r6-training').name=='analyze_r6_training.py'


@pytest.mark.parametrize('fault',['missing','duplicate','drift','acceptance','plan','fake_probability','fake_rng'])
def test_training_report_cannot_escape_closed_inventory(bound,fault):
    b=bound;r=b.report
    if fault=='missing':r['inputs'].pop()
    if fault=='duplicate':r['inputs'].append(r['inputs'][0])
    if fault=='drift':r['inputs'][0]['sha256']='0'*64
    if fault=='acceptance':r['summary']['strict_checkpoint_restore_performed']=True
    if fault=='fake_rng':r['summary']['rng_initialization_audit']={'status':'PASS'}
    if fault=='fake_probability':r['summary'].update(probability_gate_recomputed=True,probability_replay={'status':'PASS','sample_count':512})
    if fault=='plan':r['plan_sha256']='0'*64
    with pytest.raises(RecoveryError):bind_analysis(b.entry,b.key,b.pin('bad-report.json',r),b.store)
