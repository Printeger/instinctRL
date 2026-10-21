"""Real Store/journal adapters; frozen authority, extraction and Popen are CPU substitutes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rlflow import experiments, h1
from rlflow.store import Conflict
from tools import r6_render_job as gate
from tools.r6_recovery import RecoveryError, file_pin
from tools.test_r6_submit_real_contract import make_bound, invoke

REAL_INSPECT = h1.inspect_authority


@pytest.fixture
def bound(tmp_path, monkeypatch):
    return make_bound(tmp_path, monkeypatch, 'research-r6-video-replay')


def test_missing_completed_extraction_stops_before_claim(bound):
    with pytest.raises(RecoveryError, match='EXACT_EXTRACTION_REQUIRED'): invoke(bound)
    assert not bound.store.replay()['jobs']
    assert 'submission_claim' not in bound.journal.read()['events'][bound.key]
    bound.popen.assert_not_called()


def test_reviewed_real_render_store_path_and_restart_do_not_resubmit(bound, monkeypatch):
    check=Mock(return_value='CPU_EXTRACTION'); monkeypatch.setattr(gate,'validate',check)
    before=bound.journal.read()['budget']
    with pytest.raises(OSError, match='CPU stop before Popen'): invoke(bound)
    assert bound.store.replay()['jobs'][bound.jid]['status']=='START_INTENT'
    assert invoke(bound)['job_id']==bound.jid
    assert bound.popen.call_count==1 and bound.journal.read()['budget']==before
    check.assert_called_once()


@pytest.mark.parametrize('field', ['allocated_gpu_seconds','live_wall_seconds','simulator_initializations','live_starts',
                                  'output_actual_plus_unreleased_reservations_bytes'])
def test_render_reservation_cannot_omit_real_resources(bound, monkeypatch, field):
    monkeypatch.setattr(gate,'validate',lambda *a: 'CPU_EXTRACTION')
    with bound.journal.writer():
        state=bound.journal.read(); e=state['events'][bound.key]; e['resources'].pop(field)
        next(r for r in state['budget']['reservations'] if r['identity']==e['identity'])['resources']=dict(e['resources'])
        bound.journal.save(state)
    with pytest.raises(RecoveryError, match='RESERVATION_INCOMPLETE'): invoke(bound)
    bound.popen.assert_not_called()


@pytest.mark.parametrize('field', ['seed','method','evaluation','argv','product','gpu','transition','attempt','experiment'])
def test_real_h1_replay_identity_is_exact(bound, field):
    spec=copy.deepcopy(bound.spec)
    if field=='seed': spec['seed']=1
    elif field=='method': spec['method']='training'
    elif field=='evaluation': spec['evaluation']='PASS'
    elif field=='argv': spec['argv']+=['--mode','research-r6-evaluation']
    elif field=='product': spec['pins']['product']=spec['pins']['workload']
    elif field=='gpu': spec['resources']['gpu_seconds']=0
    elif field=='transition': spec['resources']['transitions']=1
    elif field=='attempt': bound.observed['authorization']['attempt']+='/other'
    elif field=='experiment': spec['id']+='-OTHER'
    with pytest.raises((Conflict, RecoveryError)): experiments.validate(spec, bound.store.root)
    bound.popen.assert_not_called()


def test_render_native_dispatch_uses_both_exact_volumes_and_frozen_pythonpath(bound):
    auth=bound.observed['authorization']; auth['parameters'].update(native_volume={'native':'CPU'},result_volume={'result':'CPU'})
    options=h1.native_dispatch_options(bound.spec)
    assert options['native_volume_pin']=={'native':'CPU'}
    assert options['result_volume_pin']=={'result':'CPU'}
    assert options['native_workload_output_root']==str(Path(bound.spec['h1']['attempt'])/'results')
    assert options['job_output_cap_bytes']==auth['parameters']['output_cap_bytes']
    cfg_path=Path(bound.spec['h1']['workload_root'])/'training_workloads/unified_h1/candidate_config.json'
    cfg=json.loads(cfg_path.read_text());cfg['execution']['isaacsim_asset_root']='/CPU_NOT_ASSETS';cfg_path.write_text(json.dumps(cfg))
    env=h1.environment(bound.spec)
    assert str(Path(bound.spec['cwd'])/'tools/rlflow') in env['PYTHONPATH'].split(':')
    assert env['CUDA_VISIBLE_DEVICES']=='0'


@pytest.fixture
def extraction(monkeypatch,tmp_path):
    result={'path':'/CPU/result.json','sha256':'a'*64}
    receipt_path=tmp_path/'receipt.json';receipt_path.write_text(json.dumps(dict(outputs={'result.json':result})))
    receipt=file_pin(receipt_path)
    event=dict(identity={'role':gate.extraction.ROLE},receipt=receipt,request={'controller_key':'controller'})
    state=dict(events={'extract':event,'controller':dict(identity={'role':'r6-controller'},cursor=dict(
        video_extraction_key='extract',video_trajectories=dict(event_key='extract',status='VIDEO_TRAJECTORIES_READY_NOT_RENDERED',
            result=result,receipt=receipt)))})
    journal=SimpleNamespace(require_writer=Mock(),read=lambda:state)
    checked=Mock(return_value=dict(outputs={'result.json':result}));monkeypatch.setattr(gate.extraction,'verified_receipt',checked)
    return journal,state,dict(extraction=result),checked


def test_old_failed_extraction_does_not_block_verified_replacement(extraction,tmp_path):
    journal,state,p,checked=extraction
    path=tmp_path/'failed.json';path.write_text(json.dumps(dict(exit_code=1,outputs={})))
    state['events']['old_failed']=dict(identity={'role':gate.extraction.ROLE},receipt=file_pin(path))
    assert gate.validate(journal,p,'confirmation_delivery')=='extract'
    checked.assert_called_once_with(journal,'extract')


def test_registered_extraction_and_cursor_binding(extraction):
    journal,state,p,checked=extraction
    assert gate.validate(journal,p,'confirmation_delivery')=='extract'
    checked.assert_called_once_with(journal,'extract');journal.require_writer.assert_called_once()


@pytest.mark.parametrize('fault',['purpose','missing_receipt','result','cursor','cursor_result','cursor_receipt','failed','duplicate'])
def test_extraction_provenance_cannot_be_relabelled(extraction,fault):
    journal,state,p,checked=extraction;purpose='confirmation_delivery'
    if fault=='purpose':purpose='development'
    elif fault=='missing_receipt':state['events']['extract'].pop('receipt')
    elif fault=='result':p['extraction']={}
    elif fault=='cursor':state['events']['controller']['cursor']['video_extraction_key']='other'
    elif fault=='cursor_result':state['events']['controller']['cursor']['video_trajectories']['result']={}
    elif fault=='cursor_receipt':state['events']['controller']['cursor']['video_trajectories']['receipt']={}
    elif fault=='failed':checked.side_effect=RecoveryError('CPU_VIDEO_FAILED_PRESERVED')
    elif fault=='duplicate':state['events']['duplicate']=copy.deepcopy(state['events']['extract'])
    with pytest.raises(RecoveryError):gate.validate(journal,p,purpose)


def test_builder_produces_exact_render_spec_without_issuing_permission(bound):
    from tools.r6_build_spec import build
    from tools.test_r6_build_spec import inputs
    args=inputs(bound)
    # Config changed while preparing the CPU fixture; keep its exact auth binding.
    auth=bound.observed['authorization'];auth['config']=args['config_pin']
    path=Path(args['authorization_pin']['path']);path.write_text(json.dumps(auth));args['authorization_pin']=file_pin(path)
    before=bound.store.replay();spec=build(**args)
    assert spec['argv'][3]=='tools.run_r6_render' and '--mode' not in spec['argv']
    assert spec['resources']['transitions']==0 and spec['resources']['gpu_seconds']==70
    assert bound.store.replay()==before
    with pytest.raises(Conflict,match='no unique exact'):experiments.authorized(before,spec,bound.store.root)


def test_replay_collects_zero_transition_counters_without_changing_native_failure(bound,tmp_path):
    output=Path(bound.spec['h1']['attempt'])/'results'/'research-r6-video-replay';output.mkdir()
    directory=tmp_path/'closed';directory.mkdir()
    counters=dict(live_process_starts=1,simulator_initializations=1,training_transitions=0,nontraining_transitions=0,
        real_optimizer_updates=0,real_checkpoint_writes=0,initialization_physics_ticks=0,explicit_resets=0,
        automatic_resets=0,attempted_policy_steps=0,completed_policy_steps=0,real_environment_transitions=0)
    for name,value in [('status.json',dict(status='PASS',counters=counters)),
            ('process_start.json',dict(authorization_sha256=bound.spec['pins']['authorization']['sha256'])),
            ('process_identity.json',dict(pid=12,pgid=12))]:
        (output/name).write_text(json.dumps(value))
    (directory/'h1-authority.json').write_text(json.dumps(bound.observed))
    receipt=dict(status='FAILED',allocated_seconds=90)
    out=h1.collect(bound.spec,directory,receipt)
    assert out['counter_completeness']=='FINAL' and out['transitions']==0
    assert out['exit_fact']=='FAILED' and out['allocation_gpu_seconds']==90
    assert out['scientific_validity']=='UNASSESSED'


def test_register_render_reserves_once_and_requires_provenance(bound,monkeypatch):
    from tools.test_r6_register_job import fresh
    from tools.r6_register_job import register
    b=fresh.__wrapped__(bound)
    with b.journal.writer(),pytest.raises(RecoveryError,match='EXACT_EXTRACTION_REQUIRED'):
        register(b.journal,b.spec_pin,b.store,purpose='confirmation_delivery',free_disk_bytes=500*1024**3)
    monkeypatch.setattr(gate,'validate',Mock(return_value='CPU_EXTRACTION'))
    with b.journal.writer():
        key=register(b.journal,b.spec_pin,b.store,purpose='confirmation_delivery',free_disk_bytes=500*1024**3)
        before=b.journal.read()
        assert register(b.journal,b.spec_pin,b.store,purpose='confirmation_delivery',free_disk_bytes=500*1024**3)==key
        assert b.journal.read()==before and before['events'][key]['resources']==b.minimum
    b.popen.assert_not_called()


def test_render_closure_routes_to_inventory_postrun_without_model_decision(bound,monkeypatch):
    from tools import r6_run_job as runner, r6_finalize_job as finalizer
    key=bound.key
    with bound.journal.writer():
        s=bound.journal.read();s['events'][key]['submission_claim']={'queue_key':bound.queue_key};bound.journal.save(s)
    closed=dict(event_key=key,closure={'native_status':'FAILED'})
    monkeypatch.setattr(runner,'register_and_close',Mock(return_value=closed))
    finalized=Mock(return_value={'inventory':'CPU','postrun':'CPU'})
    monkeypatch.setattr(finalizer,'finalize',finalized)
    decision=Mock(side_effect=AssertionError('No render model decision'));monkeypatch.setattr(runner,'finish_and_decide',decision)
    spec_pin=bound.journal.read()['events'][key]['request']['spec']
    with bound.journal.writer():
        out=runner.run(bound.journal,spec_pin,bound.store,{}, {},purpose='confirmation_delivery',free_disk_bytes=500*1024**3)
    assert out['status']=='RENDER_JOB_CLOSED_REQUIRES_VIDEO_VALIDATION'
    assert out['closure']['native_status']=='FAILED' and not out['scientific_acceptance']
    finalized.assert_called_once();decision.assert_not_called()


def test_render_exact_issuance_reuses_existing_store_transaction(bound,monkeypatch,tmp_path):
    from tools.test_r6_issue_next import ready, issue
    from rlflow.store import sha
    b=ready.__wrapped__(bound,monkeypatch)
    auth=b.observed['authorization'];auth['parameters']['experiment_id']=b.newspec['id']
    ap=Path(b.newspec['pins']['authorization']['path']);ap.write_text(json.dumps(auth));b.newspec['pins']['authorization']=file_pin(ap)
    sp=tmp_path/'render-spec.json';sp.write_text(json.dumps(b.newspec))
    b.plan.update(spec_pin=file_pin(sp),purpose='confirmation_delivery')
    pp=tmp_path/'render-plan.json';pp.write_text(json.dumps(b.plan));b.pp=file_pin(pp);b.args['plan_pin']=b.pp
    b.jid='J-'+sha(dict(id=b.newspec['id'],spec=b.newspec))[:16];b.review.return_value=(b.plan,b.newspec,b.jid)
    before=b.store.replay()
    with pytest.raises(RecoveryError,match='EXACT_EXTRACTION_REQUIRED'):issue(b)
    assert b.store.replay()==before
    provenance=Mock(return_value='CPU_EXTRACTION');monkeypatch.setattr(gate,'validate',provenance)
    first=issue(b);state=b.store.replay();budget=b.journal.read()['budget']
    assert issue(b)==first and state==b.store.replay() and b.journal.read()['budget']==budget
    assert b.jid not in state['jobs'];b.popen.assert_not_called()


def inspection_inputs(b):
    work=Path(b.spec['cwd']);cfg=b.observed['config']
    cfg['product']['source_sha256']={};cfg['workload']=dict(path=str(work),source_sha256={})
    cfg['execution'].update(python_path=b.spec['pins']['executable']['path'],python_sha256=b.spec['pins']['executable']['sha256'])
    (work/'training_workloads/unified_h1/candidate_config.json').write_text(json.dumps(cfg))
    pins={}
    files={name:'tools/'+file for name,file in [('contract','r6_render_contract.py'),('entry','run_r6_render.py'),
        ('renderer','render_r6_isaac_video.py'),('presentation','r6_video_presentation.py'),
        ('recovery','r6_recovery.py'),('budget','r6_budget.py'),('trajectory','r6_video_trajectory.py')]}
    files.update(launcher='training_workloads/scale_003/resources.py',prelaunch='training_workloads/unified_h1/robustness_prelaunch.py')
    files.update({'rlflow:'+name:'tools/rlflow/rlflow/'+name for name in ('__init__.py','store.py','jobs.py')})
    for name,file in files.items():
        path=work/file;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# inert CPU source fixture\n');path.chmod(0o444);pins[name]=file_pin(path)
    auth=b.observed['authorization'];auth['parameters']['implementation']=pins
    path=Path(b.spec['pins']['authorization']['path']);path.write_text(json.dumps(auth))
    return pins,path


@pytest.mark.parametrize('fault',['hash','writable','path','missing_helper'])
def test_replay_source_drift_rejected_before_importing_snapshot_validator(bound,monkeypatch,fault):
    pins,path=inspection_inputs(bound)
    if fault=='hash':pins['contract']['sha256']='0'*64
    elif fault=='writable':Path(pins['contract']['path']).chmod(0o644)
    elif fault=='path':pins['contract']['path']=pins['entry']['path']
    else:pins.pop('recovery')
    path.write_text(json.dumps(bound.observed['authorization']))
    run=Mock();monkeypatch.setattr(h1.subprocess,'run',run)
    with pytest.raises(Conflict,match='replay source'):REAL_INSPECT(bound.spec)
    run.assert_not_called()


def test_isolated_inspection_program_selects_replay_validator(bound,monkeypatch):
    import contextlib,io,sys
    from training_workloads.unified_h1 import config
    from tools import r6_render_contract
    inspection_inputs(bound)
    monkeypatch.setattr(config,'load_config',lambda:bound.observed['config'])
    validator=Mock(return_value=bound.observed['authorization'])
    monkeypatch.setattr(r6_render_contract,'authorization',validator)
    def run(argv,**kwargs):
        assert argv[1:4]==['-I','-B','-c'] and kwargs['env']['CUDA_VISIBLE_DEVICES']==''
        previous_argv=sys.argv;previous_path=list(sys.path);stream=io.StringIO()
        try:
            sys.argv=['-c',*argv[5:]]
            with contextlib.redirect_stdout(stream):exec(argv[4],{})
        finally:sys.argv=previous_argv;sys.path[:]=previous_path
        return SimpleNamespace(returncode=0,stdout=stream.getvalue(),stderr='')
    monkeypatch.setattr(h1.subprocess,'run',run)
    assert REAL_INSPECT(bound.spec)==bound.observed
    validator.assert_called_once()
    assert validator.call_args.args[1]==Path(bound.spec['h1']['attempt'])
