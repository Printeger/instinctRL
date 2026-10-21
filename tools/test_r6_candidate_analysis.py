"""Synthetic raw/closed-job fixtures; selection review is explicitly substituted."""
import copy
import json
from pathlib import Path
import pytest
from tools import r6_candidate_analysis as m
from tools.r6_recovery import file_pin, RecoveryError
from training_workloads.unified_h1 import robustness_analysis as replay


@pytest.fixture
def case(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tools.test_r6_training_plan import method_fixture
    method=method_fixture();update=896;seed=11
    protocol=json.loads(Path('docs/experiments/r6-robustness-auto/TEST_PROTOCOL.json').read_text());protocol['frozen']=True
    def write(path,value):
        path=tmp_path/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(replay.encoded(value));return file_pin(path)
    pp=write('protocol.json',protocol);attempt=tmp_path/'attempt';output=attempt/'results'/m.MODE;output.mkdir(parents=True)
    comparison=dict(protocol=pp,seed=17,split='development',block_ids=[b['id'] for b in protocol['blocks']])
    value=dict(role='trained_candidate',baseline_bindings={'fixture':True},method={'fixture':'method'},comparison={'fixture':'comparison'},
        training_plan={'recipe':{'sha256':'a'*64}})
    model=dict(checkpoint={'path':'/fixture/u896.pt','sha256':'b'*64},update=update,training_seed=seed,source_identity={'fixture':True},method_hashes={'fixture':True})
    selected=dict(selection=value,model=model,method=method,comparison=comparison)
    monkeypatch.setattr(m.selection,'validate',lambda value:copy.deepcopy(selected))
    p=dict(r6=dict(comparison,run_id='CPU-FIXTURE'),evaluation_selection=value,training_seed=seed,evaluation_seed=20260909,
        baseline_bindings=value['baseline_bindings'],wall_seconds=60)
    auth=dict(mode=m.MODE,attempt=str(attempt),parameters=p,budgets=m.base.budget_from_parameters(p))
    ap=write('auth.json',auth);ident=m.identity(selected)
    binding=dict(checkpoint=model['checkpoint'],checkpoint_source_identity=model['source_identity'],method_hashes=model['method_hashes'],
        evaluation_selection_sha256=ident['selection_sha256'],comparison=ident['comparison'],method=ident['method'],
        original_evaluation_label=ident['original_label'],ics_clearance_margin_m=method['ics_clearance_margin_m'],
        physical_history_continuity_claimed=False,actor_input_extension=False,
        metadata=dict(counters=dict(update=update,transition=update*512,global_policy_step=update*16,actor_optimizer=update*8,critic_optimizer=update*8),
            training_seed=seed,curriculum=dict(version='curriculum-h1-r6-v1',completed_policy_steps=update*16,recipe_sha256='a'*64,
                origin_checkpoint_sha256=m.base.BASELINE_SHA,branch_seed=seed)),
        actual_method_binding=dict(schema_version='r6-actual-method-binding/1',method_sha256=m.base.object_sha(method),
            actual_ppo=dict(method['ppo'],permutation_seed=seed),mean_prior_weight=method['mean_prior_weight'],
            unchanged_weights_adam_moments_and_counters=True,strict_resume=True,
            actual_environment=dict(pure_hover=False,**{k:method[k] for k in ('residual_neutrality_weight','anchor_weight','ics_clearance_margin_m')},
                pipeline_margins=[method['ics_clearance_margin_m']]*32,actor_optimizer_lrs=[method['ppo']['actor_learning_rate']],
                critic_optimizer_lrs=[method['ppo']['critic_learning_rate']])) )
    binding=copy.deepcopy(binding)
    bp=write(output/'r6_model_binding.json',binding)
    return SimpleNamespace(root=tmp_path,output=output,auth=auth,ap=ap,binding=binding,bp=bp,selected=selected,write=write)


def test_missing_raw_keeps_all_blocks_and_selected_identity(case):
    report,summary=m.analyze(case.output,case.ap)
    assert len(report['blocks'])==16 and not any(r['replay_complete'] for r in report['blocks'])
    assert report['candidate']['update']==896 and not report['candidate_binding_issues']
    assert summary['scientific_acceptance']is False and summary['learning_gain_established']is False
    assert {case.ap['path'],case.bp['path']}<={p['path'] for p in report['inputs']}


@pytest.mark.parametrize('fault',['checkpoint','label','seed','method','margin','actor','counter'])
def test_binding_drift_is_reported_without_acceptance(case,fault):
    b=case.binding;m.validate_binding(b,case.selected)
    if fault=='checkpoint':b['checkpoint']['sha256']='c'*64
    if fault=='label':b['original_evaluation_label']='r6_baseline_u640_original'
    if fault=='seed':b['metadata']['training_seed']+=1
    if fault=='method':b['actual_method_binding']['method_sha256']='0'*64
    if fault=='margin':b['actual_method_binding']['actual_environment']['pipeline_margins'][0]=0
    if fault=='actor':b['actor_input_extension']=True
    if fault=='counter':b['metadata']['counters']['update']+=1
    case.write(case.output/'r6_model_binding.json',b)
    report,summary=m.analyze(case.output,case.ap)
    assert report['candidate_binding_issues'] and not summary['scientific_acceptance']


@pytest.mark.parametrize('fault',['mode','attempt','seed','budget'])
def test_wrong_authorization_never_replays(case,monkeypatch,fault):
    auth=case.auth
    if fault=='mode':auth['mode']='research-r6-evaluation'
    if fault=='attempt':auth['attempt']='/other'
    if fault=='seed':auth['parameters']['r6']['seed']+=1
    if fault=='budget':auth['budgets']['optimizer_updates']=1
    pin=case.write('changed-auth.json',auth)
    monkeypatch.setattr(replay,'analyze',lambda *a,**k:pytest.fail('replay before validation'))
    with pytest.raises(RecoveryError,match='AUTHORIZATION'):m.analyze(case.output,pin)


def test_actual_original_scoring_uses_candidate_update_and_same_160_denominator(case):
    from tools.test_r6_analysis import AnalysisTests
    b=AnalysisTests();b.setUp()
    try:
        raw=b.original_fixture();target=case.output/'evaluation_raw'/m.identity(case.selected)['original_label'];target.parent.mkdir()
        raw.rename(target)
        saved=json.loads((b.root/'r6_original_suite.json').read_text());saved['update']=896
        case.write(case.output/'r6_original_suite.json',saved)
        case.write(case.output/'r6_original_wall_motion.json',dict(per_lane=[False]*32,all32=False))
        report,summary=m.analyze(case.output,case.ap)
        assert summary['original']['replay_complete'] and summary['original']['saved_result_equal']
        assert report['original']['scoring']['update']==896 and len(report['original']['scoring']['episodes'])==160
        assert not summary['original']['all_gates_pass'] and not summary['original']['wall_motion']['all32']
    finally:b.doCleanups()


@pytest.fixture
def closed(case,monkeypatch):
    from types import SimpleNamespace
    from rlflow.store import sha
    from tools.r6_analysis_binding import bind_analysis
    b=case;key='CPU-CANDIDATE';spec=dict(id='CPU',backend='real',h1=dict(attempt=b.auth['attempt'],mode=m.MODE),
        pins=dict(authorization=b.ap),argv=['python','--authorization',b.ap['path'],'--attempt',b.auth['attempt'],'--mode',m.MODE])
    # analysis_context requires the exact Store attempts parent.
    store_root=b.root/'store';store_root.mkdir();attempt=store_root/'attempts/CPU';attempt.parent.mkdir()
    Path(b.auth['attempt']).rename(attempt);b.output=attempt/'results'/m.MODE;b.auth['attempt']=str(attempt)
    b.ap=b.write('auth.json',b.auth);spec['h1']['attempt']=str(attempt);spec['pins']['authorization']=b.ap
    spec['argv']=['python','--authorization',b.ap['path'],'--attempt',str(attempt),'--mode',m.MODE]
    sp=b.write('spec.json',spec);jid='J-'+sha(dict(id='CPU',spec=spec))[:16]
    closure=b.write('close.json',dict(event_key=key,job_id=jid,spec_sha256=sha(spec),native_status='FAILED',native_exit_code=-9))
    report,summary=m.analyze(b.output,b.ap);rp=b.write('report.json',report)
    inputs=[p for p in report['inputs'] if p['path']!=b.auth['parameters']['r6']['protocol']['path']]
    inv=b.write('inventory.json',dict(event_key=key,job_id=jid,spec=sp,closure=closure,roots={'attempt':str(attempt)},
        files=inputs,output_inventory_complete=True))
    receipts=[b.write(f'query{i}.json',{}) for i in range(2)]
    post=b.write('postrun.json',dict(job_event_key=key,job_id=jid,closure=closure,inventory=inv,postrun_complete=True,postrun_queries_used=2,receipts=receipts))
    entry=dict(identity={'job_id':jid},request={'spec':sp},closure=closure,inventory=inv,postrun=post,kind='job',purpose='development')
    store=SimpleNamespace(root=store_root,replay=lambda:dict(jobs={jid:dict(id=jid,spec=spec,status='FAILED')}))
    monkeypatch.setattr('tools.r6_analysis_binding.cleanup_evidence',lambda *a:({},dict(status='FAILED',exit_code=-9)))
    b.entry=entry;b.key=key;b.store=store;b.report=report;b.rp=rp;b.spec=spec
    b.bind=lambda:bind_analysis(entry,key,b.write('report.json',b.report),store)
    return b


def test_closed_candidate_replay_preserves_native_failure_and_missing_science(closed):
    result=closed.bind()
    assert result['native_exit_code']==-9 and not result['scientific_acceptance']
    assert result['model_binding'] and 'MODEL_BINDING_MISSING' not in result['issues']
    assert 'MOTION_EVIDENCE_INCOMPLETE_OR_MISMATCHED' in result['issues']


@pytest.mark.parametrize('fault',['selection','authorization','label','method_claim','missing_auth','missing_binding','other_input'])
def test_closed_candidate_cannot_switch_model_or_inputs(closed,fault):
    r=closed.report
    if fault=='selection':r['candidate']['selection_sha256']='0'*64
    if fault=='authorization':r['authorization']['sha256']='0'*64
    if fault=='label':r['candidate']['original_label']='r6_baseline_u640_original'
    if fault=='method_claim':r['summary']['runtime_method_execution_verified']=True
    if fault=='missing_auth':r['inputs']=[p for p in r['inputs'] if p['path']!=closed.ap['path']]
    if fault=='missing_binding':r['inputs']=[p for p in r['inputs'] if not p['path'].endswith('/r6_model_binding.json')]
    if fault=='other_input':r['inputs'].append(replay.pin(closed.write('unrelated.json',{})['path']))
    with pytest.raises((RecoveryError,ValueError)):closed.bind()

