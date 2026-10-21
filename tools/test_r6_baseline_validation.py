"""Pinned complete score fixtures; parent, native binding and frozen config are explicit substitutes."""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools import r6_baseline_validation as v
from tools import r6_final_models as models
from tools import r6_controller as controller
from tools import r6_final_freeze_review as review
from tools.test_r6_profile_preparation import case
from tools.test_r6_final_freeze_review import SCRIPT
from tools.r6_batch import model_handoff
from tools.r6_holdout import verify_model_review
from tools.r6_recovery import RecoveryError,file_pin


def report_fixture(protocol):
    rows=[]
    for block in protocol['blocks']:
        passed=block['domain']=='target'
        lanes=[dict(lane=i,subgroup=v.lane_role(block,i)[0],motion_pass=passed) for i in range(32)]
        groups={name:dict(total=4,successes=4 if passed else 0) for name in {r['subgroup'] for r in lanes}}
        score=dict(block_id=block['id'],domain=block['domain'],total=32,successes=32 if passed else 0,
            motion_pass=passed,lanes=lanes,subgroups=groups)
        rows.append(dict(block_id=block['id'],domain=block['domain'],total=32,replay_complete=True,
            saved_result_equal=True,measurement_replay={'bit_exact_delivery':True},
            physics_replay={'staging_reconstructed':True},scoring=score))
    return dict(split='development',scientific_acceptance=False,blocks=rows,summary=dict(original=dict(
        replay_complete=True,saved_result_equal=True,saved_wall_equal=True,all_gates_pass=True,
        wall_motion=dict(all32=True,per_lane=[True]*32))))


@pytest.fixture
def ready(case,monkeypatch):
    b=case
    b.protocol=json.loads(Path('docs/experiments/r6-robustness-auto/freeze-candidate-01/protocol.json').read_text())
    b.split=dict(schema_version='r6-split-manifest/1',final={'seed_realization_generated':False},development={'seed':17})
    b.protocol['split_manifest']=b.write('baseline-split.json',b.split)
    b.protocol_pin=b.write('baseline-protocol.json',b.protocol)
    b.protocol_review=b.write('baseline-protocol-review.json',dict(decision='PASS_R6_PROTOCOL',reviewer='fixture',must_fix=[],
        protocol_sha256=b.protocol_pin['sha256'],evidence_pins=[b.protocol_pin,b.protocol['split_manifest']]))
    b.baseline=dict(selected_model=b.write('u640.pt',{'fixture':'baseline'}),source_identity={'fixture':'source'},method_hashes={'fixture':'method'})
    b.parameters=dict(baseline_bindings=b.context,protocol_review=b.protocol_review,
        r6=dict(protocol=b.protocol_pin,split='development',seed=17))
    b.report=report_fixture(b.protocol)
    snapshot=b.write('baseline-snapshot.json',dict(config=b.context,snapshot='/frozen-baseline-fixture'))
    context=b.write('baseline-context.json',dict(snapshot_receipt=snapshot))
    b.spec=dict(h1=dict(mode='research-r6-evaluation',workload_root='/frozen-baseline-fixture',config_sha256=b.context['sha256']))
    b.auth=dict(mode='research-r6-evaluation',parameters=b.parameters)
    b.binding=dict(checkpoint=b.baseline['selected_model'],checkpoint_source_identity=b.baseline['source_identity'],
        method_hashes=b.baseline['method_hashes'],ics_clearance_margin_m=.15,actor_input_extension=False)
    b.bound=dict(schema_version='r6-bound-motion-analysis/1',issues=[],protocol=b.protocol_pin,
        native_status='FAILED',native_exit_code=-9)
    def refresh():
        b.bound.update(report=b.write('baseline-report.json',b.report),spec=b.write('baseline-spec.json',b.spec),
            authorization=b.write('baseline-auth.json',b.auth),model_binding=b.write('baseline-model-binding.json',b.binding))
        b.analysis=b.write('baseline-bound.json',b.bound)
    b.refresh=refresh;refresh()
    with b.j.writer():
        initial=b.write('baseline-initial.json',dict(plan_pin=b.context,master_pin=b.context,review_executable=file_pin(b.builder)))
        request=b.write('baseline-controller.json',dict(initial_request=initial,profile_context=b.context))
        b.controller,_=controller.register(b.j,request,b.store,free_disk_bytes=500*1024**3)
        state=b.j.read();state['events'][b.controller]['cursor']=dict(phase='FINAL_MODEL_PREPARATION',
            baseline_validation=dict(parent_key=b.parent,job_key=b.key,context=context));b.j.save(state)
    monkeypatch.setattr(v,'parent_decision',Mock(return_value=dict(decision={'decision':'PROPOSE_BASELINE_VALIDATION'},job_event_key=b.key)))
    monkeypatch.setattr(v,'bind_analysis',lambda *a:copy.deepcopy(b.bound))
    monkeypatch.setattr(v.base,'baseline',lambda p:copy.deepcopy(b.baseline))
    monkeypatch.setattr(v,'verify_frozen',Mock())
    monkeypatch.setattr(b.j,'canonical_job',lambda key:{'analysis':b.analysis})
    return b


def run(b):
    with b.j.writer():return models.prepare(b.j,b.controller,b.store,free_disk_bytes=500*1024**3)


def test_passing_baseline_freezes_zero_training_and_all_three_final_streams(ready):
    b=ready;result=run(b);budget=b.j.read()['budget'];assert run(b)==result and b.j.read()['budget']==budget
    freeze=json.loads(v.checked(result['freeze']).read_text())
    assert freeze['branch']=='BASELINE_SUFFICIENT' and freeze['models']=={'baseline':b.baseline['selected_model']}
    assert freeze['selection_rule']['final_realization_indices']==[0,1,2]
    assert freeze['selection_rule']['training_transitions']==0 and not result['scientific_acceptance']
    assert freeze['evidence']['native_status']=='FAILED'
    assert any(r['scoring']['successes']==0 for r in b.report['blocks'])
    with b.j.writer():assert models.validate(b.j,freeze,b.store)==freeze


@pytest.mark.parametrize('fault',['target','lane','subgroup','domain','denominator','motion_replay','measurement','physics','original','wall','missing_block','duplicate'])
def test_all_target_lanes_subgroups_and_full_stress_evidence_required(ready,fault):
    b=ready;r=b.report;row=r['blocks'][0]
    if fault=='target':
        row['scoring']['lanes'][0]['motion_pass']=False;row['scoring']['successes']=31;row['scoring']['motion_pass']=False
        row['scoring']['subgroups'][row['scoring']['lanes'][0]['subgroup']]['successes']=3
    if fault=='lane':row['scoring']['lanes'].pop()
    if fault=='subgroup':row['scoring']['subgroups'].pop(next(iter(row['scoring']['subgroups'])))
    if fault=='domain':row['domain']='stress'
    if fault=='denominator':r['blocks'][-1]['scoring']['total']=31
    if fault=='motion_replay':r['blocks'][-1]['replay_complete']=False
    if fault=='measurement':row['measurement_replay']['bit_exact_delivery']=False
    if fault=='physics':row['physics_replay']['staging_reconstructed']=False
    if fault=='original':r['summary']['original']['all_gates_pass']=False
    if fault=='wall':r['summary']['original']['wall_motion']['per_lane'][31]=False
    if fault=='missing_block':r['blocks'].pop()
    if fault=='duplicate':r['blocks'][1]=r['blocks'][0]
    b.refresh()
    with pytest.raises(RecoveryError):run(b)
    assert not list((b.j.directory/'final-models').rglob('freeze.json'))


@pytest.mark.parametrize('fault',['candidate','ics','binding','stream','revealed','review','snapshot','training'])
def test_changed_policy_or_contaminated_split_cannot_use_baseline_branch(ready,fault):
    b=ready
    if fault=='candidate':b.auth['mode']='research-r6-candidate-evaluation'
    if fault=='ics':b.binding['ics_clearance_margin_m']=.2
    if fault=='binding':b.bound['issues']=['MISSING']
    if fault=='stream':b.parameters['r6']['seed']=18
    if fault=='revealed':
        b.split['final']['seed_realization_generated']=True
        b.protocol['split_manifest']=b.write('revealed-split.json',b.split)
        b.protocol_pin=b.write('revealed-protocol.json',b.protocol);b.parameters['r6']['protocol']=b.protocol_pin;b.bound['protocol']=b.protocol_pin
        b.parameters['protocol_review']=b.write('revealed-review.json',dict(decision='PASS_R6_PROTOCOL',reviewer='fixture',must_fix=[],protocol_sha256=b.protocol_pin['sha256'],evidence_pins=[b.protocol_pin,b.protocol['split_manifest']]))
    if fault=='review':b.parameters['protocol_review']=b.context
    if fault=='snapshot':b.spec['h1']['config_sha256']='0'*64
    if fault=='training':
        with b.j.writer():
            s=b.j.read();s['budget']['used']['training_transitions']=512;b.j.save(s)
    b.refresh()
    with pytest.raises((RecoveryError,PermissionError)):run(b)


def test_baseline_branch_receives_registered_review_without_training(ready,monkeypatch):
    b=ready;freeze=run(b)['freeze'];b.builder.write_text('#!'+sys.executable+'\n'+SCRIPT)
    monkeypatch.setattr(review,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    with b.j.writer():out=review.prepare(b.j,freeze,b.store,executable=file_pin(b.builder),free_disk_bytes=500*1024**3)
    assert verify_model_review(b.j,out['review'])==out['event_key']
    assert not out['holdout_released'] and not out['scientific_acceptance']
