"""Actual compressed streams and safe CPU tensor deserialization; synthetic training data."""
import json
from pathlib import Path
import pytest
import torch
from tools import r6_training_analysis as a
from tools.r6_recovery import file_pin
from tools.test_r6_training_plan import case,update_recipe,curriculum_fixture
from training_workloads.unified_h1.lossless_evidence import EvidencePool
from instinctlab.training.checkpoint_h1 import CHECKPOINT_H1_SCHEMA,CHECKPOINT_H1_VERSION


@pytest.fixture
def evidence(case,tmp_path):
    b=case;curriculum=curriculum_fixture();curriculum['phases'][0]['end_offset']=1;curriculum['phases'][1]['end_offset']=2
    b.recipe.update(branch_updates=2,checkpoint_offsets=[2],curriculum=b.pin('short.json',curriculum));update_recipe(b)
    b.plan.update(target_update=642,checkpoint_updates=[642])
    b.output=tmp_path/'attempt/results/research-r6-training';b.output.mkdir(parents=True)
    b.auth=b.pin('authorization.json',dict(mode='research-r6-training',attempt=str(b.output.parent.parent),
        parameters={'r6_training_plan':b.plan},budgets=a.contract.required_budget(b.plan)))
    def plain(relative,value):
        path=b.output/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    b.plain=plain
    b.rows={'learning_log/plan.jsonl':[dict(plan=b.plan,source_identity=b.recipe['source_identity'],physical_continuity=False)]}
    b.phases=a.slices(curriculum,640,642)
    for phase in b.phases:
        prefix=f"phases/{phase['index']}-{phase['start_update']*16}-{phase['target_update']*16}/learning_log"
        b.rows[prefix+'/phase_complete.jsonl']=[dict(phase=phase,counters={})]
        b.rows[prefix+'/updates.jsonl']=[dict(completed_updates=phase['target_update'])]
        for name,count in [('rollouts',1),('compact_rollouts',16),('minibatches',8),('accounting',1)]:b.rows[prefix+'/'+name+'.jsonl']=[{}]*count
        b.rows[prefix+'/episodes.jsonl']=[dict(env=i,start_type='center',reason='phase_end',length=16,**{'return':float(i)},right_censored=True) for i in range(32)]
        if phase['kind']=='robust':
            for name in ('curriculum','reset','force','measurement'):b.rows[prefix+'/'+name+'.jsonl']=[{}]
    b.extra={'rng/initialization.pt':b'synthetic tensor evidence',
        'pilot_diagnostic/first_update_probe.pt':b'synthetic probe',
        **{f'pilot_diagnostic/update_{n}_full_input.pt':b'synthetic input' for n in (641,642)}}
    def write_streams():
        pool=EvidencePool(b.output/'compressed_evidence',run_id=b.plan['experiment_id'],maximum_bytes=8*1024**2,maximum_raw_file_bytes=8*1024**2)
        for name,rows in b.rows.items():
            stream=pool.open(name)
            for row in rows:stream.write(json.dumps(row).encode()+b'\n')
        for name,data in b.extra.items():pool.open(name).write(data)
        pool.close()
    b.write_streams=write_streams
    plain('status.json',dict(status='PASS',counters=dict(training_transitions=1024,nontraining_transitions=0,
        completed_policy_steps=32,real_environment_transitions=1024,real_optimizer_updates=2,completed_updates=2,
        actor_optimizer_steps=16,critic_optimizer_steps=16,real_checkpoint_writes=1,explicit_resets=2)))
    plain('pilot_diagnostic/first_update_gate.json',dict(status='PASS',sample_count=512,minibatches=8,policy_rng_consumed=False,starting_update=640,first_absolute_update=641))
    b.checkpoint=b.output/'checkpoints/u642.pt';b.checkpoint.parent.mkdir()
    hashes=dict(b.plan['initialization']['method_hashes'],r6_recipe=b.plan['recipe']['sha256'],r6_method=b.recipe['method']['sha256'],curriculum='curriculum-h1-r6-v1')
    b.value=dict(version=CHECKPOINT_H1_VERSION,schema=CHECKPOINT_H1_SCHEMA,network_config={},
        ppo_config=dict(a.contract.pinned_json(b.recipe['method'])['ppo'],permutation_seed=11),method_hashes=hashes,
        source_identity=b.recipe['source_identity'],actor={'fixture':torch.ones(1)},critic={'fixture':torch.ones(1)},
        actor_optimizer={},critic_optimizer={},counters=dict(update=642,transition=642*512,global_policy_step=642*16,actor_optimizer=642*8,critic_optimizer=642*8),
        curriculum=dict(version='curriculum-h1-r6-v1',completed_policy_steps=642*16,recipe_sha256=b.plan['recipe']['sha256'],origin_checkpoint_sha256=a.contract.BASELINE_SHA,branch_seed=11),training_seed=11,reset_number=1,rng={})
    def save_checkpoint():
        torch.save(b.value,b.checkpoint);plain('checkpoints/u642.json',dict(file_pin(b.checkpoint),bytes=b.checkpoint.stat().st_size))
    b.save_checkpoint=save_checkpoint;save_checkpoint()
    return b


def run(b):return a.analyze(b.output,b.auth)


def test_complete_structural_evidence_is_not_learning_or_physics_acceptance(evidence,monkeypatch):
    b=evidence;b.write_streams()
    def forbidden(*args,**kwargs):raise AssertionError('CPU analysis must not probe CUDA')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden);monkeypatch.setattr(torch.cuda,'device_count',forbidden)
    report,summary=run(b)
    assert [i['input'] for i in report['issues']]==['first_update_probability_replay']
    assert summary['checkpoints'][0]['metadata_verified']
    assert summary['phases'][0]['episodes']['center:phase_end']['count']==32
    assert summary['phases'][0]['episodes']['center:phase_end']['return_sum']==sum(range(32))
    assert not any(summary[k] for k in ('scientific_acceptance','learning_gain_established','physical_replay_performed','strict_checkpoint_restore_performed','rng_state_semantics_verified','probability_gate_recomputed'))


@pytest.mark.parametrize('fault',['missing','truncated','wrong_run','duplicate_update','phase','gate','counter','checkpoint_clock','checkpoint_nan','sidecar','json_nan','empty_status','status_list','empty_accounting'])
def test_incomplete_or_corrupt_evidence_is_reported_without_scientific_pass(evidence,fault):
    b=evidence;name=next(n for n in b.rows if n.endswith('/updates.jsonl'))
    if fault=='missing':del b.rows[name]
    if fault=='empty_status':b.plain('status.json',{})
    if fault=='status_list':b.plain('status.json',[])
    if fault=='empty_accounting':b.rows[next(n for n in b.rows if n.endswith('/accounting.jsonl'))]=[]
    if fault=='duplicate_update':b.rows[name]*=2
    if fault=='phase':b.rows[next(n for n in b.rows if n.endswith('/phase_complete.jsonl'))][0]['phase']={}
    if fault=='json_nan':b.rows[name][0]['extra']=float('nan')
    if fault=='gate':b.plain('pilot_diagnostic/first_update_gate.json',{'status':'FAILED'})
    if fault=='counter':b.plain('status.json',{'status':'PASS','counters':{}})
    if fault=='checkpoint_clock':b.value['counters']['actor_optimizer']-=1;b.save_checkpoint()
    if fault=='checkpoint_nan':b.value['actor']['fixture'][0]=float('nan');b.save_checkpoint()
    if fault=='sidecar':b.plain('checkpoints/u642.json',{})
    b.write_streams()
    path=b.output/'compressed_evidence'/('rng/initialization.pt.r6z')
    if fault=='truncated':path.write_bytes(path.read_bytes()[:-1])
    if fault=='wrong_run':path.write_bytes(path.read_bytes().replace(b'R6-TRAINING-CPU',b'R6-TRAINING-BAD'))
    report,summary=run(b)
    assert report['issues'] and summary['issue_count'] and not summary['scientific_acceptance']


def test_empty_failed_attempt_retains_missing_evidence(evidence):
    b=evidence;b.plain('status.json',dict(status='FAILED',counters={}))
    report,summary=run(b)
    assert summary['complete_stream_count']==0 and summary['issue_count']>10
    assert any(i['reason']=='WORKLOAD_NOT_PASS' for i in report['issues'])


def test_wrong_authorization_attempt_rejected_before_read(evidence):
    b=evidence
    with pytest.raises(ValueError,match='EXACT_ATTEMPT'):a.analyze(b.output.parent,b.auth)
