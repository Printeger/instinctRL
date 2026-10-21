"""CPU phase orchestration with substituted physics/PPO/serializer boundaries."""
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
from tools.test_r6_training_plan import case,update_recipe,curriculum_fixture,metadata
from tools.test_r6_training_rng import restored,preserve_global_rng
from training_workloads.unified_h1 import robustness_training_execution as ex
from training_workloads.unified_h1.budget import ExecutionCounters
from training_workloads.unified_h1.robustness_training_storage import R6LearningLog


@pytest.fixture
def setup(case,monkeypatch,tmp_path):
    curriculum=curriculum_fixture();curriculum['phases'][0]['end_offset']=1;curriculum['phases'][1]['end_offset']=2
    case.recipe.update(branch_updates=2,checkpoint_offsets=[2],curriculum=case.pin('short.json',curriculum));update_recipe(case)
    case.plan.update(target_update=642,checkpoint_updates=[642])
    t,g,_=restored();m=metadata();m['reset_number']=19
    events=[]
    env=SimpleNamespace(device='cpu',_actions=torch.zeros(32,1),_h1_pure_hover_training=False,
        _robot=SimpleNamespace(data=SimpleNamespace(root_state_w=torch.cat((torch.zeros(32,3),torch.tensor([[1.,0,0,0]]*32),torch.zeros(32,6)),1))))
    env.set_h1_evaluation_case=lambda value:events.append(('case',value))
    env.set_h1_initial_command=lambda command,start:setattr(env,'_h1_global_policy_step',start)
    env.stage_r6_world_command=lambda command:None
    def unused(*a,**kw):raise AssertionError('CPU loop substitute must not call physics')
    env.prepare_h1_step=unused;env.step=unused
    def configure(**kw):
        events.append(('install',kw['start_policy_step'],kw['end_policy_step']))
        assert backend.journal.pool.writers['rng/initialization.pt'].complete
        env._r6_physics={'curriculum':kw['curriculum']};env._r6_measurements={}
    def finish():
        events.append(('finish',));env._r6_physics=None;env._r6_measurements=None
    env.configure_r6_training_phase=configure;env.finish_r6_training_phase=finish
    output=tmp_path/'output';output.mkdir()
    backend=SimpleNamespace(environment=env,authorization=dict(mode='research-r6-training',parameters={'r6_training_plan':case.plan},
        budgets=ex.contract.required_budget(case.plan)),trainer=t,curriculum_generator=g,output=output,
        journal=None,driver=None,loop=None,counters=ExecutionCounters(),resource_check=lambda stage,n:events.append((stage,n)))
    def reset(**kwargs):
        backend.counters.reserve_reset(automatic=False,maximum=2)
        assert backend.journal.pool.writers['rng/initialization.pt'].complete
        events.append(('reset',env._h1_global_policy_step))
        return {'policy':{'state_vec':torch.zeros(32,52)}},{}
    backend._preserving_reset=reset
    loops=[]
    class Loop:
        def __init__(self,driver,trainer,journal,writer,**kw):
            self.driver=driver;self.trainer=trainer;self.journal=journal;self.writer=writer;self.kw=kw;loops.append(self)
        def run_update(self,checkpoint_label=None):
            self.kw['before_rollout'](self.trainer.update_counter+1)
            self.kw['before_optimizer'](self.trainer.update_counter+1)
            self.trainer.update_counter+=1
            for _ in range(16):self.driver.clock.complete_step()
            events.append(('update',self.trainer.update_counter))
            if checkpoint_label:self.writer(checkpoint_label,self.trainer,self.driver)
    monkeypatch.setattr(ex,'UnifiedH1TrainingLoop',Loop)
    def save(path,trainer,**kwargs):
        events.append(('checkpoint',trainer.update_counter,kwargs));Path(path).write_bytes(b'CPU serializer substitute')
    monkeypatch.setattr('instinctlab.training.checkpoint_h1.save_checkpoint_h1',save)
    return SimpleNamespace(b=backend,p=case.plan,m=m,identity=case.identity,events=events,loops=loops)


def run(s):
    return ex.run_loaded(s.b,plan=s.p,metadata=s.m,source_identity=s.identity,
        method_hashes={'r6_recipe':s.p['recipe']['sha256']},expected_mass_kg=1.16)


def test_loaded_phase_sequence_shared_diagnostic_and_exact_checkpoint(setup):
    s=setup;run(s)
    assert [e for e in s.events if e[0]=='reset']==[('reset',640*16),('reset',641*16)]
    assert len(s.loops)==2 and s.loops[0].kw['diagnostic'] is s.loops[1].kw['diagnostic']
    assert s.loops[0].kw['diagnostic'].starting_update==640
    assert all(loop.kw['optimizer_step_origin']==(5120,5120) for loop in s.loops)
    assert isinstance(s.loops[0].driver,ex.UnifiedH1Driver) and isinstance(s.loops[1].driver,ex.R6TrainingDriver)
    ck=[e for e in s.events if e[0]=='checkpoint'];assert len(ck)==1 and ck[0][1]==642
    assert ck[0][2]['curriculum']['version']=='curriculum-h1-r6-v1'
    assert ck[0][2]['training_seed']==11 and ck[0][2]['reset_number']==21
    assert s.b.counters.real_checkpoint_writes==1 and s.b.journal.evidence_finalized
    assert s.b.journal.fixed_reset_types.count('near_x_wall')==8
    for lane in range(32):assert s.b.journal.next_reset_start_type(lane,{})==s.b.journal.fixed_reset_types[lane]


@pytest.mark.parametrize('bad',['mode','budget','hover','method'])
def test_preconditions_refuse_before_rng_reset_and_output(setup,bad):
    s=setup;before=s.b.curriculum_generator.get_state().clone()
    if bad=='mode':s.b.authorization['mode']='research-r6-evaluation'
    elif bad=='budget':s.b.authorization['budgets']['explicit_resets']=1
    elif bad=='hover':s.b.environment._h1_pure_hover_training=True
    else:
        from dataclasses import replace
        s.b.trainer.config=replace(s.b.trainer.config,minibatch_size=128)
    with pytest.raises((PermissionError,ValueError)):run(s)
    assert torch.equal(before,s.b.curriculum_generator.get_state()) and not s.events
    assert list(s.b.output.iterdir())==[]


def test_rng_evidence_failure_prevents_any_reset(setup,monkeypatch):
    s=setup
    def fail(*a,**kw):raise OSError('disk quota')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError):run(s)
    assert not s.events and s.b.counters.explicit_resets==0
    assert not s.b.journal.pool.writers['rng/initialization.pt'].complete
    s.b.journal.abort_evidence()


def test_failed_phase_does_not_advance_or_checkpoint(setup,monkeypatch):
    s=setup
    def fail(**kw):raise RuntimeError('fixture first update gate failure')
    original=ex.UnifiedH1TrainingLoop
    def loop(*a,**kw):
        result=original(*a,**kw);result.run_update=fail;return result
    monkeypatch.setattr(ex,'UnifiedH1TrainingLoop',loop)
    with pytest.raises(RuntimeError,match='gate failure'):run(s)
    assert s.b.trainer.update_counter==640 and s.b.counters.real_checkpoint_writes==0
    assert not [e for e in s.events if e[0]=='install']
    s.b.journal.abort_evidence()


def test_episode_rows_keep_sampled_reset_labels_after_automatic_reset(tmp_path):
    import json
    from training_workloads.unified_h1.lossless_evidence import EvidencePool,ChunkReader
    pool=EvidencePool(tmp_path/'pool',run_id='CPU',maximum_bytes=1024**2,maximum_raw_file_bytes=1024**2)
    labels=tuple('near_x_wall' if i==0 else 'center' for i in range(32))
    log=R6LearningLog(tmp_path/'log',pool=pool,initial_start_types=labels,fixed_reset_types=labels)
    transition=SimpleNamespace(reward=torch.zeros(32),reset_boundary=torch.ones(32,dtype=torch.bool))
    root=torch.zeros(32,13);root[:,2]=2
    for step in range(2):
        log._record_blocks_and_episodes(transition,{'pre_reset':{'root_state_w':root,
            'task':{'exact_clearance_m':torch.ones(32,1),'actual_velocity_b':torch.zeros(32,3)}},'_h1_driver':{'global_policy_step':10240+step}},
            torch.zeros(32,3),update=641,step=step)
    log.finalize_evidence()
    with ChunkReader(pool.root/'learning_log/episodes.jsonl.r6z',run_id='CPU',logical_name='learning_log/episodes.jsonl',maximum_raw_bytes=1024**2) as reader:
        rows=[json.loads(line) for line in reader.read().splitlines()]
    assert len(rows)==64
    assert all(row['start_type']==labels[row['env']] for row in rows)
