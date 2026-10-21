"""CPU curriculum sampling, not simulated physical exposure."""
from collections import Counter
import pytest
import torch
from test_r6_training_plan import curriculum_fixture,case,update_recipe
from training_workloads.unified_h1 import robustness_curriculum as c
from training_workloads.unified_h1 import robustness_training_plan as plan


def sampler(seed=17,sink=None):
    events=[]
    obj=c.PhaseCurriculum(curriculum_fixture(),branch_updates=1024,phase_index=1,
        generator=torch.Generator().manual_seed(seed),audit_sink=sink or (lambda e,v:events.append((e,v))))
    return obj,events


def test_real_stream_replay_distinct_streams_and_stratified_counts():
    a,events=sampler();b,_=sampler();d,_=sampler(18)
    assert a.assignments==b.assignments and a.assignments!=d.assignments
    for half in (0,16):assert Counter(a.assignments[half:half+16])=={'wall':8,'pulse':8}
    assert not torch.equal(events[0][1]['rng_before'],events[0][1]['rng_after'])
    assert events[0][1]['physical_exposure_verified'] is False
    args=dict(global_policy_step=a.start,mount_position_m=[0,0,0])
    first=a.reset(0,**args);assert first==b.reset(0,**args)
    second=a.reset(0,**args);assert second['seed']!=first['seed']
    assert second==b.reset(0,**args)
    first['seed']=-1
    assert events[1][1]['spec']['seed']>=0


def test_world_commands_and_repeated_pulse_boundaries():
    a,_=sampler()
    world=a.world_command(a.start)
    assert world.shape==(32,3) and torch.all(world[:,2]==0)
    assert torch.all(torch.linalg.vector_norm(world,dim=1)<=.200001)
    for tick in (999,1040,3839):assert torch.count_nonzero(a.acceleration_world(tick))==0
    for tick in (1000,1039,3840+1000):
        force=a.acceleration_world(tick)
        for lane,label in enumerate(a.assignments):
            assert torch.linalg.vector_norm(force[lane]).item()==pytest.approx(.5 if label=='pulse' else 0,abs=1e-7)
    with pytest.raises(ValueError):a.world_command(a.end+1)
    with pytest.raises(ValueError):a.acceleration_world((a.end-a.start)*20)


def test_measurement_resume_alignment_and_reset_frame():
    a,_=sampler();settings=a.measurement_settings(start_policy_step=a.start+7,end_policy_step=a.start+28)
    assert settings==dict(sigma_m=.01,delay_cycles=1,drop_steps=[7,14,21],frame_limit=22)
    assert 0 not in settings['drop_steps']


def test_audit_mutation_isolated_and_failure_blocks_sampling():
    def mutate(event,value):value.clear()
    a,_=sampler(sink=mutate);assert len(a.assignments)==32
    def fail(event,value):raise OSError('disk full')
    a.sink=fail
    with pytest.raises(OSError):a.reset(0,global_policy_step=a.start,mount_position_m=[0,0,0])
    assert a.reset_counts[0]==0
    with pytest.raises(ValueError):a.world_command(a.start)
    with pytest.raises(ValueError):a.acceleration_world(1000)


@pytest.mark.parametrize('fault',['coverage','wall','counts','noise','drop','original','cycle','pulse'])
def test_invalid_curriculum_rejected(fault):
    v=curriculum_fixture();phase=v['phases'][1]
    if fault=='coverage':phase['end_offset']=1023
    elif fault=='wall':phase['lane_counts_per_half']={'pulse':16}
    elif fault=='counts':phase['lane_counts_per_half']['wall']=7
    elif fault=='noise':phase['noise_std_m']=float('nan')
    elif fault=='drop':phase['drop_every_steps']=1
    elif fault=='original':v['phases'][0]['delay_cycles']=1
    elif fault=='cycle':phase['cycle_steps']=191
    else:v['conditions']['wall']['pulse_acceleration_m_s2']=.5
    with pytest.raises(ValueError):c.validate(v,branch_updates=1024)


def test_phase_slices_and_reset_budget(case):
    v=curriculum_fixture()
    assert [x['kind'] for x in c.slices(v,640,896)]==['original','robust']
    assert len(c.slices(v,896,1152))==1
    assert plan.required_budget(case.plan)['explicit_resets']==2
    with pytest.raises(PermissionError,match='PHYSICS_DECIMATION'):plan.expanded_budget(case.plan,physics_decimation=4)
    v['phases'][1]['end_offset']=1000
    case.recipe['curriculum']=case.pin('bad-curriculum.json',v);update_recipe(case)
    with pytest.raises(ValueError,match='COVERAGE'):plan.required_budget(case.plan)
