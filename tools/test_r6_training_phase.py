"""Actual adapter composition with CPU plant fixtures, no Isaac integration."""
from types import SimpleNamespace
import pytest
import torch
from test_r6_physics_adapter import Env as PlantEnv
from test_r6_curriculum import sampler
from instinctlab.observations.actor import RAY_COUNT
from instinctlab.observations.robustness_physics import R6PhysicsMixin
from instinctlab.observations.robustness_adapter import R6MeasurementMixin


class Base:
    def _apply_external_disturbance(self):pass
    def _reset_policy_sensor_frame(self,indices):pass
    def _policy_sensor_frame(self,ranges,valid,time):return ranges,valid,time


class Env(R6PhysicsMixin,R6MeasurementMixin,Base):
    set_wrench=PlantEnv.set_wrench
    tick=PlantEnv.tick
    def __init__(self,seed=17):
        PlantEnv.__init__(self,configure=False)
        self._actions=torch.zeros(32,1);self._command_b=torch.zeros(32,3)
        self._validate_h1_command=lambda name,value:None
        self.curriculum,self.events=sampler(seed)
        self.start=self.curriculum.start+48;self.end=self.start+16
        self._h1_global_policy_step=self.start
        self.measurements=[];self.times=torch.zeros(32,1)
    def install(self):
        self.configure_r6_training_phase(curriculum=self.curriculum,start_policy_step=self.start,end_policy_step=self.end,
            reset_sink=self.reset_rows.append,force_sink=self.force_rows.append,
            measurement_sink=lambda sequence,frame:self.measurements.append((sequence,frame)),expected_mass_kg=1.16)
    def reset(self,indices=None):
        if indices is None:indices=torch.arange(32)
        self.times[indices]=0
        self._reset_policy_sensor_frame(indices);self._reset_r6_physics(indices)
    def observe(self):
        root=self._robot.data.root_state_w
        self._ray_caster=SimpleNamespace(data=SimpleNamespace(pos_w=root[:,:3].clone(),quat_w=root[:,3:7].clone()))
        return self._policy_sensor_frame(torch.full((32,RAY_COUNT),2.),torch.ones(32,RAY_COUNT,dtype=torch.bool),self.times.clone())
    def step(self,reset=None):
        self.stage_r6_world_command(self.curriculum.world_command(self._h1_global_policy_step+1))
        for _ in range(20):self.tick()
        self.times+=.1;self._h1_global_policy_step+=1
        if reset is not None:self.reset(reset)
        return self.observe()


def test_real_adapter_composition_resume_pulse_reset_noise_and_completion():
    e=Env();e.install();e.reset();e.observe()
    assert len(e.reset_rows)==32
    old=e._r6_physics['specs'][0]['seed']
    for step in range(16):e.step(torch.tensor([0]) if step==2 else None)
    assert e._r6_physics['specs'][0]['seed']!=old
    assert len(e.reset_rows)==33 and len(e.measurements)==17
    active=[x for x in e.force_rows if x['active']]
    assert [x['block_tick'] for x in active]==list(range(40,80))
    for row in active:
        expected=e.curriculum.acceleration_world(1000)*1.16
        assert torch.allclose(row['additional_world_n'].sum(dim=1),expected,atol=1e-6)
        assert torch.equal(row['staged_torques_local_nm'],e._wrench.torques_body_nm)
    assert not e.measurements[3][1]['frame_available'][0]
    assert e.measurements[3][1]['frame_available'][1]
    noisy=e.measurements[2][1]
    assert noisy['frame_available'].all()
    assert 0<float((noisy['delivered_ranges']-2.).abs().max())<=.030001
    assert torch.equal(e.staged[0],e._wrench.forces_body_n)
    result=e.finish_r6_training_phase()
    assert result['block_ticks']==320 and e._r6_measurements is None and e._r6_physics is None
    assert e.events[-1][0]=='PHASE_COMPLETE'


def test_actual_noise_replay_and_new_streams():
    outputs=[]
    for seed in (17,17,18):
        e=Env(seed);e.install();e.reset();e.observe();e.step();frame=e.step()
        assert frame[1].all()
        outputs.append(frame[0])
    assert torch.equal(outputs[0],outputs[1])
    assert not torch.equal(outputs[0],outputs[2])


def test_install_and_finish_boundaries_preserve_incomplete_evidence():
    e=Env();e._h1_global_policy_step+=1
    with pytest.raises(ValueError,match='global clock'):e.install()
    e=Env();e.install()
    with pytest.raises(RuntimeError):e.install()
    e.reset();e.observe();e.step()
    with pytest.raises(RuntimeError,match='incomplete'):e.finish_r6_training_phase()
    assert e._r6_physics is not None and e._r6_measurements is not None
    e._h1_global_policy_step+=1
    with pytest.raises(RuntimeError,match='clock mismatch'):e.observe()
    assert e._r6_physics['failed']


def test_sample_sink_failure_prevents_root_write_and_further_physics():
    e=Env();e.install();before=e._robot.data.root_state_w.clone()
    def fail(event,value):raise OSError('quota')
    e.curriculum.sink=fail
    with pytest.raises(OSError):e.reset()
    assert torch.equal(before,e._robot.data.root_state_w)
    assert e._r6_physics['failed']
    with pytest.raises(RuntimeError):e.tick()


def test_actual_geometry_mismatch_is_retained_and_stops_phase():
    e=Env();e.install();e.reset()
    root=e._robot.data.root_state_w
    e._ray_caster=SimpleNamespace(data=SimpleNamespace(pos_w=root[:,:3].clone()+.1,quat_w=root[:,3:7].clone()))
    with pytest.raises(ValueError,match='RESET_POSITION_MISMATCH'):
        e._policy_sensor_frame(torch.full((32,RAY_COUNT),2.),torch.ones(32,RAY_COUNT,dtype=torch.bool),e.times)
    assert len(e.reset_rows)==1 and e._r6_physics['failed']


def test_measurement_audit_failure_poison_blocks_physics_and_finish():
    e=Env();e.install();e.reset()
    def fail(sequence,frame):raise OSError('measurement quota')
    e._r6_measurements['sink']=fail
    with pytest.raises(OSError):e.observe()
    assert e._r6_physics['failed'] and e._r6_measurements['failed']
    with pytest.raises(RuntimeError):e.tick()
    with pytest.raises(RuntimeError):e.finish_r6_training_phase()


def test_completion_audit_failure_preserves_both_adapters():
    e=Env();e.install();e.reset();e.observe()
    for _ in range(16):e.step()
    def fail(event,value):raise OSError('final audit quota')
    e.curriculum.sink=fail
    with pytest.raises(OSError):e.finish_r6_training_phase()
    assert e._r6_physics['failed'] and e._r6_measurements is not None


def test_no_physics_with_wrong_global_policy_clock():
    e=Env();e.install();e.reset();e.observe();e._h1_global_policy_step+=1
    with pytest.raises(RuntimeError,match='clock mismatch'):e.tick()
    assert e._r6_physics['ticks']==0 and e._r6_physics['failed']
