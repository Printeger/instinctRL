"""CPU size samples for the real control/reward payloads in R6 step audit.

Synthetic measurements and states, no physics or learned policy execution. This
fills the step-record gap; sampled maxima are estimates, never storage bounds.
"""
from dataclasses import asdict,replace
import json
from pathlib import Path
import tempfile
import time
import torch

from instinctlab.observations.actor import ActorObservationHistory,RAY_COUNT
from instinctlab.control.pipeline import ControlCorePipeline,ControlCoreConfig
from instinctlab.task.rewards_h1 import compute_reward_h1
from training_workloads.unified_h1.budget import ExecutionCounters
from training_workloads.unified_h1.robustness_evaluation import lane_sample
from training_workloads.unified_h1.robustness_audit import AuditLog,read_audit
from training_workloads.unified_h1.logging import _jsonable
from training_workloads.unified_h1.robustness_analysis import pin

ROOT=Path(__file__).resolve().parents[1]


def run():
    generator=torch.Generator(device='cpu').manual_seed(2026102003)
    history=ActorObservationHistory();base=ControlCoreConfig()
    control=ControlCorePipeline(replace(base,ics=replace(base.ics,clearance_margin_m=.15)))
    counters=ExecutionCounters();samples=[];payloads=[]
    previous=torch.zeros(32,3);root=torch.zeros(32,13);root[:,3]=1;root[:,2]=2
    masks={k:torch.zeros(32,1,dtype=torch.bool) for k in ('collision','below_height','above_height')}
    with tempfile.TemporaryDirectory(prefix='r6-step-capacity-') as directory:
        path=Path(directory)/'audit.bin'
        with AuditLog(path,run_id='CPU-SIZE',block_id='synthetic',maximum_bytes=16*1024**2) as log:
            for step in range(8):
                command=torch.rand(32,3,generator=generator)*.4-.2;command[:,2]=0
                ranges=torch.rand(32,RAY_COUNT,generator=generator)*5.2+.8
                observation=history.update(ranges_m=ranges,validity=torch.ones(32,RAY_COUNT,dtype=torch.bool),
                    sensor_frame_time_s=torch.full((32,1),step*.1),policy_time_s=torch.full((32,1),step*.1),
                    body_angular_velocity_rad_s=torch.rand(32,3,generator=generator)*.01,
                    gravity_direction_body=torch.tensor([[0.,0.,-1.]]*32),
                    commanded_body_velocity_m_s=command,previous_issued_body_velocity_m_s=previous)
                action=torch.rand(32,4,generator=generator);reset=torch.full((32,),step==0,dtype=torch.bool)
                result=control(observation['lidar_grid'],observation['state_vec'],action,.1,reset,reset,root[:,3:7])
                velocity=torch.rand(32,3,generator=generator)*.3-.15
                reward=compute_reward_h1(lidar_grid=observation['lidar_grid'],command_b=command,
                    v_gov_b=result.v_gov_b,v_final_b=result.v_final_b,previous_issued_v_final_b=previous,
                    actual_velocity_b=velocity,anchor_loss=result.anchor_metrics['huber_loss'],
                    anchor_active=result.anchor_metrics['active'],anchor_usable_fraction=result.anchor_metrics['usable_fraction'],
                    ics_beta=result.ics_metrics['beta'],exact_clearance_m=torch.full((32,1),2.),
                    v_corr_b=result.v_corr_b,residual_neutrality_weight=.05,anchor_weight=4.,**masks)
                previous=result.v_final_b.clone();counters.attempt_step()
                attempt=dict(policy_step=step,counters=counters.snapshot(),command_w_m_s=command,
                    command_b_m_s=command,attitude_wxyz=root[:,3:7],action=action,input_measurement_sequence=step)
                root[:,7:10]=velocity;root[:,:3]+=velocity*.1;counters.complete_step(32,kind='nontraining')
                sensor=root[:,:3]+torch.tensor([0.,0.,.05])
                lanes=[lane_sample(sensor,root,masks,i,0,time_s=(step+1)*.1) for i in range(32)]
                for lane,row in enumerate(lanes):row.update(policy_step=step,command_w_m_s=command[lane].tolist())
                complete=dict(policy_step=step,counters=counters.snapshot(),lanes=lanes,root_state_w=root,
                    reset_mask=torch.zeros(32,dtype=torch.bool),control=asdict(result),reward_diagnostics=reward.diagnostics,
                    output_measurement_sequence=step+1,
                    timing='action_t/command_t before physics; root, source and reward diagnostics are pre-reset t+1; delivery archive t+1 may reflect reset')
                for kind,payload in [('step_attempt',attempt),('step_complete',complete)]:
                    payloads.append(_jsonable(payload));before=log.written;start=time.monotonic();log.append(kind,payload)
                    samples.append(dict(kind=kind,step=step,bytes=log.written-before,seconds=time.monotonic()-start))
            count=log.records
        decoded=list(read_audit(path,run_id='CPU-SIZE',block_id='synthetic',expected_records=count))
        if [r['payload'] for r in decoded]!=payloads:raise AssertionError('step roundtrip')
    protocol_path=ROOT/'docs/experiments/r6-robustness-auto/TEST_PROTOCOL.json'
    protocol=json.loads(protocol_path.read_text());steps=sum(b['steps'] for b in protocol['blocks'])
    per_step=sum(max(s['bytes'] for s in samples if s['kind']==kind) for kind in ('step_attempt','step_complete'))
    return dict(status='CPU_FIXTURE_NOT_CAPACITY_ACCEPTANCE',capacity_pass=False,seed=2026102003,
        samples=samples,exact_json_roundtrip=True,policy_steps_per_model=steps,
        sampled_pair_bytes=per_step,per_model_step_audit_bytes=steps*per_step,final_four_step_audit_bytes=4*steps*per_step,
        protocol=pin(protocol_path),code=[pin(ROOT/p) for p in ['tools/measure_r6_step_capacity.py',
            'training_workloads/unified_h1/robustness_evaluation.py','training_workloads/unified_h1/robustness_audit.py',
            'isaaclab-training/instinctlab/control/pipeline.py','isaaclab-training/instinctlab/task/rewards_h1.py']],
        limitations='Actual CPU control/reward functions on full21240beam synthetic history. No simulator or policy model. Samples do not bound all numeric values/compression or reset events. Excludes range, force, reset, motion, original suite, training, native, snapshot, video and metadata.',
        gpu_queries=0,simulator_initializations=0)


if __name__=='__main__':print(json.dumps(run(),indent=2))
