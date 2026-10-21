"""Real lossless AuditLog decoding with synthetic 32-lane recorded poses, CPU only."""
import copy,json
import pytest
from training_workloads.unified_h1.robustness_audit import AuditLog
from tools import r6_video_trajectory as m
from tools.r6_recovery import file_pin,RecoveryError

@pytest.fixture
def trajectory(tmp_path):
    steps=200;run='fixture';block='reverse';root=[0.,0.,2.,1.,0.,0.,0.,.1,0.,0.,0.,0.,0.]
    sample=dict(sensor_position_w_m=[0.,0.,2.],velocity_w_m_s=[.1,0.,0.],episode_id=0,time_s=0.,collision=False,height_violation=False,missing=False)
    traces=[dict(lane=i,run_id=run,block_id=block,env_origin_w_m=[0.,0.,0.],initial=copy.deepcopy(sample),rows=[]) for i in range(32)]
    initial=dict(root_state_w=[root]*32,sensor_position_w_m=[[0.,0.,2.]]*32,env_origins_w_m=[[0.,0.,0.]]*32,lanes=[t['initial'] for t in traces])
    records=[('initial',initial)]
    for step in range(steps):
        attempt=dict(policy_step=step,input_measurement_sequence=step,command_w_m_s=[[.2,0.,0.]]*32,command_b_m_s=[[.2,0.,0.]]*32,attitude_wxyz=[[1.,0.,0.,0.]]*32)
        rows=[]
        for trace in traces:
            row=dict(sample,time_s=(step+1)*.1,policy_step=step,command_w_m_s=[.2,0.,0.],episode_id=int(step==steps-1),collision=step==steps-1)
            trace['rows'].append(row);rows.append(row)
        complete=dict(policy_step=step,output_measurement_sequence=step+1,lanes=rows,root_state_w=[root]*32,reset_mask=[step==steps-1]*32)
        records.extend([('step_attempt',attempt),('step_complete',complete)])
    def write(fault=None):
        path=tmp_path/'audit.bin'
        if path.exists():path.unlink()
        with AuditLog(path,run_id=run,block_id=block,maximum_bytes=4*1024**2) as log:
            for kind,payload in records:log.append(kind,payload)
        def pin(name,value):
            p=tmp_path/name;p.write_text(json.dumps(value));return file_pin(p)
        raw={'audit.bin':file_pin(path),'motion_traces.json':pin('traces.json',traces),
            'result.json':pin('summary.json',dict(run_id=run,block_id=block,seed=17,split='final',audit_records=len(records),force_records=0,reset_records=0))}
        return dict(lane=0,frame_count=steps,first_policy_step=0,last_policy_step=steps-1,run_id=run,block_id=block,assignment={'evaluation_seed':17},audit_records=len(records),raw=raw)
    return records,traces,write


def test_exact_pose_timing_and_last_step_reset_are_preserved(trajectory):
    _,_,write=trajectory;clip=write();out=m.decode(clip)
    assert len(out['frames'])==200 and not out['physics_reintegrated']
    assert out['frames'][0]['control_time_s']==0 and out['frames'][0]['state_time_s']==.1
    assert out['frames'][-1]['reset_after_state'] and out['frames'][-1]['collision']
    assert out['frames'][-1]['episode_counter_after_step']==1
    assert out['frames'][0]['quaternion_wxyz']==[1.,0.,0.,0.]


@pytest.mark.parametrize('fault',['command','quaternion','order','time','episode','reset','nonfinite','truncated','identity'])
def test_wrong_or_partial_trajectory_rejected(trajectory,fault):
    records,traces,write=trajectory
    if fault=='command':records[1][1]['command_w_m_s']=[[9.,0.,0.]]*32
    elif fault=='quaternion':records[2][1]['root_state_w']=[[0.,0.,2.,0.,0.,0.,0.,.1,0.,0.,0.,0.,0.]]*32
    elif fault=='order':records[1][1]['policy_step']=1
    elif fault=='time':traces[0]['rows'][0]['time_s']=9
    elif fault=='episode':traces[0]['rows'][-1]['episode_id']=0
    elif fault=='reset':records[-1][1]['reset_mask']=[0]*32
    elif fault=='nonfinite':records[1][1]['command_b_m_s']=[['NaN',0.,0.]]*32
    elif fault=='truncated':records.pop()
    elif fault=='identity':traces[0]['run_id']='wrong'
    with pytest.raises((RecoveryError,ValueError)):m.decode(write())
