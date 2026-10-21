"""CPU-only extraction of one full, hash-bound R6 recorded trajectory.

Uses the existing lossless audit reader. Pose is never integrated or inferred;
command t and pre-reset state t+1 remain separate, including terminal resets.
"""
import json
import math
from tools.r6_recovery import checked, file_pin, RecoveryError


def require(value,reason):
    if not value:raise RecoveryError('VIDEO_TRAJECTORY_'+reason)


def vector(value,n):
    require(isinstance(value,list) and len(value)==n and all(type(v) in (int,float) and math.isfinite(v) for v in value),'FINITE_VECTOR')
    return list(value)


def matrix_lane(value,lane,n):
    require(isinstance(value,list) and len(value)==32,'LANES')
    return vector(value[lane],n)


def bounded_json(pin,cap):
    path=checked(pin);require(path.stat().st_size<=cap,'JSON_CAP')
    return json.loads(path.read_text())


def decode(clip):
    from training_workloads.unified_h1 import robustness_audit as audit
    from training_workloads.unified_h1.robustness import surface_distance
    lane=clip['lane'];steps=clip['frame_count']
    require(type(lane)is int and 0<=lane<32 and type(steps)is int and steps in (200,330),'LANE_OR_STEPS')
    require(clip['first_policy_step']==0 and clip['last_policy_step']==steps-1,'FULL_CLIP')
    require(set(clip['raw'])=={'audit.bin','motion_traces.json','result.json'},'RAW_SET')
    summary=bounded_json(clip['raw']['result.json'],16*1024**2)
    require(summary['run_id']==clip['run_id'] and summary['block_id']==clip['block_id']
        and summary['seed']==clip['assignment']['evaluation_seed'] and summary['split']=='final'
        and summary['audit_records']==clip['audit_records'],'IDENTITY')
    count=clip['audit_records'];require(type(count)is int and 0<count<=64*(steps+1)+1024,'RECORD_CAP')
    traces=bounded_json(clip['raw']['motion_traces.json'],32*1024**2)
    require(isinstance(traces,list) and len(traces)==32,'TRACES')
    trace=traces[lane]
    require(trace['lane']==lane and trace['run_id']==clip['run_id'] and trace['block_id']==clip['block_id']
        and len(trace['rows'])==steps,'TRACE_IDENTITY')
    origin=vector(trace['env_origin_w_m'],3)
    path=checked(clip['raw']['audit.bin']);require(path.stat().st_size<=2*1024**3,'AUDIT_CAP')
    initial=None;pending=None;frames=[];counts={'force':0,'reset':0};episode=0
    for record in audit.read_audit(path,run_id=clip['run_id'],block_id=clip['block_id'],expected_records=count):
        kind=record['kind'];p=record['payload']
        if kind in counts:counts[kind]+=1;continue
        if kind=='initial':
            require(initial is None and not frames and pending is None,'INITIAL_ORDER')
            require(matrix_lane(p['env_origins_w_m'],lane,3)==origin and p['lanes'][lane]==trace['initial'],'INITIAL_TRACE')
            root=matrix_lane(p['root_state_w'],lane,13);sensor=matrix_lane(p['sensor_position_w_m'],lane,3)
            require(sensor==trace['initial']['sensor_position_w_m'] and root[7:10]==trace['initial']['velocity_w_m_s'],'INITIAL_STATE')
            require(abs(sum(v*v for v in root[3:7])-1)<=1e-5,'QUATERNION')
            initial=dict(root_state_w=root,sensor_position_w_m=sensor,env_origin_w_m=origin,
                geometry_distance_m=surface_distance(sensor,origin),episode_id=0)
        elif kind=='step_attempt':
            require(initial is not None and pending is None and p['policy_step']==len(frames)
                and p['input_measurement_sequence']==len(frames),'ATTEMPT_ORDER')
            attitude=matrix_lane(p['attitude_wxyz'],lane,4)
            require(abs(sum(v*v for v in attitude)-1)<=1e-5,'CONTROL_QUATERNION')
            pending=dict(command_w_m_s=matrix_lane(p['command_w_m_s'],lane,3),command_b_m_s=matrix_lane(p['command_b_m_s'],lane,3),
                control_attitude_wxyz=attitude)
        elif kind=='step_complete':
            step=len(frames)
            require(initial is not None and pending is not None and step<steps and p['policy_step']==step
                and p['output_measurement_sequence']==step+1,'COMPLETE_ORDER')
            require(isinstance(p['reset_mask'],list) and len(p['reset_mask'])==32 and type(p['reset_mask'][lane])is bool,'RESET_MASK')
            reset=p['reset_mask'][lane];sample=p['lanes'][lane]
            require(sample==trace['rows'][step] and sample['policy_step']==step
                and abs(sample['time_s']-(step+1)*.1)<1e-9,'TRACE_STEP')
            require(type(sample['collision'])is bool and type(sample['height_violation'])is bool,'FAILURE_FLAGS')
            episode+=int(reset);require(sample['episode_id']==episode,'EPISODE')
            root=matrix_lane(p['root_state_w'],lane,13);sensor=vector(sample['sensor_position_w_m'],3)
            require(root[7:10]==sample['velocity_w_m_s'] and sample['command_w_m_s']==pending['command_w_m_s'],'STATE_COMMAND')
            require(abs(sum(v*v for v in root[3:7])-1)<=1e-5,'QUATERNION')
            frames.append(dict(frame=step,policy_step=step,control_time_s=step*.1,state_time_s=sample['time_s'],
                root_state_w=root,position_local_m=[root[i]-origin[i] for i in range(3)],
                quaternion_wxyz=root[3:7],sensor_position_local_m=[sensor[i]-origin[i] for i in range(3)],
                velocity_w_m_s=root[7:10],geometry_distance_m=surface_distance(sensor,origin),
                reset_after_state=reset,episode_counter_after_step=episode,collision=sample['collision'],
                height_violation=sample['height_violation'],**pending))
            pending=None
        else:raise RecoveryError('VIDEO_TRAJECTORY_UNKNOWN_RECORD')
    require(initial is not None and pending is None and len(frames)==steps,'COMPLETE_TRAJECTORY')
    require(counts=={'force':summary['force_records'],'reset':summary['reset_records']},'PHYSICS_RECORD_COUNTS')
    for pin in clip['raw'].values():checked(pin)
    return dict(schema_version='r6-video-trajectory/1',clip=clip,initial=initial,frames=frames,
        decoder=file_pin(__file__),audit_reader=file_pin(audit.__file__),physics_reintegrated=False,
        state_semantics='RECORDED_PRE_RESET_POST_STEP_STATE',command_semantics='RECORDED_PRE_STEP_COMMAND',
        reset_semantics='DISPLAY_FAILURE_STATE_BEFORE_RESET_AND_BREAK_TRAIL_BEFORE_NEXT_FRAME',scientific_acceptance=False)
