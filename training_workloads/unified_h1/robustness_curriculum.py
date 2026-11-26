"""Registered R6 phase layout and privileged training-condition sampling.

Sampling is not evidence of physical exposure. Existing physics/measurement
adapters must verify actual resets and retain delivered frames/forces. Neither
lane IDs, sampled geometry nor this curriculum state enter actor observations.
"""
import copy
import math
import re
from .robustness import command,lane_role
from .robustness_scenarios import reset_spec


def require(value,reason):
    if not value:raise ValueError('R6_CURRICULUM_'+reason)


def validate(value,*,branch_updates):
    require(isinstance(value,dict) and set(value)=={'schema_version','conditions','phases'}
        and value['schema_version']=='r6-training-curriculum/1','FIELDS')
    conditions=value['conditions'];phases=value['phases']
    require(isinstance(conditions,dict) and 1<=len(conditions)<=16,'CONDITIONS')
    for name,c in conditions.items():
        require(isinstance(name,str) and re.fullmatch('[A-Za-z0-9_-]{1,64}',name),'CONDITION_ID')
        require(isinstance(c,dict) and set(c)=={'id','task','speed_m_s','sensor_to_wall_distance_m','pulse_acceleration_m_s2'}
            and c['id']==name and c['task'] in ('stop','reverse','wall_modes','pulse_hold'),'CONDITION')
        for field in ('speed_m_s','pulse_acceleration_m_s2'):
            require(type(c[field])in (int,float) and math.isfinite(c[field]),'SCALAR')
        require(0<c['speed_m_s']<=.3,'SPEED')
        if c['task']=='wall_modes':
            distance=c['sensor_to_wall_distance_m']
            require(type(distance)in (int,float) and math.isfinite(distance) and 1.3<=distance<=1.8,'WALL_DISTANCE')
        else:require(c['sensor_to_wall_distance_m'] is None,'NONWALL_DISTANCE')
        require(c['pulse_acceleration_m_s2'] in (0.,.5,1.)
            and (c['task']=='pulse_hold' or c['pulse_acceleration_m_s2']==0),'PULSE')
    require(isinstance(phases,list) and 2<=len(phases)<=32,'PHASES')
    previous=0;seen=set()
    for phase in phases:
        required = {'kind','end_offset','cycle_steps','lane_counts_per_half',
                    'noise_std_m','delay_cycles','drop_every_steps'}
        require(isinstance(phase,dict) and set(phase) in (required, required | {'tube_drive_steps'}),'PHASE_FIELDS')
        require(phase['kind'] in ('original','robust') and type(phase['end_offset'])is int
            and previous<phase['end_offset']<=branch_updates,'PHASE_RANGE')
        previous=phase['end_offset'];seen.add(phase['kind'])
        if phase['kind']=='original':
            require(phase['lane_counts_per_half']=={} and all(type(phase[k])is int and phase[k]==0
                for k in ('cycle_steps','noise_std_m','delay_cycles','drop_every_steps')) and
                'tube_drive_steps' not in phase,'ORIGINAL_UNIMPAIRED')
            continue
        counts=phase['lane_counts_per_half']
        require(isinstance(counts,dict) and counts and set(counts)<=set(conditions)
            and all(type(v)is int and v>0 for v in counts.values()) and sum(counts.values())==16,'LANE_COUNTS')
        require(any(conditions[k]['task']=='wall_modes' for k in counts),'NEARWALL_EXPOSURE')
        require(type(phase['cycle_steps'])is int and 192<=phase['cycle_steps']<=4096
            and phase['cycle_steps']%16==0,'CYCLE')
        if 'tube_drive_steps' in phase:
            drive_steps = phase['tube_drive_steps']
            require(type(drive_steps) is int and 100 <= drive_steps <= phase['cycle_steps']-40, 'TUBE_DRIVE_STEPS')
        require(type(phase['noise_std_m'])in (int,float) and math.isfinite(phase['noise_std_m'])
            and 0<=phase['noise_std_m']<=.03,'NOISE')
        require(type(phase['delay_cycles'])is int and phase['delay_cycles'] in (0,1),'DELAY')
        drop=phase['drop_every_steps'];require(type(drop)is int and (drop==0 or drop>=2),'DROP')
    require(previous==branch_updates and seen=={'original','robust'},'REPLAY_AND_COVERAGE')
    return copy.deepcopy(value)


def slices(value,start_update,target_update,*,branch_start_update=640):
    require(type(branch_start_update)is int and type(start_update)is int and type(target_update)is int
        and 640<=branch_start_update<=start_update<target_update,'SEGMENT')
    result=[];start=branch_start_update
    for index,phase in enumerate(value['phases']):
        end=branch_start_update+phase['end_offset'];left=max(start,start_update);right=min(end,target_update)
        if left<right:result.append(dict(index=index,start_update=left,target_update=right,phase_start_update=start,
            phase_end_update=end,kind=phase['kind']))
        start=end
    require(result and result[0]['start_update']==start_update and result[-1]['target_update']==target_update,'SEGMENT_COVERAGE')
    return result


class PhaseCurriculum:
    def __init__(self,value,*,branch_updates,phase_index,generator,audit_sink,tube_lanes=(),branch_start_update=640):
        import torch
        self.value=validate(value,branch_updates=branch_updates)
        require(type(phase_index)is int and 0<=phase_index<len(value['phases']),'PHASE_INDEX')
        self.phase=copy.deepcopy(value['phases'][phase_index]);self.index=phase_index
        require(self.phase['kind']=='robust','ORIGINAL_USES_LEGACY_DRIVER')
        require(isinstance(generator,torch.Generator) and str(generator.device)=='cpu','GENERATOR')
        require(callable(audit_sink),'AUDIT_SINK')
        self.generator=generator;self.sink=audit_sink;self.failed=False
        self.tube_lanes = set(tube_lanes)
        require('tube_drive_steps' not in self.phase or self.tube_lanes, 'TUBE_DRIVE_REQUIRES_TUBES')
        require(type(branch_start_update)is int and branch_start_update>=640,'BRANCH_START')
        self.start=(branch_start_update+(value['phases'][phase_index-1]['end_offset'] if phase_index else 0))*16
        self.end=(branch_start_update+self.phase['end_offset'])*16
        self.reset_counts=[0]*32
        labels=[k for k,n in sorted(self.phase['lane_counts_per_half'].items()) for _ in range(n)]
        self.assignments=[]
        before=generator.get_state().clone()
        for half in range(2):
            permutation=torch.randperm(16,generator=generator).tolist()
            self.assignments.extend(labels[i] for i in permutation)
        if tube_lanes:
            from .recovery_training_scene import TUBE_LANES

            require(tuple(tube_lanes) == TUBE_LANES, 'TUBE_SCENE_LANES')
            for half in range(2):
                target = [lane for lane in tube_lanes if lane//16 == half]
                relative = list(range(16*half, 16*(half+1)))
                vacant = [lane for lane in target if self.assignments[lane] != 'stop']
                donors = [lane for lane in relative if lane not in target and self.assignments[lane] == 'stop']
                require(len(vacant) <= len(donors), 'TUBE_STOP_COVERAGE')
                for lane, donor in zip(vacant, donors):
                    self.assignments[lane], self.assignments[donor] = self.assignments[donor], self.assignments[lane]
            self.record('TUBE_TRAINING_LANES', dict(lanes=list(tube_lanes), task='stop'))
        self.record('LANE_ASSIGNMENTS',dict(phase_index=phase_index,start_policy_step=self.start,end_policy_step=self.end,
            requested_counts_per_half=self.phase['lane_counts_per_half'],actual_sampled_conditions=self.assignments,
            rng_before=before,rng_after=generator.get_state().clone(),physical_exposure_verified=False))

    def record(self,event,value):
        if self.failed:raise RuntimeError('R6_CURRICULUM_AUDIT_FAILED')
        try:self.sink(event,copy.deepcopy(value))
        except BaseException:
            self.failed=True
            raise

    def require_step(self,step):
        require(not self.failed,'FAILED')
        require(type(step)is int and self.start<=step<=self.end,'STEP')

    def world_command(self,step,*,device='cpu',dtype=None):
        import torch
        self.require_step(step);local=(step-self.start)%self.phase['cycle_steps']
        tube_drive_steps = self.phase.get('tube_drive_steps', 100)
        return torch.tensor([command(self.value['conditions'][name], lane,
                                     min(local, 99) if lane in self.tube_lanes and local < tube_drive_steps else local)
            for lane,name in enumerate(self.assignments)],device=device,dtype=dtype or torch.float32)

    def reset(self,lane,*,global_policy_step,mount_position_m):
        import torch
        self.require_step(global_policy_step);require(type(lane)is int and 0<=lane<32,'LANE')
        seed=int(torch.randint(0,2**31,(),generator=self.generator).item())
        condition=self.value['conditions'][self.assignments[lane]]
        spec=reset_spec(condition,seed=seed,split='training',lane=lane,mount_position_m=mount_position_m)
        self.record('RESET_SAMPLE',dict(lane=lane,reset_ordinal=self.reset_counts[lane],condition_id=condition['id'],
            global_policy_step=global_policy_step,spec=spec,physical_exposure_verified=False))
        self.reset_counts[lane]+=1
        return copy.deepcopy(spec)

    def acceleration_world(self,physics_tick,*,device='cpu',dtype=None):
        import torch
        require(type(physics_tick)is int and 0<=physics_tick<(self.end-self.start)*20,'PHYSICS_TICK')
        require(not self.failed,'FAILED')
        active=1000<=physics_tick%(self.phase['cycle_steps']*20)<1040
        values=[]
        for lane,name in enumerate(self.assignments):
            condition=self.value['conditions'][name];_,direction=lane_role(condition,lane)
            scale=condition['pulse_acceleration_m_s2'] if active and condition['task']=='pulse_hold' else 0
            values.append([scale*x for x in direction])
        return torch.tensor(values,device=device,dtype=dtype or torch.float32)

    def measurement_settings(self,*,start_policy_step,end_policy_step):
        self.require_step(start_policy_step);self.require_step(end_policy_step)
        require(start_policy_step<end_policy_step,'MEASUREMENT_INTERVAL')
        frames=end_policy_step-start_policy_step+1;period=self.phase['drop_every_steps']
        # Frame 0 is a fresh physical reset. Other drops retain acquisition time
        # through the existing shared actor/ICS measurement delivery adapter.
        drops=[i for i in range(1,frames) if period and (start_policy_step-self.start+i)%period==0]
        return dict(sigma_m=self.phase['noise_std_m'],delay_cycles=self.phase['delay_cycles'],
                    drop_steps=drops,frame_limit=frames)
