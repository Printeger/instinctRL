"""Opt-in R6 measurement wiring; no simulator imports or privileged inputs.

The workload installs one finite block and resets before taking observations.
Only ranges, validity and sensor time pass to history; clean frames go to the
explicit audit sink. ICS continues to extract the same history as the actor.
"""
import torch
from .actor import RAY_COUNT
from .robustness import RangeDelivery


class R6MeasurementMixin:
    def configure_r6_measurements(self, *, seed, stream_id, sigma_m, delay_cycles,
                                  drop_steps, frame_limit, audit_sink):
        if getattr(self,'_r6_measurements',None) is not None:
            raise RuntimeError('finish the existing R6 measurement block first')
        if getattr(self,'_h1_evaluation_case',None) is not None:
            raise ValueError('original suite cannot use R6 impairment')
        if type(frame_limit)is not int or frame_limit<=0:
            raise ValueError('finite frame limit required, including reset observation')
        if not isinstance(drop_steps,(list,tuple)) or any(type(s)is not int or not 0<=s<frame_limit for s in drop_steps):
            raise ValueError('drop indices must refer to registered block observations')
        if len(set(drop_steps))!=len(drop_steps) or any(s+1 in drop_steps for s in drop_steps):
            raise ValueError('only distinct isolated frame drops are registered')
        if not callable(audit_sink):raise TypeError('durable measurement audit sink required')
        chain=RangeDelivery(lanes=self.num_envs,rays=RAY_COUNT,seed=seed,stream_id=stream_id,
            sigma_m=sigma_m,delay_cycles=delay_cycles,device=self._actions.device,dtype=self._actions.dtype)
        self._r6_measurements=dict(chain=chain,drop_steps=frozenset(drop_steps),frame_limit=frame_limit,
            sequence=0,sink=audit_sink,reset_lanes=set(),failed=False)

    def finish_r6_measurements(self):
        block=getattr(self,'_r6_measurements',None)
        if block is None:raise RuntimeError('no active R6 block')
        if block['failed'] or block['sequence']!=block['frame_limit']:
            raise RuntimeError('incomplete R6 measurement block must be preserved')
        self._r6_measurements=None

    def _reset_policy_sensor_frame(self,indices):
        super()._reset_policy_sensor_frame(indices)
        block=getattr(self,'_r6_measurements',None)
        if block is not None:
            block['chain'].reset(indices)
            block['reset_lanes'].update(indices.detach().cpu().tolist())

    def _policy_sensor_frame(self,ranges_m,validity,policy_time):
        block=getattr(self,'_r6_measurements',None)
        if block is None:return super()._policy_sensor_frame(ranges_m,validity,policy_time)
        if getattr(self,'_h1_evaluation_case',None) is not None:
            raise RuntimeError('R6 block cannot contaminate original suite')
        if block['failed']:raise RuntimeError('failed measurement block cannot continue')
        if block['reset_lanes']!=set(range(self.num_envs)):
            raise RuntimeError('reset all lanes after installing R6 measurement block')
        if block['sequence']>=block['frame_limit']:raise RuntimeError('R6 measurement frame quota')
        drop=torch.full((self.num_envs,1),block['sequence'] in block['drop_steps'],
                        device=policy_time.device,dtype=torch.bool)
        try:
            audit=block['chain'].deliver(ranges_m,validity,policy_time,drop)
            # Sink cannot modify the measurement passed into actor/ICS history.
            frame=tuple(audit[k].clone() for k in ('delivered_ranges','delivered_validity','acquisition_time'))
            block['sink'](block['sequence'],audit)
        except BaseException:
            block['failed']=True
            raise
        block['sequence']+=1
        return frame
