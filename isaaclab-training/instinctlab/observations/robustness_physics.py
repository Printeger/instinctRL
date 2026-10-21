"""Opt-in R6 privileged reset/disturbance wiring; never actor inputs."""
import copy
import math
import torch
from instinctlab.control.lee_velocity_controller import quaternion_to_rotation_matrix_wxyz
from instinctlab.control.lee_velocity_controller import rotate_world_to_body_wxyz
from training_workloads.unified_h1.robustness_scenarios import reset_spec,verify_reset


def distributed_pulse(masses,body_quaternions,acceleration_world,rotor_forces,rotor_torques):
    """Mass-proportional force at each body COM: zero net added COM torque.

    Isaac's public setter takes local link-frame vectors and applies them at
    body COM when positions=None. Preserve the existing rotor torque buffers.
    """
    n=masses.shape[0]
    if masses.shape!=(n,5) or body_quaternions.shape!=(n,5,4) or acceleration_world.shape!=(n,3):
        raise ValueError('R6 public plant tensor shape')
    if rotor_forces.shape!=(n,5,3) or rotor_torques.shape!=rotor_forces.shape:
        raise ValueError('R6 rotor wrench tensor shape')
    values=(masses,body_quaternions,acceleration_world,rotor_forces,rotor_torques)
    if any(not bool(torch.isfinite(x).all()) for x in values) or bool((masses<=0).any()):
        raise ValueError('R6 finite positive measured plant required')
    if any(x.device!=rotor_forces.device or x.dtype!=rotor_forces.dtype for x in values):
        raise ValueError('R6 force dtype/device mismatch')
    norms=torch.linalg.vector_norm(body_quaternions,dim=-1,keepdim=True)
    if bool(((norms-1).abs()>1e-5).any()):raise ValueError('R6 body quaternion norm')
    rotation=quaternion_to_rotation_matrix_wxyz((body_quaternions/norms).reshape(-1,4)).reshape(n,5,3,3)
    world=masses[:,:,None]*acceleration_world[:,None,:]
    local=torch.matmul(rotation.transpose(-1,-2),world.unsqueeze(-1)).squeeze(-1)
    return rotor_forces+local,rotor_torques.clone(),world,local


class R6PhysicsMixin:
    def configure_r6_physics(self, *, block, seed, split, reset_sink, force_sink,expected_mass_kg):
        if getattr(self,'_r6_physics',None) is not None:raise RuntimeError('R6 physics already active')
        if self.num_envs!=32 or getattr(self,'_h1_evaluation_case',None) is not None:
            raise ValueError('R6 requires32 lanes and separate original suite')
        if getattr(self,'_enhanced_b0_active',False):raise ValueError('R6/enhanced-B0 conflict')
        if not callable(reset_sink) or not callable(force_sink):raise TypeError('R6 audit sinks required')
        if type(block.get('steps'))is not int or block['steps']<=0:raise ValueError('finite R6 step budget required')
        acceleration=block.get('pulse_acceleration_m_s2',0.)
        if acceleration not in (0.,.5,1.) or (acceleration!=0 and block['task']!='pulse_hold'):
            raise ValueError('unregistered R6 force level')
        if type(expected_mass_kg)not in (int,float) or not 0<expected_mass_kg<100:
            raise ValueError('pinned baseline total mass required')
        specs=[reset_spec(block,seed=seed,split=split,lane=i,mount_position_m=self.cfg.ray_caster.offset.pos) for i in range(32)]
        self._r6_physics=dict(block=copy.deepcopy(block),specs=specs,reset_sink=reset_sink,force_sink=force_sink,
            pending=set(),ready=set(),ticks=0,last_sim_tick=None,failed=False,expected_mass=float(expected_mass_kg),world_command=None)

    def configure_r6_training_phase(self, *, curriculum, start_policy_step, end_policy_step,
                                    reset_sink, force_sink, measurement_sink, expected_mass_kg):
        """Install a bounded robust phase; caller must perform a fresh full reset.

        Separate measurement streams are drawn from the registered curriculum
        RNG. A resumed segment starts new physical/history/noise state and must
        retain this explicit fork, rather than claim continuous physical replay.
        """
        from training_workloads.unified_h1.robustness_curriculum import PhaseCurriculum
        if not isinstance(curriculum,PhaseCurriculum):raise TypeError('R6 phase curriculum required')
        settings=curriculum.measurement_settings(start_policy_step=start_policy_step,end_policy_step=end_policy_step)
        if start_policy_step%16 or end_policy_step%16:raise ValueError('R6 phase update boundaries required')
        if getattr(self,'_r6_measurements',None) is not None:raise RuntimeError('R6 measurements already active')
        if self._h1_global_policy_step!=start_policy_step:raise ValueError('R6 phase global clock mismatch')
        if not callable(measurement_sink):raise TypeError('R6 measurement audit required')
        # Reuse the same plant/sink/mass guards. The placeholder specifications
        # are replaced by actual sampled specifications before any root write.
        block=dict(next(iter(curriculum.value['conditions'].values())),steps=end_policy_step-start_policy_step)
        self.configure_r6_physics(block=block,seed=0,split='training',reset_sink=reset_sink,
                                  force_sink=force_sink,expected_mass_kg=expected_mass_kg)
        state=self._r6_physics
        state.update(curriculum=curriculum,start_policy_step=start_policy_step,end_policy_step=end_policy_step,
                     specs=[None]*32)
        try:
            before=curriculum.generator.get_state().clone()
            seed=int(torch.randint(0,2**31,(),generator=curriculum.generator).item())
            stream=f'training/phase-{curriculum.index}/segment-{start_policy_step}-{end_policy_step}'
            self.configure_r6_measurements(seed=seed,stream_id=stream,audit_sink=measurement_sink,**settings)
            curriculum.record('MEASUREMENT_STREAM',dict(seed=seed,stream_id=stream,rng_before=before,
                rng_after=curriculum.generator.get_state().clone(),settings=settings,
                lane_rng_states=self._r6_measurements['chain'].rng_states(),fresh_physical_reset=True))
            self.stage_r6_world_command(curriculum.world_command(start_policy_step,
                device=self._actions.device,dtype=self._actions.dtype))
        except BaseException:
            state['failed']=True
            raise

    def finish_r6_training_phase(self):
        state=getattr(self,'_r6_physics',None);measurement=getattr(self,'_r6_measurements',None)
        if state is None or 'curriculum' not in state:raise RuntimeError('R6 training phase required')
        if (measurement is None or measurement['failed'] or measurement['sequence']!=measurement['frame_limit']
                or self._h1_global_policy_step!=state['end_policy_step'] or state['failed'] or state['pending']
                or state['ticks']!=state['block']['steps']*20 or state['last_sim_tick'] is None
                or self.sim.current_time_step_index!=state['last_sim_tick']+1):
            raise RuntimeError('incomplete R6 training phase')
        try:
            state['curriculum'].record('PHASE_COMPLETE',dict(start_policy_step=state['start_policy_step'],
                end_policy_step=state['end_policy_step'],physics_ticks=state['ticks'],
                measurement_frames=measurement['sequence'],lane_rng_states=measurement['chain'].rng_states(),
                curriculum_rng_state=state['curriculum'].generator.get_state().clone(),
                evidence_status='RESET_VERIFIED_WRENCH_STAGING_AND_CLOCK_ONLY_NOT_INTEGRATION_PROOF'))
        except BaseException:
            state['failed']=True
            raise
        completion=self.finish_r6_physics()
        self.finish_r6_measurements()
        return completion

    def stage_r6_world_command(self,command):
        """Command for the next observation boundary, not another actor field."""
        state=getattr(self,'_r6_physics',None)
        if state is None or state['failed']:raise RuntimeError('active R6 block required')
        self._validate_h1_command('R6 world command',command)
        if bool((torch.linalg.vector_norm(command,dim=-1)>.300001).any()) or bool(command[:,2].ne(0).any()):
            raise ValueError('R6 low-speed horizontal command boundary')
        state['world_command']=command.detach().clone()

    def _reset_r6_physics(self,indices):
        state=getattr(self,'_r6_physics',None)
        if state is None:return False
        if state['failed']:raise RuntimeError('failed R6 physics block')
        if getattr(self,'_h1_evaluation_case',None) is not None:raise RuntimeError('original suite/R6 reset conflict')
        lanes=indices.detach().cpu().tolist()
        if not lanes:return True
        root=self._robot.data.root_state_w[indices].clone()
        specs=state['specs']
        if 'curriculum' in state:
            try:
                if state['ticks']%20:raise RuntimeError('R6 reset outside policy boundary')
                step=state['start_policy_step']+state['ticks']//20
                if self._h1_global_policy_step!=step:raise RuntimeError('R6 reset global clock mismatch')
                for lane in lanes:
                    specs[lane]=state['curriculum'].reset(lane,global_policy_step=step,
                        mount_position_m=self.cfg.ray_caster.offset.pos)
            except BaseException:
                state['failed']=True
                raise
        root[:,:3]=torch.tensor([specs[i]['root_position_local_m'] for i in lanes],device=root.device,dtype=root.dtype)+self.scene.env_origins[indices]
        root[:,3:7]=torch.tensor([specs[i]['root_quaternion_wxyz'] for i in lanes],device=root.device,dtype=root.dtype)
        root[:,7:13]=0.
        state['pending'].update(lanes);state['ready'].difference_update(lanes)
        try:
            self._robot.write_root_pose_to_sim(root[:,:7],indices)
            self._robot.write_root_velocity_to_sim(root[:,7:],indices)
        except BaseException:
            state['failed']=True
            raise
        return True

    def _policy_sensor_frame(self,ranges_m,validity,policy_time):
        state=getattr(self,'_r6_physics',None)
        if state is not None and 'curriculum' in state:
            if state['failed']:raise RuntimeError('failed R6 training phase')
            step=state['start_policy_step']+state['ticks']//20
            if state['ticks']%20 or self._h1_global_policy_step!=step:
                state['failed']=True
                raise RuntimeError('R6 observation global clock mismatch')
        if state is not None and state['pending']:
            # Called after the actual sensor producer refresh, not from planned
            # coordinates or a cached expected sensor location.
            ray=self._ray_caster.data;root=self._robot.data.root_state_w
            rotation=quaternion_to_rotation_matrix_wxyz(ray.quat_w)
            mount=torch.tensor(self.cfg.ray_caster.offset.pos,device=root.device,dtype=root.dtype)
            sensor=ray.pos_w+torch.matmul(rotation,mount)
            try:
                for lane in sorted(state['pending']):
                    actual=dict(sensor_position_local_m=(sensor[lane]-self.scene.env_origins[lane]).detach().cpu().tolist(),
                        root_position_local_m=(root[lane,:3]-self.scene.env_origins[lane]).detach().cpu().tolist(),
                        root_quaternion_wxyz=root[lane,3:7].detach().cpu().tolist(),
                        root_linear_velocity_w_m_s=root[lane,7:10].detach().cpu().tolist(),
                        root_angular_velocity_w_rad_s=root[lane,10:13].detach().cpu().tolist())
                    # Persist failures as well as successful reset observations.
                    # Float32 world origins can lose more than1e-5m on subtraction.
                    # Bound representational error explicitly, without rounding
                    # observed coordinates or altering geometric safety metrics.
                    magnitude=max(1.,float(root[lane,:3].abs().max()),float(sensor[lane].abs().max()))
                    position_atol=max(1e-5,2*torch.finfo(root.dtype).eps*magnitude)
                    state['reset_sink'](dict(lane=lane,block_tick=state['ticks'],expected=copy.deepcopy(state['specs'][lane]),actual=actual,
                        dtype=str(root.dtype),env_origin_w_m=self.scene.env_origins[lane].detach().cpu().tolist(),
                        root_position_w_m=root[lane,:3].detach().cpu().tolist(),sensor_position_w_m=sensor[lane].detach().cpu().tolist(),
                        position_atol_m=position_atol,position_tolerance_rule='max(1e-5,2*dtype_epsilon*max(1,abs(actual_world_coordinates)))'))
                    verify_reset(state['specs'][lane],actual,position_atol=position_atol)
                state['ready'].update(state['pending']);state['pending'].clear()
            except BaseException:
                state['failed']=True
                raise
        if state is not None and state['world_command'] is not None:
            # Rewards already used c_t. Convert c_(t+1) with actual t+1 attitude
            # before history.update, so both current and retained history agree.
            self._command_b.copy_(rotate_world_to_body_wxyz(self._robot.data.root_state_w[:,3:7],state['world_command']))
        try:
            return super()._policy_sensor_frame(ranges_m,validity,policy_time)
        except BaseException:
            if state is not None and 'curriculum' in state:state['failed']=True
            raise

    def _apply_external_disturbance(self):
        super()._apply_external_disturbance()
        state=getattr(self,'_r6_physics',None)
        if state is None:return
        if state['failed'] or state['ready']!=set(range(32)) or state['pending']:
            raise RuntimeError('R6 reset must be observed before physics')
        tick=state['ticks'];sim_tick=self.sim.current_time_step_index
        if 'curriculum' in state:
            measurement=getattr(self,'_r6_measurements',None)
            if (measurement is None or measurement['failed']
                    or self._h1_global_policy_step!=state['start_policy_step']+tick//20):
                state['failed']=True
                raise RuntimeError('R6 training measurement/physics clock mismatch')
        physics_dt=float(self.sim.get_physics_dt())
        if not math.isfinite(physics_dt) or abs(physics_dt-.005)>1e-12:raise RuntimeError('R6 actual physics dt differs')
        if type(sim_tick)is not int or (state['last_sim_tick'] is not None and sim_tick!=state['last_sim_tick']+1):
            raise RuntimeError('R6 actual physics clock discontinuity')
        if tick>=state['block']['steps']*20:raise RuntimeError('R6 physics tick quota')
        try:
            mass=self._robot.root_physx_view.get_masses().to(self._wrench.forces_body_n)
            if bool((mass.sum(dim=1)-state['expected_mass']).abs().gt(1e-5).any()):
                raise RuntimeError('R6 measured mass differs from pinned baseline')
            q=self._robot.data.body_link_state_w[:,:,3:7].to(mass)
            directions=torch.tensor([s['direction_world'] for s in state['specs']],device=mass.device,dtype=mass.dtype)
            active=state['block']['task']=='pulse_hold' and 1000<=tick<1040
            acceleration=directions*(state['block'].get('pulse_acceleration_m_s2',0.) if active else 0.)
            if 'curriculum' in state:
                phase_tick=(state['start_policy_step']-state['curriculum'].start)*20+tick
                acceleration=state['curriculum'].acceleration_world(phase_tick,device=mass.device,dtype=mass.dtype)
                active=bool(acceleration.ne(0).any())
            forces,torques,world,local=distributed_pulse(mass,q,acceleration,self._wrench.forces_body_n,self._wrench.torques_body_nm)
            if active:self._robot.set_external_force_and_torque(forces,torques,is_global=False)
            # Setter stages buffers; physics application is verified separately
            # from the subsequent trajectory and actual simulator tick audit.
            state['force_sink'](dict(block_tick=tick,sim_tick=sim_tick,active=active,
                physics_dt_s=physics_dt,dtype=str(mass.dtype),body_quaternion_wxyz=q.detach().clone(),
                rotor_forces_local_n=self._wrench.forces_body_n.detach().clone(),
                rotor_torques_local_nm=self._wrench.torques_body_nm.detach().clone(),
                mass_kg=mass.detach().clone(),additional_world_n=world.detach().clone(),
                additional_local_n=local.detach().clone(),staged_forces_local_n=forces.detach().clone(),
                staged_torques_local_nm=torques.detach().clone()))
        except BaseException:
            state['failed']=True
            raise
        state['last_sim_tick']=sim_tick;state['ticks']+=1

    def finish_r6_physics(self):
        state=getattr(self,'_r6_physics',None)
        if state is None or state['failed'] or state['pending'] or state['ticks']!=state['block']['steps']*20:
            raise RuntimeError('incomplete R6 physics block')
        if state['last_sim_tick'] is None or self.sim.current_time_step_index!=state['last_sim_tick']+1:
            raise RuntimeError('last R6 staged wrench has no completed physics tick')
        completion=dict(block_ticks=state['ticks'],last_staged_sim_tick=state['last_sim_tick'],
            completed_sim_tick=self.sim.current_time_step_index,physics_dt_s=float(self.sim.get_physics_dt()),
            evidence_status='STAGING_AND_CLOCK_ONLY_NOT_INTEGRATION_PROOF')
        self._r6_physics=None
        return completion
