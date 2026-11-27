"""Apply the frozen option to motor requests while leaving PPO samples intact."""

from dataclasses import replace
import math

import torch

from instinctlab.control.escape_option import EscapeOption, METHOD_VERSION
from instinctlab.control.escape_recovery import (
    ALIGNED_INTERCEPT_25_VERSION, BRAKE_FORWARD_25_VERSION, BRAKE_TUBE_25_VERSION, BRAKE_TUBE_CONFIRM_25_VERSION,
    CLEAR_PASS_25_VERSION, COMMAND_HANDOFF_25_VERSION, COMMITTED_INTERCEPT_25_VERSION, CONFIRMED_STANDOFF_25_VERSION,
    CURVED_PASS_25_VERSION,
    EARLY_RETREAT_25_VERSION, EscapeRecovery,
    FALLBACK_25_VERSION, FAST_LATERAL_VERSION,
    FRONT_CLEARANCE_25_VERSION, FRONT_SWITCH_25_VERSION, HOVER_PATH_25_VERSION, INTERCEPT_25_VERSION,
    LATERAL_VERSION, MIXED_VERSION, MOUNT_PITCH_DEGREES, NEIGHBOR_HISTORY_25_VERSION,
    OFFSET_STOP_25_VERSION,
    PRIORITY_INTERCEPT_25_VERSION, RANKED_RETURN_25_VERSION, REAR_CLEARANCE_25_VERSION, REAR_SWITCH_25_VERSION,
    RECOVERY_REARM_25_VERSION, RECOVERY_RETREAT_25_VERSION,
    RECOVERY_VERSION, REDIRECT_25_VERSION, RESERVE_INTERCEPT_25_VERSION,
    SIDE_HOVER_RETREAT_25_VERSION, STOP_RELEASE_25_VERSION, TURN_FIT_25_VERSION,
)
from instinctlab.envs.unified_h1_direct_env import UnifiedH1DirectEnv


NEIGHBOR_HISTORY_VERSIONS = (NEIGHBOR_HISTORY_25_VERSION, CONFIRMED_STANDOFF_25_VERSION)
OFFSET_STOPS = (OFFSET_STOP_25_VERSION, *NEIGHBOR_HISTORY_VERSIONS)
COMMAND_HANDOFFS = (COMMAND_HANDOFF_25_VERSION, *OFFSET_STOPS)
PRIORITY_INTERCEPTS = (PRIORITY_INTERCEPT_25_VERSION, *COMMAND_HANDOFFS)
COMMITTED_INTERCEPTS = (COMMITTED_INTERCEPT_25_VERSION, *PRIORITY_INTERCEPTS)
RESERVE_INTERCEPTS = (RESERVE_INTERCEPT_25_VERSION, *COMMITTED_INTERCEPTS)
ALIGNED_INTERCEPTS = (ALIGNED_INTERCEPT_25_VERSION, *RESERVE_INTERCEPTS)
INTERCEPTS = (INTERCEPT_25_VERSION, *ALIGNED_INTERCEPTS)
STOP_RELEASE_VERSIONS = (STOP_RELEASE_25_VERSION, TURN_FIT_25_VERSION, *INTERCEPTS)


class EscapeOptionEnv(UnifiedH1DirectEnv):
    escape_option = None
    lidar_mount_pitch_degrees = 30
    lidar_rays_body = None

    def _set_lidar_mount(self, pitch):
        if pitch == self.lidar_mount_pitch_degrees:
            return
        sensor = self._ray_caster
        rays = sensor.ray_directions
        if not hasattr(self, '_nominal_lidar_rays'):
            if not torch.equal(rays, rays[:1].expand_as(rays)):
                raise ValueError('Per-lane ray calibration differs before mount adjustment')
            self._nominal_lidar_rays = rays[0].clone()
            self._nominal_lidar_rotation = tuple(sensor.cfg.offset.rot)
        original = self._nominal_lidar_rays
        calibrated = original
        if pitch != 30:
            calibrated = original.clone()
            delta = math.radians(pitch - 30)
            cosine, sine = math.cos(delta), math.sin(delta)
            calibrated[:, 0] = cosine*original[:, 0] + sine*original[:, 2]
            calibrated[:, 2] = -sine*original[:, 0] + cosine*original[:, 2]
        rays.copy_(calibrated[None].expand_as(rays))
        half_angle = math.radians(pitch) / 2.
        rotation = self._nominal_lidar_rotation if pitch == 30 else (math.cos(half_angle), 0., math.sin(half_angle), 0.)
        sensor.cfg.offset.rot = self.cfg.ray_caster.offset.rot = rotation
        self.lidar_mount_pitch_degrees = pitch
        self.lidar_rays_body = None if pitch == 30 else rays[0]
        for pipeline in self._control_pipelines:
            pipeline.lidar_rays_body = self.lidar_rays_body

    def set_escape_option(self, enabled, *, version=METHOD_VERSION):
        pitch = MOUNT_PITCH_DEGREES.get(version, 30) if enabled else 30
        self._set_lidar_mount(pitch)
        self.escape_option_version = version if enabled else None
        kwargs = dict(lanes=self.num_envs, device=self.device, dtype=self._actions.dtype)
        if not enabled:
            self.escape_option = None
        elif version == METHOD_VERSION:
            self.escape_option = EscapeOption(**kwargs)
        elif version in (MIXED_VERSION, RECOVERY_VERSION, LATERAL_VERSION, FAST_LATERAL_VERSION, *MOUNT_PITCH_DEGREES):
            rank_return = version in (RANKED_RETURN_25_VERSION, CURVED_PASS_25_VERSION, HOVER_PATH_25_VERSION,
                                       *STOP_RELEASE_VERSIONS)
            recenter_rearm = rank_return or version == RECOVERY_REARM_25_VERSION
            clear_pass = recenter_rearm or version == CLEAR_PASS_25_VERSION
            early_retreat = clear_pass or version in (EARLY_RETREAT_25_VERSION, SIDE_HOVER_RETREAT_25_VERSION)
            brake_forward = early_retreat or version in (
                BRAKE_FORWARD_25_VERSION, BRAKE_TUBE_25_VERSION, BRAKE_TUBE_CONFIRM_25_VERSION, RECOVERY_RETREAT_25_VERSION,
            )
            rear_clearance = brake_forward or version in (REAR_CLEARANCE_25_VERSION, FRONT_CLEARANCE_25_VERSION)
            self.escape_option = EscapeRecovery(**kwargs, recenter=version != MIXED_VERSION,
                                                lateral=version in (LATERAL_VERSION, FAST_LATERAL_VERSION, *MOUNT_PITCH_DEGREES),
                                                fast_lateral=version in (FAST_LATERAL_VERSION, *MOUNT_PITCH_DEGREES),
                                                redirect=rear_clearance or version in (REDIRECT_25_VERSION, FALLBACK_25_VERSION,
                                                                                      FRONT_SWITCH_25_VERSION, REAR_SWITCH_25_VERSION),
                                                fallback=rear_clearance or version in (FALLBACK_25_VERSION, FRONT_SWITCH_25_VERSION,
                                                                                      REAR_SWITCH_25_VERSION),
                                                front_switch=version == FRONT_SWITCH_25_VERSION,
                                                rear_switch=version == REAR_SWITCH_25_VERSION,
                                                rear_clearance=rear_clearance,
                                                front_clearance=version == FRONT_CLEARANCE_25_VERSION,
                                                brake_forward_escape=brake_forward,
                                                recovery_retreat=version == RECOVERY_RETREAT_25_VERSION,
                                                early_retreat=early_retreat,
                                                side_hover_retreat=version == SIDE_HOVER_RETREAT_25_VERSION,
                                                clear_pass=clear_pass, recenter_rearm=recenter_rearm,
                                                rank_return=rank_return, curved_pass=version == CURVED_PASS_25_VERSION,
                                                hover_path=version in (HOVER_PATH_25_VERSION, *STOP_RELEASE_VERSIONS),
                                                stop_release=version in STOP_RELEASE_VERSIONS,
                                                turn_fallback=version in (TURN_FIT_25_VERSION, *INTERCEPTS),
                                                intercept=version in INTERCEPTS, align_intercept=version in ALIGNED_INTERCEPTS,
                                                intercept_reserve=version in RESERVE_INTERCEPTS,
                                                command_handoff=version in COMMAND_HANDOFFS,
                                                commit_intercept=version in COMMITTED_INTERCEPTS,
                                                retreat_priority=version in PRIORITY_INTERCEPTS,
                                                offset_stop=version in OFFSET_STOPS)
        else:
            raise ValueError('ESCAPE_OPTION_VERSION')
        if self.lidar_rays_body is not None:
            detector = self.escape_option.detector
            detector._full_rays.copy_(self.lidar_rays_body)
            detector._rays.copy_(self.lidar_rays_body.reshape(59, 360, 3)[::3].reshape(-1, 3))

    def set_h1_ics_clearance_margin(self, margin_m):
        super().set_h1_ics_clearance_margin(margin_m)
        version = getattr(self, 'escape_option_version', None)
        confirm = version in (BRAKE_TUBE_CONFIRM_25_VERSION, RECOVERY_RETREAT_25_VERSION, EARLY_RETREAT_25_VERSION,
                              SIDE_HOVER_RETREAT_25_VERSION, CLEAR_PASS_25_VERSION, RECOVERY_REARM_25_VERSION,
                              RANKED_RETURN_25_VERSION, CURVED_PASS_25_VERSION, HOVER_PATH_25_VERSION,
                              *STOP_RELEASE_VERSIONS)
        radius = .60 if confirm or version == BRAKE_TUBE_25_VERSION else 0.
        neighbor_history = version in NEIGHBOR_HISTORY_VERSIONS
        standoff = .45 if version == CONFIRMED_STANDOFF_25_VERSION else 0.
        for pipeline in self._control_pipelines:
            ics = pipeline.config.ics
            actual = (ics.motion_tube_radius_m, ics.motion_tube_confirm_closure,
                      ics.motion_tube_neighbor_history, ics.confirmed_standoff_m)
            if actual != (radius, confirm, neighbor_history, standoff):
                pipeline.config = replace(pipeline.config, ics=replace(ics, motion_tube_radius_m=radius,
                                                                      motion_tube_confirm_closure=confirm,
                                                                      motion_tube_neighbor_history=neighbor_history,
                                                                      confirmed_standoff_m=standoff))

    def enable_clearance_capture(self, radius_by_body):
        names = self._robot.body_names
        if set(names) != set(radius_by_body):
            raise ValueError('collision envelope bodies differ from the live articulation')
        self._capture_body_radii = self._command_b.new_tensor([radius_by_body[name] for name in names])
        self._clearance_samples = self._command_b.new_empty((self.cfg.decimation+1, self.num_envs, 7))

    def _append_clearance_sample(self):
        root = self._robot.data.root_link_pos_w
        links = self._robot.data.body_link_pos_w
        mover = self._h1_moving_sphere.asset.root_physx_view.get_transforms()[:, :3]
        sample = self._clearance_samples[self._captured]
        sample[:, :3].copy_(root)
        sample[:, 3:6].copy_(mover)
        radius = (links-root[:, None]).norm(dim=-1)+self._capture_body_radii
        sample[:, 6].copy_(radius.amax(dim=1))
        self._captured += 1

    def _pre_physics_step(self, actions):
        if hasattr(self, '_clearance_samples'):
            self._captured = 0
        if isinstance(self.escape_option, EscapeRecovery):
            actions = self.escape_option.apply_recovery(self._policy, actions, self._control_pipelines[0].config)
        elif self.escape_option is not None:
            actions = self.escape_option.apply(self._policy, actions)
        super()._pre_physics_step(actions)

    def _apply_action(self):
        if hasattr(self, '_clearance_samples'):
            self._append_clearance_sample()
        super()._apply_action()

    def _get_rewards(self):
        reward = super()._get_rewards()
        if hasattr(self, '_clearance_samples'):
            if self._captured != self.cfg.decimation:
                raise RuntimeError('clearance capture missed a physics tick')
            self._append_clearance_sample()
            self.extras['pre_reset']['clearance_samples'] = self._clearance_samples
        if getattr(self, 'capture_ics_limiter', False):
            self.extras['pre_reset']['ics_limiter'] = self._ics_limiter_witness()
        if self.escape_option is not None:
            self.extras['pre_reset']['escape_option'] = {
                'active': self.escape_option.applied.detach().clone(),
                'direction_b': self.escape_option.direction.detach().clone(),
                'applied_action': self._actions.detach().clone(),
            }
            if isinstance(self.escape_option, EscapeRecovery):
                self.extras['pre_reset']['escape_option']['recovery'] = self.escape_option.recovering.detach().clone()
                self.extras['pre_reset']['escape_option']['redirected'] = self.escape_option.redirected.detach().clone()
                if self.escape_option.clear_pass:
                    self.extras['pre_reset']['escape_option']['holding'] = self.escape_option.holding.detach().clone()
                if self.escape_option.recenter_rearm:
                    self.extras['pre_reset']['escape_option']['rearm_steps'] = self.escape_option.rearm_steps.detach().clone()
                if self.escape_option.rank_return:
                    self.extras['pre_reset']['escape_option']['ranked'] = self.escape_option.ranked.detach().clone()
                if self.escape_option.stop_release:
                    self.extras['pre_reset']['escape_option']['released'] = self.escape_option.released.detach().clone()
                if self.escape_option.intercept:
                    self.extras['pre_reset']['escape_option']['intercepted'] = self.escape_option.intercepted.detach().clone()
                if self.escape_option.commit_intercept:
                    self.extras['pre_reset']['escape_option']['intercept_continued'] = (
                        self.escape_option.intercept_continued.detach().clone())
                if self.escape_option.command_handoff:
                    self.extras['pre_reset']['escape_option']['handoff'] = self.escape_option.handed_off.detach().clone()
                if self.escape_option.offset_stop:
                    self.extras['pre_reset']['escape_option']['offset_stop'] = self.escape_option.offset_stopped.detach().clone()
                if self.escape_option.brake_forward_escape:
                    self.extras['pre_reset']['escape_option']['braking'] = self.escape_option.braking.detach().clone()
                if self.escape_option.recovery_retreat or self.escape_option.early_retreat:
                    self.extras['pre_reset']['escape_option']['retreating'] = self.escape_option.retreating.detach().clone()
                if getattr(self, 'mirrored_pipe_pairs', False):
                    self.extras['pre_reset']['escape_option']['retreat_basis'] = self.escape_option.retreat_basis.detach().clone()
        return reward

    def _ics_limiter_witness(self):
        caches = [pipeline._last_ics_cache for pipeline in self._control_pipelines]
        active = torch.cat([cache['active_mask'] for cache in caches])
        approach = torch.cat([cache['approach_speed_m_s'] for cache in caches])
        safe = torch.cat([cache['v_safe_m_s'] for cache in caches])
        closing = approach.clamp_min(1e-6)
        ratio = torch.where(active, safe/closing, torch.inf)
        index = ratio.argmin(dim=1)
        lanes = torch.arange(self.num_envs, device=self.device)
        distance = self._policy['lidar_grid'][:, 9].flatten(1)[lanes, index]
        hit = distance[:, None] * self.lidar_rays_body[index]
        return dict(limiter_beam=torch.where(active.any(dim=1), index, -1), limiter_hit_b=hit)

    def _reset_idx(self, env_ids):
        if self.escape_option is not None:
            self.escape_option.reset(self._validate_reset_indices(env_ids))
        super()._reset_idx(env_ids)
        if self.lidar_rays_body is not None:
            for lane in self._validate_reset_indices(env_ids).detach().cpu().tolist():
                self._control_pipelines[lane].lidar_rays_body = self.lidar_rays_body


class EscapeHoldoutEnv(EscapeOptionEnv):
    def __init__(self, cfg, *, layout_seed, mirrored_pipe_pairs=False, mirrored_trunk_pairs=False, command_routes=False):
        self.layout_seed = layout_seed
        self.mirrored_pipe_pairs = mirrored_pipe_pairs
        self.mirrored_trunk_pairs = mirrored_trunk_pairs
        self.command_routes = command_routes
        super().__init__(cfg)

    def _setup_scene(self):
        from .escape_holdout_scene import scene_geometry

        super()._setup_scene()
        self._replace_mesh(scene_geometry(self.scene.env_origins.detach().cpu(), self.layout_seed,
                                          mirrored_pipe_pairs=self.mirrored_pipe_pairs,
                                          mirrored_trunk_pairs=self.mirrored_trunk_pairs,
                                          command_routes=self.command_routes))

    def _replace_mesh(self, geometry):
        from instinctlab.envs.unified_h1_direct_env import _define_triangle_mesh
        from instinctlab.task.scene_h1 import H1_RAY_MESH_PATH

        self.holdout_geometry = geometry
        points, faces, walls, wall_faces = geometry
        _define_triangle_mesh(H1_RAY_MESH_PATH, points, faces, collision=False)
        _define_triangle_mesh('/World/H1CollisionWalls/Walls', walls, wall_faces, collision=True)

    def holdout_preflight(self, output, *, scene='room-door-trunks-street-tube'):
        from training_workloads.scale_003.persistence import atomic_json_replace
        from .preflight import (initial_source, live_geometry_readback, persist_initial_source,
                                validate_initial_observability)

        geometry = live_geometry_readback(self)
        atomic_json_replace(output, 'initial_preflight/geometry.json', geometry)
        source = initial_source(self)
        persist_initial_source(output, source)
        observability = validate_initial_observability(source, output, full_rank=False)
        atomic_json_replace(output, 'initial_preflight.json', {
            'schema_version': 'unified-h1-preflight.v1', 'status': 'PASS',
            'scene': scene, 'num_envs': 32,
            'geometry': geometry, 'observability': observability,
        })


class EscapeRecoveryTrainingEnv(EscapeHoldoutEnv):
    from .recovery_training_scene import TUBE_LANES as tube_training_lanes
    single_pass_tube_movers = False

    def __init__(self, cfg):
        super().__init__(cfg, layout_seed=None)

    def _setup_scene(self):
        from .recovery_training_scene import geometry

        EscapeOptionEnv._setup_scene(self)
        self._replace_mesh(geometry(self.scene.env_origins.detach().cpu()))

    def _reset_r6_physics(self, indices):
        if not self.single_pass_tube_movers:
            return super()._reset_r6_physics(indices)
        mover = self._h1_moving_sphere
        was_running = mover.enabled
        result = super()._reset_r6_physics(indices)
        if not was_running and mover.enabled:
            state = self._r6_physics
            turn_seconds = mover.half_period_s
            # Match the first pass, then hold the same endpoint past the end
            # of the phase. Each physical reset starts the same path again.
            mask = torch.zeros_like(mover.active)
            mask[list(self.tube_training_lanes)] = True
            mask &= mover.active
            longer_than_phase = state['block']['steps'] * .1 + turn_seconds
            mover.half_period_s = torch.where(mask, longer_than_phase, turn_seconds).to(mover.age.dtype)
            mover.end_travel_s[mask] = turn_seconds
            state['curriculum'].record('TUBE_SINGLE_PASS_MOVERS', dict(
                lanes=mask.nonzero().flatten(), first_pass_seconds=turn_seconds,
                held_at_endpoint_after_seconds=turn_seconds, half_period_s=mover.half_period_s,
                physical_reset_repeats_first_pass=True))
        return result
