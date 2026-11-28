"""Apply the frozen option to motor requests while leaving PPO samples intact."""

from dataclasses import replace
import math

import torch

from instinctlab.control.escape_option import EscapeOption, METHOD_VERSION
from instinctlab.control.escape_recovery import (
    ALIGNED_INTERCEPT_25_VERSION, ALONG_INTERCEPT_25_VERSION, BRAKE_FORWARD_25_VERSION, BRAKE_TUBE_25_VERSION, BRAKE_TUBE_CONFIRM_25_VERSION,
    CLEAR_PASS_25_VERSION, COMMAND_HANDOFF_25_VERSION, COMMITTED_INTERCEPT_25_VERSION, CONFIRMED_STANDOFF_25_VERSION,
    CONFIRMED_TUBE_25_VERSION,
    CURVED_PASS_25_VERSION,
    EARLY_DIAGONAL_25_VERSION, EARLY_RETREAT_25_VERSION, EDGE_CONTINUATION_25_VERSION, EscapeRecovery,
    FALLBACK_25_VERSION, FAST_LATERAL_VERSION, FINITE_EDGE_25_VERSION, FINITE_INTERCEPT_25_VERSION,
    FRONT_CLEARANCE_25_VERSION, FRONT_SWITCH_25_VERSION, HOVER_PATH_25_VERSION, INTERCEPT_25_VERSION,
    INTERCEPT_CHOICE_25_VERSION,
    LATERAL_VERSION, MIXED_VERSION, MOUNT_PITCH_DEGREES, NEIGHBOR_HISTORY_25_VERSION,
    OFFSET_STOP_25_VERSION, OPENING_STEERING_25_VERSION,
    PRIORITY_INTERCEPT_25_VERSION, RANKED_RETURN_25_VERSION, REAR_CLEARANCE_25_VERSION, REAR_SWITCH_25_VERSION,
    RECOVERY_REARM_25_VERSION, RECOVERY_RETREAT_25_VERSION,
    RECOVERY_VERSION, REDIRECT_25_VERSION, RESERVE_DIRECTION_25_VERSION, RESERVE_INTERCEPT_25_VERSION, ROTATION_TUBE_25_VERSION,
    SETTLED_OFFSET_STOP_25_VERSION, SIDE_HOVER_RETREAT_25_VERSION, STOP_RELEASE_25_VERSION, TANGENT_MAGNITUDE_25_VERSION,
    TURN_FIT_25_VERSION, VERIFIED_RESUME_25_VERSION, VERIFIED_ZERO_25_VERSION,
    WALL_TANGENT_25_VERSION, CROSSING_FALLBACK_25_VERSION, CROSSING_STOP_25_VERSION, CROSSING_STOP_25_V2, CROSSING_STOP_25_V3, CROSSING_STOP_25_V4, CROSSING_STOP_25_V5, CROSSING_STOP_25_V6,
    CROSSING_STOP_25_V7, CROSSING_STOP_25_V8, CROSSING_STOP_25_V9, CROSSING_STOP_25_V10,
    CROSSING_STOP_25_V11, CROSSING_STOP_25_V12, CROSSING_STOP_25_V13, CROSSING_STOP_25_V14, CROSSING_STOP_25_V15,
    CROSSING_STOP_25_V16, CROSSING_STOP_25_V17, CROSSING_STOP_25_V18, CROSSING_STOP_25_V19, CROSSING_STOP_25_V20, CROSSING_STOP_25_V21,
    CROSSING_STOP_25_V22, CROSSING_STOP_25_V23, CROSSING_STOP_25_V24, CROSSING_STOP_25_V25,
    CROSSING_STOP_25_V26, CROSSING_STOP_25_V27, CROSSING_STOP_25_V28, CROSSING_STOP_25_V29, CROSSING_STOP_25_V30, CROSSING_STOP_25_V31, CROSSING_STOP_25_V32, CROSSING_STOP_25_V33,
    CROSSING_STOP_25_V34, CROSSING_STOP_25_V35, CROSSING_STOP_25_V36, CROSSING_STOP_25_V37, CROSSING_STOP_25_V38,
    CROSSING_STOP_25_V39, CROSSING_STOP_25_V40, CROSSING_STOP_25_V41, RECENT_COMMAND_HANDOFF_25, STATIC_RETREAT_HANDOFF_25,
    GRAVITY_ALIGNED_PROTECTION_25, RELEASE_STOP_CERTIFICATES_25,
)
from instinctlab.envs.unified_h1_direct_env import UnifiedH1DirectEnv


GRAVITY_ALIGNED_PROTECTION_VERSIONS = (GRAVITY_ALIGNED_PROTECTION_25, RELEASE_STOP_CERTIFICATES_25)
STATIC_RETREAT_HANDOFF_VERSIONS = (STATIC_RETREAT_HANDOFF_25, *GRAVITY_ALIGNED_PROTECTION_VERSIONS)
RECENT_HANDOFF_VERSIONS = (RECENT_COMMAND_HANDOFF_25, *STATIC_RETREAT_HANDOFF_VERSIONS)
COMPOSED_CROSSING_VERSIONS = (CROSSING_STOP_25_V41, *RECENT_HANDOFF_VERSIONS)
CURRENT_REQUEST_RECOVERY_VERSIONS = (CROSSING_STOP_25_V40, *COMPOSED_CROSSING_VERSIONS)
SUPPORTED_WALL_VERSIONS = (CROSSING_STOP_25_V39, *CURRENT_REQUEST_RECOVERY_VERSIONS)
COMPOSED_MOTION_VERSIONS = (CROSSING_STOP_25_V34, CROSSING_STOP_25_V36, CROSSING_STOP_25_V38, *SUPPORTED_WALL_VERSIONS)
FACETED_WALL_STOPPING_VERSIONS = (CROSSING_STOP_25_V28, *COMPOSED_MOTION_VERSIONS)
WALL_EDGE_REPROOF_VERSIONS = (CROSSING_STOP_25_V30, *COMPOSED_MOTION_VERSIONS)
BRAKE_TRANSITION_VERSIONS = (CROSSING_STOP_25_V27, *COMPOSED_MOTION_VERSIONS)
FRONT_WALL_BRAKING_VERSIONS = (CROSSING_STOP_25_V25, CROSSING_STOP_25_V28, *WALL_EDGE_REPROOF_VERSIONS)
LOCAL_WALL_STOPPING_VERSIONS = (CROSSING_STOP_25_V22, *FRONT_WALL_BRAKING_VERSIONS)
RETREAT_STOP_VERSIONS = (CROSSING_STOP_25_V37, CROSSING_STOP_25_V38, *SUPPORTED_WALL_VERSIONS)
RETREAT_ONSET_SIDE_VERSIONS = (CROSSING_STOP_25_V35, CROSSING_STOP_25_V36, *RETREAT_STOP_VERSIONS)
RANKED_REVERSAL_STOP_VERSIONS = (CROSSING_STOP_25_V32, CROSSING_STOP_25_V34, *RETREAT_ONSET_SIDE_VERSIONS)
ZERO_CROSSING_WITHDRAWAL_VERSIONS = (CROSSING_STOP_25_V31, *RANKED_REVERSAL_STOP_VERSIONS)
OLDEST_PARTIAL_TURN_VERSIONS = (CROSSING_STOP_25_V29, CROSSING_STOP_25_V33, *ZERO_CROSSING_WITHDRAWAL_VERSIONS)
RETREAT_CONTINUATION_VERSIONS = (CROSSING_STOP_25_V26, *OLDEST_PARTIAL_TURN_VERSIONS)
OVERLAPPING_STOP_VERSIONS = (CROSSING_STOP_25_V23, *RETREAT_CONTINUATION_VERSIONS)
CROSSING_RESTART_VERSIONS = (CROSSING_STOP_25_V21, CROSSING_STOP_25_V24, CROSSING_STOP_25_V27, *OVERLAPPING_STOP_VERSIONS,
                             *LOCAL_WALL_STOPPING_VERSIONS)
HOVER_REFERENCE_RESERVE_VERSIONS = (CROSSING_STOP_25_V20, *CROSSING_RESTART_VERSIONS)
CLIPPED_RETREAT_ONSET_VERSIONS = (CROSSING_STOP_25_V18, *HOVER_REFERENCE_RESERVE_VERSIONS)
VERIFIED_RETREAT_ONSET_VERSIONS = (CROSSING_STOP_25_V17, *CLIPPED_RETREAT_ONSET_VERSIONS)
FAR_CROSSING_REFERENCE_VERSIONS = (CROSSING_STOP_25_V16, CROSSING_STOP_25_V19, *VERIFIED_RETREAT_ONSET_VERSIONS)
HOVER_OWN_SIDE_VERSIONS = (CROSSING_STOP_25_V15, *FAR_CROSSING_REFERENCE_VERSIONS)
SAMPLED_TURN_VERSIONS = (CROSSING_STOP_25_V13, CROSSING_STOP_25_V14, *HOVER_OWN_SIDE_VERSIONS)
OBSERVED_HEADING_VERSIONS = (CROSSING_STOP_25_V12, *SAMPLED_TURN_VERSIONS)
SHALLOW_STEERING_VERSIONS = (CROSSING_STOP_25_V11, *OBSERVED_HEADING_VERSIONS)
CROSSING_REFERENCE_VERSIONS = (CROSSING_STOP_25_V10, *SHALLOW_STEERING_VERSIONS)
CROSSING_SETTLED_VERSIONS = (CROSSING_STOP_25_V9, *CROSSING_REFERENCE_VERSIONS)
CROSSING_EXTENDED_RETREAT_VERSIONS = (CROSSING_STOP_25_V8, *CROSSING_SETTLED_VERSIONS)
CROSSING_RECEDING_HOVER_VERSIONS = (CROSSING_STOP_25_V7, *CROSSING_EXTENDED_RETREAT_VERSIONS)
CROSSING_VERIFIED_HOVER_VERSIONS = (CROSSING_STOP_25_V6, *CROSSING_RECEDING_HOVER_VERSIONS)
CROSSING_COMPATIBLE_VERSIONS = (CROSSING_STOP_25_V5, *CROSSING_VERIFIED_HOVER_VERSIONS)
CROSSING_EARLY_VERSIONS = (CROSSING_STOP_25_V4, *CROSSING_COMPATIBLE_VERSIONS)
CROSSING_SHALLOW_VERSIONS = (CROSSING_STOP_25_V3, *CROSSING_EARLY_VERSIONS)
CROSSING_CONTINUATION_VERSIONS = (CROSSING_STOP_25_V2, *CROSSING_SHALLOW_VERSIONS)
CROSSING_STOP_VERSIONS = (CROSSING_STOP_25_VERSION, *CROSSING_CONTINUATION_VERSIONS)
FINITE_INTERCEPT_VERSIONS = (FINITE_INTERCEPT_25_VERSION, *CROSSING_STOP_VERSIONS)
ALONG_INTERCEPT_VERSIONS = (ALONG_INTERCEPT_25_VERSION, *FINITE_INTERCEPT_VERSIONS)
OPENING_STEERING_VERSIONS = (OPENING_STEERING_25_VERSION, *ALONG_INTERCEPT_VERSIONS)
CROSSING_FALLBACK_VERSIONS = (CROSSING_FALLBACK_25_VERSION, *OPENING_STEERING_VERSIONS)
RESERVE_DIRECTION_VERSIONS = (RESERVE_DIRECTION_25_VERSION, *CROSSING_FALLBACK_VERSIONS)
EARLY_DIAGONAL_VERSIONS = (EARLY_DIAGONAL_25_VERSION, *RESERVE_DIRECTION_VERSIONS)
INTERCEPT_CHOICE_VERSIONS = (INTERCEPT_CHOICE_25_VERSION, *EARLY_DIAGONAL_VERSIONS)
VERIFIED_ZERO_VERSIONS = (VERIFIED_ZERO_25_VERSION, *INTERCEPT_CHOICE_VERSIONS)
ROTATION_TUBE_VERSIONS = (ROTATION_TUBE_25_VERSION, VERIFIED_RESUME_25_VERSION, *VERIFIED_ZERO_VERSIONS)
FINITE_EDGE_VERSIONS = (FINITE_EDGE_25_VERSION, EDGE_CONTINUATION_25_VERSION,
                        TANGENT_MAGNITUDE_25_VERSION, *ROTATION_TUBE_VERSIONS)
CONFIRMED_TUBE_VERSIONS = (CONFIRMED_TUBE_25_VERSION, *FINITE_EDGE_VERSIONS)
WALL_TANGENT_VERSIONS = (WALL_TANGENT_25_VERSION, *CONFIRMED_TUBE_VERSIONS)
SETTLED_OFFSET_STOP_VERSIONS = (SETTLED_OFFSET_STOP_25_VERSION, *WALL_TANGENT_VERSIONS)
NEIGHBOR_HISTORY_VERSIONS = (NEIGHBOR_HISTORY_25_VERSION, CONFIRMED_STANDOFF_25_VERSION, *SETTLED_OFFSET_STOP_VERSIONS)
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
    bilateral_stopping = False
    tube_stopping = False
    local_wall_stopping = False
    front_wall_braking = False
    faceted_wall_stopping = False
    wall_edge_reproof = False
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
        self.bilateral_stopping = enabled and version == CROSSING_STOP_25_V14
        self.tube_stopping = enabled and version == CROSSING_STOP_25_V19
        self.local_wall_stopping = enabled and version in LOCAL_WALL_STOPPING_VERSIONS
        self.front_wall_braking = enabled and version in FRONT_WALL_BRAKING_VERSIONS
        self.faceted_wall_stopping = enabled and version in FACETED_WALL_STOPPING_VERSIONS
        self.wall_edge_reproof = enabled and version in WALL_EDGE_REPROOF_VERSIONS
        align_planar = enabled and version in GRAVITY_ALIGNED_PROTECTION_VERSIONS
        for pipeline in self._control_pipelines:
            if pipeline.config.align_planar_protection != align_planar:
                pipeline.config = replace(pipeline.config, align_planar_protection=align_planar)
            if hasattr(pipeline, 'allow_bilateral_stopping'):
                pipeline.allow_bilateral_stopping = False
            if hasattr(pipeline, 'allow_tube_stopping'):
                pipeline.allow_tube_stopping = False
            if hasattr(pipeline, 'allow_front_wall_brake'):
                pipeline.allow_front_wall_brake = False
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
                                                retry_command_handoff=version in CURRENT_REQUEST_RECOVERY_VERSIONS,
                                                commit_intercept=version in COMMITTED_INTERCEPTS,
                                                retreat_priority=version in PRIORITY_INTERCEPTS,
                                                offset_stop=version in OFFSET_STOPS,
                                                settled_offset_stop=version in SETTLED_OFFSET_STOP_VERSIONS,
                                                wall_tangent=version in WALL_TANGENT_VERSIONS,
                                                finite_edge=version in FINITE_EDGE_VERSIONS,
                                                edge_continuation=version == EDGE_CONTINUATION_25_VERSION,
                                                tangent_magnitude=version in (TANGENT_MAGNITUDE_25_VERSION, CROSSING_STOP_25_V33,
                                                                              CROSSING_STOP_25_V36, CROSSING_STOP_25_V38,
                                                                              *SUPPORTED_WALL_VERSIONS),
                                                verified_resume=version == VERIFIED_RESUME_25_VERSION,
                                                verified_zero=version in VERIFIED_ZERO_VERSIONS,
                                                intercept_choice=version in INTERCEPT_CHOICE_VERSIONS,
                                                receding_stop=version in INTERCEPT_CHOICE_VERSIONS,
                                                early_diagonal=version in EARLY_DIAGONAL_VERSIONS,
                                                reserve_direction=version in RESERVE_DIRECTION_VERSIONS,
                                                zero_recheck=version in CROSSING_FALLBACK_VERSIONS,
                                                reverse_diagonal=version in CROSSING_FALLBACK_VERSIONS,
                                                opening_steering=version in OPENING_STEERING_VERSIONS,
                                                along_intercept=version in ALONG_INTERCEPT_VERSIONS,
                                                finite_intercept=version in FINITE_INTERCEPT_VERSIONS,
                                                crossing_stop=version in CROSSING_STOP_VERSIONS,
                                                crossing_stop_continuation=version in CROSSING_CONTINUATION_VERSIONS,
                                                crossing_stop_shallow=version in CROSSING_SHALLOW_VERSIONS,
                                                crossing_stop_early=version in CROSSING_EARLY_VERSIONS,
                                                crossing_stop_compatible=version in CROSSING_COMPATIBLE_VERSIONS,
                                                crossing_stop_verified_hover=version in CROSSING_VERIFIED_HOVER_VERSIONS,
                                                crossing_stop_receding_hover=version in CROSSING_RECEDING_HOVER_VERSIONS,
                                                crossing_retreat_own_side=version in CROSSING_RECEDING_HOVER_VERSIONS,
                                                crossing_retreat_extended=version in CROSSING_EXTENDED_RETREAT_VERSIONS,
                                                settled_crossing_mount=tuple(self.cfg.ray_caster.offset.pos)
                                                if version in CROSSING_SETTLED_VERSIONS else None,
                                                static_scan_stop=version in CROSSING_REFERENCE_VERSIONS,
                                                crossing_reference=version in CROSSING_REFERENCE_VERSIONS,
                                                shallow_steering=version in SHALLOW_STEERING_VERSIONS,
                                                observed_heading_steering=version in OBSERVED_HEADING_VERSIONS,
                                                tracked_resume=version in OBSERVED_HEADING_VERSIONS,
                                                crossing_turn_between_samples=version in SAMPLED_TURN_VERSIONS,
                                                hover_escape_own_side=version in HOVER_OWN_SIDE_VERSIONS,
                                                far_crossing_reference=version in FAR_CROSSING_REFERENCE_VERSIONS,
                                                verified_retreat_onset=version in VERIFIED_RETREAT_ONSET_VERSIONS,
                                                retain_only_clipped_retreat=version in CLIPPED_RETREAT_ONSET_VERSIONS,
                                                verify_hover_reference_reserve=version in HOVER_REFERENCE_RESERVE_VERSIONS,
                                                verify_crossing_restart=version in CROSSING_RESTART_VERSIONS,
                                                verify_overlapping_stop=version in OVERLAPPING_STOP_VERSIONS,
                                                verify_brake_recovery=version in (CROSSING_STOP_25_V24, *BRAKE_TRANSITION_VERSIONS),
                                                verify_retreat_continuation=version in RETREAT_CONTINUATION_VERSIONS,
                                                verify_brake_transition=version in BRAKE_TRANSITION_VERSIONS,
                                                verify_outward_brake=version in CURRENT_REQUEST_RECOVERY_VERSIONS,
                                                verify_oldest_partial_turn=version in OLDEST_PARTIAL_TURN_VERSIONS,
                                                verify_zero_crossing_withdrawal=version in ZERO_CROSSING_WITHDRAWAL_VERSIONS,
                                                verify_ranked_reversal_stop=version in RANKED_REVERSAL_STOP_VERSIONS,
                                                verify_retreat_onset_side=version in RETREAT_ONSET_SIDE_VERSIONS,
                                                verify_retreat_stop=version in RETREAT_STOP_VERSIONS,
                                                verify_composed_crossing=version in COMPOSED_CROSSING_VERSIONS,
                                                verify_recent_handoff=version in RECENT_HANDOFF_VERSIONS,
                                                verify_static_retreat_handoff=version in STATIC_RETREAT_HANDOFF_VERSIONS,
                                                align_planar_protection=align_planar,
                                                verify_release_stop=version == RELEASE_STOP_CERTIFICATES_25)
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
        inner = .41 if version in CONFIRMED_TUBE_VERSIONS else 0.
        align = version in ROTATION_TUBE_VERSIONS
        for pipeline in self._control_pipelines:
            ics = pipeline.config.ics
            actual = (ics.motion_tube_radius_m, ics.motion_tube_confirm_closure,
                      ics.motion_tube_neighbor_history, ics.confirmed_standoff_m, ics.confirmed_tube_radius_m,
                      ics.confirmed_tube_align_rotation)
            if actual != (radius, confirm, neighbor_history, standoff, inner, align):
                pipeline.config = replace(pipeline.config, ics=replace(ics, motion_tube_radius_m=radius,
                                                                      motion_tube_confirm_closure=confirm,
                                                                      motion_tube_neighbor_history=neighbor_history,
                                                                      confirmed_standoff_m=standoff,
                                                                      confirmed_tube_radius_m=inner,
                                                                      confirmed_tube_align_rotation=align))

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
            option = self.escape_option
            config = self._control_pipelines[0].config
            actions = option.apply_recovery(self._policy, actions, config)
            if option.verify_release_stop:
                from instinctlab.control.tube_stopping import TubeStoppingPipeline

                bounds = [pipeline.oldest_speed_bound if isinstance(pipeline, TubeStoppingPipeline)
                          else pipeline.config.ics.velocity_limit_m_s for pipeline in self._control_pipelines]
                option.reprove_wall_stop_at_release(self._policy['lidar_grid'], self._policy['state_vec'],
                                                   actions, config, bounds)
            if self.bilateral_stopping or self.tube_stopping or self.local_wall_stopping:
                from instinctlab.control.bilateral_stopping import BilateralStoppingPipeline
                from instinctlab.control.local_wall_stopping import (
                    FacetedWallBrakingPipeline, FacetedWallEdgeReproofPipeline, FrontWallBrakingPipeline, LocalWallStoppingPipeline,
                    SupportedWallEdgeReproofPipeline, WallEdgeReproofPipeline,
                )
                from instinctlab.control.tube_stopping import TubeStoppingPipeline

                protected = (option.applied | option.retreating | option.braking | option.ranked | option.holding
                             | option.intercepted | option.recovering)
                finite_stopping = self.tube_stopping or self.local_wall_stopping
                if finite_stopping:
                    protected |= option.intercept_continued
                if option.verify_composed_crossing:
                    protected |= option.composed_crossing_verified
                if self.local_wall_stopping:
                    protected |= option.crossing_restart_verified
                    if self.escape_option_version in SUPPORTED_WALL_VERSIONS:
                        pipeline_type = SupportedWallEdgeReproofPipeline
                    elif self.wall_edge_reproof and self.faceted_wall_stopping:
                        pipeline_type = FacetedWallEdgeReproofPipeline
                    elif self.wall_edge_reproof:
                        pipeline_type = WallEdgeReproofPipeline
                    elif self.faceted_wall_stopping:
                        pipeline_type = FacetedWallBrakingPipeline
                    else:
                        pipeline_type = FrontWallBrakingPipeline if self.front_wall_braking else LocalWallStoppingPipeline
                else:
                    pipeline_type = TubeStoppingPipeline if self.tube_stopping else BilateralStoppingPipeline
                if self.front_wall_braking:
                    ordinary = (option.applied & ~option.retreating & ~option.braking & ~option.ranked & ~option.holding
                                & ~option.intercepted & ~option.intercept_continued & ~option.recovering
                                & ~option.handed_off & ~option.static_stop_verified & ~option.crossing_restart_verified)
                    if option.verify_composed_crossing:
                        ordinary &= ~option.composed_crossing_verified
                    front_wall = ordinary.tolist()
                if self.wall_edge_reproof:
                    fresh_edge = (option.detour_used & ~option.shallow_used & ~option.observed_heading_used).tolist()
                    other_detour = (option.detour_used & (option.shallow_used | option.observed_heading_used)).tolist()
                for index, allowed in enumerate((~protected).tolist()):
                    pipeline = self._control_pipelines[index]
                    if type(pipeline) is not pipeline_type:
                        pipeline = pipeline_type(pipeline)
                        self._control_pipelines[index] = pipeline
                    if finite_stopping:
                        pipeline.allow_tube_stopping = allowed
                    else:
                        pipeline.allow_bilateral_stopping = allowed
                    if self.front_wall_braking:
                        pipeline.allow_front_wall_brake = front_wall[index]
                    if self.wall_edge_reproof:
                        pipeline.fresh_edge = fresh_edge[index]
                        pipeline.other_detour = other_detour[index]
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
                if self.escape_option.wall_tangent:
                    self.extras['pre_reset']['escape_option']['wall_tangent'] = self.escape_option.tangent_used.detach().clone()
                if self.escape_option.finite_edge:
                    self.extras['pre_reset']['escape_option']['finite_edge'] = self.escape_option.detour_used.detach().clone()
                if self.escape_option.shallow_steering:
                    self.extras['pre_reset']['escape_option']['shallow_steering'] = self.escape_option.shallow_used.detach().clone()
                if self.escape_option.observed_heading_steering:
                    self.extras['pre_reset']['escape_option']['observed_heading'] = self.escape_option.observed_heading_used.detach().clone()
                if self.escape_option.tracked_resume:
                    self.extras['pre_reset']['escape_option']['tracked_resume'] = self.escape_option.tracked_resume_used.detach().clone()
                if self.escape_option.crossing_turn_between_samples:
                    self.extras['pre_reset']['escape_option']['settled_crossing'] = self.escape_option.settled_crossing_verified.detach().clone()
                if self.escape_option.verify_oldest_partial_turn:
                    self.extras['pre_reset']['escape_option']['oldest_partial_turn_verified'] = (
                        self.escape_option.oldest_partial_turn_verified.detach().clone())
                if self.escape_option.verify_zero_crossing_withdrawal:
                    option = self.extras['pre_reset']['escape_option']
                    option['zero_crossing_mode'] = self.escape_option.zero_crossing_mode.detach().clone()
                    option['zero_crossing_floor'] = self.escape_option.zero_crossing_floor.detach().clone()
                if self.escape_option.verify_ranked_reversal_stop:
                    self.extras['pre_reset']['escape_option']['ranked_reversal_verified'] = (
                        self.escape_option.ranked_reversal_verified.detach().clone())
                if self.escape_option.verify_retreat_onset_side:
                    self.extras['pre_reset']['escape_option']['retreat_onset_side_floor'] = (
                        self.escape_option.retreat_onset_side_floor.detach().clone())
                if self.escape_option.verify_retreat_stop:
                    self.extras['pre_reset']['escape_option']['retreat_stop_floor'] = (
                        self.escape_option.retreat_stop_floor.detach().clone())
                if self.escape_option.verify_composed_crossing:
                    self.extras['pre_reset']['escape_option']['composed_crossing_floor'] = (
                        self.escape_option.composed_crossing_floor.detach().clone())
                if self.escape_option.verify_static_retreat_handoff:
                    self.extras['pre_reset']['escape_option']['static_retreat_witness'] = (
                        self.escape_option.static_retreat_witness.detach().clone())
                if self.escape_option.hover_escape_own_side:
                    self.extras['pre_reset']['escape_option']['hover_own_side'] = self.escape_option.hover_escape_redirected.detach().clone()
                if self.escape_option.far_crossing_reference:
                    self.extras['pre_reset']['escape_option']['far_crossing_refined'] = self.escape_option.far_crossing_refined.detach().clone()
                if self.escape_option.verified_retreat_onset:
                    self.extras['pre_reset']['escape_option']['retreat_request_kept'] = self.escape_option.retreat_request_kept.detach().clone()
                if self.escape_option.verify_crossing_restart:
                    option = self.extras['pre_reset']['escape_option']
                    option['crossing_restart_verified'] = self.escape_option.crossing_restart_verified.detach().clone()
                    option['crossing_restart_previous_b'] = self.escape_option.crossing_restart_previous_b.detach().clone()
                if self.escape_option.verify_overlapping_stop:
                    self.extras['pre_reset']['escape_option']['overlapping_stop_verified'] = (
                        self.escape_option.overlapping_stop_verified.detach().clone())
                if self.escape_option.verify_brake_recovery:
                    option = self.extras['pre_reset']['escape_option']
                    option['brake_recovery_verified'] = self.escape_option.brake_recovery_verified.detach().clone()
                    option['brake_recovery_bounds'] = self.escape_option.brake_recovery_bounds.detach().clone()
                if self.escape_option.verify_retreat_continuation:
                    option = self.extras['pre_reset']['escape_option']
                    option['retreat_continuation_verified'] = self.escape_option.retreat_continuation_verified.detach().clone()
                    option['retreat_continuation_floor'] = self.escape_option.retreat_continuation_floor.detach().clone()
                if self.bilateral_stopping:
                    used = [pipeline.bilateral_stopping_used for pipeline in self._control_pipelines]
                    self.extras['pre_reset']['escape_option']['bilateral_stopping'] = self.escape_option.applied.new_tensor(used)
                if self.tube_stopping or self.local_wall_stopping:
                    target = torch.zeros_like(self.escape_option.direction)
                    hold_seconds = target.new_zeros((self.num_envs,))
                    baseline_velocity = torch.zeros_like(target)
                    baseline_beta = torch.zeros_like(hold_seconds)
                    gap = torch.zeros_like(hold_seconds)
                    required_gap = torch.zeros_like(hold_seconds)
                    cleared = target.new_zeros((self.num_envs,),dtype=torch.long)
                    for lane, pipeline in enumerate(self._control_pipelines):
                        if pipeline.tube_stopping is not None:
                            decision, velocity, beta = pipeline.tube_stopping
                            target[lane] = decision.target[0]
                            hold_seconds[lane] = decision.hold_s
                            baseline_velocity[lane] = velocity[0]
                            baseline_beta[lane] = beta[0,0]
                            gap[lane] = decision.observed_gap_m
                            required_gap[lane] = decision.required_gap_m
                            cleared[lane] = decision.cleared_limiting_beams
                    prefix = 'local_wall' if self.local_wall_stopping else 'tube'
                    receipt = dict(stopping_target_b=target, stopping_hold_s=hold_seconds,
                                   baseline_velocity_b=baseline_velocity, baseline_beta=baseline_beta,
                                   gap_m=gap, required_gap_m=required_gap, cleared_limiting_beams=cleared)
                    self.extras['pre_reset']['escape_option'].update({prefix+'_'+key:value for key,value in receipt.items()})
                    if self.faceted_wall_stopping:
                        oblique = [p.tube_stopping is not None and p.tube_stopping[0].used_oblique_facets
                                   for p in self._control_pipelines]
                        self.extras['pre_reset']['escape_option']['faceted_wall_stopping_used'] = (
                            self.escape_option.applied.new_tensor(oblique))
                if self.front_wall_braking:
                    baseline_velocity = torch.zeros_like(self.escape_option.direction)
                    gap = baseline_velocity.new_zeros((self.num_envs,))
                    required = torch.zeros_like(gap)
                    attributed = gap.new_zeros((self.num_envs,), dtype=torch.long)
                    verified = self.escape_option.applied.new_zeros((self.num_envs,))
                    for lane, pipeline in enumerate(self._control_pipelines):
                        if pipeline.front_wall_brake is not None:
                            observed, needed, closing, original = pipeline.front_wall_brake
                            baseline_velocity[lane] = original[0]
                            gap[lane], required[lane], attributed[lane] = observed, needed, closing
                            verified[lane] = True
                    self.extras['pre_reset']['escape_option'].update(front_wall_brake_verified=verified,
                        front_wall_brake_previous_b=baseline_velocity, front_wall_brake_gap_m=gap,
                        front_wall_brake_required_m=required, front_wall_brake_attributed_beams=attributed)
                if self.wall_edge_reproof:
                    proofs = [p.edge_reproof or (0, 0., 0., 0.) for p in self._control_pipelines]
                    self.extras['pre_reset']['escape_option']['wall_edge_reproof'] = (
                        self.escape_option.direction.new_tensor(proofs))
                if self.escape_option.edge_continuation:
                    self.extras['pre_reset']['escape_option']['edge_continued'] = self.escape_option.edge_continued.detach().clone()
                if self.escape_option.verified_resume:
                    self.extras['pre_reset']['escape_option']['verified_resume'] = self.escape_option.resumed.detach().clone()
                if self.escape_option.opening_steering:
                    self.extras['pre_reset']['escape_option']['opening_steered'] = (
                        self.escape_option.opening_steered & self.escape_option.applied & ~self.escape_option.intercepted).detach().clone()
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
