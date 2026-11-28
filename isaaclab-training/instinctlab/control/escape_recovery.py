"""Test bounded command recovery in the same governor and ICS action space."""

import math

import torch

from .escape_option import DURATION_STEPS, SPEED_M_S, EscapeOption
from .governor import decode_governor, extract_latest_commands
from .ics import attenuate_command


MIXED_VERSION = 'm4-escape-mixed-v1'
RECOVERY_VERSION = 'm4-escape-recovery-v1'
LATERAL_VERSION = 'm4-escape-lateral-recovery-v1'
FAST_LATERAL_VERSION = 'm4-escape-lateral-recovery-v2'
MOUNT_25_VERSION = 'm4-escape-lateral-recovery-p25-v1'
MOUNT_28_VERSION = 'm4-escape-lateral-recovery-p28-v1'
REDIRECT_25_VERSION = 'm4-escape-redirect-p25-v1'
FALLBACK_25_VERSION = 'm4-escape-fallback-p25-v1'
FRONT_SWITCH_25_VERSION = 'm4-escape-front-switch-p25-v1'
REAR_SWITCH_25_VERSION = 'm4-escape-rear-switch-p25-v1'
REAR_CLEARANCE_25_VERSION = 'm4-escape-rear-clearance-p25-v1'
FRONT_CLEARANCE_25_VERSION = 'm4-escape-front-clearance-p25-v1'
BRAKE_FORWARD_25_VERSION = 'm4-escape-brake-forward-p25-v1'
BRAKE_TUBE_25_VERSION = 'm4-escape-brake-tube-p25-v1'
BRAKE_TUBE_CONFIRM_25_VERSION = 'm4-escape-brake-tube-p25-v2'
RECOVERY_RETREAT_25_VERSION = 'm4-escape-brake-tube-p25-recovery-retreat-v1'
EARLY_RETREAT_25_VERSION = 'm4-escape-brake-tube-p25-early-retreat-v1'
SIDE_HOVER_RETREAT_25_VERSION = 'm4-escape-brake-tube-p25-side-hover-v1'
CLEAR_PASS_25_VERSION = 'm4-escape-brake-tube-p25-clear-pass-v1'
RECOVERY_REARM_25_VERSION = 'm4-escape-brake-tube-p25-recovery-rearm-v1'
RANKED_RETURN_25_VERSION = 'm4-escape-brake-tube-p25-ranked-return-v1'
CURVED_PASS_25_VERSION = 'm4-escape-brake-tube-p25-curved-pass-v1'
HOVER_PATH_25_VERSION = 'm4-escape-brake-tube-p25-hover-path-v1'
STOP_RELEASE_25_VERSION = 'm4-escape-brake-tube-p25-stop-release-v1'
TURN_FIT_25_VERSION = 'm4-escape-brake-tube-p25-turn-fit-v1'
INTERCEPT_25_VERSION = 'm4-escape-brake-tube-p25-intercept-v1'
ALIGNED_INTERCEPT_25_VERSION = 'm4-escape-brake-tube-p25-intercept-v2'
RESERVE_INTERCEPT_25_VERSION = 'm4-escape-brake-tube-p25-intercept-v3'
COMMITTED_INTERCEPT_25_VERSION = 'm4-escape-brake-tube-p25-intercept-v4'
PRIORITY_INTERCEPT_25_VERSION = 'm4-escape-brake-tube-p25-intercept-v5'
COMMAND_HANDOFF_25_VERSION = 'm4-escape-brake-tube-p25-command-handoff-v1'
OFFSET_STOP_25_VERSION = 'm4-escape-brake-tube-p25-offset-stop-v1'
NEIGHBOR_HISTORY_25_VERSION = 'm4-escape-brake-tube-p25-neighbor-history-v1'
CONFIRMED_STANDOFF_25_VERSION = 'm4-escape-brake-tube-p25-confirmed-standoff-v1'
SETTLED_OFFSET_STOP_25_VERSION = 'm4-escape-brake-tube-p25-settled-offset-stop-v1'
WALL_TANGENT_25_VERSION = 'm4-escape-brake-tube-p25-wall-tangent-v1'
CONFIRMED_TUBE_25_VERSION = 'm4-escape-brake-tube-p25-confirmed-inner-tube-v1'
FINITE_EDGE_25_VERSION = 'm4-escape-brake-tube-p25-finite-edge-v1'
EDGE_CONTINUATION_25_VERSION = 'm4-escape-brake-tube-p25-finite-edge-continued-v1'
TANGENT_MAGNITUDE_25_VERSION = 'm4-escape-brake-tube-p25-tangent-magnitude-v1'
ROTATION_TUBE_25_VERSION = 'm4-escape-brake-tube-p25-rotation-aligned-v1'
VERIFIED_RESUME_25_VERSION = 'm4-escape-brake-tube-p25-verified-resume-v1'
VERIFIED_ZERO_25_VERSION = 'm4-escape-brake-tube-p25-verified-zero-v1'
INTERCEPT_CHOICE_25_VERSION = 'm4-escape-brake-tube-p25-intercept-choice-v1'
EARLY_DIAGONAL_25_VERSION = 'm4-escape-brake-tube-p25-early-diagonal-v1'
RESERVE_DIRECTION_25_VERSION = 'm4-escape-brake-tube-p25-reserve-direction-v1'
CROSSING_FALLBACK_25_VERSION = 'm4-escape-brake-tube-p25-crossing-fallback-v1'
OPENING_STEERING_25_VERSION = 'm4-escape-brake-tube-p25-opening-steering-v1'
ALONG_INTERCEPT_25_VERSION = 'm4-escape-brake-tube-p25-along-intercept-v1'
FINITE_INTERCEPT_25_VERSION = 'm4-escape-brake-tube-p25-finite-intercept-v1'
CROSSING_STOP_25_VERSION = 'm4-escape-brake-tube-p25-crossing-stop-v1'
CROSSING_STOP_25_V2 = 'm4-escape-brake-tube-p25-crossing-stop-v2'
CROSSING_STOP_25_V3 = 'm4-escape-brake-tube-p25-crossing-stop-v3'
CROSSING_STOP_25_V4 = 'm4-escape-brake-tube-p25-crossing-stop-v4'
CROSSING_STOP_25_V5 = 'm4-escape-brake-tube-p25-crossing-stop-v5'
CROSSING_STOP_25_V6 = 'm4-escape-brake-tube-p25-crossing-stop-v6'
CROSSING_STOP_25_V7 = 'm4-escape-brake-tube-p25-crossing-stop-v7'
CROSSING_STOP_25_V8 = 'm4-escape-brake-tube-p25-crossing-stop-v8'
CROSSING_STOP_25_V9 = 'm4-escape-brake-tube-p25-crossing-stop-v9'
CROSSING_STOP_25_V10 = 'm4-escape-brake-tube-p25-crossing-stop-v10'
CROSSING_STOP_25_V11 = 'm4-escape-brake-tube-p25-crossing-stop-v11'
CROSSING_STOP_25_V12 = 'm4-escape-brake-tube-p25-crossing-stop-v12'
CROSSING_STOP_25_V13 = 'm4-escape-brake-tube-p25-crossing-stop-v13'
CROSSING_STOP_25_V14 = 'm4-escape-brake-tube-p25-crossing-stop-v14'
CROSSING_STOP_25_V15 = 'm4-escape-brake-tube-p25-crossing-stop-v15'
CROSSING_STOP_25_V16 = 'm4-escape-brake-tube-p25-crossing-stop-v16'
CROSSING_STOP_25_V17 = 'm4-escape-brake-tube-p25-crossing-stop-v17'
CROSSING_STOP_25_V18 = 'm4-escape-brake-tube-p25-crossing-stop-v18'
CROSSING_STOP_25_V19 = 'm4-escape-brake-tube-p25-crossing-stop-v19'
CROSSING_STOP_25_V20 = 'm4-escape-brake-tube-p25-crossing-stop-v20'
CROSSING_STOP_25_V21 = 'm4-escape-brake-tube-p25-crossing-stop-v21'
CROSSING_STOP_25_V22 = 'm4-escape-brake-tube-p25-crossing-stop-v22'
CROSSING_STOP_25_V23 = 'm4-escape-brake-tube-p25-crossing-stop-v23'
CROSSING_STOP_25_V24 = 'm4-escape-brake-tube-p25-crossing-stop-v24'
CROSSING_STOP_25_V25 = 'm4-escape-brake-tube-p25-crossing-stop-v25'
CROSSING_STOP_25_V26 = 'm4-escape-brake-tube-p25-crossing-stop-v26'
CROSSING_STOP_25_V27 = 'm4-escape-brake-tube-p25-crossing-stop-v27'
CROSSING_STOP_25_V28 = 'm4-escape-brake-tube-p25-crossing-stop-v28'
CROSSING_STOP_25_V29 = 'm4-escape-brake-tube-p25-crossing-stop-v29'
CROSSING_STOP_25_V30 = 'm4-escape-brake-tube-p25-crossing-stop-v30'
CROSSING_STOP_25_V31 = 'm4-escape-brake-tube-p25-crossing-stop-v31'
CROSSING_STOP_25_V32 = 'm4-escape-brake-tube-p25-crossing-stop-v32'
CROSSING_STOP_25_V33 = 'm4-escape-brake-tube-p25-crossing-stop-v33'
CROSSING_STOP_25_V34 = 'm4-escape-brake-tube-p25-crossing-stop-v34'
CROSSING_STOP_25_V35 = 'm4-escape-brake-tube-p25-crossing-stop-v35'
CROSSING_STOP_25_V36 = 'm4-escape-brake-tube-p25-crossing-stop-v36'
CROSSING_STOP_25_V37 = 'm4-escape-brake-tube-p25-crossing-stop-v37'
CROSSING_STOP_25_V38 = 'm4-escape-brake-tube-p25-crossing-stop-v38'
CROSSING_STOP_25_V39 = 'm4-escape-brake-tube-p25-crossing-stop-v39'
CROSSING_STOP_25_V40 = 'm4-escape-brake-tube-p25-crossing-stop-v40'
CROSSING_STOP_25_V41 = 'm4-escape-brake-tube-p25-crossing-stop-v41'
RECENT_COMMAND_HANDOFF_25 = 'm4-escape-brake-tube-p25-crossing-stop-v42'
STATIC_RETREAT_HANDOFF_25 = 'm4-escape-brake-tube-p25-crossing-stop-v43'
GRAVITY_ALIGNED_PROTECTION_25 = 'm4-escape-brake-tube-p25-crossing-stop-v44'
RELEASE_STOP_CERTIFICATES_25 = 'm4-escape-brake-tube-p25-crossing-stop-v45'
STEADY_HANDOFF_25 = 'm4-escape-brake-tube-p25-crossing-stop-v46'
TANGENT_RESPONSE_HEADROOM_25 = 'm4-escape-brake-tube-p25-crossing-stop-v47'
NEUTRAL_RETRY_25 = 'm4-escape-brake-tube-p25-crossing-stop-v48'
OBSERVED_TURN_BRAKING_25 = 'm4-escape-brake-tube-p25-crossing-stop-v49'
FRONT_WALL_HANDOFF_25 = 'm4-escape-brake-tube-p25-crossing-stop-v50'
COLLECTIVE_PROJECTION_25 = 'm4-escape-brake-tube-p25-crossing-stop-v51'
STATIC_TURN_ONSET_25 = 'm4-escape-brake-tube-p25-crossing-stop-v52'
CORRIDOR_PROJECTION_25 = 'm4-escape-brake-tube-p25-crossing-stop-v53'
MOUNT_PITCH_DEGREES = {MOUNT_25_VERSION: 25, MOUNT_28_VERSION: 28,
                       REDIRECT_25_VERSION: 25, FALLBACK_25_VERSION: 25, FRONT_SWITCH_25_VERSION: 25,
                       REAR_SWITCH_25_VERSION: 25, REAR_CLEARANCE_25_VERSION: 25, FRONT_CLEARANCE_25_VERSION: 25,
                       BRAKE_FORWARD_25_VERSION: 25, BRAKE_TUBE_25_VERSION: 25, BRAKE_TUBE_CONFIRM_25_VERSION: 25,
                       RECOVERY_RETREAT_25_VERSION: 25, EARLY_RETREAT_25_VERSION: 25,
                       SIDE_HOVER_RETREAT_25_VERSION: 25, CLEAR_PASS_25_VERSION: 25, RECOVERY_REARM_25_VERSION: 25,
                       RANKED_RETURN_25_VERSION: 25, CURVED_PASS_25_VERSION: 25, HOVER_PATH_25_VERSION: 25,
                       STOP_RELEASE_25_VERSION: 25, TURN_FIT_25_VERSION: 25, INTERCEPT_25_VERSION: 25,
                       ALIGNED_INTERCEPT_25_VERSION: 25, RESERVE_INTERCEPT_25_VERSION: 25,
                       COMMITTED_INTERCEPT_25_VERSION: 25, PRIORITY_INTERCEPT_25_VERSION: 25,
                       COMMAND_HANDOFF_25_VERSION: 25, OFFSET_STOP_25_VERSION: 25,
                       NEIGHBOR_HISTORY_25_VERSION: 25, CONFIRMED_STANDOFF_25_VERSION: 25,
                       SETTLED_OFFSET_STOP_25_VERSION: 25, WALL_TANGENT_25_VERSION: 25,
                       CONFIRMED_TUBE_25_VERSION: 25, FINITE_EDGE_25_VERSION: 25, EDGE_CONTINUATION_25_VERSION: 25,
                       TANGENT_MAGNITUDE_25_VERSION: 25, ROTATION_TUBE_25_VERSION: 25, VERIFIED_RESUME_25_VERSION: 25,
                       VERIFIED_ZERO_25_VERSION: 25, INTERCEPT_CHOICE_25_VERSION: 25, EARLY_DIAGONAL_25_VERSION: 25,
                       RESERVE_DIRECTION_25_VERSION: 25, CROSSING_FALLBACK_25_VERSION: 25,
                       OPENING_STEERING_25_VERSION: 25, ALONG_INTERCEPT_25_VERSION: 25,
                       FINITE_INTERCEPT_25_VERSION: 25, CROSSING_STOP_25_VERSION: 25,
                       CROSSING_STOP_25_V2: 25, CROSSING_STOP_25_V3: 25,
                       CROSSING_STOP_25_V4: 25, CROSSING_STOP_25_V5: 25, CROSSING_STOP_25_V6: 25,
                       CROSSING_STOP_25_V7: 25, CROSSING_STOP_25_V8: 25, CROSSING_STOP_25_V9: 25,
                       CROSSING_STOP_25_V10: 25, CROSSING_STOP_25_V11: 25, CROSSING_STOP_25_V12: 25,
                       CROSSING_STOP_25_V13: 25, CROSSING_STOP_25_V14: 25, CROSSING_STOP_25_V15: 25,
                       CROSSING_STOP_25_V16: 25, CROSSING_STOP_25_V17: 25, CROSSING_STOP_25_V18: 25,
                       CROSSING_STOP_25_V19: 25, CROSSING_STOP_25_V20: 25, CROSSING_STOP_25_V21: 25,
                       CROSSING_STOP_25_V22: 25, CROSSING_STOP_25_V23: 25, CROSSING_STOP_25_V24: 25,
                       CROSSING_STOP_25_V25: 25, CROSSING_STOP_25_V26: 25, CROSSING_STOP_25_V27: 25,
                       CROSSING_STOP_25_V28: 25, CROSSING_STOP_25_V29: 25, CROSSING_STOP_25_V30: 25,
                       CROSSING_STOP_25_V31: 25, CROSSING_STOP_25_V32: 25, CROSSING_STOP_25_V33: 25,
                       CROSSING_STOP_25_V34: 25, CROSSING_STOP_25_V35: 25, CROSSING_STOP_25_V36: 25,
                       CROSSING_STOP_25_V37: 25, CROSSING_STOP_25_V38: 25, CROSSING_STOP_25_V39: 25,
                       CROSSING_STOP_25_V40: 25, CROSSING_STOP_25_V41: 25, RECENT_COMMAND_HANDOFF_25: 25,
                       STATIC_RETREAT_HANDOFF_25: 25, GRAVITY_ALIGNED_PROTECTION_25: 25,
                       RELEASE_STOP_CERTIFICATES_25: 25, STEADY_HANDOFF_25: 25,
                       TANGENT_RESPONSE_HEADROOM_25: 25, NEUTRAL_RETRY_25: 25, OBSERVED_TURN_BRAKING_25: 25,
                       FRONT_WALL_HANDOFF_25: 25, COLLECTIVE_PROJECTION_25: 25, STATIC_TURN_ONSET_25: 25,
                       CORRIDOR_PROJECTION_25: 25}
FAST_LATERAL_SPEED_M_S = .45
RECOVERY_DELAY = 28
RECOVERY_LIMIT = 65
RECENTER_REARM_STEPS = 10
OUTWARD_BRAKE_RENEWAL_STEPS = 10
COMPOSED_CROSSING_RECHECKS = 5


class EscapeRecovery(EscapeOption):
    def __init__(self, *, lanes, device, dtype, recenter, lateral=False, fast_lateral=False, redirect=False,
                 fallback=False, front_switch=False, rear_switch=False, rear_clearance=False, front_clearance=False,
                 brake_forward_escape=False, recovery_retreat=False, early_retreat=False, side_hover_retreat=False,
                 clear_pass=False, recenter_rearm=False, rank_return=False, curved_pass=False, hover_path=False,
                 stop_release=False, turn_fallback=False, intercept=False, align_intercept=False,
                 intercept_reserve=False, command_handoff=False, commit_intercept=False, retreat_priority=False,
                 offset_stop=False, settled_offset_stop=False, wall_tangent=False, finite_edge=False,
                 edge_continuation=False, tangent_magnitude=False, verified_resume=False, verified_zero=False,
                 intercept_choice=False, receding_stop=False, early_diagonal=False, reserve_direction=False,
                 zero_recheck=False, reverse_diagonal=False, opening_steering=False, along_intercept=False,
                 finite_intercept=False, crossing_stop=False, crossing_stop_continuation=False,
                 crossing_stop_shallow=False, crossing_stop_early=False, crossing_stop_compatible=False,
                 crossing_stop_verified_hover=False, crossing_stop_receding_hover=False,
                 crossing_retreat_own_side=False, crossing_retreat_extended=False, settled_crossing_mount=None,
                 static_scan_stop=False, crossing_reference=False, shallow_steering=False,
                 observed_heading_steering=False, tracked_resume=False, crossing_turn_between_samples=False,
                 hover_escape_own_side=False, far_crossing_reference=False, verified_retreat_onset=False,
                 retain_only_clipped_retreat=False, verify_hover_reference_reserve=False,
                 verify_crossing_restart=False, verify_overlapping_stop=False, verify_brake_recovery=False,
                 verify_retreat_continuation=False, verify_brake_transition=False, verify_oldest_partial_turn=False,
                 verify_zero_crossing_withdrawal=False, verify_ranked_reversal_stop=False,
                 verify_retreat_onset_side=False, verify_retreat_stop=False, retry_command_handoff=False,
                 verify_outward_brake=False, verify_composed_crossing=False, verify_recent_handoff=False,
                 verify_static_retreat_handoff=False, align_planar_protection=False,
                 verify_release_stop=False, verify_steady_handoff=False, tangent_response_headroom=False,
                 verify_neutral_retry=False, observed_turn_braking=False, verify_static_turn=False):
        super().__init__(lanes=lanes, device=device, dtype=dtype)
        self.recenter = recenter
        self.lateral = lateral
        self.fast_lateral = fast_lateral
        self.redirect = redirect
        self.fallback = fallback
        self.front_switch = front_switch
        self.front_switched = torch.zeros_like(self.applied)
        self.rear_switch = rear_switch
        self.rear_clearance = rear_clearance
        self.front_clearance = front_clearance
        self.brake_forward_escape = brake_forward_escape
        self.braking = torch.zeros_like(self.applied)
        self.recovery_retreat = recovery_retreat
        self.early_retreat = early_retreat
        self.verified_retreat_onset = verified_retreat_onset
        self.retain_only_clipped_retreat = retain_only_clipped_retreat
        self.verify_static_turn = verify_static_turn
        if verify_static_turn:
            if not verified_retreat_onset:
                raise ValueError('Static turn recheck requires verified retreat onset')
            self.static_turn_floor = self.direction.new_zeros(lanes)
        if retain_only_clipped_retreat and not verified_retreat_onset:
            raise ValueError('Clipped retreat handoff requires verified retreat onset')
        if verified_retreat_onset:
            if not (early_retreat and align_intercept):
                raise ValueError('Verified retreat onset requires early retreat and aligned command history')
            self.retreat_request_kept = torch.zeros_like(self.applied)
        self.side_hover_retreat = side_hover_retreat
        self.clear_pass = clear_pass
        self.curved_pass = curved_pass
        self.hover_path = hover_path
        self.turn_fallback = turn_fallback
        self.intercept = intercept
        self.align_intercept = align_intercept
        self.intercept_reserve = intercept_reserve
        self.commit_intercept = commit_intercept
        self.retreat_priority = retreat_priority
        if commit_intercept:
            self.intercept_steps = torch.zeros_like(self.remaining)
            self.intercept_direction = torch.zeros_like(self.direction)
            self.intercept_continued = torch.zeros_like(self.applied)
        self.command_handoff = command_handoff
        self.align_planar_protection = align_planar_protection
        self.verify_release_stop = verify_release_stop
        if verify_release_stop:
            if not (verify_composed_crossing and crossing_stop_receding_hover and verify_static_retreat_handoff):
                raise ValueError('ESCAPE_RELEASE_STOP_REQUIRES_CURRENT_PROTECTION')
            self.release_wall_witness = torch.zeros_like(self.applied)
        self.verify_neutral_retry = verify_neutral_retry
        if verify_neutral_retry:
            if not (verify_release_stop and crossing_stop_verified_hover):
                raise ValueError('ESCAPE_NEUTRAL_RETRY_REQUIRES_VERIFIED_RELEASE')
            from .neutral_retry import NeutralRetryObserver

            self.neutral_retry_observer = NeutralRetryObserver(lanes, device)
            self.neutral_retry_pending = torch.zeros_like(self.applied)
            self.neutral_retry_floor = self.direction.new_zeros(lanes)
        self.observed_turn_braking = observed_turn_braking
        if observed_turn_braking:
            if not (verify_neutral_retry and verify_steady_handoff and align_intercept):
                raise ValueError('ESCAPE_OBSERVED_TURN_REQUIRES_CURRENT_PROTECTION')
            from .observed_turn_braking import ObservedTurnBrake

            self.turn_brake = ObservedTurnBrake(lanes)
            self.observed_turn_brake_floor = self.direction.new_zeros(lanes)
        self.retry_command_handoff = retry_command_handoff
        self.verify_recent_handoff = verify_recent_handoff
        if verify_recent_handoff and not retry_command_handoff:
            raise ValueError('ESCAPE_RECENT_HANDOFF_REQUIRES_COMMAND_RETRY')
        self.verify_steady_handoff = verify_steady_handoff
        if verify_steady_handoff:
            if not (verify_recent_handoff and verify_composed_crossing):
                raise ValueError('ESCAPE_STEADY_HANDOFF_REQUIRES_RECENT_AND_CROSSING_PROOFS')
            self.steady_handoff_witness = torch.zeros_like(self.applied)
            self.steady_handoff_used = torch.zeros_like(self.applied)
        if retry_command_handoff:
            if not (command_handoff and clear_pass and static_scan_stop and commit_intercept):
                raise ValueError('Retrying a handoff requires evidence for the active protections')
            self.handoff_pending = torch.zeros_like(self.applied)
        self.offset_stop = offset_stop
        self.settled_offset_stop = settled_offset_stop
        self.wall_tangent = wall_tangent
        self.tangent_magnitude = tangent_magnitude
        self.tangent_response_headroom = tangent_response_headroom
        if tangent_response_headroom and not (wall_tangent and tangent_magnitude):
            raise ValueError('ESCAPE_TANGENT_HEADROOM_REQUIRES_VERIFIED_TANGENT')
        self.finite_edge = finite_edge
        self.edge_continuation = edge_continuation
        self.verified_resume = verified_resume
        self.verified_zero = verified_zero
        self.intercept_choice = intercept_choice
        self.receding_stop = receding_stop
        self.early_diagonal = early_diagonal
        self.reverse_diagonal = reverse_diagonal
        self.along_intercept = along_intercept
        self.finite_intercept = finite_intercept
        self.crossing_stop = crossing_stop
        self.crossing_stop_continuation = crossing_stop_continuation
        self.crossing_stop_shallow = crossing_stop_shallow
        self.crossing_stop_early = crossing_stop_early
        self.crossing_stop_compatible = crossing_stop_compatible
        self.crossing_stop_verified_hover = crossing_stop_verified_hover
        self.crossing_stop_receding_hover = crossing_stop_receding_hover
        self.crossing_retreat_own_side = crossing_retreat_own_side
        self.crossing_retreat_extended = crossing_retreat_extended
        self.settled_crossing_mount = settled_crossing_mount
        self.crossing_turn_between_samples = crossing_turn_between_samples
        self.verify_hover_reference_reserve = verify_hover_reference_reserve
        if verify_hover_reference_reserve and not (crossing_turn_between_samples and crossing_reference):
            raise ValueError('Hover reserve requires sampled reversals and a static reference')
        self.verify_oldest_partial_turn = verify_oldest_partial_turn
        if verify_oldest_partial_turn:
            if not verify_hover_reference_reserve:
                raise ValueError('Partial oldest turn requires sampled reversals and a static reference')
            self.oldest_partial_turn_verified = torch.zeros_like(self.applied)
        self.verify_zero_crossing_withdrawal = verify_zero_crossing_withdrawal
        if verify_zero_crossing_withdrawal:
            if not (verify_oldest_partial_turn and commit_intercept):
                raise ValueError('Zero crossing withdrawal requires the observed crossing controller')
            self.zero_crossing_ticks = torch.zeros_like(self.remaining)
            self.zero_crossing_commit = torch.zeros_like(self.remaining)
            self.zero_crossing_mode = torch.zeros_like(self.remaining)
            self.zero_crossing_floor = self.direction.new_zeros(lanes)
        self.verify_ranked_reversal_stop = verify_ranked_reversal_stop
        if verify_ranked_reversal_stop:
            if not verify_zero_crossing_withdrawal:
                raise ValueError('Ranked reversal proof requires verified crossing withdrawal')
            self.ranked_reversal_verified = torch.zeros_like(self.applied)
        self.verify_retreat_onset_side = verify_retreat_onset_side
        if verify_retreat_onset_side:
            if not (crossing_retreat_own_side and reserve_direction):
                raise ValueError('Retreat onset side proof requires same-side retreat and a reserve observation')
            self.retreat_onset_side_floor = self.direction.new_zeros(lanes)
        self.verify_retreat_stop = verify_retreat_stop
        if verify_retreat_stop:
            if not crossing_stop_verified_hover:
                raise ValueError('Retreat stop proof requires verified crossing hover')
            self.retreat_stop_floor = self.direction.new_zeros(lanes)
        self.verify_crossing_restart = verify_crossing_restart
        if verify_crossing_restart:
            if not (verify_hover_reference_reserve and tracked_resume and align_intercept and hover_escape_own_side):
                raise ValueError('Crossing restart requires the tracked crossing controller')
            self.crossing_restart_verified = torch.zeros_like(self.applied)
            self.crossing_restart_previous_b = torch.zeros_like(self.direction)
            self.crossing_restart_steps = torch.zeros_like(self.remaining)
        self.verify_brake_recovery = verify_brake_recovery
        if verify_brake_recovery:
            if not verify_crossing_restart:
                raise ValueError('Brake recovery requires verified crossing restart')
            self.brake_recovery_verified = torch.zeros_like(self.applied)
            self.brake_recovery_bounds = self.direction.new_zeros((lanes, 2))
        self.verify_brake_transition = verify_brake_transition
        if verify_brake_transition:
            if not verify_brake_recovery:
                raise ValueError('Brake transition requires verified brake recovery')
            self.brake_transition_steps = torch.zeros_like(self.remaining)
        self.verify_outward_brake = verify_outward_brake
        if verify_outward_brake:
            if not verify_brake_transition:
                raise ValueError('Outward brake renewal requires verified brake transition')
            self.outward_brake_steps = torch.zeros_like(self.remaining)
            self.outward_brake_anchor = self.direction.new_zeros((lanes, 7))
        self.verify_composed_crossing = verify_composed_crossing
        if verify_composed_crossing:
            if not (align_intercept and finite_edge and clear_pass):
                raise ValueError('Composed crossing verification requires aligned recovery and finite edges')
            self.composed_crossing_steps = torch.zeros_like(self.remaining)
            self.composed_crossing_request = torch.zeros_like(self.direction)
            self.composed_crossing_verified = torch.zeros_like(self.applied)
            self.composed_crossing_floor = self.direction.new_zeros(lanes)
        self.verify_retreat_continuation = verify_retreat_continuation
        if verify_retreat_continuation:
            if not (verify_overlapping_stop and retreat_priority and commit_intercept):
                raise ValueError('Retreat continuation requires the verified crossing controller')
            self.retreat_continuation_steps = torch.zeros_like(self.remaining)
            self.retreat_continuation_verified = torch.zeros_like(self.applied)
            self.retreat_continuation_floor = self.direction.new_zeros((lanes,))
        self.hover_escape_own_side = hover_escape_own_side
        if hover_escape_own_side:
            if not (crossing_turn_between_samples and crossing_stop and rank_return):
                raise ValueError('Hover escape direction requires the settled crossing profile')
            self.hover_escape_redirected = torch.zeros_like(self.applied)
            self.hover_escape_continuation = torch.zeros_like(self.applied)
            self.hover_escape_direction = torch.zeros_like(self.direction)
        if crossing_turn_between_samples and settled_crossing_mount is None:
            raise ValueError('Settled crossing turn requires the calibrated mount')
        self.static_scan_stop = static_scan_stop
        self.verify_overlapping_stop = verify_overlapping_stop
        if verify_overlapping_stop:
            if not (static_scan_stop and verify_crossing_restart):
                raise ValueError('Overlapping scan stop requires the static and crossing checks')
            self.overlapping_stop_verified = torch.zeros_like(self.applied)
        self.crossing_reference = crossing_reference
        self.far_crossing_reference = far_crossing_reference
        if far_crossing_reference:
            if not (crossing_reference and crossing_stop_early):
                raise ValueError('Far crossing direction requires early crossing and plane support')
            self.far_crossing_refined = torch.zeros_like(self.applied)
        self.shallow_steering = shallow_steering
        self.observed_heading_steering = observed_heading_steering
        self.tracked_resume = tracked_resume
        if observed_heading_steering and not shallow_steering:
            raise ValueError('Observed heading steering requires the shallow steering profile')
        if tracked_resume and not (crossing_reference and commit_intercept and rank_return and clear_pass
                                   and static_scan_stop and command_handoff and crossing_stop_early):
            raise ValueError('Tracked recovery requires crossing, held escape and stop checks')
        if shallow_steering:
            if not (crossing_reference and finite_edge and crossing_stop_early and wall_tangent
                    and command_handoff and rank_return):
                raise ValueError('Shallow steering requires crossing, finite edge, wall and ranked escape checks')
            self.shallow_used = torch.zeros_like(self.applied)
        if observed_heading_steering:
            self.observed_heading_used = torch.zeros_like(self.applied)
        if tracked_resume:
            self.tracked_resume_used = torch.zeros_like(self.applied)
            self.tracked_resume_witness = torch.zeros_like(self.applied)
        if static_scan_stop:
            self.static_stop_verified = torch.zeros_like(self.applied)
        if settled_crossing_mount is not None:
            if not (crossing_stop_verified_hover and rank_return):
                raise ValueError('Settled crossing requires verified hover and ranked escape')
            self.settled_crossing_verified = torch.zeros_like(self.applied)
        if crossing_retreat_extended and not crossing_retreat_own_side:
            raise ValueError('Extended crossing retreat requires a certified same-side retreat')
        if crossing_retreat_own_side and not (retreat_priority and crossing_stop_early):
            raise ValueError('Crossing retreat requires retreat priority and early crossing')
        if crossing_stop_receding_hover and not crossing_stop_verified_hover:
            raise ValueError('Receding hover requires verified hover')
        if crossing_stop_verified_hover and not crossing_stop_compatible:
            raise ValueError('Verified crossing hover requires compatible crossing continuation')
        if crossing_stop_compatible and not crossing_stop_early:
            raise ValueError('Compatible crossing continuation requires the earlier crossing profile')
        if crossing_stop_early:
            if not crossing_stop_shallow:
                raise ValueError('Early crossing requires the confirmed crossing fallback')
            self.early_crossing = torch.zeros_like(self.applied)
        if crossing_stop_shallow and not crossing_stop_continuation:
            raise ValueError('Crossing stop shallow retry requires same-side continuation')
        if crossing_stop_continuation and not crossing_stop:
            raise ValueError('Crossing stop continuation requires crossing stop')
        if crossing_stop and not finite_intercept:
            raise ValueError('Crossing stop requires finite interception')
        if finite_intercept and not (along_intercept and reserve_direction):
            raise ValueError('Finite interception requires recalled along-track steering')
        if along_intercept:
            if not (intercept_choice and commit_intercept and intercept_reserve):
                raise ValueError('Along-track interception requires ranked and committed reserve actions')
            self.intercept_speed = self.direction.new_full((lanes,), SPEED_M_S)
        self.opening_steering = opening_steering
        if opening_steering:
            self.opening_steered = torch.zeros_like(self.applied)
        self.reserve_direction = reserve_direction
        if reserve_direction:
            self.reserve_direction_seen = torch.zeros_like(self.applied)
        if verified_resume:
            self.resume_witness = torch.zeros_like(self.applied)
            self.resumed = torch.zeros_like(self.applied)
        if wall_tangent:
            self.tangent_used = torch.zeros_like(self.applied)
        if finite_edge:
            self.detour_used = torch.zeros_like(self.applied)
        if edge_continuation:
            self.edge_steps = torch.zeros_like(self.remaining)
            self.edge_side = torch.zeros_like(self.applied)
            self.edge_continued = torch.zeros_like(self.applied)
        if offset_stop:
            self.offset_stopped = torch.zeros_like(self.applied)
        if command_handoff:
            self.handed_off = torch.zeros_like(self.applied)
        if intercept:
            self.intercepted = torch.zeros_like(self.applied)
        if align_intercept:
            self.command_offsets = self.direction.new_zeros((lanes, 4, 2))
            self.command_coast = self.direction.new_zeros((lanes, 2))
        self.stop_release = stop_release
        self.zero_recheck = zero_recheck
        if stop_release:
            self.release_checks = torch.zeros_like(self.remaining)
            self.released = torch.zeros_like(self.applied)
        if clear_pass:
            self.holding = torch.zeros_like(self.applied)
        self.recenter_rearm = recenter_rearm
        if recenter_rearm:
            self.rearm_steps = torch.zeros_like(self.remaining)
        self.rank_return = rank_return
        if rank_return:
            self.ranked = torch.zeros_like(self.applied)
        self.retreat_hold_steps = torch.zeros_like(self.remaining)
        self.retreat_hold_direction = torch.zeros_like(self.direction)
        self.retreating = torch.zeros_like(self.applied)
        self.verify_static_retreat_handoff = verify_static_retreat_handoff
        if verify_static_retreat_handoff:
            if not (early_retreat and align_intercept):
                raise ValueError('ESCAPE_STATIC_HANDOFF_REQUIRES_ALIGNED_RETREAT')
            self.static_retreat_witness = torch.zeros_like(self.applied)
        self.rear_switched = torch.zeros_like(self.applied)
        self.escape_speed = self.direction.new_full((lanes,), SPEED_M_S)
        self.retreat_basis = torch.zeros_like(self.direction)
        self.redirected = torch.zeros_like(self.applied)
        self.since_escape = torch.full_like(self.remaining, RECOVERY_LIMIT)
        self.recovering = torch.zeros_like(self.applied)

    def reset(self, lanes):
        super().reset(lanes)
        if self.static_scan_stop:
            self.static_stop_verified[lanes] = False
        if self.verify_overlapping_stop:
            self.overlapping_stop_verified[lanes] = False
        if self.settled_crossing_mount is not None:
            self.settled_crossing_verified[lanes] = False
        if self.verify_oldest_partial_turn:
            self.oldest_partial_turn_verified[lanes] = False
        if self.verify_zero_crossing_withdrawal:
            self.zero_crossing_ticks[lanes] = 0
            self.zero_crossing_commit[lanes] = 0
            self.zero_crossing_mode[lanes] = 0
            self.zero_crossing_floor[lanes] = 0.
        if self.verify_ranked_reversal_stop:
            self.ranked_reversal_verified[lanes] = False
        if self.verify_retreat_onset_side:
            self.retreat_onset_side_floor[lanes] = 0.
        if self.verify_retreat_stop:
            self.retreat_stop_floor[lanes] = 0.
        if self.opening_steering:
            self.opening_steered[lanes] = False
        self.since_escape[lanes] = RECOVERY_LIMIT
        self.recovering[lanes] = False
        self.escape_speed[lanes] = SPEED_M_S
        self.retreat_basis[lanes] = 0.
        self.redirected[lanes] = False
        self.front_switched[lanes] = False
        self.rear_switched[lanes] = False
        self.braking[lanes] = False
        self.retreat_hold_steps[lanes] = 0
        self.retreat_hold_direction[lanes] = 0.
        self.retreating[lanes] = False
        if self.verify_static_retreat_handoff:
            self.static_retreat_witness[lanes] = False
        if self.verify_release_stop:
            self.release_wall_witness[lanes] = False
        if self.verify_neutral_retry:
            self.neutral_retry_observer.reset(lanes)
            self.neutral_retry_pending[lanes] = False
            self.neutral_retry_floor[lanes] = 0.
        if self.observed_turn_braking:
            self.turn_brake.reset(lanes)
            self.observed_turn_brake_floor[lanes] = 0.
        if self.verify_steady_handoff:
            self.steady_handoff_witness[lanes] = False
            self.steady_handoff_used[lanes] = False
        if self.verified_retreat_onset:
            self.retreat_request_kept[lanes] = False
        if self.verify_static_turn:
            self.static_turn_floor[lanes] = 0.
        if self.verify_crossing_restart:
            self.crossing_restart_verified[lanes] = False
            self.crossing_restart_previous_b[lanes] = 0.
            self.crossing_restart_steps[lanes] = 0
        if self.verify_brake_recovery:
            self.brake_recovery_verified[lanes] = False
            self.brake_recovery_bounds[lanes] = 0.
        if self.verify_brake_transition:
            self.brake_transition_steps[lanes] = 0
        if self.verify_outward_brake:
            self.outward_brake_steps[lanes] = 0
            self.outward_brake_anchor[lanes] = 0.
        if self.verify_composed_crossing:
            self.composed_crossing_steps[lanes] = 0
            self.composed_crossing_request[lanes] = 0.
            self.composed_crossing_verified[lanes] = False
            self.composed_crossing_floor[lanes] = 0.
        if self.verify_retreat_continuation:
            self.retreat_continuation_steps[lanes] = 0
            self.retreat_continuation_verified[lanes] = False
            self.retreat_continuation_floor[lanes] = 0.
        if self.recenter_rearm:
            self.rearm_steps[lanes] = 0
        if self.rank_return:
            self.ranked[lanes] = False
        if self.stop_release:
            self.release_checks[lanes] = 0
            self.released[lanes] = False
        if self.intercept:
            self.intercepted[lanes] = False
        if self.shallow_steering:
            self.shallow_used[lanes] = False
        if self.observed_heading_steering:
            self.observed_heading_used[lanes] = False
        if self.tracked_resume:
            self.tracked_resume_used[lanes] = False
            self.tracked_resume_witness[lanes] = False
        if self.hover_escape_own_side:
            self.hover_escape_redirected[lanes] = False
            self.hover_escape_continuation[lanes] = False
            self.hover_escape_direction[lanes] = 0.
        if self.commit_intercept:
            self.intercept_steps[lanes] = 0
            self.intercept_direction[lanes] = 0.
            self.intercept_continued[lanes] = False
        if self.along_intercept:
            self.intercept_speed[lanes] = SPEED_M_S
        if self.crossing_stop_early:
            self.early_crossing[lanes] = False
        if self.far_crossing_reference:
            self.far_crossing_refined[lanes] = False
        if self.reserve_direction:
            self.reserve_direction_seen[lanes] = False
        if self.align_intercept:
            self.command_offsets[lanes] = 0.
            self.command_coast[lanes] = 0.
        if self.command_handoff:
            self.handed_off[lanes] = False
        if self.retry_command_handoff:
            self.handoff_pending[lanes] = False
        if self.offset_stop:
            self.offset_stopped[lanes] = False
        if self.wall_tangent:
            self.tangent_used[lanes] = False
        if self.finite_edge:
            self.detour_used[lanes] = False
        if self.edge_continuation:
            self.edge_steps[lanes] = 0
            self.edge_side[lanes] = False
            self.edge_continued[lanes] = False
        if self.verified_resume:
            self.resume_witness[lanes] = False
            self.resumed[lanes] = False

    def issued(self, lidar, state, action, config):
        if self.align_planar_protection:
            from .escape_passage import align_planar_protective_action

            action = align_planar_protective_action(action, state, config.governor)
        request = decode_governor(action, state, config.governor).v_gov_b
        return attenuate_command(lidar, self.detector._full_rays, request, .1, config.ics, state_vec=state)

    def select_lateral(self, lidar, state, selected, config):
        # Keep the same trigger and hold. Select using the command this path can
        # actually issue at the escape speed, including the shared emergency stop.
        lanes = (self.applied & (self.remaining == DURATION_STEPS - 1)).nonzero().flatten()
        if not len(lanes):
            return
        local_lidar, local_state = lidar[lanes], state[lanes]
        retreat = self.detector.retreat_direction(local_lidar, local_state, hover_only=False)
        if self.redirect:
            self.retreat_basis[lanes] = retreat
        side = torch.stack((-retreat[:, 1], retreat[:, 0], retreat[:, 2]), dim=1)
        baseline = self.issued(local_lidar, local_state, selected[lanes], config).v_final_b
        best_score = (baseline * side).sum(1).abs() + .035
        best_direction = self.direction[lanes]
        for axial in (0., .45, .85):
            for sign in (1., -1.):
                direction = (sign * side + axial * retreat) / (1. + axial**2)**.5
                trial = torch.cat((torch.zeros_like(local_state[:, :1]), .5 + SPEED_M_S * direction), dim=1)
                issued = self.issued(local_lidar, local_state, trial, config).v_final_b
                score = (issued * side).sum(1).abs()
                better = score > best_score
                best_direction = torch.where(better[:, None], direction, best_direction)
                best_score = torch.where(better, score, best_score)
        if self.rear_clearance:
            best_direction = self.choose_rear_clearance(local_lidar, local_state, retreat, side, best_direction, config)
        if self.front_clearance:
            best_direction = self.choose_front_clearance(local_lidar, local_state, retreat, side, best_direction, config)
        if self.side_hover_retreat:
            best_direction = self.choose_side_hover_retreat(local_lidar, local_state, retreat, best_direction, config)
        if self.rank_return:
            best_direction = self.choose_after_return(lidar, state, lanes, retreat, best_direction, config)
        self.direction[lanes] = best_direction
        selected[lanes, 1:] = .5 + SPEED_M_S * best_direction
        if self.fast_lateral:
            self.escape_speed[lanes] = SPEED_M_S
            command, _ = extract_latest_commands(local_state)
            rear_hover = ((command.norm(dim=1) <= .001) & (retreat[:, 0] > .65)
                          & ((side * best_direction).sum(1).abs() > .5))
            if bool(rear_hover.any()):
                trial = torch.cat((torch.zeros_like(local_state[:, :1]),
                                   .5 + FAST_LATERAL_SPEED_M_S * best_direction), dim=1)
                issued = self.issued(local_lidar[rear_hover], local_state[rear_hover], trial[rear_hover], config).v_final_b
                useful = (issued * side[rear_hover]).sum(1).abs() > best_score[rear_hover] + .035
                self.escape_speed[lanes[rear_hover][useful]] = FAST_LATERAL_SPEED_M_S
        if self.opening_steering:
            self.steer_initial_escape(lidar, state, selected, config, lanes)

    def steer_initial_escape(self, lidar, state, selected, config, lanes):
        from .escape_tangent import wall_opening_direction

        command, previous = extract_latest_commands(state)
        speed = command[lanes].norm(dim=1)
        lanes = lanes[(speed >= .16) & (speed <= .30)]
        if not len(lanes):
            return
        scans, history = lidar[lanes], state[lanes]
        confirmed, heading, opening = wall_opening_direction(scans, history, self.detector._full_rays)
        opposite = (self.direction[lanes]*opening).sum(dim=1) <= -.35
        eligible = (confirmed & opposite).nonzero().flatten()
        if not len(eligible):
            return
        lanes, heading, opening = lanes[eligible], heading[eligible], opening[eligible]
        speed = command[lanes].norm(dim=1).clamp_max(SPEED_M_S-.0001)
        direction = (heading+opening)*.5**.5
        trial = self.action_for_velocity(speed[:, None]*direction, previous[lanes], config.governor)
        issued = self.issued(lidar[lanes], state[lanes], trial, config)
        valid = ((issued.metrics['beta'][:, 0] >= .90) & (issued.metrics['emergency'][:, 0] == 0.)
                 & ((issued.v_final_b*heading).sum(dim=1) >= .60*speed)
                 & ((issued.v_final_b*opening).sum(dim=1) >= .12))
        chosen = lanes[valid]
        self.direction[chosen] = direction[valid]
        self.escape_speed[chosen] = speed[valid]
        selected[chosen] = trial[valid]
        self.opening_steered[chosen] = True

    def release_opening_on_command_change(self, state, selected, actor):
        self.opening_steered &= self.applied & (self.remaining < DURATION_STEPS-1)
        recent = state.reshape(-1, 4, 13)[:, -3:, 6:9]
        changed = (recent[:, :-1]-recent[:, -1:]).norm(dim=2).amax(dim=1) > .04
        release = self.opening_steered & changed
        selected[release] = actor[release]
        self.applied[release] = False
        self.remaining[release] = 0
        self.opening_steered[release] = False

    def choose_rear_clearance(self, lidar, state, retreat, side, direction, config):
        from .escape_clearance import wider_side

        command, _ = extract_latest_commands(state)
        previous_side = (side * direction).sum(dim=1)
        eligible = (command.norm(dim=1) <= .001) & (retreat[:, 0] > .65) & (previous_side.abs() > .5)
        lanes = eligible.nonzero().flatten()
        if not len(lanes):
            return direction
        sign = wider_side(lidar[lanes], state[lanes], self.detector._full_rays, retreat[lanes], side[lanes])
        changed = sign * previous_side[lanes] < 0.
        lanes, sign = lanes[changed], sign[changed]
        if not len(lanes):
            return direction
        preferred = side[lanes] * sign[:, None]
        trial = torch.cat((torch.zeros_like(state[lanes, :1]), .5 + SPEED_M_S * preferred), dim=1)
        command = self.issued(lidar[lanes], state[lanes], trial, config)
        viable = ((command.v_final_b * preferred).sum(dim=1) > .08) & (command.metrics['emergency'].flatten() == 0.)
        direction[lanes[viable]] = preferred[viable]
        return direction

    def choose_front_clearance(self, lidar, state, retreat, side, direction, config):
        command, _ = extract_latest_commands(state)
        lateral = (direction * side).sum(dim=1)
        lanes = ((command.norm(dim=1) <= .001) & (retreat[:, 0] < -.65) & (lateral.abs() > .85)).nonzero().flatten()
        if not len(lanes):
            return direction
        baseline = side[lanes] * lateral[lanes, None].sign()
        gain = torch.zeros_like(state[lanes, :1])
        left, right = (self.issued(lidar[lanes], state[lanes], torch.cat((gain, .5 + sign*SPEED_M_S*baseline), dim=1),
                                   config).metrics for sign in (1., -1.))
        unattenuated = (left['beta'] == 1.) & (right['beta'] == 1.) & (left['emergency'] == 0.) & (right['emergency'] == 0.)
        switch = (unattenuated & (left['active_beam_count'] > 0.) & (right['active_beam_count'] == 0.)).flatten()
        direction[lanes[switch]] = -baseline[switch]
        return direction

    def choose_side_hover_retreat(self, lidar, state, retreat, direction, config):
        command, _ = extract_latest_commands(state)
        alignment = (direction*retreat).sum(dim=1)
        lanes = ((command.norm(dim=1) <= .001) & (retreat[:, 0].abs() < .45)
                 & (retreat.norm(dim=1) > .1) & (alignment < .35)).nonzero().flatten()
        if not len(lanes):
            return direction
        away = retreat[lanes]
        action = torch.cat((torch.zeros_like(away[:, :1]), .5 + SPEED_M_S*away), dim=1)
        result = self.issued(lidar[lanes], state[lanes], action, config)
        viable = ((result.v_final_b * away).sum(1) > .08) & (result.metrics['emergency'].flatten() == 0.)
        direction[lanes[viable]] = away[viable]
        return direction

    def choose_after_return(self, lidar, state, lanes, retreat, direction, config):
        from .escape_passage import approach_scores

        raw, _ = extract_latest_commands(state[lanes])
        speed = raw.norm(dim=1)
        local = ((self.rearm_steps[lanes] > 0) & (speed >= .08)).nonzero().flatten()
        if not len(local):
            return direction
        indices = lanes[local]
        choices = torch.stack((direction[local], retreat[local], -raw[local]/speed[local,None]), dim=1)
        scans, history = lidar[indices], state[indices]
        issued = []
        for proposed in choices.unbind(dim=1):
            trial = torch.cat((torch.zeros_like(proposed[:, :1]), .5+SPEED_M_S*proposed), dim=1)
            issued.append(self.issued(scans, history, trial, config).v_final_b)
        final = torch.stack(issued, dim=1)
        supported, scores = approach_scores(scans, history, self.detector._full_rays, final)
        usable = (final[:, 1:]*choices[:, 1:]).sum(dim=2) >= .08
        baseline = scores[:, 0]
        gain, which = (scores[:, 1:]-baseline[:, None]).masked_fill(~usable, -torch.inf).max(dim=1)
        replace = supported & (baseline < .35) & (gain >= .08)
        direction[local[replace]] = choices[replace, which[replace]+1]
        self.ranked[indices[replace]] = True
        return direction

    def redirect_clipped(self, lidar, state, selected, config):
        self.redirected.zero_()
        continuing = self.applied & (self.remaining < DURATION_STEPS - 1)
        if self.rear_switch:
            self.rear_switched &= continuing
        x, y, z = self.retreat_basis.unbind(dim=1)
        yaw = -.1 * state[:, 41]
        rotated = torch.stack((x*yaw.cos() - y*yaw.sin(), x*yaw.sin() + y*yaw.cos(), z), dim=1)
        self.retreat_basis = torch.where(continuing[:, None], rotated, self.retreat_basis)
        command, _ = extract_latest_commands(state)
        lanes = (self.applied & (command.norm(dim=1) <= .001) & (self.retreat_basis[:, 0] > .65)
                 & (self.escape_speed >= FAST_LATERAL_SPEED_M_S)).nonzero().flatten()
        if not len(lanes):
            return
        original = self.issued(lidar[lanes], state[lanes], selected[lanes], config)
        blocked = (original.metrics['beta'].flatten() == 0.) & (original.metrics['emergency'].flatten() == 0.)
        lanes = lanes[blocked]
        if not len(lanes):
            return
        retreat, direction = self.retreat_basis[lanes], self.direction[lanes]
        side = torch.stack((-retreat[:, 1], retreat[:, 0], retreat[:, 2]), dim=1)
        signed_side = side * (direction * side).sum(dim=1, keepdim=True).sign()
        proposal = direction + .45 * retreat
        proposal /= proposal.norm(dim=1, keepdim=True).clamp_min(.01)
        trial = torch.cat((selected[lanes, :1], .5 + self.escape_speed[lanes, None] * proposal), dim=1)
        issued = self.issued(lidar[lanes], state[lanes], trial, config).v_final_b
        releases_side = (issued * signed_side).sum(dim=1) > .08
        selected[lanes[releases_side]] = trial[releases_side]
        self.redirected[lanes[releases_side]] = True
        if self.fallback:
            unresolved = ~releases_side & (signed_side.norm(dim=1) > .5)
            self.try_steeper_retreat(lidar, state, selected, config, lanes[unresolved], signed_side[unresolved])
            if self.rear_switch:
                switch = unresolved & continuing[lanes] & ~self.redirected[lanes] & ~self.rear_switched[lanes]
                self.try_rear_opposite(lidar,state,selected,config,lanes[switch], -signed_side[switch])

    def try_steeper_retreat(self, lidar, state, selected, config, lanes, side):
        # Search only while the held action and the original redirection both
        # fail. Preserve side commitment and use the least axial legal trial.
        for axial in (.85, 1.4, 2.2):
            if not len(lanes):
                break
            direction = (side + axial * self.retreat_basis[lanes]) / (1. + axial**2)**.5
            trial = torch.cat((selected[lanes, :1], .5 + self.escape_speed[lanes, None] * direction), dim=1)
            result = self.issued(lidar[lanes], state[lanes], trial, config)
            viable = ((result.v_final_b * side).sum(dim=1) > .08)
            viable &= result.metrics['emergency'].flatten() == 0.
            selected[lanes[viable]] = trial[viable]
            self.redirected[lanes[viable]] = True
            lanes, side = lanes[~viable], side[~viable]

    def try_rear_opposite(self, lidar, state, selected, config, lanes, opposite):
        """Commit a single previously untried side when the rear option is blocked."""
        for axial in (1.4, 2.2):
            if not len(lanes):
                break
            direction = (opposite + axial*self.retreat_basis[lanes])/(1.+axial**2)**.5
            trial = torch.cat((selected[lanes, :1], .5+self.escape_speed[lanes, None]*direction), dim=1)
            result = self.issued(lidar[lanes],state[lanes],trial,config)
            viable = ((result.v_final_b*opposite).sum(dim=1)>.08) & (result.metrics['emergency'].flatten()==0.)
            used = lanes[viable]
            selected[used] = trial[viable]
            self.direction[used] = direction[viable]
            self.rear_switched[used] = self.redirected[used] = True
            lanes, opposite = lanes[~viable], opposite[~viable]

    def switch_blocked_front(self, lidar, state, selected, config):
        # Give a blocked frontal hover response one new side. A chosen
        # feasible side stays committed until the current option ends.
        continuing = self.applied & (self.remaining < DURATION_STEPS - 1)
        self.front_switched &= continuing
        command, _ = extract_latest_commands(state)
        lanes = (continuing & ~self.front_switched & (command.norm(dim=1) <= .001)
                 & (self.retreat_basis[:, 0] < -.65)).nonzero().flatten()
        if not len(lanes):
            return
        retreat = self.retreat_basis[lanes]
        side = torch.stack((-retreat[:, 1], retreat[:, 0], retreat[:, 2]), dim=1)
        signed = (side * self.direction[lanes]).sum(dim=1)
        original = self.issued(lidar[lanes], state[lanes], selected[lanes], config)
        blocked = ((original.metrics['beta'].flatten() == 0.)
                   & (original.metrics['emergency'].flatten() == 0.) & (signed.abs() > .5))
        lanes = lanes[blocked]
        if not len(lanes):
            return
        opposite = -signed[blocked, None].sign() * side[blocked]
        trial = torch.cat((selected[lanes, :1], .5 + self.escape_speed[lanes, None] * opposite), dim=1)
        issued = self.issued(lidar[lanes], state[lanes], trial, config)
        viable = ((issued.v_final_b * opposite).sum(dim=1) > .08) & (issued.metrics['emergency'].flatten() == 0.)
        used = lanes[viable]
        self.direction[used] = opposite[viable]
        selected[used] = trial[viable]
        self.front_switched[used] = self.redirected[used] = True

    def hold_recovery_retreat(self, state, selected):
        self.retreating = self.retreat_hold_steps > 0
        if not self.early_retreat:
            self.retreating &= ~self.applied
        self.retreat_hold_steps = torch.where(self.retreating, self.retreat_hold_steps-1, 0)
        lanes = self.retreating.nonzero().flatten()
        if not len(lanes):
            return
        yaw = -.1 * state[lanes, 41]
        x, y, z = self.retreat_hold_direction[lanes].unbind(dim=1)
        direction = torch.stack((x*yaw.cos()-y*yaw.sin(), x*yaw.sin()+y*yaw.cos(), z), dim=1)
        self.retreat_hold_direction[lanes] = direction
        selected[lanes, 0] = 0.
        selected[lanes, 1:] = .5 + SPEED_M_S * direction

    def start_recovery_retreat(self, lidar, state, selected, config, lanes, heading, *, brake_when_blocked=True):
        reverse = -heading
        trial = torch.cat((torch.zeros_like(heading[:, :1]), .5 + SPEED_M_S * reverse), dim=1)
        result = self.issued(lidar[lanes], state[lanes], trial, config)
        usable = ((result.v_final_b * reverse).sum(1) > .08) & (result.metrics['emergency'].flatten() == 0.)
        if brake_when_blocked:
            selected[lanes] = selected.new_tensor([0., .5, .5, .5])
        used = lanes[usable]
        selected[used] = trial[usable]
        self.retreating[used] = True
        self.retreat_hold_steps[used] = 14
        self.retreat_hold_direction[used] = reverse[usable]

    def start_early_retreat(self, lidar, state, selected, config, command, speed):
        if self.verified_retreat_onset:
            self.retreat_request_kept.zero_()
        if self.verify_static_turn:
            self.static_turn_floor.zero_()
        lanes = ((speed >= .08) & ~self.applied & ~self.retreating & (state[:, -1] <= .22)).nonzero().flatten()
        if not len(lanes):
            return
        heading = command[lanes]/speed[lanes, None]
        away = self.detector.retreat_direction(lidar[lanes], state[lanes], hover_only=False)
        alignment = (away*heading).sum(dim=1)
        oblique = (alignment > -.45) & (alignment < -.12)
        if bool(oblique.any()):
            candidates = lanes[oblique]
            if self.verified_retreat_onset:
                current = selected[candidates].clone()
            self.start_recovery_retreat(lidar, state, selected, config, candidates, heading[oblique], brake_when_blocked=False)
            if self.verified_retreat_onset:
                self.keep_request_before_retreat(lidar, state, selected, config, candidates, current, command, speed)
            if self.verify_static_turn:
                self.keep_request_on_static_turn(lidar, state, selected, config, candidates, current)

    def keep_request_on_static_turn(self, lidar, state, selected, config, candidates, current):
        from .escape_passage import advance_command_response
        from .static_turn_onset import static_turn_margin

        protected = (self.applied | self.braking | self.holding | self.ranked | self.intercepted
                     | self.intercept_continued | (self.intercept_steps > 0))
        usable = self.retreating[candidates] & ~protected[candidates]
        for position in usable.nonzero().flatten().tolist():
            lane = int(candidates[position])
            index = slice(lane, lane+1)
            scans, frames = lidar[index], state[index]
            original = current[position:position+1]
            candidate = self.issued(scans, frames, original, config)
            retreat = self.issued(scans, frames, selected[index], config)
            response = advance_command_response(self.command_offsets[index], self.command_coast[index], frames)
            margin = static_turn_margin(self.detector, scans, frames, candidate, retreat, response)
            if margin:
                selected[index] = original
                self.retreating[lane] = False
                self.retreat_hold_steps[lane] = 0
                self.retreat_hold_direction[lane] = 0.
                self.retreat_request_kept[lane] = True
                self.static_turn_floor[lane] = margin

    def keep_request_before_retreat(self, lidar, state, selected, config, candidates, current, command, speed):
        from .escape_passage import advance_command_response, intercept_scores, moving_intercepts, observed_path_clearance

        # This check runs before the response history is advanced for normal interception.
        scans, states = lidar[candidates], state[candidates]
        history = states.reshape(-1, 4, 13)
        eligible = (self.retreating[candidates] & (speed[candidates] <= .30)
                    & (history[:, :, 12] <= .04).all(dim=1)
                    & (history[:, :, :3].norm(dim=2).amax(dim=1) <= .50))
        if self.retain_only_clipped_retreat:
            retreat = self.issued(scans, states, selected[candidates], config)
            eligible &= (retreat.metrics['beta'][:, 0] < .60) & (retreat.metrics['emergency'][:, 0] == 0.)
        if not bool(eligible.any()):
            return
        response = advance_command_response(self.command_offsets[candidates], self.command_coast[candidates], states)
        rays = self.detector._full_rays
        fits, position, motion, coast, radius = moving_intercepts(
            scans, states, rays, eligible, command_response=response, maximum_range_m=2.65)
        if not len(fits):
            return

        lanes, scans, states = candidates[fits], scans[fits], states[fits]
        checked = self.issued(scans, states, current[fits], config)
        issued = checked.v_final_b[:, None]
        follows = (issued[:, 0]*command[lanes]).sum(dim=1) >= .66*speed[lanes].square()
        usable = follows & (checked.metrics['beta'][:, 0] >= .90) & (checked.metrics['emergency'][:, 0] == 0.)
        hold = 2.4
        # Bound the static path for every release time, allowing for coasting and the preview tail.
        minimum = observed_path_clearance(scans, rays, torch.zeros_like(coast), issued, hold_s=hold)[:, 0]
        minimum -= .35*coast.norm(dim=1)+.003
        mover_speed = motion.norm(dim=1, keepdim=True)
        along = motion/mover_speed
        normal = torch.stack((-along[:, 1], along[:, 0]), dim=1)
        for lateral_margin in (-.04, 0., .04):
            direction = torch.nn.functional.normalize(along+lateral_margin*normal, dim=1)
            across = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
            resting = ((position-.35*coast)*across).sum(dim=1)
            after_hold = resting-hold*(issued[:, 0, :2]*across).sum(dim=1)
            parked = torch.minimum(resting.abs(), resting.sign()*after_hold)-radius
            minimum = torch.minimum(minimum, parked)
            for speed_margin in (-.12, 0., .12):
                uncertain = direction*(mover_speed+speed_margin).clamp_min(.20)
                bound = intercept_scores(position, uncertain, coast, radius, issued, hold_s=hold,
                                         preview_steps=121, release_early=True)
                minimum = torch.minimum(minimum, bound[:, 0])
        queued = .10*states[:, 48:51].norm(dim=1)
        retain = usable & (minimum-queued >= .43)
        kept = lanes[retain]
        selected[kept] = current[fits[retain]]
        self.retreating[kept] = False
        self.retreat_hold_steps[kept] = 0
        self.retreat_hold_direction[kept] = 0.
        self.retreat_request_kept[kept] = True

    def hold_during_clear_pass(self, lidar, state, selected):
        from .escape_passage import clear_hover_passage

        self.holding.zero_()
        onset = (self.applied & (self.remaining == DURATION_STEPS-1) & (state[:, 45:48].norm(dim=1) <= .001)
                 & (self.retreat_hold_steps == 0))
        lanes = onset.nonzero().flatten()
        if not len(lanes):
            return
        clear = clear_hover_passage(lidar[lanes], state[lanes], self.detector._full_rays, curved=self.curved_pass)
        self.hold_neutral(lanes[clear], selected)

    def hold_neutral(self, held, selected):
        self.holding[held] = True
        self.applied[held] = False
        self.remaining[held] = 0
        self.cooldown[held] = 0
        selected[held] = selected.new_tensor([0., .5, .5, .5])

    def choose_hover_path(self, lidar, state, selected, config):
        from .escape_passage import hover_escape_scores

        onset = (self.applied & (self.remaining == DURATION_STEPS-1) & (state[:, 45:48].norm(dim=1) <= .001)
                 & (self.retreat_hold_steps == 0))
        lanes = onset.nonzero().flatten()
        if not len(lanes):
            return
        scans, history = lidar[lanes], state[lanes]
        away = self.detector.retreat_direction(scans, history, hover_only=False)
        tangent = torch.stack((-away[:, 1], away[:, 0], away[:, 2]), dim=1)
        directions = torch.stack((away, tangent, -tangent), dim=1)
        trials = torch.cat((history.new_zeros((len(lanes), 3, 1)), .5+SPEED_M_S*directions), dim=2)
        original = self.issued(scans, history, selected[lanes], config).v_final_b
        alternatives = torch.stack([self.issued(scans, history, action, config).v_final_b
                                    for action in trials.unbind(dim=1)], dim=1)
        issued = torch.cat((original[:, None], torch.zeros_like(original[:, None]), alternatives), dim=1)
        supported, scores = hover_escape_scores(scans, history, self.detector._full_rays, issued,
                                               turn_fallback=self.turn_fallback)
        improves = supported & (scores[:, 0] < .45)
        hold = improves & (scores[:, 1] >= .45) & (scores[:, 1]-scores[:, 0] >= .08)
        self.hold_neutral(lanes[hold], selected)
        feasible = (alternatives*directions).sum(dim=2) >= .08
        best, option = scores[:, 2:].masked_fill(~feasible, -torch.inf).max(dim=1)
        switch = improves & ~hold & (best-scores[:, 0] >= .08)
        changed = lanes[switch]
        self.direction[changed] = directions[switch, option[switch]]
        self.escape_speed[changed] = SPEED_M_S
        self.ranked[changed] = True
        selected[changed] = trials[switch, option[switch]]

    def release_older_escape(self, lidar, state, selected):
        from .escape_passage import clear_to_end_escape

        previous = state[:, 32:35].norm(dim=1) > .03
        zero = state[:, 45:48].norm(dim=1) <= .001
        continuing = self.applied & (self.remaining < DURATION_STEPS-1)
        eligible = zero & continuing & ~self.ranked & (self.retreat_hold_steps == 0)
        if self.zero_recheck:
            # A sustained zero request can outlast the initial assessment.
            # Each retry still needs fresh proof before altering the escape.
            self.release_checks = torch.where(eligible & previous, 1, self.release_checks)
        else:
            self.release_checks = torch.where(eligible & previous, 3, (self.release_checks-1).clamp_min(0))
        self.release_checks = torch.where(eligible, self.release_checks, 0)
        self.released.zero_()
        lanes = (self.release_checks > 0).nonzero().flatten()
        if not len(lanes):
            return
        if self.verified_zero:
            clear = clear_to_end_escape(lidar[lanes], state[lanes], allow_distant_returns=True)
        else:
            clear = clear_to_end_escape(lidar[lanes], state[lanes])
        if self.receding_stop:
            from .escape_passage import advance_command_response, clear_receding_return_at_stop

            receding = ~clear & ~self.braking[lanes]
            unresolved = lanes[receding]
            if len(unresolved):
                track = advance_command_response(self.command_offsets[unresolved], self.command_coast[unresolved],
                                                  state[unresolved])
                clear[receding] = clear_receding_return_at_stop(lidar[unresolved], state[unresolved],
                                                               self.detector._full_rays, track)
        ended = lanes[clear]
        self.released[ended] = True
        self.release_checks[ended] = 0
        self.hold_neutral(ended, selected)

    def hold_after_static_stop(self, lidar, state, selected):
        from .escape_passage import advance_command_response, clear_overlapping_scans_at_stop, clear_static_scan_at_stop

        zero = state[:, 45:48].norm(dim=1) <= 1.e-6
        continuing_escape = self.applied & (self.remaining < DURATION_STEPS-1)
        if self.verify_overlapping_stop:
            self.overlapping_stop_verified.zero_()
        self.static_stop_verified &= zero
        eligible = (self.applied & (continuing_escape | self.static_stop_verified)
                    & zero & ~self.ranked & ~self.braking & (self.retreat_hold_steps == 0))
        lanes = eligible.nonzero().flatten()
        self.static_stop_verified[lanes] = False
        if not len(lanes):
            return
        response = advance_command_response(self.command_offsets[lanes], self.command_coast[lanes], state[lanes])
        proved = clear_static_scan_at_stop(lidar[lanes], state[lanes], self.detector._full_rays, response)
        if self.verify_overlapping_stop:
            first_zero = state[lanes, 32:35].norm(dim=1) > .03
            checks = (~proved & first_zero & continuing_escape[lanes] & ~self.retreating[lanes]).nonzero().flatten()
            if len(checks):
                checked = lanes[checks]
                confirmed = clear_overlapping_scans_at_stop(lidar[checked], state[checked], self.detector._full_rays,
                                                           tuple(value[checks] for value in response))
                proved[checks] = confirmed
                self.overlapping_stop_verified[checked] = confirmed
        held = lanes[proved]
        self.static_stop_verified[held] = True
        self.released[held] = True
        self.hold_neutral(held, selected)

    def update_recenter_rearm(self, lidar, config):
        depth, valid, reliability = lidar[:, -3], lidar[:, -2], lidar[:, -1]
        threshold = config.ics.emergency_clearance_m+config.ics.clearance_margin_m
        emergency = ((depth < threshold) & (valid > .5) & (reliability >= config.ics.min_reliability)).flatten(1).any(dim=1)
        self.rearm_steps = torch.where(emergency, RECENTER_REARM_STEPS, (self.rearm_steps-1).clamp_min(0))

    def handoff_changed_command(self, lidar, state, selected, action, config):
        from .escape_passage import clear_to_end_escape, moving_intercepts

        self.handed_off.zero_()
        if self.verify_steady_handoff:
            prior_steady = self.steady_handoff_witness.clone()
            self.steady_handoff_witness.zero_()
            self.steady_handoff_used.zero_()
        previous, latest_command = state[:, 32:35], state[:, 45:48]
        ordinary = (self.applied & (self.remaining < DURATION_STEPS-1) & ~self.ranked
                    & ~self.braking & ~self.retreating)
        if self.retry_command_handoff:
            protected = (self.holding | self.recovering | self.static_stop_verified | self.intercepted
                         | self.intercept_continued | (self.intercept_steps > 0))
            steady = (previous-latest_command).norm(dim=1) <= .04
            retry = self.handoff_pending & ordinary & steady & ~protected
            self.handoff_pending.zero_()
            previous = torch.where(retry[:, None], state[:, 19:22], previous)
        recent = torch.stack((previous, latest_command), dim=1)
        speed = recent.norm(dim=2)
        older, latest = (recent/speed[:, :, None].clamp_min(.001)).unbind(dim=1)
        changed = (older*latest).sum(dim=1) <= .25
        basis = self.retreat_basis
        moved_away = ((basis*older).sum(dim=1) <= -.35) & ((basis*latest).sum(dim=1) >= -.15)
        angular_handoff = changed & moved_away & (speed >= .08).all(dim=1)
        eligible = angular_handoff
        if self.verify_recent_handoff:
            older_requests = state.reshape(-1, 4, 13)[:, :2, 6:9]
            recent_change = (older_requests-latest_command[:, None]).norm(dim=2).amax(dim=1) > .04
            same_request_ready = steady & ~protected & (speed[:, 1] >= .08) & (speed[:, 1] <= .30)
            recent_handoff = recent_change & same_request_ready
            eligible = eligible | recent_handoff
            if self.verify_steady_handoff:
                steady_handoff = (same_request_ready & ~recent_change & (self.remaining > 0)
                                  & ~self.composed_crossing_verified)
                eligible |= steady_handoff
        lanes = (eligible & ordinary).nonzero().flatten()
        if not len(lanes):
            return
        scans, history, heading = lidar[lanes], state[lanes], latest[lanes]
        observed = ((scans[:, 1::3] > .5) & (scans[:, 2::3] > .55)).all(dim=1).flatten(1)
        forward = (heading @ self.detector._full_rays.T) >= .70
        coverage = (forward & observed).sum(dim=1)/forward.sum(dim=1).clamp_min(1)
        clear = (coverage >= .45) & clear_to_end_escape(scans, history, minimum_coverage=.50)
        if self.retry_command_handoff:
            # Only a fresh change denied by the scan proof gets one recheck.
            self.handoff_pending[lanes] = angular_handoff[lanes] & ~retry[lanes] & ~protected[lanes] & ~clear
        lanes, scans, history, heading = lanes[clear], scans[clear], history[clear], heading[clear]
        if not len(lanes):
            return
        actor = self.issued(scans, history, action[lanes], config)
        held = self.issued(scans, history, selected[lanes], config)
        progress = (actor.v_final_b*heading).sum(dim=1)
        old_progress = (held.v_final_b*heading).sum(dim=1)
        usable = (actor.metrics['beta'].flatten() >= .90) & (actor.metrics['emergency'].flatten() == 0.)
        usable &= (progress >= .60*speed[lanes, 1]) & (progress-old_progress >= .12)
        release_ok = usable & angular_handoff[lanes]
        if self.verify_recent_handoff:
            recent = usable & ~angular_handoff[lanes] & recent_handoff[lanes] & (old_progress < 0.)
            if self.verify_steady_handoff:
                repeated = usable & ~angular_handoff[lanes] & steady_handoff[lanes] & (old_progress < 0.)
                recent |= repeated
            if bool(recent.any()):
                moving, *_ = moving_intercepts(scans, history, self.detector._full_rays, recent, maximum_range_m=2.65)
                recent[moving] = False
            if self.verify_steady_handoff:
                proved = repeated & recent
                self.steady_handoff_witness[lanes] = proved
                renewed = proved & prior_steady[lanes]
                self.steady_handoff_used[lanes] = renewed
                recent &= ~repeated | renewed
            release_ok |= recent
        release = lanes[release_ok]
        selected[release] = action[release]
        self.applied[release] = False
        self.remaining[release] = 0
        self.cooldown[release] = 0
        self.since_escape[release] = RECOVERY_LIMIT
        self.handed_off[release] = True

    def continue_intercept(self, lidar, state, selected, config, observed):
        self.intercept_continued.zero_()
        missing = state[:, -1] <= .04
        missing[observed] = False
        if self.clear_pass:
            missing &= ~self.holding
        lanes = (missing & (self.intercept_steps > 0)).nonzero().flatten()
        self.intercept_steps = torch.where(missing, (self.intercept_steps-1).clamp_min(0), 0)
        if not len(lanes):
            return
        x, y, z = self.intercept_direction[lanes].unbind(dim=1)
        yaw = -.05*(state[lanes, 28]+state[lanes, 41])
        direction = torch.stack((x*yaw.cos()-y*yaw.sin(), x*yaw.sin()+y*yaw.cos(), z), dim=1)
        speed = self.intercept_speed[lanes, None] if self.along_intercept else SPEED_M_S
        trial = torch.cat((state.new_zeros((len(lanes), 1)), .5+speed*direction), dim=1)
        result = self.issued(lidar[lanes], state[lanes], trial, config)
        usable = ((result.v_final_b*direction).sum(dim=1) >= .12) & (result.metrics['emergency'].flatten() == 0.)
        kept = lanes[usable]
        self.intercept_steps[lanes[~usable]] = 0
        self.intercept_direction[kept] = direction[usable]
        selected[kept] = trial[usable]
        self.intercepted[kept] = True
        self.intercept_continued[kept] = True

    def usable_recovery_retreat(self, lidar, state, selected, config):
        lanes = self.retreating.nonzero().flatten()
        if not len(lanes):
            return lanes
        current = self.issued(lidar[lanes], state[lanes], selected[lanes], config)
        along_retreat = (current.v_final_b*self.retreat_hold_direction[lanes]).sum(dim=1)
        usable = (along_retreat >= .12) & (current.metrics['emergency'].flatten() == 0.)
        return lanes[usable]

    def stop_for_offset_pass(self, lidar, state, selected, lanes, position, motion, coast, radius):
        zero = state[lanes, 45:48].norm(dim=1) <= .001
        ordinary = self.applied[lanes] & ~self.ranked[lanes] & ~self.retreating[lanes] & ~self.braking[lanes]
        normal = torch.stack((-motion[:, 1], motion[:, 0]), dim=1)/motion.norm(dim=1, keepdim=True)
        cross = (position*normal).sum(dim=1)
        after_stop = cross-.35*(coast*normal).sum(dim=1)
        same_side = cross*after_stop > 0.
        surface = torch.minimum(cross.abs(), after_stop.abs())-radius
        scans = lidar[lanes]
        reliable = (scans[:, 10] > .5) & (scans[:, 11] > .55)
        nearest = scans[:, 9].masked_fill(~reliable, torch.inf).flatten(1).amin(dim=1)
        eligible = zero & ordinary & same_side & (nearest >= .85)
        cancel = eligible & (surface >= .65)
        if self.settled_offset_stop:
            from .escape_passage import clear_hover_passage

            history = state[lanes].reshape(-1, 4, 13)
            settled = ((history[:, :, 6:9].norm(dim=2) <= .001).all(dim=1)
                       & (history[:, :, 9:12].norm(dim=2) <= .08).all(dim=1)
                       & (history[:, :, 12] <= .04).all(dim=1))
            onset = self.remaining[lanes] == DURATION_STEPS-1
            check = (eligible & ~cancel & nearest.isfinite() & onset & settled & (surface >= .45)).nonzero().flatten()
            if len(check):
                proof = dict(curved=True, require_curvature=True)
                if self.verified_zero:
                    proof['turn_fallback'] = True
                cancel[check] = clear_hover_passage(scans[check], state[lanes[check]], self.detector._full_rays, **proof)
        self.hold_neutral(lanes[cancel], selected)
        self.offset_stopped[lanes[cancel]] = True
        return cancel

    def resume_latest_request(self, lidar, state, selected, action, config, eligible, tracks):
        from .escape_passage import clear_to_end_escape, intercept_scores

        previous = self.resume_witness.clone()
        self.resume_witness.zero_()
        self.resumed.zero_()
        history = state.reshape(-1, 4, 13)
        command = history[:, -1, 6:9]
        speed = command.norm(dim=1)
        continuing = self.applied & (self.remaining < DURATION_STEPS-1) & ~self.retreating
        ready = eligible & continuing & (history[:, :, 12] <= .04).all(dim=1) & (speed >= .08) & (speed <= .30)
        candidates = ready.nonzero().flatten()
        if not len(candidates):
            return
        scans, histories = lidar[candidates], state[candidates]
        actor = self.issued(scans, histories, action[candidates], config)
        held = self.issued(scans, histories, selected[candidates], config)
        heading = command[candidates]/speed[candidates, None]
        projection = (actor.v_final_b*heading).sum(dim=1)
        gain = ((actor.v_final_b-held.v_final_b)*heading).sum(dim=1)
        observed = ((scans[:, 1::3] > .5) & (scans[:, 2::3] > .55)).all(dim=1).flatten(1)
        forward = (heading@self.detector._full_rays.T) >= .70
        coverage = (observed & forward).sum(dim=1)/forward.sum(dim=1).clamp_min(1)
        accepted = ((actor.metrics['beta'][:, 0] >= .90) & (actor.metrics['emergency'][:, 0] == 0.)
                    & (projection >= .60*speed[candidates]) & (gain >= .075) & (coverage >= .45))
        candidates, scans, histories = candidates[accepted], scans[accepted], histories[accepted]
        if not len(candidates):
            return
        moving_lanes, position, motion, coast, radius = tracks
        clear = torch.zeros_like(self.applied)
        clear[candidates] = clear_to_end_escape(scans, histories, minimum_coverage=.50)
        if len(moving_lanes):
            velocities = torch.zeros_like(command)
            velocities[candidates] = actor.v_final_b[accepted]
            observed_lane = torch.zeros_like(self.applied)
            observed_lane[candidates] = True
            fits = observed_lane[moving_lanes]
            clearance = intercept_scores(position[fits], motion[fits], coast[fits], radius[fits],
                                         velocities[moving_lanes[fits], None])[:, 0]
            clear[moving_lanes[fits]] = clearance >= .48
        self.resume_witness[candidates] = clear[candidates]
        steady = (command-history[:, -2, 6:9]).norm(dim=1) <= .04
        release = previous & self.resume_witness & steady
        selected[release] = action[release]
        self.applied[release] = False
        self.braking[release] = False
        self.ranked[release] = False
        self.remaining[release] = 0
        self.cooldown[release] = 0
        self.since_escape[release] = RECOVERY_LIMIT
        self.resumed[release] = True

    def prove_tracked_resume(self, lidar, state, selected, actor_action, config, eligible, tracks):
        from .escape_passage import crossing_stop_components, intercept_scores

        previous = self.tracked_resume_witness.clone()
        self.tracked_resume_used.zero_()
        self.tracked_resume_witness.zero_()
        observed, position, motion, coast, radius = tracks
        if not len(observed):
            return
        history = state[observed].reshape(-1, 4, 13)
        request = history[:, -1, 6:9]
        speed = request.norm(dim=1)
        ordinary = self.applied & ~self.braking & ~self.ranked & ~self.retreating
        steady = (history[:, :3, 6:9]-request[:, None]).norm(dim=2).amax(dim=1) <= .04
        fresh = (history[:, :, 12] <= .04).all(dim=1)
        consider = ((eligible & ordinary)[observed] & steady & fresh & (speed >= .08) & (speed <= .30)).nonzero().flatten()
        if not len(consider):
            return
        lanes = observed[consider]
        scans, local = lidar[lanes], state[lanes]
        actor = self.issued(scans, local, actor_action[lanes], config)
        original = self.issued(scans, local, selected[lanes], config)
        direction = request[consider]/speed[consider, None]
        progress = (actor.v_final_b*direction).sum(dim=1)
        gain = ((actor.v_final_b-original.v_final_b)*direction).sum(dim=1)
        possible = ((actor.metrics['beta'][:, 0] >= .90) & (actor.metrics['emergency'][:, 0] == 0.)
                    & (progress >= .66*speed[consider]) & (gain >= .04))
        indices = consider[possible]
        if not len(indices):
            return
        lanes = lanes[possible]
        position, motion, coast, radius = (x[indices] for x in (position, motion, coast, radius))
        speed_mover = motion.norm(dim=1, keepdim=True)
        along = motion/speed_mover
        normal = torch.stack((-along[:, 1], along[:, 0]), dim=1)
        target = actor.v_final_b[possible, None]
        swept, settled = crossing_stop_components(scans[possible], self.detector._full_rays,
                                                  position, motion, coast, radius, target, hold_s=2.4)
        minimum = torch.minimum(swept[:, 0], settled[:, 0])
        for delta in (-.12, .12):
            estimate = intercept_scores(position, (speed_mover+delta).clamp_min(.20)*along, coast, radius,
                                        target, hold_s=2.4, preview_steps=101)[:, 0]
            minimum = torch.minimum(minimum, estimate)
        queued = .10*local[possible, 48:51].norm(dim=1)
        witnessed = ((minimum-queued >= .43) & ((position*normal).sum(dim=1).abs() > .28))
        self.tracked_resume_witness[lanes] = witnessed
        self.tracked_resume_used[lanes] = previous[lanes] & witnessed

    def resume_after_interception(self, selected, actor_action):
        protected = (self.intercepted | self.intercept_continued | self.braking | self.retreating
                     | self.ranked | self.holding | self.static_stop_verified | self.handed_off | ~self.applied)
        self.tracked_resume_used &= ~protected
        self.tracked_resume_witness &= ~protected
        selected[self.tracked_resume_used] = actor_action[self.tracked_resume_used]

    def keep_request_at_crossing_restart(self, lidar, state, selected, actor_action, config):
        from .escape_passage import crossing_stop_components, intercept_scores, moving_intercepts

        self.crossing_restart_verified.zero_()
        self.crossing_restart_previous_b.zero_()
        history = state.reshape(-1, 4, 13)
        request, previous = extract_latest_commands(state)
        speed = request.norm(dim=1)
        at_rest = (history[:, :3, 6:9].norm(dim=2) <= .001).all(dim=1)
        at_rest &= (history[:, :, 9:12].norm(dim=2) <= .02).all(dim=1)
        angular_rate = history[:, :, :3].norm(dim=2).amax(dim=1)
        at_rest &= angular_rate <= .08
        recent = self.crossing_restart_steps > 0
        released = recent & (speed <= .001)
        self.crossing_restart_steps = (self.crossing_restart_steps-1).clamp_min(0)
        ages = history[:, :, 12]
        fresh = (ages <= .04).all(dim=1) & (ages.diff(dim=1).abs() <= .03).all(dim=1) & (angular_rate <= .50)
        eligible = fresh & ((at_rest & (self.applied | self.intercepted)) | recent)
        eligible &= ((speed >= .08) & (speed <= .30)) | released
        eligible &= ~(self.braking | self.ranked | self.retreating | self.holding | self.static_stop_verified
                      | self.settled_crossing_verified | self.handed_off | self.offset_stopped
                      | self.hover_escape_redirected)
        if not bool(eligible.any()):
            return
        rays = self.detector._full_rays
        lanes, position, motion, coast, radius = moving_intercepts(lidar, state, rays, eligible,
            command_response=(self.command_offsets, self.command_coast), maximum_range_m=2.65)
        if not len(lanes):
            return

        scans, local = lidar[lanes], state[lanes]
        mover_speed = motion.norm(dim=1, keepdim=True)
        along = motion/mover_speed
        normal = torch.stack((-along[:, 1], along[:, 0]), dim=1)
        offset = (position*normal).sum(dim=1)
        own_side = -offset.sign()[:, None]*normal
        original = self.issued(scans, local, selected[lanes], config)
        toward_line = (original.v_final_b[:, :2]*own_side).sum(dim=1) <= -.12
        blocked = recent[lanes] & (original.v_final_b.norm(dim=1) <= .03)
        asks_own_side = (request[lanes, :2]*own_side).sum(dim=1) >= .08
        release = released[lanes]
        possible = ((asks_own_side | release) & (offset.abs() >= .36) & (toward_line | blocked)
                    & (original.metrics['emergency'][:, 0] == 0.))
        if not bool(possible.any()):
            return

        direction = torch.nn.functional.normalize(own_side-.267949*along, dim=1)  # 15 degrees against motion
        target = (SPEED_M_S-.0001)*torch.nn.functional.pad(direction, (0, 1))
        diagonal = self.action_for_velocity(target, previous[lanes], config.governor)
        neutral = torch.full_like(diagonal, .5)
        neutral[:, 0] = 0.
        actions = torch.stack((actor_action[lanes], diagonal, neutral), dim=1)
        checks = [self.issued(scans, local, action, config) for action in actions.unbind(dim=1)]
        velocities = torch.stack((original.v_final_b, *(check.v_final_b for check in checks)), dim=1)
        queued = .10*previous[lanes].norm(dim=1, keepdim=True)

        def floor(hold_s):
            sweep, stop = crossing_stop_components(scans, rays, position, motion, coast, radius, velocities, hold_s=hold_s)
            minimum = torch.minimum(sweep, stop)
            for delta in (-.12, .12):
                varied = (mover_speed+delta).clamp_min(.20)*along
                estimate = intercept_scores(position, varied, coast, radius, velocities, hold_s=hold_s, preview_steps=121)
                minimum = torch.minimum(minimum, estimate)
            return minimum-queued

        short, long = floor(.60), floor(2.4)
        progress = (velocities[:, 1:3]*request[lanes, None]).sum(dim=2)
        outward = (velocities[:, 1:3, :2]*own_side[:, None]).sum(dim=2)
        usable = (short[:, 1:3] >= .39) & (long[:, 1:3] >= .43) & (outward >= .12)
        usable &= torch.cat([check.metrics['beta'] >= .90 for check in checks[:2]], dim=1)
        usable &= torch.cat([check.metrics['emergency'] == 0. for check in checks[:2]], dim=1)
        usable &= (progress >= .60*speed[lanes, None].square()) | release[:, None]
        actor = usable[:, 0] & ~release
        stop = release & (short[:, 3] >= .39) & (long[:, 3] >= .39) & (velocities[:, 3].norm(dim=1) <= .001)
        choice = torch.where(stop, 2, (~actor).long())
        use = possible & (short[:, 0] < .39) & (actor | usable[:, 1] | stop)
        accepted = lanes[use]
        selected[accepted] = actions[use, choice[use]]
        self.crossing_restart_verified[accepted] = True
        self.crossing_restart_previous_b[accepted] = original.v_final_b[use]
        # A fresh departure can be reconsidered for the next eight policy ticks.
        # Each replacement requires current scans and verifies the latest request.
        fresh_departure = at_rest[accepted] & ~recent[accepted]
        self.crossing_restart_steps[accepted] = torch.where(fresh_departure, 8, self.crossing_restart_steps[accepted])
        self.applied[accepted] = False
        self.remaining[accepted] = 0
        self.cooldown[accepted] = 0
        self.since_escape[accepted] = RECOVERY_LIMIT
        self.intercepted[accepted] = False
        self.intercept_steps[accepted] = 0
        self.intercept_continued[accepted] = False
        self.early_crossing[accepted] = False
        self.reserve_direction_seen[accepted] = False
        self.tracked_resume_used[accepted] = False
        self.tracked_resume_witness[accepted] = False

    def verify_request_during_braking(self, lidar, state, selected, actor_action, config):
        from .brake_renewal import remember_crossing, renew_crossing
        from .escape_passage import braking_recovery_clearance, intercept_scores, moving_intercepts

        self.brake_recovery_verified.zero_()
        self.brake_recovery_bounds.zero_()
        if self.verify_outward_brake:
            outward_remaining = self.outward_brake_steps.clone()
            outward_anchor = self.outward_brake_anchor.clone()
            self.outward_brake_steps.zero_()
            self.outward_brake_anchor.zero_()
        history = state.reshape(-1, 4, 13)
        request = history[:, -1, 6:9]
        speed = request.norm(dim=1)
        age = history[:, :, 12]
        fresh = (age <= .04).all(dim=1) & (age.diff(dim=1).abs() <= .03).all(dim=1)
        fresh &= (history[:, :, :3].norm(dim=2).amax(dim=1) <= .50)
        protected = (self.intercepted | self.intercept_continued | self.retreating | self.ranked | self.holding
                     | self.static_stop_verified | self.handed_off | self.offset_stopped | self.crossing_restart_verified)
        eligible = self.applied & self.braking & ~protected & fresh & (speed >= .08) & (speed <= .30)
        if self.verify_brake_transition:
            remaining = self.brake_transition_steps.clone()
            self.brake_transition_steps.zero_()
            previous = history[:, -2, 6:9]
            request_change = (request-previous).norm(dim=1)
            previous_speed = previous.norm(dim=1)
            same_heading = (previous_speed >= .08) & ((request*previous).sum(dim=1) >= .98*speed*previous_speed)
            continuation = remaining > 0
            if self.verify_outward_brake:
                continuation |= outward_remaining > 0
            eligible &= (request_change >= .15) | (continuation & same_heading)
        trials = eligible.nonzero().flatten()
        if not len(trials):
            return
        scans, observed = lidar[trials], state[trials]
        actor = self.issued(scans, observed, actor_action[trials], config)
        guarded = self.issued(scans, observed, selected[trials], config)
        target = actor.v_final_b
        progress = (target*request[trials]).sum(dim=1)
        gain = ((target-guarded.v_final_b)*request[trials]).sum(dim=1)
        magnitude = speed[trials].square()
        suitable = ((actor.metrics['beta'][:, 0] >= .90) & (progress >= .60*magnitude) & (gain >= .30*magnitude)
                    & (actor.metrics['emergency'][:, 0] == 0.) & (guarded.metrics['emergency'][:, 0] == 0.))
        if not bool(suitable.any()):
            return
        lanes = trials[suitable]
        rays = self.detector._full_rays
        fitted, position, motion, coast, radius = moving_intercepts(lidar[lanes], state[lanes], rays, fresh[lanes],
            command_response=(self.command_offsets[lanes], self.command_coast[lanes]), maximum_range_m=2.65)
        closing_count = len(fitted)
        if self.verify_outward_brake:
            eligible_renewal = (outward_remaining[lanes] > 0) & same_heading[lanes]
            eligible_renewal[fitted] = False
            visible = eligible_renewal.nonzero().flatten()
            if len(visible):
                renewing_lanes = lanes[visible]
                renewed = renew_crossing(lidar[renewing_lanes], state[renewing_lanes], rays,
                    self.command_offsets[renewing_lanes], self.command_coast[renewing_lanes],
                    outward_anchor[renewing_lanes])
                if renewed is not None:
                    index, *tracks = renewed
                    current = (visible[index], *tracks)
                    if closing_count:
                        old = (fitted, position, motion, coast, radius)
                        fitted, position, motion, coast, radius = (
                            torch.cat((a, b), dim=0) for a, b in zip(old, current, strict=True))
                    else:
                        fitted, position, motion, coast, radius = current
        if not len(fitted):
            return
        observed = lanes[fitted]
        issued = target[suitable][fitted]
        floor = braking_recovery_clearance(lidar[observed], rays, state[observed], position, motion,
                                           coast, radius, issued)
        verified = (floor[:, 0] >= .39) & (floor[:, 1] >= .43)
        if self.verify_brake_transition:
            along = torch.nn.functional.normalize(motion, dim=1)
            normal = torch.stack((-along[:, 1], along[:, 0]), dim=1)
            inward = normal*(position*normal).sum(dim=1).sign()[:, None]
            latest = request[observed, :2]
            earlier = previous[observed]
            prior_clearance = intercept_scores(position, motion, coast, radius, earlier[:, None],
                                               hold_s=2.4, preview_steps=121)[:, 0]
            onset = ((request_change[observed] >= .15) & ((earlier[:, :2]*inward).sum(dim=1) >= .12)
                     & (prior_clearance < .36))
            renewing = (remaining[observed] > 0) & same_heading[observed]
            tangent = (latest*inward).sum(dim=1).abs() <= .04
            against_crossing = (latest*along).sum(dim=1) <= -.50*speed[observed]
            stock = torch.arange(len(fitted), device=state.device) < closing_count
            tangent_verified = verified & stock & (onset | renewing) & tangent & against_crossing
            # An unverified packet drops the renewal. Current observations must
            # support the exact request again on every packet.
            self.brake_transition_steps[observed[tangent_verified]] = torch.where(
                onset[tangent_verified], 7, remaining[observed[tangent_verified]]-1)
            if self.verify_outward_brake:
                outward_onset = onset & stock
                outward_renewal = (outward_remaining[observed] > 0) & same_heading[observed]
                request_away = (latest*inward).sum(dim=1)+.04*(latest*along).sum(dim=1).abs() <= -.12
                issued_away = (issued[:, :2]*inward).sum(dim=1)+.04*(issued[:, :2]*along).sum(dim=1).abs() <= -.12
                outward = verified & (outward_onset | outward_renewal) & request_away & issued_away
                remaining_outward = torch.where(outward_onset, OUTWARD_BRAKE_RENEWAL_STEPS,
                                               outward_remaining[observed]-1)
                self.outward_brake_steps[observed[outward]] = remaining_outward[outward]
                save_anchor = outward & (remaining_outward > 0)
                anchors = observed[save_anchor]
                if len(anchors):
                    fit = remember_crossing(lidar[anchors], state[anchors], rays, position[save_anchor], motion[save_anchor])
                    if fit is not None:
                        indices, values = fit
                        self.outward_brake_anchor[anchors[indices]] = values
                tangent_verified |= outward
            verified = tangent_verified
        accepted = observed[verified]
        # This command is independent of how the braking option evolves next tick.
        selected[accepted] = actor_action[accepted]
        self.brake_recovery_verified[accepted] = True
        self.brake_recovery_bounds[accepted] = floor[verified]

    def continue_retreat_on_observed_side(self, lidar, state, selected, config):
        from .escape_passage import crossing_stop_components, intercept_scores, moving_intercepts

        self.retreat_continuation_verified.zero_()
        self.retreat_continuation_floor.zero_()
        continuing = self.retreat_continuation_steps > 0
        self.retreat_continuation_steps = (self.retreat_continuation_steps-1).clamp_min(0)
        onset = self.retreating & self.intercepted & ~self.intercept_continued
        frames = state.reshape(-1, 4, 13)
        ages = frames[:, :, 12]
        fresh = (ages <= .04).all(dim=1) & (ages.diff(dim=1).abs() <= .03).all(dim=1)
        fresh &= frames[:, :, :3].norm(dim=2).amax(dim=1) <= .50
        resumed_brake = continuing & self.braking & self.retreating
        eligible = (onset | continuing) & (~self.braking | resumed_brake) & ~self.ranked & ~self.holding & fresh
        if not bool(eligible.any()):
            return
        rays = self.detector._full_rays
        lanes, position, motion, coast, radius = moving_intercepts(lidar, state, rays, eligible,
            command_response=(self.command_offsets, self.command_coast), maximum_range_m=2.65)
        if not len(lanes):
            return
        scans, local = lidar[lanes], state[lanes]
        mover_speed = motion.norm(dim=1, keepdim=True)
        along = motion/mover_speed
        normal = torch.stack((-along[:, 1], along[:, 0]), dim=1)
        lateral = (position*normal).sum(dim=1)
        own_side = -lateral.sign()[:, None]*normal
        pending = local[:, 48:51]
        original = self.issued(scans, local, selected[lanes], config)
        outward = (original.v_final_b[:, :2]*own_side).sum(dim=1)
        earlier_withdrawal = (pending[:, :2]*own_side).sum(dim=1) >= .12
        change_side = onset[lanes] & earlier_withdrawal & (outward <= -.12)
        confirm_again = continuing[lanes] & (outward < .12)
        ready = (change_side | confirm_again) & (lateral.abs() >= .20) & (original.metrics['emergency'][:, 0] == 0.)
        if not bool(ready.any()):
            return

        directions = [torch.nn.functional.pad(torch.nn.functional.normalize(own_side-angle*along, dim=1), (0, 1))
                      for angle in (.40, .60, .22, .35)]
        actions = [torch.cat((local.new_zeros((len(lanes), 1)), .5+(SPEED_M_S-.0001)*d), dim=1) for d in directions]
        checked = [self.issued(scans, local, action, config) for action in actions]
        velocities = torch.stack((original.v_final_b, *(result.v_final_b for result in checked)), dim=1)
        sweep, stop = crossing_stop_components(scans, rays, position, motion, coast, radius, velocities, hold_s=2.4)
        floor = torch.minimum(sweep, stop)
        for angle in (-.04, 0., .04):
            path = torch.nn.functional.normalize(along+angle*normal, dim=1)
            across = torch.stack((-path[:, 1], path[:, 0]), dim=1)
            parked = ((position-.35*coast)*across).sum(dim=1)
            final = parked[:, None]-2.4*(velocities[..., :2]*across[:, None]).sum(dim=2)
            floor = torch.minimum(floor, parked.sign()[:, None]*final-radius[:, None])
            for delta in (-.12, .12):
                varied = (mover_speed+delta).clamp_min(.20)*path
                predicted = intercept_scores(position, varied, coast, radius, velocities, hold_s=2.4, preview_steps=121)
                floor = torch.minimum(floor, predicted)
        floor -= .10*pending.norm(dim=1, keepdim=True)
        available = torch.cat([result.metrics['beta'] >= .90 for result in checked], dim=1)
        available &= torch.cat([result.metrics['emergency'] == 0. for result in checked], dim=1)
        available &= ((velocities[:, 1:, :2]*own_side[:, None]).sum(dim=2) >= .12) & (floor[:, 1:] >= .39)
        available[:, -1] &= resumed_brake[lanes]
        original_limit = torch.where(continuing[lanes], .39, .21)
        use = ready & (floor[:, 0] < original_limit) & available.any(dim=1)
        chosen = available.float().argmax(dim=1)[use]
        accepted = lanes[use]
        direction = torch.stack(directions, dim=1)[use, chosen]
        selected[accepted] = torch.stack(actions, dim=1)[use, chosen]
        self.retreat_hold_direction[accepted] = direction
        self.intercept_direction[accepted] = direction
        self.intercept_speed[accepted] = SPEED_M_S-.0001
        self.intercept_steps[accepted] = 4
        self.intercepted[accepted] = True
        self.intercept_continued[accepted] = False
        self.retreat_continuation_steps[accepted] = torch.where(continuing[accepted], self.retreat_continuation_steps[accepted], 18)
        self.retreat_continuation_verified[accepted] = True
        self.retreat_continuation_floor[accepted] = floor[use, chosen+1]

    def crossing_stop_trial(self, scans, history, config, position, motion, coast, radius,
                            directions, previous_direction, current, *, minimum_beta=0.):
        from .escape_passage import crossing_stop_clearance

        actions = torch.cat((history.new_zeros((*directions.shape[:2], 1)),
                             .5+(SPEED_M_S-.0001)*directions), dim=2)
        velocities, betas = [], []
        for action in actions.unbind(dim=1):
            result = self.issued(scans, history, action, config)
            velocities.append(result.v_final_b)
            betas.append(result.metrics['beta'])
        checked = torch.stack(velocities, dim=1)
        beta = torch.cat(betas, dim=1)
        velocities = torch.cat((current[:, None], checked), dim=1)
        clearance = crossing_stop_clearance(scans, self.detector._full_rays, position, motion, coast, radius,
                                            velocities, hold_s=2.6)
        available = (checked*directions).sum(dim=2) >= .12
        feasible = available & (beta >= minimum_beta) & (clearance[:, 1:] >= .39)
        confirmed = (directions*previous_direction[:, None]).sum(dim=2) >= .95
        ranking = clearance[:, 1:]+.025*confirmed
        choice = ranking.masked_fill(~feasible, -torch.inf).argmax(dim=1)
        rows = torch.arange(len(history), device=history.device)
        gain = clearance[rows, choice+1]-clearance[:, 0]
        use = feasible[rows, choice] & (clearance[:, 0] < .39) & (gain >= .075)
        return use, actions[rows, choice], checked[rows, choice], directions[rows, choice], available

    def continue_crossing_stop(self, scans, history, config, position, motion, coast, radius,
                               across, along, previous, current):
        from .escape_passage import crossing_stop_components

        side = torch.where((previous*across).sum(dim=1, keepdim=True) >= 0., across, -across)
        direction = torch.nn.functional.normalize(side-1.43*along, dim=1)
        pending = history[:, 48:51]
        action = self.action_for_velocity((SPEED_M_S-.0001)*direction, pending, config.governor)
        checked = self.issued(scans, history, action, config)
        velocity = torch.stack((current, checked.v_final_b), dim=1)
        swept, settled = crossing_stop_components(scans, self.detector._full_rays, position, motion, coast,
                                                  radius, velocity, hold_s=2.)
        access = (checked.metrics['beta'][:, 0] >= .9) & ((checked.v_final_b*side).sum(dim=1) >= .12)
        # If the normal command approaches the vehicle envelope, take only a
        # same-side turn which materially improves the sweep and can stop clear.
        use = (access & ((current*side).sum(dim=1) < .08) & ((pending*side).sum(dim=1) >= .12)
               & (swept[:, 0] < .21) & (swept[:, 1] >= .34) & (swept[:, 1]-swept[:, 0] >= .20)
               & (settled[:, 1] >= .39))
        return use, action, checked.v_final_b, direction

    def anticipate_crossing(self, lidar, state, actor_action, config, eligible, response):
        from .escape_passage import intercept_scores, moving_intercepts, observed_path_clearance

        if self.far_crossing_reference:
            self.far_crossing_refined.zero_()
        request, previous = extract_latest_commands(state)
        history = state.reshape(-1, 4, 13)
        speed = request.norm(dim=1)
        stable = (history[:, :3, 6:9]-request[:, None]).norm(dim=2).amax(dim=1) <= .04
        self.early_crossing &= self.intercept_steps > 0
        ongoing = self.early_crossing
        new_approach = ~self.applied & stable & (speed >= .12) & (speed <= .30)
        ready = eligible & (ongoing | new_approach)
        candidates = ready.nonzero().flatten()
        if not len(candidates) or config.governor.v_corr_limit <= 0.:
            return candidates[:0], None, None
        minimum = torch.where(ongoing[candidates], 0., 1.9).reshape(-1, 1, 1, 1)
        retained = tuple(tensor[candidates] for tensor in response)
        rays = self.detector._full_rays
        fitted, position, motion, coast, radius = moving_intercepts(
            lidar[candidates], state[candidates], rays, ready[candidates], command_response=retained,
            minimum_range_m=minimum, maximum_range_m=2.65)
        lanes = candidates[fitted]
        if not len(lanes):
            return lanes, None, None

        scans, states = lidar[lanes], state[lanes]
        continuing = ongoing[lanes]
        if self.far_crossing_reference:
            from .escape_passage import _return_seed
            from .escape_reference import crossing_direction_from_planes

            along = torch.nn.functional.normalize(motion, dim=1)
            normal = torch.stack((-along[:, 1], along[:, 0]), dim=1)
            near_line = ((position*normal).sum(dim=1).abs() <= .28) & ~continuing
            checks = near_line.nonzero().flatten()
            if len(checks):
                scan, history = scans[checks], states[checks]
                # Fit the same distant return that triggered the early preview.
                seed = _return_seed(scan, history, minimum_range_m=1.9, maximum_range_m=2.65)
                refined = crossing_direction_from_planes(scan, history, rays, motion[checks], seeded=seed)
                self.far_crossing_refined[lanes[checks]] = (refined != motion[checks]).any(dim=1)
                motion[checks] = refined
        speed_mover = motion.norm(dim=1, keepdim=True)
        along = motion/speed_mover
        normal = torch.stack((-along[:, 1], along[:, 0]), dim=1)
        offset = (position*normal).sum(dim=1)
        side = offset.sign()
        toward = (request[lanes, :2]*normal).sum(dim=1)*side
        direction = torch.nn.functional.pad(-side[:, None]*normal-.22*along, (0, 1))
        direction = torch.nn.functional.normalize(direction, dim=1)
        intent = self.issued(scans, states, actor_action[lanes], config)
        heading = (intent.v_final_b[:, :2]*normal).sum(dim=1)*side
        stop_risk = intercept_scores(position, motion, coast, radius, intent.v_final_b[:, None],
                                     hold_s=.6, preview_steps=121)[:, 0]
        time_to_pass = -(position*along).sum(dim=1)/speed_mover[:, 0]
        # Only count coasting and pipeline travel that reduce the lateral reserve.
        drift = (side*(coast*normal).sum(dim=1)).clamp_min(0.)
        queued = (side*(previous[lanes, :2]*normal).sum(dim=1)).clamp_min(0.)
        standstill = offset.abs()-.35*drift-.1*queued
        same_side = (direction*self.intercept_direction[lanes]).sum(dim=1) >= .20
        onset = (standstill >= .08) & (toward >= .12) & (heading >= .12) & (time_to_pass > 0.) & (time_to_pass <= 5.5)
        relevant = (stop_risk < .36) & (intent.metrics['emergency'][:, 0] == 0.)
        relevant &= torch.where(continuing, same_side, onset)
        active = relevant.nonzero().flatten()
        use = torch.zeros_like(relevant)
        actions = self.action_for_velocity((SPEED_M_S-.0001)*direction, previous[lanes], config.governor)
        if len(active):
            checked = self.issued(scans[active], states[active], actions[active], config)
            velocity = checked.v_final_b[:, None]
            seen = observed_path_clearance(scans[active], rays, coast[active], velocity, hold_s=2.6)[:, 0]
            lateral = -side[active]*(checked.v_final_b[:, :2]*normal[active]).sum(dim=1)
            stopped = standstill[active]+2.6*lateral-radius[active]
            predicted = torch.stack([
                intercept_scores(position[active], (speed_mover[active]+delta).clamp_min(.20)*along[active],
                                 coast[active], radius[active], velocity, hold_s=2.6, preview_steps=121)[:, 0]
                for delta in (-.12, 0., .12)
            ]).amin(dim=0)
            use[active] = ((checked.metrics['beta'][:, 0] >= .90) & (lateral >= .12)
                           & (seen >= .39) & (predicted >= .39) & (stopped >= .39))
        self.early_crossing[lanes] = use
        self.intercept_steps[lanes[continuing & ~use]] = 0
        return lanes[use], actions[use], direction[use]

    def end_at_verified_hover(self, lidar, state, selected, config, eligible):
        from .escape_passage import clear_fitted_crossing_at_stop, clear_opening_crossing_at_stop, retreat_stop_clearance

        if self.verify_neutral_retry:
            prior_retry = self.neutral_retry_pending.clone()
            self.neutral_retry_pending.zero_()
            self.neutral_retry_floor.zero_()
        zero = state[:, 45:48].norm(dim=1) <= 1.e-6
        leaving_hover = (self.applied | self.retreating) & ~self.braking & ~self.ranked
        lanes = (eligible & zero & leaving_hover).nonzero().flatten()
        if not len(lanes):
            return lanes
        history = state[lanes].reshape(-1, 4, 13)
        settled = ((history[:, :, 6:9].norm(dim=2) <= 1.e-6).all(dim=1)
                   & (history[:, :, 9:12].norm(dim=2) <= .08).all(dim=1))
        onset = (self.remaining[lanes] == DURATION_STEPS-1) & ~self.retreating[lanes]
        response = self.command_offsets[lanes], self.command_coast[lanes]
        emergency_m = config.ics.emergency_clearance_m+config.ics.clearance_margin_m
        proof = clear_fitted_crossing_at_stop(lidar[lanes], state[lanes], self.detector._full_rays,
                                             response, settled & onset, emergency_m)
        if self.crossing_stop_receding_hover:
            opening_allowed = ~onset
            if self.verify_release_stop:
                protected = (self.holding | self.intercepted | self.intercept_continued | self.recovering)[lanes]
                opening_allowed |= onset & ~protected & (history[:, -2, 6:9].norm(dim=1) > .03)
            ordinary = ~proof & opening_allowed & ~self.retreating[lanes]
            pending = lanes[ordinary]
            if len(pending):
                pending_response = self.command_offsets[pending], self.command_coast[pending]
                proof[ordinary] = clear_opening_crossing_at_stop(lidar[pending], state[pending],
                    self.detector._full_rays, pending_response, emergency_m)
        if self.verify_retreat_stop:
            ages = history[:, :, 12]
            ready = ~proof & self.retreating[lanes] & (self.retreat_hold_steps[lanes] > 0)
            ready &= history[:, -2, 6:9].norm(dim=1) > .03
            ready &= (ages <= .04).all(dim=1) & (ages.diff(dim=1).abs() <= .03).all(dim=1)
            ready &= (history[:, :, :3].norm(dim=2).amax(dim=1) <= .18)
            ready &= history[:, :, 9:12].norm(dim=2).amax(dim=1) <= .31
            ready &= (history[:, -1, 9:12]*self.retreat_hold_direction[lanes]).sum(dim=1) >= .12
            for index in ready.nonzero().flatten():
                lane = lanes[index].reshape(1)
                scan, local = lidar[lane], state[lane]
                stop_action = selected.new_tensor([[0., .5, .5, .5]])
                neutral = self.issued(scan, local, stop_action, config)
                if bool(neutral.metrics['emergency'][0, 0]) or float(neutral.v_final_b.norm()) > .001:
                    continue
                response = self.command_offsets[lane], self.command_coast[lane]
                bound = retreat_stop_clearance(scan, local, self.detector._full_rays, response, neutral.v_final_b)
                if bound is not None:
                    proof[index] = True
                    self.retreat_stop_floor[lane] = bound
        if self.verify_neutral_retry:
            protected = (self.holding | self.intercepted | self.intercept_continued | self.recovering
                         | self.composed_crossing_verified)[lanes]
            prior_speed = history[:, -2, 6:9].norm(dim=1)
            ready = ~proof & ~protected
            self.neutral_retry_pending[lanes] = ready & (prior_speed > .03)
            second = ready & prior_retry[lanes] & (prior_speed <= 1.e-6) & (history[:, -3, 6:9].norm(dim=1) > .03)
            for index in second.nonzero().flatten():
                lane = lanes[index].reshape(1)
                scan, local = lidar[lane], state[lane]
                action = selected.new_tensor([[0., .5, .5, .5]])
                neutral = self.issued(scan, local, action, config)
                if bool(neutral.metrics['emergency'][0, 0]) or float(neutral.v_final_b.norm()) > .001:
                    continue
                floor = self.neutral_retry_observer.clearance(int(lane[0]), scan, local, self.detector._full_rays)
                if floor is not None:
                    proof[index] = True
                    self.neutral_retry_floor[lane] = floor
        held = lanes[proof]
        self.hold_neutral(held, selected)
        self.retreating[held] = False
        self.retreat_hold_steps[held] = 0
        self.early_crossing[held] = False
        self.intercept_steps[held] = 0
        self.offset_stopped[held] = True
        return held

    def keep_settled_hover_at_reversal(self, lidar, state, selected, config, eligible):
        from .escape_passage import clear_reversing_crossing_at_hover

        onset = self.applied & (self.remaining == DURATION_STEPS-1)
        ordinary = onset & ~self.ranked if self.crossing_turn_between_samples else None
        if ordinary is None:
            onset &= self.ranked
        was_verified = self.settled_crossing_verified
        lanes = (eligible & (onset | was_verified) & ~self.braking & ~self.retreating).nonzero().flatten()
        prior = was_verified[lanes]
        was_verified.zero_()
        partial = None
        if self.verify_oldest_partial_turn:
            self.oldest_partial_turn_verified.zero_()
            partial = prior.new_zeros(len(lanes))
        ranked = None
        if self.verify_ranked_reversal_stop:
            self.ranked_reversal_verified.zero_()
            ranked = prior.new_zeros(len(lanes))
        if not len(lanes):
            return lanes
        response = self.command_offsets[lanes], self.command_coast[lanes]
        emergency_m = config.ics.emergency_clearance_m+config.ics.clearance_margin_m
        proof = clear_reversing_crossing_at_hover(lidar[lanes], state[lanes], self.detector._full_rays,
            response, self.settled_crossing_mount, emergency_m, prior,
            ordinary_onset=None if ordinary is None else ordinary[lanes],
            verify_reference_reserve=self.verify_hover_reference_reserve, partial_oldest_verified=partial,
            ranked_onset=None if ranked is None else (onset & self.ranked)[lanes], ranked_reversal_verified=ranked)
        if partial is not None:
            self.oldest_partial_turn_verified[lanes] = partial
        if ranked is not None:
            self.ranked_reversal_verified[lanes] = ranked
        was_verified[lanes[proof]] = True
        held = lanes[proof]
        self.hold_neutral(held, selected)
        self.ranked[held] = False
        self.intercept_steps[held] = 0
        self.early_crossing[held] = False
        self.offset_stopped[held] = True
        return held

    def verify_zero_crossing(self, lidar, state, selected, config):
        from .escape_passage import crossing_stop_components, intercept_scores, moving_intercepts

        history = state.reshape(-1, 4, 13)
        zero = history[:, -1, 6:9].norm(dim=1) <= 1.e-6
        previous_motion = history[:, -2, 6:9].norm(dim=1) > .03
        onset = zero & previous_motion & self.applied & ~self.intercepted
        ticks = self.zero_crossing_ticks.clone()
        self.zero_crossing_ticks.sub_(1).clamp_min_(0)
        self.zero_crossing_ticks.masked_fill_(~zero, 0)
        ages = history[:, :, 12]
        fresh = (ages <= .04).all(dim=1) & (ages.diff(dim=1).abs() <= .03).all(dim=1)
        limit = torch.where(onset, .50, .80)
        fresh &= history[:, :, :3].norm(dim=2).amax(dim=1) <= limit
        protected = (self.retreating | self.braking | self.ranked | self.holding | self.recovering
                     | self.crossing_restart_verified)
        eligible = zero & fresh & ~protected & (onset | (ticks > 0))
        if not bool(eligible.any()):
            return
        rays = self.detector._full_rays
        lanes, position, motion, coast, radius = moving_intercepts(lidar, state, rays, eligible,
            command_response=(self.command_offsets, self.command_coast))
        if not len(lanes):
            return
        scans, local = lidar[lanes], state[lanes]
        original = self.issued(scans, local, selected[lanes], config)
        mover_speed = motion.norm(dim=1, keepdim=True)
        along = motion/mover_speed
        normal = torch.stack((-along[:, 1], along[:, 0]), dim=1)
        side = -(position*normal).sum(dim=1).sign()[:, None]*normal
        lateral = torch.nn.functional.pad(side, (0, 1))
        tangent = torch.nn.functional.pad(along, (0, 1))
        original_outward = (original.v_final_b*lateral).sum(dim=1)
        emergency = original.metrics['emergency'][:, 0] != 0.
        ready = ~emergency & (~onset[lanes] | (original_outward <= -.12))
        if not bool(ready.any()):
            return
        headings = [torch.nn.functional.normalize(lateral+angle*tangent, dim=1)
                    for angle in (.22, 0., .40, -.22, -.40)]
        amounts = (.15, .20, .24, .2999)
        targets = [speed*heading for speed in amounts for heading in headings]
        actions = [self.action_for_velocity(target, local[:, 48:51], config.governor) for target in targets]
        checked = [self.issued(scans, local, action, config) for action in actions]
        velocities = torch.stack((original.v_final_b, torch.zeros_like(original.v_final_b),
                                  *(trial.v_final_b for trial in checked)), dim=1)
        swept, stopped = crossing_stop_components(scans, rays, position, motion, coast, radius, velocities, hold_s=.60)
        floor = torch.minimum(swept, stopped)
        for variation in (-.12, .12):
            varied = (mover_speed+variation).clamp_min(.20)*along
            estimate = intercept_scores(position, varied, coast, radius, velocities, hold_s=.60, preview_steps=121)
            floor = torch.minimum(floor, estimate)
        floor -= .10*local[:, 48:51].norm(dim=1, keepdim=True)

        ordinary_safe = floor[:, 0] >= .39
        stop = ready & (~onset[lanes] | ~ordinary_safe) & (floor[:, 1] >= .39)
        stay_away = self.intercepted[lanes] & ordinary_safe & (original_outward >= .12)
        can_move = ready & ~stop & ~stay_away & (~onset[lanes] | ~ordinary_safe)
        can_move &= onset[lanes] | (ticks[lanes] > 29)
        usable = torch.cat([((trial.metrics['beta'] >= .90) & (trial.metrics['emergency'] == 0.))
                            for trial in checked], dim=1)
        usable &= (floor[:, 2:] >= .39) & ((floor[:, 2:]-floor[:, :1]) >= .075)
        usable &= (velocities[:, 2:]*lateral[:, None]).sum(dim=2) >= .12
        usable[:, torch.arange(len(targets), device=state.device) % len(headings) != 0] &= ~onset[lanes, None]
        moving = can_move & usable.any(dim=1)
        choices = usable.to(torch.int32).argmax(dim=1)[moving]
        withdrawn = lanes[moving]
        selected[withdrawn] = torch.stack(actions, dim=1)[moving, choices]
        withdrawn_velocity = velocities[moving, choices+2]
        self.intercept_direction[withdrawn] = torch.nn.functional.normalize(withdrawn_velocity, dim=1)
        self.intercept_speed[withdrawn] = withdrawn_velocity.norm(dim=1)
        self.intercept_steps[withdrawn] = 2
        self.intercepted[withdrawn] = True
        self.intercept_continued[withdrawn] = False
        self.zero_crossing_commit[withdrawn] = 2
        self.zero_crossing_mode[withdrawn] = 1
        self.zero_crossing_floor[withdrawn] = floor[moving, choices+2]
        self.hold_neutral(lanes[stop], selected)
        self.intercepted[lanes[stop]] = False
        self.intercept_steps[lanes[stop]] = 0
        self.intercept_continued[lanes[stop]] = False
        self.zero_crossing_commit[lanes[stop]] = 0
        self.zero_crossing_mode[lanes[stop]] = 2
        self.zero_crossing_floor[lanes[stop]] = floor[stop, 1]
        self.zero_crossing_ticks[lanes[onset[lanes] & (moving | stop)]] = 40

    def keep_side_at_retreat_start(self, scans, history, selected, config, lanes, position, motion, coast, radius):
        from .escape_passage import crossing_stop_components, intercept_scores

        unit = motion/motion.norm(dim=1, keepdim=True)
        normal = torch.stack((-unit[:, 1], unit[:, 0]), dim=1)
        same_side = -((position*normal).sum(dim=1)).sign()[:, None]*normal
        same_side = torch.nn.functional.pad(same_side, (0, 1))
        request, previous = extract_latest_commands(history)
        current = self.issued(scans, history, selected[lanes], config)
        conflicts = ((request*same_side).sum(dim=1) >= .025) & ((current.v_final_b*same_side).sum(dim=1) <= -.08)
        candidates = conflicts.nonzero().flatten()
        if not len(candidates):
            return lanes[:0]
        scans, history, direction = scans[candidates], history[candidates], same_side[candidates]
        velocity = (SPEED_M_S-.0001)*direction
        trial = self.action_for_velocity(velocity, previous[candidates], config.governor)
        checked = self.issued(scans, history, trial, config)
        issued = torch.stack((current.v_final_b[candidates], checked.v_final_b), dim=1)
        points = (position[candidates], motion[candidates], coast[candidates], radius[candidates])
        # One current request and fourteen held requests precede the brake preview.
        swept, settled = crossing_stop_components(scans, self.detector._full_rays, *points, issued, hold_s=1.5)
        tracked_speed = points[1].norm(dim=1, keepdim=True)
        robust = torch.stack([
            intercept_scores(points[0], unit[candidates]*(tracked_speed+delta).clamp_min(.20),
                             points[2], points[3], checked.v_final_b[:, None],
                             hold_s=1.5, preview_steps=101)[:, 0]
            for delta in (-.12, .12)
        ]).amin(dim=0)
        queued = .10*previous[candidates, :2].norm(dim=1)
        access = (checked.metrics['beta'][:, 0] >= .90) & (checked.metrics['emergency'][:, 0] == 0.)
        access &= (checked.v_final_b*direction).sum(dim=1) >= .12
        crossed = access & (swept[:, 0] < .21) & (settled[:, 0] < -points[3])
        use = (crossed & (torch.minimum(torch.minimum(swept[:, 1], settled[:, 1]), robust)-queued >= .39)
               & (swept[:, 1]-swept[:, 0] >= .20))
        remaining = torch.full_like(candidates, 14)
        if self.verify_retreat_onset_side:
            recent = self.reserve_direction_seen[lanes[candidates]]
            previous_heading = self.intercept_direction[lanes[candidates]]
            recent &= (previous_heading*direction).sum(dim=1) >= .75
            recent &= ((previous[candidates]*direction).sum(dim=1) >= .12)
            recent &= ((request[candidates]*direction).sum(dim=1) >= .12)
            fresh_turn = history.reshape(-1, 4, 13)[:, :, :3].norm(dim=2).amax(dim=1) <= .28
            offset = (position*normal).sum(dim=1).abs()
            old_motion = current.v_final_b[candidates]
            inward = ((old_motion*direction).sum(dim=1) <= -.12) & (current.metrics['emergency'][candidates, 0] == 0.)
            examine = (recent & fresh_turn & (offset[candidates] >= .32) & inward & access
                       & ~crossed & (swept[:, 0] < .21) & (settled[:, 0] < 0.)).nonzero().flatten()
            if len(examine):
                target = checked.v_final_b[examine, None]
                position_now, motion_now, coast_now, radius_now = (value[examine] for value in points)
                forward = unit[candidates[examine]]
                across = normal[candidates[examine]]
                minimum = torch.minimum(swept[examine, 1], settled[examine, 1])
                for angle in (-.04, 0., .04):
                    heading = torch.nn.functional.normalize(forward+angle*across, dim=1)
                    lateral = torch.stack((-heading[:, 1], heading[:, 0]), dim=1)
                    start = ((position_now-.35*coast_now)*lateral).sum(dim=1)
                    end = start-1.5*(target[:, 0, :2]*lateral).sum(dim=1)
                    minimum = torch.minimum(minimum, start.sign()*end-radius_now)
                    for delta in (-.12, 0., .12):
                        varied = (tracked_speed[examine]+delta).clamp_min(.20)*heading
                        sweep = intercept_scores(position_now, varied, coast_now, radius_now, target,
                                                 hold_s=1.5, preview_steps=121)[:, 0]
                        minimum = torch.minimum(minimum, sweep)
                bound = minimum-.10*previous[candidates[examine]].norm(dim=1)
                verified = (bound >= .39) & (bound-swept[examine, 0] >= .20)
                approved = examine[verified]
                use[approved] = True
                self.retreat_onset_side_floor[lanes[candidates[approved]]] = bound[verified]
        if self.crossing_retreat_extended:
            retry = (crossed & ~use).nonzero().flatten()
            if len(retry):
                local = tuple(point[retry] for point in points)
                approved = checked.v_final_b[retry, None]
                extended_swept, extended_settled = crossing_stop_components(
                    scans[retry], self.detector._full_rays, *local, approved, hold_s=1.8)
                extended_robust = torch.stack([
                    intercept_scores(local[0], unit[candidates[retry]]*(tracked_speed[retry]+delta).clamp_min(.20),
                                     local[2], local[3], approved, hold_s=1.8, preview_steps=101)[:, 0]
                    for delta in (-.12, .12)
                ]).amin(dim=0)
                clearance = torch.minimum(torch.minimum(extended_swept[:, 0], extended_settled[:, 0]), extended_robust)
                use[retry] = ((clearance-queued[retry] >= .39)
                              & (extended_swept[:, 0]-swept[retry, 0] >= .20))
                remaining[retry] = 17
        changed = lanes[candidates[use]]
        selected[changed] = trial[use]
        self.retreat_hold_direction[changed] = direction[use]
        self.retreat_hold_steps[changed] = remaining[use]
        self.intercept_steps[changed] = 0
        self.early_crossing[changed] = False
        return changed

    def redirect_hover_escape_from_crossing(self, lidar, state, selected, config):
        from .escape_passage import crossing_stop_components, intercept_scores, moving_intercepts, rotate_scan_rays

        continuing = self.hover_escape_continuation.clone()
        self.hover_escape_continuation.zero_()
        self.hover_escape_redirected.zero_()
        history = state.reshape(-1, 4, 13)
        zero = (history[:, :, 6:9].norm(dim=2) <= 1.e-6).all(dim=1)
        fresh = (history[:, :, 12] <= .04).all(dim=1)
        angular_rate = history[:, :, :3].norm(dim=2).amax(dim=1)
        settled = (history[:, :, 9:12].norm(dim=2) <= .02).all(dim=1) & (angular_rate <= .08)
        starting = self.applied & (self.remaining == DURATION_STEPS-1) & settled
        protected = self.ranked | self.retreating | self.braking | self.holding | self.intercepted
        eligible = zero & fresh & (angular_rate <= .50) & (starting | continuing) & ~protected
        if not bool(eligible.any()):
            return
        rays = self.detector._full_rays
        lanes, position, motion, coast, radius = moving_intercepts(lidar, state, rays, eligible,
            command_response=(self.command_offsets, self.command_coast))
        if not len(lanes):
            return
        scans, local = lidar[lanes], state[lanes]
        speed = motion.norm(dim=1, keepdim=True)
        along = motion/speed
        normal = torch.stack((-along[:, 1], along[:, 0]), dim=1)
        own_side = -(position*normal).sum(dim=1).sign()[:, None]*normal
        reverse = torch.nn.functional.normalize(own_side-.84*along, dim=1)
        direction = torch.nn.functional.pad(reverse, (0, 1))
        action = torch.cat((local.new_zeros((len(lanes), 1)), .5+(SPEED_M_S-.0001)*direction), dim=1)
        candidate = self.issued(scans, local, action, config)
        original = self.issued(scans, local, selected[lanes], config)
        choices = torch.stack((original.v_final_b, torch.zeros_like(original.v_final_b), candidate.v_final_b), dim=1)
        swept, settled = crossing_stop_components(scans, rays, position, motion, coast, radius, choices, hold_s=2.4)
        minimum = torch.minimum(swept, settled)
        for difference in (-.12, 0., .12):
            varied = (speed+difference).clamp_min(.20)*along
            robust = intercept_scores(position, varied, coast, radius, choices, hold_s=2.4, preview_steps=121)
            minimum = torch.minimum(minimum, robust)
        floor = minimum-.10*local[:, 48:51].norm(dim=1, keepdim=True)
        toward_crossing = (original.v_final_b[:, :2]*own_side).sum(dim=1) <= -.12
        hidden_by_stop_reserve = (floor[:, 1] >= .36) & (floor[:, 1] < .42)
        begin = starting[lanes] & toward_crossing & (floor[:, 0] < .21) & hidden_by_stop_reserve
        stop = continuing[lanes] & (floor[:, 1] >= .39)
        rotation = -.05*(history[lanes, -2, :3]+history[lanes, -1, :3])
        previous = rotate_scan_rays(self.hover_escape_direction[lanes, None], rotation)[:, 0]
        same_heading = (previous*direction).sum(dim=1) >= .95
        proceed = (begin | (continuing[lanes] & ~stop & same_heading)) & (floor[:, 2] >= .39)
        proceed &= (candidate.v_final_b[:, :2]*own_side).sum(dim=1) >= .12
        proceed &= (candidate.metrics['beta'][:, 0] >= .90) & (original.metrics['emergency'][:, 0] == 0.)
        chosen = lanes[proceed]
        selected[chosen] = action[proceed]
        self.hover_escape_redirected[chosen] = True
        self.hover_escape_continuation[chosen] = True
        self.hover_escape_direction[chosen] = direction[proceed]
        self.intercepted[chosen] = True
        self.applied[chosen] = False
        self.remaining[chosen] = 0
        self.cooldown[chosen] = 1
        self.hold_neutral(lanes[stop], selected)

    def avoid_interception(self, lidar, state, selected, config, *, actor_action=None):
        from .escape_passage import intercept_scores, moving_intercepts, observed_path_clearance

        self.intercepted.zero_()
        if self.offset_stop:
            self.offset_stopped.zero_()
        eligible = (state[:, -1] <= .04) & (state.reshape(-1, 4, 13)[:, :, :3].norm(dim=2).amax(dim=1) <= .50)
        if self.clear_pass:
            eligible &= ~self.holding
        if self.crossing_stop_verified_hover:
            held = self.end_at_verified_hover(lidar, state, selected, config, eligible)
            eligible[held] = False
        if self.settled_crossing_mount is not None:
            held = self.keep_settled_hover_at_reversal(lidar, state, selected, config, eligible)
            eligible[held] = False
        protected_retreat = torch.zeros_like(self.applied)
        if self.retreat_priority:
            protected_retreat[self.usable_recovery_retreat(lidar, state, selected, config)] = True
        response = (self.command_offsets, self.command_coast) if self.align_intercept else None
        early_lanes = state.new_empty(0, dtype=torch.long)
        if self.crossing_stop_early:
            early_lanes, early_actions, early_directions = self.anticipate_crossing(
                lidar, state, actor_action, config, eligible & ~protected_retreat, response)
            eligible[early_lanes] = False
        lanes, position, motion, coast, radius = moving_intercepts(
            lidar, state, self.detector._full_rays, eligible, command_response=response)
        if self.crossing_reference and len(lanes):
            from .escape_reference import crossing_direction_from_planes

            history = state[lanes].reshape(-1, 4, 13)
            request = history[:, -1, 6:9]
            speed = request.norm(dim=1)
            normal = torch.stack((-motion[:, 1], motion[:, 0]), dim=1)/motion.norm(dim=1, keepdim=True)
            offset = (position*normal).sum(dim=1)
            stable = (history[:, :3, 6:9]-request[:, None]).norm(dim=2).amax(dim=1) <= .06
            uncertain = (offset.abs() <= .28) & (speed >= .08) & (speed <= .30) & stable
            candidates = uncertain.nonzero().flatten()
            if len(candidates):
                checks = lanes[candidates]
                motion[candidates] = crossing_direction_from_planes(
                    lidar[checks], state[checks], self.detector._full_rays, motion[candidates])
        reoriented = lanes[:0]
        if self.crossing_retreat_own_side and len(lanes):
            starting = (protected_retreat[lanes] & (self.retreat_hold_steps[lanes] == 14) & ~self.applied[lanes]
                        & (state[lanes].reshape(-1, 4, 13)[:, :, 12] <= .04).all(dim=1))
            initial = lanes[starting]
            if len(initial):
                reoriented = self.keep_side_at_retreat_start(lidar[initial], state[initial], selected, config, initial,
                    position[starting], motion[starting], coast[starting], radius[starting])
        previously_confirmed = ((self.intercept_steps > 0) | self.intercept_continued) if self.intercept_choice else None
        if self.reserve_direction:
            reserve_seen = self.reserve_direction_seen.clone()
            self.reserve_direction_seen.zero_()
        if self.verified_resume and actor_action is not None:
            self.resume_latest_request(lidar, state, selected, actor_action, config, eligible,
                                       (lanes, position, motion, coast, radius))
        if self.tracked_resume:
            self.prove_tracked_resume(lidar, state, selected, actor_action, config, eligible,
                                      (lanes, position, motion, coast, radius))
        if self.commit_intercept:
            observed = torch.cat((lanes, early_lanes)) if len(early_lanes) else lanes
            if self.retreat_priority:
                conflicting = (self.intercept_direction*self.retreat_hold_direction).sum(dim=1) < .4
                observed = torch.cat((observed, (protected_retreat & conflicting).nonzero().flatten()))
            self.continue_intercept(lidar, state, selected, config, observed)
        if len(early_lanes):
            selected[early_lanes] = early_actions
            self.intercepted[early_lanes] = True
            self.intercept_steps[early_lanes] = 4
            self.intercept_direction[early_lanes] = early_directions
            self.intercept_speed[early_lanes] = SPEED_M_S-.0001
        if len(reoriented):
            keep = ~torch.isin(lanes, reoriented)
            lanes, position, motion, coast, radius = (x[keep] for x in (lanes, position, motion, coast, radius))
        if self.shallow_steering:
            untracked = eligible & ~protected_retreat & ~self.intercepted
            untracked[lanes] = False
            self.steer_along_observed_opening(lidar, state, selected, actor_action, config, untracked)
        if not len(lanes):
            return
        if self.offset_stop:
            keep = ~self.stop_for_offset_pass(lidar, state, selected, lanes, position, motion, coast, radius)
            lanes, position, motion, coast, radius = (x[keep] for x in (lanes, position, motion, coast, radius))
            if not len(lanes):
                return
        scans, history = lidar[lanes], state[lanes]
        original = self.issued(scans, history, selected[lanes], config)
        score = intercept_scores(position, motion, coast, radius, original.v_final_b[:, None])[:, 0]
        exposed = score < .36
        consider = exposed.clone()
        if self.intercept_reserve:
            stopping = intercept_scores(position, motion, coast, radius, torch.zeros_like(original.v_final_b[:, None]))[:, 0]
            consider |= stopping < .36
        consider &= original.metrics['emergency'].flatten() == 0.
        if not bool(consider.any()):
            return
        lanes, scans, history, exposed = lanes[consider], scans[consider], history[consider], exposed[consider]
        position, motion, coast, radius, score = (x[consider] for x in (position, motion, coast, radius, score))
        side = torch.stack((-motion[:, 1], motion[:, 0], torch.zeros_like(radius)), dim=1)
        side /= side.norm(dim=1, keepdim=True)
        zero = torch.zeros_like(side)
        directions = torch.stack((zero, side, -side), dim=1)
        if self.intercept_choice:
            along = torch.cat((motion/motion.norm(dim=1, keepdim=True), zero[:, :1]), dim=1)
            diagonal = torch.stack((side+.60*along, -side+.60*along), dim=1)
            diagonal /= diagonal.norm(dim=2, keepdim=True)
            directions = torch.cat((directions, self.intercept_direction[lanes, None], diagonal), dim=1)
        target = SPEED_M_S*directions
        if self.along_intercept:
            target[:, 3] = self.intercept_speed[lanes, None]*directions[:, 3]
        trials = torch.cat((history.new_zeros((*directions.shape[:2], 1)), .5+target), dim=2)
        velocity = torch.stack([self.issued(scans, history, a, config).v_final_b
                                for a in trials.unbind(dim=1)], dim=1)
        scores = intercept_scores(position, motion, coast, radius, velocity)
        hold = exposed & (scores[:, 0] >= .42) & (scores[:, 0]-score >= .12)
        transverse = (velocity[:, 1:]*directions[:, 1:]).sum(dim=2)
        admissible = transverse >= .12
        if self.intercept_reserve:
            current = (original.v_final_b[consider, None]*directions[:, 1:]).sum(dim=2)
            continues = (current >= .12) & (transverse-current >= .04)
            admissible &= exposed[:, None] | continues
        if self.retreat_priority:
            # A fresh forecast can override an existing withdrawal when it
            # predicts contact within the vehicle envelope.
            keep = protected_retreat[lanes] & (score >= .21)
            aligned = (velocity[:, 1:]*self.retreat_hold_direction[lanes, None]).sum(dim=2) >= .12
            admissible &= ~keep[:, None] | aligned
            hold &= ~keep
        gain = torch.where(exposed, .15, .04)
        if self.intercept_choice:
            if self.reserve_direction:
                # Reserve-only choices can be ranked on a fresh exposure. They
                # never grant the missing-observation continuation lease.
                previously_confirmed[lanes] |= reserve_seen[lanes] & exposed
            admissible[:, 2] &= previously_confirmed[lanes]
            if not self.early_diagonal:
                standard = admissible[:, :3] & (scores[:, 1:4] >= .28)
                standard &= scores[:, 1:4]-score[:, None] >= gain[:, None]
                admissible[:, 3:] &= ((score < .21) & ~standard.any(dim=1))[:, None]
        best, side_id = scores[:, 1:].masked_fill(~admissible, -torch.inf).max(dim=1)
        if self.early_diagonal:
            prior = scores[:, 3]
            prefer_prior = (admissible[:, 2] & (prior >= .28) & (prior-score >= gain) & (prior >= best-.04))
            best = torch.where(prefer_prior, prior, best)
            side_id = torch.where(prefer_prior, 2, side_id)
        change = hold | ((best >= .28) & (best-score >= gain))
        choice = torch.where(hold, 0, side_id+1)
        rows = torch.arange(len(lanes), device=lanes.device)
        move = trials[rows, choice]
        issued = velocity[rows, choice]
        direction = directions[rows, choice]
        if self.finite_intercept:
            renewed = torch.zeros_like(change)
        if self.reverse_diagonal:
            fallback = ((score < .21) & ~change).nonzero().flatten()
            if len(fallback):
                reverse = .22*along[fallback]
                options = torch.stack((side[fallback]-reverse, -side[fallback]-reverse), dim=1)
                options /= options.norm(dim=2, keepdim=True)
                # Leave room below the ICS threshold after action encoding.
                speed = SPEED_M_S-.0001
                actions = torch.cat((history.new_zeros((len(fallback), 2, 1)), .5+speed*options), dim=2)
                checked = torch.stack([self.issued(scans[fallback], history[fallback], a, config).v_final_b
                                       for a in actions.unbind(dim=1)], dim=1)
                predicted = intercept_scores(position[fallback], motion[fallback], coast[fallback], radius[fallback],
                                             checked)
                feasible = (checked*options).sum(dim=2) >= .12
                separation, option = predicted.masked_fill(~feasible, -torch.inf).max(dim=1)
                proceed = (separation >= .28) & (separation-score[fallback] >= gain[fallback])
                accepted, option = fallback[proceed], option[proceed]
                move[accepted] = actions[proceed, option]
                issued[accepted] = checked[proceed, option]
                direction[accepted] = options[proceed, option]
                change[accepted] = True
        if self.along_intercept:
            zero = history[:, 45:48].norm(dim=1) <= .001
            known_retreat = protected_retreat[lanes] & (score >= .21)
            witness = (zero & (stopping[consider] < .21) & ~known_retreat).nonzero().flatten()
            if len(witness):
                pace = along[witness]
                action = torch.cat((history.new_zeros((len(witness), 1)), .5+(SPEED_M_S-.0001)*pace), dim=1)
                checked = self.issued(scans[witness], history[witness], action, config)
                current = torch.where(change[witness, None], issued[witness], original.v_final_b[consider][witness])
                velocities = torch.stack((current, checked.v_final_b), dim=1)
                tracked = intercept_scores(position[witness], motion[witness], coast[witness], radius[witness], velocities)
                observed = observed_path_clearance(scans[witness], self.detector._full_rays, coast[witness], velocities)
                # Vehicle envelope, clearance floor and inter-sample travel.
                buffered_body = .21+.15+.03
                threatened = (observed[:, 0] < buffered_body) | (~change[witness] & (score[witness] < .21))
                combined = torch.minimum(tracked, observed)
                if self.finite_intercept:
                    pending = lanes[witness]
                    remembered = previously_confirmed[pending] | reserve_seen[pending]
                    same_direction = (self.intercept_direction[pending]*pace).sum(dim=1) >= .95
                    finite = intercept_scores(position[witness], motion[witness], coast[witness], radius[witness],
                                              velocities, hold_s=.60)
                    second_clearance = finite[:, 1] >= .36
                    threatened |= remembered & same_direction & (finite[:, 0] < .36) & second_clearance
                approved = ((observed[:, 1] >= buffered_body) & (tracked[:, 1] >= .28)
                            & (combined[:, 1]-combined[:, 0] >= gain[witness])
                            & ((checked.v_final_b*pace).sum(dim=1) >= .12))
                use = threatened & approved
                accepted = witness[use]
                move[accepted] = action[use]
                issued[accepted] = checked.v_final_b[use]
                direction[accepted] = pace[use]
                change[accepted] = True
                if self.finite_intercept:
                    renewed[accepted] = second_clearance[use] & (combined[use, 1] >= .45)
        if self.crossing_stop:
            # A pause that avoids the next encounter can leave the vehicle on
            # the mover's line. Recheck a bounded lateral maneuver and its stop.
            neutral = history[:, 45:48].norm(dim=1) <= 1.e-6 if self.crossing_stop_continuation else zero
            witness = (neutral & (stopping[consider] < .36) & ~known_retreat).nonzero().flatten()
            if len(witness):
                pending = lanes[witness]
                reverse = .22*along[witness]
                across = side[witness]
                previous = self.intercept_direction[pending]
                options = torch.stack((across-reverse, -across-reverse, previous), dim=1)
                options = torch.nn.functional.normalize(options, dim=2)
                current = torch.where(change[witness, None], issued[witness], original.v_final_b[consider][witness])
                use, action, checked, escape, available = self.crossing_stop_trial(
                    scans[witness], history[witness], config, position[witness], motion[witness],
                    coast[witness], radius[witness], options, previous, current)
                if self.crossing_stop_continuation:
                    prior = previously_confirmed[pending]
                    use &= ~prior | ((previous*escape).sum(dim=1) >= 0.)
                    if self.crossing_stop_shallow:
                        retry = (prior & ~use & ~available[:, 2]).nonzero().flatten()
                        if len(retry):
                            subset = witness[retry]
                            same_side = torch.where((previous[retry]*side[subset]).sum(dim=1, keepdim=True) >= 0.,
                                                    side[subset], -side[subset])
                            offset = .08749*along[subset]  # tangent of five degrees
                            options = torch.nn.functional.normalize(torch.stack((same_side-offset, same_side+offset), dim=1), dim=2)
                            confirmed, shifted, sent, bearing, _ = self.crossing_stop_trial(
                                scans[subset], history[subset], config, position[subset], motion[subset], coast[subset],
                                radius[subset], options, previous[retry], current[retry], minimum_beta=.9)
                            chosen = retry[confirmed]
                            use[chosen] = True
                            action[chosen], checked[chosen], escape[chosen] = shifted[confirmed], sent[confirmed], bearing[confirmed]
                    continued = (prior & ~use).nonzero().flatten()
                    if len(continued):
                        subset = witness[continued]
                        continuation, move_on, checked_on, direction_on = self.continue_crossing_stop(
                            scans[subset], history[subset], config, position[subset], motion[subset],
                            coast[subset], radius[subset], side[subset], along[subset], previous[continued], current[continued])
                        chosen = continued[continuation]
                        use[chosen] = True
                        action[chosen], checked[chosen], escape[chosen] = (
                            move_on[continuation], checked_on[continuation], direction_on[continuation])
                accepted = witness[use]
                move[accepted], issued[accepted], direction[accepted] = action[use], checked[use], escape[use]
                change[accepted] = True
                renewed[accepted] = True
        if self.crossing_stop_compatible:
            request = history[:, 45:48]
            speed = request.norm(dim=1)
            nominal = original.v_final_b[consider]
            prior_motion = (velocity[:, 3]*self.intercept_direction[lanes]).sum(dim=1)
            blocked = (nominal.norm(dim=1) <= .02) & (prior_motion < .12)
            compatible = ((speed >= .08) & (speed <= .30) & blocked & ~known_retreat & ~change
                          & previously_confirmed[lanes] & (stopping[consider] < .36)).nonzero().flatten()
            if len(compatible):
                previous = self.intercept_direction[lanes[compatible]]
                across = side[compatible]
                retained = torch.where((previous*across).sum(dim=1, keepdim=True) >= 0., across, -across)
                target = torch.nn.functional.normalize(retained-.22*along[compatible], dim=1)
                use, action, checked, heading, _ = self.crossing_stop_trial(
                    scans[compatible], history[compatible], config, position[compatible], motion[compatible],
                    coast[compatible], radius[compatible], target[:, None], previous, nominal[compatible], minimum_beta=.9)
                progress = (checked*request[compatible]).sum(dim=1)
                required = .60*speed[compatible]*torch.maximum(speed[compatible], checked.norm(dim=1))
                use &= progress >= required
                accepted = compatible[use]
                move[accepted], issued[accepted], direction[accepted] = action[use], checked[use], heading[use]
                change[accepted] = True
                renewed[accepted] = True
        selected[lanes[change]] = move[change]
        self.intercepted[lanes[change]] = True
        if self.commit_intercept:
            turn = (issued-original.v_final_b[consider]).norm(dim=1)
            commit = change & exposed & ~hold & (turn >= .12)
            if self.finite_intercept:
                commit |= renewed
            self.intercept_steps[lanes[commit]] = 4
            if self.reserve_direction:
                reserve = change & ~exposed & ~hold & (turn >= .12)
                self.reserve_direction_seen[lanes[reserve]] = True
                commit |= reserve
            self.intercept_direction[lanes[commit]] = direction[commit]
            if self.along_intercept:
                speed = (move[commit, 1:]-.5).norm(dim=1)
                self.intercept_speed[lanes[commit]] = torch.where(speed < SPEED_M_S-.00005, SPEED_M_S-.0001, SPEED_M_S)

    @staticmethod
    def action_for_velocity(target, previous, governor):
        smoothing = governor.smoothing
        correction = (target-smoothing*previous)/(1.-smoothing)
        scale = 2.*governor.v_corr_limit
        return torch.cat((torch.zeros_like(correction[:, :1]), (.5+correction/scale).clamp(0., 1.)), dim=1)

    def retain_wall_tangent(self, lidar, state, action, config):
        from .escape_tangent import surface_tangent

        self.tangent_used.zero_()
        if config.governor.v_corr_limit <= 0.:
            return
        command, previous = extract_latest_commands(state)
        speed = command.norm(dim=1)
        ordinary = ~self.applied & ~self.retreating & ~self.recovering & ~self.braking
        lanes = (ordinary & (speed >= .10) & (speed <= .30) & (state[:, -1] <= .04)).nonzero().flatten()
        if not len(lanes):
            return
        scans, history = lidar[lanes], state[lanes]
        original = self.issued(scans, history, action[lanes], config)
        clipped = (original.metrics['beta'][:, 0] <= .25) & (original.metrics['emergency'][:, 0] == 0.)
        blocked = clipped.nonzero().flatten()
        if not len(blocked):
            return
        limiting = original.cache['active_mask'][blocked] & (original.cache['v_safe_m_s'][blocked] < .09)
        confirmed, target = surface_tangent(scans[blocked], history[blocked], self.detector._full_rays, limiting)
        if not bool(confirmed.any()):
            return
        selected = lanes[blocked[confirmed]]
        trial = self.action_for_velocity(target[confirmed], previous[selected], config.governor)
        issued = self.issued(lidar[selected], state[selected], trial, config)
        head = command[selected]/speed[selected, None]
        progress = ((issued.v_final_b-original.v_final_b[blocked[confirmed]])*head).sum(dim=1)
        use = (issued.metrics['beta'][:, 0] >= .9) & (progress >= .055)
        velocities = issued.v_final_b
        if self.tangent_magnitude:
            tangent = target[confirmed]
            full_speed = tangent/tangent.norm(dim=1, keepdim=True)*speed[selected, None]
            minimum = .60*speed[selected]
            projection = (issued.v_final_b*head).sum(dim=1)
            boost = (use & (projection < minimum) & ((full_speed*head).sum(dim=1) >= minimum)).nonzero().flatten()
            if len(boost):
                lanes = selected[boost]
                expanded = self.action_for_velocity(full_speed[boost], previous[lanes], config.governor)
                checked = self.issued(lidar[lanes], state[lanes], expanded, config)
                along = (checked.v_final_b*head[boost]).sum(dim=1)
                accept = (checked.metrics['beta'][:, 0] >= .9) & (along >= minimum[boost])
                trial[boost[accept]] = expanded[accept]
                if self.tangent_response_headroom:
                    velocities = velocities.clone()
                    velocities[boost[accept]] = checked.v_final_b[accept]
        if self.tangent_response_headroom:
            from .escape_passage import moving_intercepts

            ratio = (velocities*head).sum(dim=1)/speed[selected]
            close = (use & (ratio >= .60) & (ratio < .605)).nonzero().flatten()
            if len(close):
                lanes = selected[close]
                target = velocities[close]*(.605002/ratio[close, None])
                capped = target.norm(dim=1) <= speed[lanes].clamp_max(.2999)
                close, lanes, target = close[capped], lanes[capped], target[capped]
                if len(close):
                    candidate_scans, candidate_state = lidar[lanes], state[lanes]
                    static = torch.ones(len(close), dtype=torch.bool, device=state.device)
                    moving, *_ = moving_intercepts(candidate_scans, candidate_state, self.detector._full_rays,
                                                    static, maximum_range_m=2.65)
                    static[moving] = False
                    close, lanes, target = close[static], lanes[static], target[static]
                if len(close):
                    proposed = self.action_for_velocity(target, previous[lanes], config.governor)
                    checked = self.issued(lidar[lanes], state[lanes], proposed, config)
                    along = (checked.v_final_b*head[close]).sum(dim=1)
                    aligned = torch.nn.functional.cosine_similarity(velocities[close], checked.v_final_b)
                    valid = ((checked.metrics['beta'][:, 0] >= .90) & (checked.metrics['emergency'][:, 0] == 0.)
                             & (along >= .605*speed[lanes]) & (aligned >= .999999))
                    trial[close[valid]] = proposed[valid]
        action[selected[use]] = trial[use]
        self.tangent_used[selected[use]] = True

    def progress_around_edge(self, lidar, state, action, config):
        from .escape_tangent import finite_edge_targets
        from .ics import _fresh_history

        self.detour_used.zero_()
        if self.edge_continuation:
            self.edge_continued.zero_()
        if config.governor.v_corr_limit <= 0.:
            return
        command, previous = extract_latest_commands(state)
        speed = command.norm(dim=1)
        ordinary = ~self.applied & ~self.retreating & ~self.recovering & ~self.braking
        if self.wall_tangent:
            ordinary &= ~self.tangent_used
        eligible = ordinary & (speed >= .10) & (speed <= .30) & (state[:, -1] <= .04)
        if self.edge_continuation:
            self.edge_steps.sub_(1).clamp_min_(0)
            previous_requests = state.reshape(-1, 4, 13)[:, :3, 6:9]
            steady = (previous_requests-command[:, None]).norm(dim=2).amax(dim=1) <= .04
            self.edge_steps.masked_fill_(~eligible | ~steady, 0)
        lanes = eligible.nonzero().flatten()
        if not len(lanes):
            return
        scans, history = lidar[lanes], state[lanes]
        original = self.issued(scans, history, action[lanes], config)
        blocked = ((original.metrics['beta'][:, 0] <= .25) & (original.metrics['emergency'][:, 0] == 0.)).nonzero().flatten()
        if self.edge_continuation:
            self.edge_steps[lanes[original.metrics['beta'][:, 0] > .25]] = 0
            self.edge_steps[lanes[original.metrics['emergency'][:, 0] != 0.]] = 0
        if not len(blocked):
            return
        limiting = original.cache['active_mask'][blocked] & (original.cache['v_safe_m_s'][blocked] < .09)
        targets, available = finite_edge_targets(scans[blocked], history[blocked], self.detector._full_rays, limiting)
        fresh = available
        if self.edge_continuation:
            subset = lanes[blocked]
            latest = scans[blocked, 9].flatten(1)
            reliable = (scans[blocked, 10].flatten(1) > .5) & (scans[blocked, 11].flatten(1) >= .55)
            nearest = latest.masked_fill(~reliable, torch.inf).amin(dim=1)
            clear = nearest.isfinite() & (nearest >= .50)
            self.edge_steps[subset[~clear]] = 0
            valid = (self.edge_steps[subset] > 0) & clear
            valid &= _fresh_history(history[blocked], command[subset], .1)[:, 0]
            side = torch.arange(4, device=lidar.device) >= 2
            available = available | (valid[:, None] & (side[None] == self.edge_side[subset, None]))
        indices, choices = available.nonzero(as_tuple=True)
        if not len(indices):
            return
        checked = lanes[blocked[indices]]
        trial = self.action_for_velocity(targets[indices, choices], previous[checked], config.governor)
        issued = self.issued(lidar[checked], state[checked], trial, config)
        head = command[checked]/speed[checked, None]
        projection = (issued.v_final_b*head).sum(dim=1)
        previous_projection = (original.v_final_b[blocked[indices]]*head).sum(dim=1)
        usable = (issued.metrics['beta'][:, 0] >= .90) & (projection >= .6*speed[checked])
        usable &= projection-previous_projection >= .075
        scores = targets.new_full(available.shape, -torch.inf)
        scores[indices, choices] = projection.masked_fill(~usable, -torch.inf)
        gain, best = scores.max(dim=1)
        row = best[indices] == choices
        selected = row & gain[indices].isfinite()
        action[checked[selected]] = trial[selected]
        self.detour_used[checked[selected]] = True
        if self.edge_continuation:
            observed_now = fresh[indices, choices]
            renew = selected & observed_now
            self.edge_steps[checked[renew]] = 9
            self.edge_side[checked[renew]] = choices[renew] >= 2
            self.edge_continued[checked[selected & ~observed_now]] = True

    def steer_along_observed_opening(self, lidar, state, selected, actor_action, config, eligible):
        self.shallow_used.zero_()
        self.steer_on_observed_path(lidar, state, selected, actor_action, config, eligible, bracket_actor=True)
        if self.observed_heading_steering:
            self.observed_heading_used.zero_()
            self.steer_on_observed_path(lidar, state, selected, actor_action, config, eligible, bracket_actor=False)

    def steer_on_observed_path(self, lidar, state, selected, actor_action, config, eligible, *, bracket_actor):
        from .ics import _fresh_history, _rotate_inner_tube, _visible_inner_tube

        command, previous = extract_latest_commands(state)
        speed = command.norm(dim=1)
        protected = self.retreating | self.braking | self.ranked | self.detour_used | self.handed_off
        if bracket_actor:
            protected |= self.tangent_used
        else:
            protected |= self.applied
        moving = speed >= .10 if bracket_actor else speed > .03
        lanes = (eligible & ~protected & moving & (speed <= .30)).nonzero().flatten()
        radius = config.ics.confirmed_tube_radius_m
        if not len(lanes) or radius <= 0.:
            return
        local = self.issued(lidar[lanes], state[lanes], selected[lanes], config)
        minimum = .60*speed[lanes].square()
        stalled = (local.v_final_b*command[lanes]).sum(dim=1) < minimum
        lanes = lanes[stalled & (local.metrics['emergency'][:, 0] == 0.)]
        if not len(lanes):
            return

        scans, history = lidar[lanes], state[lanes]
        governed = decode_governor(actor_action[lanes], history, config.governor).v_gov_b
        ranges = scans[:, 3::3].flatten(2)
        reliable = ((scans[:, 4::3] > .5) & (scans[:, 5::3] >= .65)).flatten(2)
        rays = self.detector._full_rays
        gravity = history[:, 42:45]
        fresh = _fresh_history(history, governed, .1)[:, 0]
        rotating = (_fresh_history(history, governed, .1, maximum_gyro=.38)[:, 0] & ~fresh).nonzero().flatten()

        def observed_clear(heading):
            clear = fresh & _visible_inner_tube(ranges, reliable, rays, heading, gravity, radius)
            if len(rotating):
                clear[rotating] = _rotate_inner_tube(ranges[rotating], reliable[rotating], rays,
                                                    heading[rotating], history[rotating], .1, radius)
            return clear

        observed_actor = torch.ones_like(fresh)
        if bracket_actor:
            actor_heading = governed/governed.norm(dim=1, keepdim=True).clamp_min(1.e-6)
            observed_actor = observed_clear(actor_heading)
            if not bool(observed_actor.any()):
                return
        request = command[lanes]
        heading = request-(request*gravity).sum(dim=1, keepdim=True)*gravity
        heading = heading/heading.norm(dim=1, keepdim=True).clamp_min(1.e-6)
        sideways = torch.cross(-gravity, heading, dim=1)
        best_action = selected[lanes]
        best_continuity = speed[lanes].new_full((len(lanes),), -torch.inf)
        angles = (-.30, .30) if bracket_actor else (-.30, .30, -.50, .50, -.73, .73)
        for angle in angles:
            direction = math.cos(angle)*heading+math.sin(angle)*sideways
            witnesses = (observed_actor & observed_clear(direction)).nonzero().flatten()
            if not len(witnesses):
                continue
            checked_lanes = lanes[witnesses]
            target = speed[checked_lanes, None].clamp_max(SPEED_M_S-.0001)*direction[witnesses]
            trial = self.action_for_velocity(target, previous[checked_lanes], config.governor)
            checked = self.issued(scans[witnesses], history[witnesses], trial, config)
            projection = (checked.v_final_b*command[checked_lanes]).sum(dim=1)
            continuity = (checked.v_final_b*previous[checked_lanes]).sum(dim=1)
            if not bracket_actor:
                continuity = projection+.10*continuity
            accept = ((checked.metrics['beta'][:, 0] >= .90)
                      & (projection >= .60*speed[checked_lanes].square())
                      & (continuity > best_continuity[witnesses]))
            best_action[witnesses[accept]] = trial[accept]
            best_continuity[witnesses[accept]] = continuity[accept]
        use = best_continuity.isfinite()
        chosen = lanes[use]
        selected[chosen] = best_action[use]
        self.detour_used[chosen] = True
        if bracket_actor:
            self.shallow_used[chosen] = True
            self.applied[chosen] = False
            self.remaining[chosen] = 0
            self.cooldown[chosen] = 0
            self.since_escape[chosen] = RECOVERY_LIMIT
        else:
            self.tangent_used[chosen] = False
            self.observed_heading_used[chosen] = True

    @torch.no_grad()
    def verify_composed_crossing_action(self, lidar, state, selected, actor_action, config):
        """Check the merged recovery command against a freshly observed crossing."""
        from .escape_passage import crossing_stop_components, intercept_scores, moving_intercepts

        self.composed_crossing_verified.zero_()
        self.composed_crossing_floor.zero_()
        history = state.reshape(-1, 4, 13)
        request = history[:, -1, 6:9]
        speed = request.norm(dim=1)
        continuing = ((self.composed_crossing_steps > 0)
                      & ((request-self.composed_crossing_request).norm(dim=1) <= .05))
        self.composed_crossing_steps = torch.where(continuing, self.composed_crossing_steps-1, 0)
        ages = history[:, :, 12]
        recovery = self.recovering | self.detour_used
        ready = ((recovery | continuing) & (ages <= .04).all(dim=1)
                 & (ages.diff(dim=1).abs() <= .03).all(dim=1)
                 & (history[:, :, :3].norm(dim=2).amax(dim=1) <= .50)
                 & (speed >= .12) & (speed <= .30))
        ready &= ~(self.holding | self.retreating | self.ranked | self.braking)
        if not bool(ready.any()):
            return
        rays = self.detector._full_rays
        lanes, position, motion, coast, radius = moving_intercepts(lidar, state, rays, ready,
            command_response=(self.command_offsets, self.command_coast), minimum_range_m=1.9, maximum_range_m=2.65)
        if not len(lanes):
            return
        scans, local = lidar[lanes], state[lanes]
        actual = self.issued(scans, local, selected[lanes], config)
        actor = self.issued(scans, local, actor_action[lanes], config)
        along = torch.nn.functional.normalize(motion, dim=1)
        normal = torch.stack((-along[:, 1], along[:, 0]), dim=1)
        signed = (position*normal).sum(dim=1)
        own = -signed.sign()[:, None]*normal
        definitions = torch.stack((own-.60*along, own-.35*along), dim=1)
        directions = torch.nn.functional.pad(torch.nn.functional.normalize(definitions, dim=2), (0, 1))
        actions = torch.cat((local.new_zeros((len(lanes), 2, 1)), .5+(SPEED_M_S-.0001)*directions), dim=2)
        checks = [self.issued(scans, local, trial, config) for trial in actions.unbind(dim=1)]
        velocities = torch.stack((actual.v_final_b, checks[0].v_final_b, checks[1].v_final_b), dim=1)
        hold = 1.5
        swept, stopped = crossing_stop_components(scans, rays, position, motion, coast, radius, velocities, hold_s=hold)
        candidates = velocities[:, 1:]
        floor = torch.minimum(swept[:, 1:], stopped[:, 1:])
        mover_speed = motion.norm(dim=1, keepdim=True)
        for angle in (-.04, 0., .04):
            heading = torch.nn.functional.normalize(along+angle*normal, dim=1)
            across = torch.stack((-heading[:, 1], heading[:, 0]), dim=1)
            parked = ((position-.35*coast)*across).sum(dim=1)
            settled = parked[:, None]-hold*(candidates[:, :, :2]*across[:, None]).sum(dim=2)
            floor = torch.minimum(floor, parked[:, None].sign()*settled-radius[:, None])
            for change in (-.12, .12):
                moved = (mover_speed+change).clamp_min(.20)*heading
                sweep = intercept_scores(position, moved, coast, radius, candidates, hold_s=hold, preview_steps=121)
                floor = torch.minimum(floor, sweep)
        floor -= .10*local[:, 48:51].norm(dim=1, keepdim=True)

        beta = torch.cat([result.metrics['beta'] for result in checks], dim=1)
        emergency = torch.cat([result.metrics['emergency'] for result in checks], dim=1)
        outward = (candidates[:, :, :2]*own[:, None]).sum(dim=2)
        valid = (beta >= .90) & (emergency == 0.) & (outward >= .12) & (floor >= .39)
        # Recheck the shallower direction only if the running withdrawal was clipped.
        valid[:, 1] &= continuing[lanes] & (beta[:, 0] < .90)
        danger = torch.minimum(swept[:, 0], stopped[:, 0])
        inward = -(actual.v_final_b[:, :2]*own).sum(dim=1)
        onset = (recovery[lanes] & (inward >= .08) & (danger < .21) & (actor.metrics['beta'][:, 0] < .60))
        renewal = continuing[lanes] & (danger < .39)
        use = ((onset | renewal) & (signed.abs() >= .28) & (actual.metrics['emergency'][:, 0] == 0.)
               & valid.any(dim=1))
        choices = (~valid[use, 0]).long()
        accepted = lanes[use]
        selected[accepted] = actions[use, choices]
        self.composed_crossing_verified[accepted] = True
        self.composed_crossing_floor[accepted] = floor[use, choices]
        fresh = accepted[~continuing[accepted]]
        self.composed_crossing_steps[fresh] = COMPOSED_CROSSING_RECHECKS
        self.composed_crossing_request[fresh] = request[fresh]

    @torch.no_grad()
    def handoff_static_retreat(self, lidar, state, selected, action, config):
        """End a continuing retreat after two fresh static path checks."""
        from .escape_passage import _return_seed, clear_static_scan, observed_path_clearance

        previous = self.static_retreat_witness.clone()
        self.static_retreat_witness.zero_()
        history = state.reshape(-1, 4, 13)
        request = history[:, -1, 6:9]
        speed = request.norm(dim=1)
        steady = (history[:, :, 6:9]-request[:, None]).norm(dim=2).amax(dim=1) <= .04
        continuing = self.retreating & (self.retreat_hold_steps > 0) & (self.retreat_hold_steps < 14)
        protected = self.applied | self.braking | self.ranked | self.intercepted | self.intercept_continued | self.holding
        lanes = (continuing & ~protected & steady & (speed >= .08) & (speed <= .30)).nonzero().flatten()
        if not len(lanes):
            return

        rays = self.detector._full_rays
        scans, local = lidar[lanes], state[lanes]
        actor = self.issued(scans, local, action[lanes], config)
        held = self.issued(scans, local, selected[lanes], config)
        heading = request[lanes]/speed[lanes, None]
        along = (actor.v_final_b*heading).sum(dim=1)
        old = (held.v_final_b*heading).sum(dim=1)
        usable = ((actor.metrics['beta'][:, 0] >= .90) & (actor.metrics['emergency'][:, 0] == 0.)
                  & (held.metrics['emergency'][:, 0] == 0.) & (along >= .66*speed[lanes])
                  & (old < 0.) & (along-old >= .12))
        observed = ((scans[:, 1::3] > .5) & (scans[:, 2::3] > .55)).all(dim=1).flatten(1)
        forward = heading@rays.T >= .70
        coverage = (observed & forward).sum(dim=1)/forward.sum(dim=1).clamp_min(1)
        closing, _, _ = _return_seed(scans, local, maximum_range_m=2.65)
        candidates = (usable & (coverage >= .45) & ~closing).nonzero().flatten()
        if not len(candidates):
            return

        lanes, scans, local = lanes[candidates], scans[candidates], local[candidates]
        offsets, coast = self.command_offsets[lanes], self.command_coast[lanes]
        floor = observed_path_clearance(scans, rays, torch.zeros_like(coast),
                                        actor.v_final_b[candidates, None], hold_s=2.4)[:, 0]
        floor -= .35*coast.norm(dim=1)+.10*local[:, 48:51].norm(dim=1)+.058
        static = clear_static_scan(scans, local, rays, (offsets, coast), eligible=floor >= .43)
        self.static_retreat_witness[lanes] = static
        release = lanes[static & previous[lanes]]
        selected[release] = action[release]
        self.retreating[release] = False
        self.retreat_hold_steps[release] = 0
        self.retreat_hold_direction[release] = 0.

    @torch.no_grad()
    def reprove_wall_stop_at_release(self, lidar, state, selected, config, previous_speed_bounds):
        """Recheck a closing wall when the latest command first reaches zero."""
        from .escape_passage import rotate_scan_rays
        from .local_wall_stopping import certify_stop_at_front_wall

        previous = self.release_wall_witness.clone()
        self.release_wall_witness.zero_()
        protected = (self.ranked | self.braking | self.retreating | self.holding | self.opening_steered
                     | self.intercepted | self.intercept_continued | self.recovering | self.tracked_resume_used
                     | self.detour_used | self.tangent_used | self.brake_recovery_verified
                     | self.retreat_continuation_verified | self.crossing_restart_verified
                     | self.composed_crossing_verified | (self.intercept_steps > 0) | (self.retreat_hold_steps > 0))
        history = state.reshape(-1, 4, 13)
        speed = history[:, -1, 6:9].norm(dim=1)
        prior_request = history[:, -2, 6:9]
        ordinary = self.applied & ~protected
        rays = self.detector._full_rays
        for index in (ordinary & (speed > .03) & (speed <= .30)).nonzero().flatten():
            lane = index.reshape(1)
            proof = certify_stop_at_front_wall(lidar[lane], state[lane], rays,
                                                previous_speed_bound=previous_speed_bounds[int(index)],
                                                require_closing=True)
            self.release_wall_witness[lane] = proof is not None
        released = ((speed <= 1.e-6) & (prior_request.norm(dim=1) > .03)
                    & (self.remaining < DURATION_STEPS-1) & previous & ordinary)
        for index in released.nonzero().flatten():
            lane = index.reshape(1)
            frame = history[index]
            elapsed = .10 - (frame[-1, 12] - frame[-2, 12])
            rotation = -.5 * elapsed * (frame[-2, :3] + frame[-1, :3])
            approach = rotate_scan_rays(prior_request[lane], rotation)[0]
            proof = certify_stop_at_front_wall(lidar[lane], state[lane], rays,
                                                previous_speed_bound=previous_speed_bounds[int(index)],
                                                require_closing=False, approach=approach)
            if proof is None:
                continue
            neutral = selected.new_tensor([[0., .5, .5, .5]])
            halt = self.issued(lidar[lane], state[lane], neutral, config)
            if not bool(halt.metrics['emergency'][0, 0]):
                self.static_stop_verified[lane] = True
                self.released[lane] = True
                self.hold_neutral(lane, selected)

    def handoff_after_front_wall_brake(self, policy, selected, actor_action, config, pipelines):
        from .front_wall_handoff import tangential_wall_margin

        lidar, state = policy['lidar_grid'], policy['state_vec']
        protected = (self.ranked | self.braking | self.retreating | self.holding | self.recovering
                     | self.intercepted | self.intercept_continued | self.opening_steered
                     | self.composed_crossing_verified | self.detour_used | self.tangent_used
                     | self.tracked_resume_used | self.brake_recovery_verified | self.crossing_restart_verified
                     | self.retreat_continuation_verified | self.handed_off | self.static_stop_verified
                     | self.offset_stopped | (self.intercept_steps > 0) | (self.retreat_hold_steps > 0)
                     | (self.observed_turn_brake_floor > 0.))
        margin = state.new_zeros(len(state))
        for lane in (self.applied & ~protected).nonzero().flatten().tolist():
            pipeline = pipelines[lane]
            if getattr(pipeline, 'front_wall_brake', None) is None:
                continue
            latest, oldest = state[lane, 45:48], state[lane, 6:9]
            speed, old_speed = float(latest.norm()), float(oldest.norm())
            if (min(speed, old_speed) < .08 or max(speed, old_speed) > .30
                    or float(latest @ oldest) > .25*speed*old_speed):
                continue
            index = slice(lane, lane+1)
            held = self.issued(lidar[index], state[index], selected[index], config)
            if bool(held.metrics['emergency'][0, 0]) or float(held.v_final_b[0] @ latest) >= -.08*speed:
                continue
            checked = self.issued(lidar[index], state[index], actor_action[index], config)
            margin[lane] = tangential_wall_margin(
                lidar[index], state[index], self.detector._full_rays, checked,
                (self.command_offsets[index], self.command_coast[index]), pipeline.oldest_speed_bound)
        used = margin > 0.
        self.applied[used] = False
        self.remaining[used] = 0
        self.cooldown[used] = 0
        self.since_escape[used] = RECOVERY_LIMIT
        self.handed_off[used] = True
        selected[used] = actor_action[used]
        return margin

    def apply_recovery(self, policy, action, config):
        if self.verify_retreat_onset_side:
            self.retreat_onset_side_floor.zero_()
        if self.verify_retreat_stop:
            self.retreat_stop_floor.zero_()
        if self.verify_zero_crossing_withdrawal:
            latest = policy['state_vec'][:, 45:48]
            changed = (self.zero_crossing_commit > 0) & (latest.norm(dim=1) > 1.e-6)
            self.intercept_steps[changed] = 0
            self.zero_crossing_commit.sub_(1).clamp_min_(0)
            self.zero_crossing_mode.zero_()
            self.zero_crossing_floor.zero_()
        selected = self._apply_recovery(policy, action, config)
        if self.command_handoff:
            self.handoff_changed_command(policy['lidar_grid'], policy['state_vec'], selected, action, config)
        if self.wall_tangent:
            self.retain_wall_tangent(policy['lidar_grid'], policy['state_vec'], selected, config)
        if self.finite_edge:
            self.progress_around_edge(policy['lidar_grid'], policy['state_vec'], selected, config)
        if self.intercept:
            state = policy['state_vec']
            if self.align_intercept:
                from .escape_passage import advance_command_response

                self.command_offsets, self.command_coast = advance_command_response(
                    self.command_offsets, self.command_coast, state)
            if self.verify_neutral_retry:
                self.neutral_retry_observer.advance(state)
            if self.verified_resume:
                self.avoid_interception(policy['lidar_grid'], state, selected, config, actor_action=action)
                self.resumed &= ~self.intercepted
            else:
                self.avoid_interception(policy['lidar_grid'], state, selected, config,
                                        actor_action=action if self.crossing_stop_early else None)
            if self.hover_escape_own_side:
                self.redirect_hover_escape_from_crossing(policy['lidar_grid'], state, selected, config)
            if self.tracked_resume:
                self.resume_after_interception(selected, action)
            if self.verify_crossing_restart:
                self.keep_request_at_crossing_restart(policy['lidar_grid'], state, selected, action, config)
            if self.verify_brake_recovery:
                self.verify_request_during_braking(policy['lidar_grid'], state, selected, action, config)
            if self.verify_retreat_continuation:
                self.continue_retreat_on_observed_side(policy['lidar_grid'], state, selected, config)
            if self.verify_zero_crossing_withdrawal:
                self.verify_zero_crossing(policy['lidar_grid'], state, selected, config)
            if self.wall_tangent:
                self.tangent_used &= ~self.intercepted
            if self.finite_edge:
                self.detour_used &= ~self.intercepted
            if self.edge_continuation:
                self.edge_continued &= ~self.intercepted
                self.edge_steps[self.intercepted] = 0
        if self.verify_static_retreat_handoff:
            self.handoff_static_retreat(policy['lidar_grid'], policy['state_vec'], selected, action, config)
        if self.verify_composed_crossing:
            self.verify_composed_crossing_action(policy['lidar_grid'], policy['state_vec'], selected, action, config)
        if self.observed_turn_braking:
            protected = (self.applied | self.retreating | self.intercepted | self.intercept_continued
                         | self.holding | self.braking | self.ranked | self.recovering | self.handed_off
                         | self.crossing_restart_verified | self.offset_stopped | self.static_stop_verified
                         | self.brake_recovery_verified | self.composed_crossing_verified
                         | self.opening_steered | self.tangent_used | self.detour_used
                         | (self.retreat_stop_floor > 0.) | (self.intercept_steps > 0.))
            self.observed_turn_brake_floor = self.turn_brake.apply(
                policy['lidar_grid'], policy['state_vec'], selected, config,
                rays=self.detector._full_rays, offsets=self.command_offsets, coast=self.command_coast,
                protected=protected, issued=self.issued)
        return selected

    def _apply_recovery(self, policy, action, config):
        selected = super().apply(policy, action)
        lidar, state = policy['lidar_grid'], policy['state_vec']
        if self.opening_steering:
            self.release_opening_on_command_change(state, selected, action)
        if self.clear_pass:
            self.hold_during_clear_pass(lidar, state, selected)
        if self.rank_return:
            self.ranked &= self.applied & (self.remaining < DURATION_STEPS-1)
        if self.stop_release:
            self.release_older_escape(lidar, state, selected)
        if self.static_scan_stop:
            self.hold_after_static_stop(lidar, state, selected)
        if self.lateral:
            self.select_lateral(lidar, state, selected, config)
        if self.fast_lateral:
            held = self.applied.nonzero().flatten()
            selected[held, 1:] = .5 + self.escape_speed[held, None] * self.direction[held]
        if self.hover_path:
            self.choose_hover_path(lidar, state, selected, config)
        if self.redirect:
            self.redirect_clipped(lidar, state, selected, config)
        if self.front_switch:
            self.switch_blocked_front(lidar, state, selected, config)
        command, _ = extract_latest_commands(state)
        speed = command.norm(dim=1)
        moving = speed > .03
        self.since_escape = torch.where(self.applied, 0, (self.since_escape + 1).clamp_max(RECOVERY_LIMIT))
        self.recovering.zero_()
        self.braking &= self.applied
        mixed = (self.applied & moving).nonzero().flatten()
        if len(mixed):
            base = selected[mixed]
            trial = base.clone()
            trial[:, 0] = .25
            previous = self.issued(lidar[mixed], state[mixed], base, config).v_final_b
            issued = self.issued(lidar[mixed], state[mixed], trial, config).v_final_b
            lateral = self.direction[mixed]
            preserves_escape = (issued * lateral).sum(1) >= (previous * lateral).sum(1) - 1.e-5
            gains_forward = ((issued - previous) * command[mixed]).sum(1) > .002 * speed[mixed]
            use_trial = (preserves_escape & gains_forward)[:, None]
            if self.rank_return:
                use_trial &= ~self.ranked[mixed, None]
            if self.opening_steering:
                use_trial &= ~self.opening_steered[mixed, None]
            selected[mixed] = torch.where(use_trial, trial, base)
            if self.brake_forward_escape:
                # Compare the admissible escape with the latest human command at its trigger.
                chosen = torch.where(use_trial, issued, previous)
                projected = (chosen * command[mixed]).sum(1) / speed[mixed]
                starts = self.remaining[mixed] == DURATION_STEPS - 1
                self.braking[mixed] |= starts & (projected > speed[mixed] + .02)
        if self.brake_forward_escape:
            selected[self.braking] = selected.new_tensor([0., .5, .5, .5])
        if self.recovery_retreat or self.early_retreat:
            self.hold_recovery_retreat(state, selected)
        if self.early_retreat:
            self.start_early_retreat(lidar, state, selected, config, command, speed)
        if not self.recenter:
            return selected

        recovery = (moving & ~self.retreating & (state[:, -1] <= .22) & (self.since_escape >= RECOVERY_DELAY)
                    & (self.since_escape < RECOVERY_LIMIT))
        if self.recenter_rearm:
            self.update_recenter_rearm(lidar, config)
            recovery &= self.rearm_steps == 0
        lanes = recovery.nonzero().flatten()
        if not len(lanes):
            return selected

        local_lidar, local_state = lidar[lanes], state[lanes]
        heading = command[lanes] / speed[lanes, None]
        original = self.issued(local_lidar, local_state, selected[lanes], config)
        progress = (original.v_final_b * heading).sum(1)
        has_room = original.metrics['emergency'].flatten() == 0
        blocked = (progress < .03) & has_room
        if not bool(blocked.any()):
            return selected

        retreat = self.detector.retreat_direction(local_lidar, local_state, hover_only=False)
        alignment = (retreat * heading).sum(1)
        eligible = blocked & (alignment >= -.25)
        if self.recovery_retreat:
            closing = eligible & (retreat.norm(dim=1) > .1) & (alignment < 0.)
            if bool(closing.any()):
                self.start_recovery_retreat(lidar, state, selected, config, lanes[closing], heading[closing])
            eligible &= ~closing
        if not bool(eligible.any()):
            return selected

        lanes, heading = lanes[eligible], heading[eligible]
        local_lidar, local_state = lidar[lanes], state[lanes]
        side = torch.stack((-heading[:, 1], heading[:, 0], torch.zeros_like(heading[:, 0])), dim=1)
        side /= side.norm(dim=1, keepdim=True).clamp_min(.1)
        best = selected[lanes]
        best_score = progress[eligible].clamp_min(0.) + .015
        for alpha, offset in ((1., 0.), (1., .12), (1., -.12), (.65, .22), (.65, -.22)):
            trial = torch.cat((torch.full_like(best[:, :1], alpha), .5 + offset * side), dim=1)
            issued = self.issued(local_lidar, local_state, trial, config).v_final_b
            score = (issued * heading).sum(1) - .18 * (issued * side).sum(1).abs()
            better = score > best_score
            best = torch.where(better[:, None], trial, best)
            best_score = torch.where(better, score, best_score)
            self.recovering[lanes] |= better
        selected[lanes] = best
        return selected
