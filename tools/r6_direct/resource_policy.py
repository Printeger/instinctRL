"""Explicit new-input resource authorization; historical accounting stays intact."""
import json
import time

from tools.r6_evidence import checked
from training_workloads.unified_h1.command_replay import (
    CLEARANCE_ID as COMMAND_CLEARANCE_ID,
    CORNER_PREFIX as COMMAND_CORNER_PREFIX,
    CURVED_PREFIX as COMMAND_CURVED_PREFIX,
    CRAZY_PREFIX as COMMAND_CRAZY_PREFIX, GUARD_PREFIX as COMMAND_GUARD_PREFIX, PREFIX as COMMAND_REPLAY_PREFIX,
    EARLY_PREFIX as COMMAND_EARLY_PREFIX, ENDURANCE_PREFIX as COMMAND_ENDURANCE_PREFIX,
    HOVER_PREFIX as COMMAND_HOVER_PREFIX, LONG_PREFIX as COMMAND_LONG_PREFIX,
    LONG_OBSERVABLE_SWEEP_ID as COMMAND_LONG_OBSERVABLE_SWEEP_ID,
    HOVER_PATH_PREFIX as COMMAND_HOVER_PATH_PREFIX,
    NOVEL_ALIGNED_ID as COMMAND_NOVEL_ALIGNED_ID, NOVEL_COMMITTED_ID as COMMAND_NOVEL_COMMITTED_ID,
    NOVEL_CONFIRM_ID as COMMAND_NOVEL_CONFIRM_ID,
    NOVEL_CONFIRMED_TUBE_ID as COMMAND_NOVEL_CONFIRMED_TUBE_ID,
    NOVEL_EDGE_CONTINUATION_ID as COMMAND_NOVEL_EDGE_CONTINUATION_ID,
    NOVEL_FINITE_EDGE_ID as COMMAND_NOVEL_FINITE_EDGE_ID, NOVEL_REVALIDATE_ID as COMMAND_NOVEL_REVALIDATE_ID,
    NOVEL_HANDOFF_ID as COMMAND_NOVEL_HANDOFF_ID,
    NOVEL_INTERCEPT_ID as COMMAND_NOVEL_INTERCEPT_ID, NOVEL_NEIGHBOR_HISTORY_ID as COMMAND_NOVEL_NEIGHBOR_HISTORY_ID,
    NOVEL_PREFIX as COMMAND_NOVEL_PREFIX,
    NOVEL_STREET_SWEEP_ID as COMMAND_NOVEL_STREET_SWEEP_ID,
    NOVEL_ACCELERATION_BOUNDED_SWEEP_ID as COMMAND_NOVEL_ACCELERATION_BOUNDED_SWEEP_ID,
    NOVEL_NEUTRAL_RETRY_ID as COMMAND_NOVEL_NEUTRAL_RETRY_ID,
    NOVEL_OBSERVED_TURN_BRAKING_ID as COMMAND_NOVEL_OBSERVED_TURN_BRAKING_ID,
    NOVEL_FRONT_WALL_HANDOFF_ID as COMMAND_NOVEL_FRONT_WALL_HANDOFF_ID,
    NOVEL_COLLECTIVE_PROJECTION_ID as COMMAND_NOVEL_COLLECTIVE_PROJECTION_ID,
    NOVEL_STATIC_TURN_ONSET_ID as COMMAND_NOVEL_STATIC_TURN_ONSET_ID,
    NOVEL_CORRIDOR_PROJECTION_ID as COMMAND_NOVEL_CORRIDOR_PROJECTION_ID,
    NOVEL_OBSERVABLE_SWEEP_ID as COMMAND_NOVEL_OBSERVABLE_SWEEP_ID,
    NOVEL_MEASURED_CURVE_DEFLECTION_ID as COMMAND_NOVEL_MEASURED_CURVE_DEFLECTION_ID,
    NOVEL_MEASURED_CURVE_SLEW_ID as COMMAND_NOVEL_MEASURED_CURVE_SLEW_ID,
    NOVEL_MEASURED_CURVE_APPROACH_ID as COMMAND_NOVEL_MEASURED_CURVE_APPROACH_ID,
    NOVEL_OFFSET_STOP_ID as COMMAND_NOVEL_OFFSET_STOP_ID, NOVEL_PRIORITY_ID as COMMAND_NOVEL_PRIORITY_ID,
    NOVEL_STANDOFF_ID as COMMAND_NOVEL_STANDOFF_ID,
    NOVEL_SETTLED_STOP_ID as COMMAND_NOVEL_SETTLED_STOP_ID,
    NOVEL_TANGENT_MAGNITUDE_ID as COMMAND_NOVEL_TANGENT_MAGNITUDE_ID,
    NOVEL_WALL_TANGENT_ID as COMMAND_NOVEL_WALL_TANGENT_ID,
    NOVEL_RESERVE_ID as COMMAND_NOVEL_RESERVE_ID,
    NOVEL_ROTATION_TUBE_ID as COMMAND_NOVEL_ROTATION_TUBE_ID,
    NOVEL_VERIFIED_RESUME_ID as COMMAND_NOVEL_VERIFIED_RESUME_ID,
    NOVEL_VERIFIED_ZERO_ID as COMMAND_NOVEL_VERIFIED_ZERO_ID,
    NOVEL_ZERO_CONFIRM_ID as COMMAND_NOVEL_ZERO_CONFIRM_ID,
    NOVEL_INTERCEPT_CHOICE_ID as COMMAND_NOVEL_INTERCEPT_CHOICE_ID,
    NOVEL_TRACKING_CONFIRM_ID as COMMAND_NOVEL_TRACKING_CONFIRM_ID,
    NOVEL_EARLY_DIAGONAL_ID as COMMAND_NOVEL_EARLY_DIAGONAL_ID,
    NOVEL_RESERVE_DIRECTION_ID as COMMAND_NOVEL_RESERVE_DIRECTION_ID,
    NOVEL_CROSSING_FALLBACK_ID as COMMAND_NOVEL_CROSSING_FALLBACK_ID,
    NOVEL_OPENING_STEERING_ID as COMMAND_NOVEL_OPENING_STEERING_ID,
    NOVEL_ALONG_INTERCEPT_ID as COMMAND_NOVEL_ALONG_INTERCEPT_ID,
    NOVEL_ALONG_CONFIRM_ID as COMMAND_NOVEL_ALONG_CONFIRM_ID,
    NOVEL_FINITE_INTERCEPT_ID as COMMAND_NOVEL_FINITE_INTERCEPT_ID,
    NOVEL_CROSSING_STOP_ID as COMMAND_NOVEL_CROSSING_STOP_ID,
    NOVEL_CROSSING_STOP_CONTINUATION_ID as COMMAND_NOVEL_CROSSING_STOP_CONTINUATION_ID,
    NOVEL_CROSSING_STOP_SHALLOW_ID as COMMAND_NOVEL_CROSSING_STOP_SHALLOW_ID,
    NOVEL_CROSSING_STOP_EARLY_ID as COMMAND_NOVEL_CROSSING_STOP_EARLY_ID,
    NOVEL_CROSSING_COMPATIBLE_ID as COMMAND_NOVEL_CROSSING_COMPATIBLE_ID,
    NOVEL_CROSSING_VERIFIED_HOVER_ID as COMMAND_NOVEL_CROSSING_VERIFIED_HOVER_ID,
    NOVEL_CROSSING_RECEDING_HOVER_ID as COMMAND_NOVEL_CROSSING_RECEDING_HOVER_ID,
    NOVEL_CROSSING_RECEDING_CONFIRM_ID as COMMAND_NOVEL_CROSSING_RECEDING_CONFIRM_ID,
    NOVEL_CROSSING_EXTENDED_RETREAT_ID as COMMAND_NOVEL_CROSSING_EXTENDED_RETREAT_ID,
    NOVEL_CROSSING_SETTLED_REVERSAL_ID as COMMAND_NOVEL_CROSSING_SETTLED_REVERSAL_ID,
    NOVEL_CROSSING_SETTLED_CONFIRM_ID as COMMAND_NOVEL_CROSSING_SETTLED_CONFIRM_ID,
    NOVEL_CROSSING_REFERENCE_ID as COMMAND_NOVEL_CROSSING_REFERENCE_ID,
    NOVEL_CROSSING_REFERENCE_CONFIRM_ID as COMMAND_NOVEL_CROSSING_REFERENCE_CONFIRM_ID,
    NOVEL_SHALLOW_STEERING_ID as COMMAND_NOVEL_SHALLOW_STEERING_ID,
    NOVEL_OBSERVED_HEADING_ID as COMMAND_NOVEL_OBSERVED_HEADING_ID,
    NOVEL_OBSERVED_HEADING_CONFIRM_ID as COMMAND_NOVEL_OBSERVED_HEADING_CONFIRM_ID,
    NOVEL_CROSSING_SAMPLED_TURN_ID as COMMAND_NOVEL_CROSSING_SAMPLED_TURN_ID,
    NOVEL_CROSSING_SAMPLED_TURN_CONFIRM_ID as COMMAND_NOVEL_CROSSING_SAMPLED_TURN_CONFIRM_ID,
    NOVEL_BILATERAL_STOPPING_ID as COMMAND_NOVEL_BILATERAL_STOPPING_ID,
    NOVEL_BILATERAL_STOPPING_CONFIRM_ID as COMMAND_NOVEL_BILATERAL_STOPPING_CONFIRM_ID,
    NOVEL_HOVER_OWN_SIDE_ID as COMMAND_NOVEL_HOVER_OWN_SIDE_ID,
    NOVEL_HOVER_OWN_SIDE_CONFIRM_ID as COMMAND_NOVEL_HOVER_OWN_SIDE_CONFIRM_ID,
    NOVEL_FAR_CROSSING_REFERENCE_ID as COMMAND_NOVEL_FAR_CROSSING_REFERENCE_ID,
    NOVEL_FAR_CROSSING_REFERENCE_CONFIRM_ID as COMMAND_NOVEL_FAR_CROSSING_REFERENCE_CONFIRM_ID,
    NOVEL_VERIFIED_RETREAT_ONSET_ID as COMMAND_NOVEL_VERIFIED_RETREAT_ONSET_ID,
    NOVEL_VERIFIED_RETREAT_ONSET_CONFIRM_ID as COMMAND_NOVEL_VERIFIED_RETREAT_ONSET_CONFIRM_ID,
    NOVEL_CLIPPED_RETREAT_ONSET_ID as COMMAND_NOVEL_CLIPPED_RETREAT_ONSET_ID,
    NOVEL_CLIPPED_RETREAT_ONSET_CONFIRM_ID as COMMAND_NOVEL_CLIPPED_RETREAT_ONSET_CONFIRM_ID,
    NOVEL_TUBE_STOPPING_ID as COMMAND_NOVEL_TUBE_STOPPING_ID,
    NOVEL_TUBE_STOPPING_CONFIRM_ID as COMMAND_NOVEL_TUBE_STOPPING_CONFIRM_ID,
    NOVEL_HOVER_REFERENCE_RESERVE_ID as COMMAND_NOVEL_HOVER_REFERENCE_RESERVE_ID,
    NOVEL_HOVER_REFERENCE_RESERVE_CONFIRM_ID as COMMAND_NOVEL_HOVER_REFERENCE_RESERVE_CONFIRM_ID,
    NOVEL_CROSSING_RESTART_ID as COMMAND_NOVEL_CROSSING_RESTART_ID,
    NOVEL_CROSSING_RESTART_CONFIRM_ID as COMMAND_NOVEL_CROSSING_RESTART_CONFIRM_ID,
    NOVEL_LOCAL_WALL_STOPPING_ID as COMMAND_NOVEL_LOCAL_WALL_STOPPING_ID,
    NOVEL_OVERLAPPING_STOP_ID as COMMAND_NOVEL_OVERLAPPING_STOP_ID,
    NOVEL_OVERLAPPING_STOP_CONFIRM_ID as COMMAND_NOVEL_OVERLAPPING_STOP_CONFIRM_ID,
    NOVEL_BRAKE_RECOVERY_ID as COMMAND_NOVEL_BRAKE_RECOVERY_ID,
    NOVEL_BRAKE_TRANSITION_ID as COMMAND_NOVEL_BRAKE_TRANSITION_ID,
    NOVEL_FRONT_WALL_BRAKING_ID as COMMAND_NOVEL_FRONT_WALL_BRAKING_ID,
    NOVEL_FACETED_WALL_STOPPING_ID as COMMAND_NOVEL_FACETED_WALL_STOPPING_ID,
    NOVEL_WALL_EDGE_REPROOF_ID as COMMAND_NOVEL_WALL_EDGE_REPROOF_ID,
    NOVEL_ZERO_CROSSING_WITHDRAWAL_ID as COMMAND_NOVEL_ZERO_CROSSING_WITHDRAWAL_ID,
    NOVEL_RANKED_REVERSAL_STOP_ID as COMMAND_NOVEL_RANKED_REVERSAL_STOP_ID,
    NOVEL_RANKED_REVERSAL_STOP_CONFIRM_ID as COMMAND_NOVEL_RANKED_REVERSAL_STOP_CONFIRM_ID,
    NOVEL_ROOM_TANGENT_MAGNITUDE_ID as COMMAND_NOVEL_ROOM_TANGENT_MAGNITUDE_ID,
    NOVEL_COMBINED_CONTROL_ID as COMMAND_NOVEL_COMBINED_CONTROL_ID,
    NOVEL_RETREAT_ONSET_SIDE_ID as COMMAND_NOVEL_RETREAT_ONSET_SIDE_ID,
    NOVEL_RETREAT_ONSET_SIDE_CONFIRM_ID as COMMAND_NOVEL_RETREAT_ONSET_SIDE_CONFIRM_ID,
    NOVEL_CROSSING_AND_WALL_MOTION_ID as COMMAND_NOVEL_CROSSING_AND_WALL_MOTION_ID,
    NOVEL_CROSSING_AND_WALL_MOTION_CONFIRM_ID as COMMAND_NOVEL_CROSSING_AND_WALL_MOTION_CONFIRM_ID,
    NOVEL_RETREAT_STOP_ID as COMMAND_NOVEL_RETREAT_STOP_ID,
    NOVEL_RETREAT_STOP_CONFIRM_ID as COMMAND_NOVEL_RETREAT_STOP_CONFIRM_ID,
    NOVEL_WALL_MOTION_AND_STOP_ID as COMMAND_NOVEL_WALL_MOTION_AND_STOP_ID,
    NOVEL_WALL_MOTION_AND_STOP_CONFIRM_ID as COMMAND_NOVEL_WALL_MOTION_AND_STOP_CONFIRM_ID,
    NOVEL_SUPPORTED_WALL_APPROACH_ID as COMMAND_NOVEL_SUPPORTED_WALL_APPROACH_ID,
    NOVEL_SUPPORTED_WALL_APPROACH_CONFIRM_ID as COMMAND_NOVEL_SUPPORTED_WALL_APPROACH_CONFIRM_ID,
    NOVEL_CURRENT_REQUEST_RECOVERY_ID as COMMAND_NOVEL_CURRENT_REQUEST_RECOVERY_ID,
    NOVEL_CURRENT_REQUEST_RECOVERY_CONFIRM_ID as COMMAND_NOVEL_CURRENT_REQUEST_RECOVERY_CONFIRM_ID,
    NOVEL_COMPOSED_CROSSING_ID as COMMAND_NOVEL_COMPOSED_CROSSING_ID,
    NOVEL_STATIC_RETREAT_HANDOFF_ID as COMMAND_NOVEL_STATIC_RETREAT_HANDOFF_ID,
    NOVEL_RETREAT_CONTINUATION_ID as COMMAND_NOVEL_RETREAT_CONTINUATION_ID,
    NOVEL_RETREAT_CONTINUATION_CONFIRM_ID as COMMAND_NOVEL_RETREAT_CONTINUATION_CONFIRM_ID,
    NOVEL_OLDEST_PARTIAL_TURN_ID as COMMAND_NOVEL_OLDEST_PARTIAL_TURN_ID,
    NOVEL_OLDEST_PARTIAL_TURN_CONFIRM_ID as COMMAND_NOVEL_OLDEST_PARTIAL_TURN_CONFIRM_ID,
    NOVEL_CROSSING_VERIFIED_CONFIRM_ID as COMMAND_NOVEL_CROSSING_VERIFIED_CONFIRM_ID,
    NOVEL_DIAGONAL_CONFIRM_ID as COMMAND_NOVEL_DIAGONAL_CONFIRM_ID,
    NOVEL_TURN_ID as COMMAND_NOVEL_TURN_ID,
    PASSAGE_PREFIX as COMMAND_PASSAGE_PREFIX,
    REARM_PREFIX as COMMAND_REARM_PREFIX,
    RANK_PREFIX as COMMAND_RANK_PREFIX,
    STOP_RELEASE_PREFIX as COMMAND_STOP_RELEASE_PREFIX,
    REGRESSION_ID as COMMAND_REGRESSION_ID,
    REGRESSION_RETRY_ID as COMMAND_REGRESSION_RETRY_ID,
    REGRESSION_SUPPORTED_WALL_ID as COMMAND_REGRESSION_SUPPORTED_WALL_ID,
    REGRESSION_CORRIDOR_PROJECTION_ID as COMMAND_REGRESSION_CORRIDOR_PROJECTION_ID,
    RETURN_PREFIX as COMMAND_RETURN_PREFIX, ROUTE_PREFIX as COMMAND_ROUTE_PREFIX,
)
from training_workloads.unified_h1.dynamic_assay import FRONT_STUDY_PREFIX, NEXT_LONG_PREFIX, THREE_PREFIX, U1792_STUDY_PREFIX
from training_workloads.unified_h1.pipe_authority_probe import PIPE_CLEARANCE_RUN, PIPE_PROBE_RUN, PIPE_SWITCH_RUN


THIRD_PREFIX = 'R6-DIRECT-THIRD-'
THIRD_SCHEMA = 'instinctrl-third-training-resources/1'
CYCLE_PREFIX = 'R6-DIRECT-INSTINCT-CYCLE1-'
CYCLE_SCHEMA = 'instinctrl-first-cycle-training-resources/1'
NEXT_PREFIX = 'R6-DIRECT-INSTINCT-NEXT-'
NEXT_SCHEMA = 'instinctrl-next-cycle-training-resources/1'
GPU_STUDY_PREFIX = 'R6-DIRECT-INSTINCT-GPU-'
GPU_STUDY_SCHEMA = 'instinctrl-gpu-three-training-resources/1'
GPU_FOLLOWUP_SCHEMA = 'instinctrl-post-r3-resources/1'
GPU_RECOVERY_SCHEMA = 'instinctrl-r4-recovery-resources/1'
POST_R4_THREE_SCHEMA = 'instinctrl-post-r4-three-resources/1'
NEXT_LONG_SCHEMA = 'instinctrl-post-r4-next-long-resources/1'
PIPE_PROBE_SCHEMA = 'instinctrl-post-u1792-pipe-probe-resources/1'
U1792_STUDY_SCHEMA = 'instinctrl-post-u1792-study-resources/1'
FRONT_STUDY_SCHEMA = 'instinctrl-post-u1792-front-study-resources/1'
COMMAND_REPLAY_SCHEMA = 'instinctrl-continuous-command-resources/1'
BOUNDED_SCOPES = {
    COMMAND_HOVER_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/hover/RESOURCE_LEDGER.json'),
    COMMAND_HOVER_PATH_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/hover_path/RESOURCE_LEDGER.json'),
    COMMAND_PASSAGE_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/passage/RESOURCE_LEDGER.json'),
    COMMAND_REARM_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/rearm/RESOURCE_LEDGER.json'),
    COMMAND_RANK_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/rank/RESOURCE_LEDGER.json'),
    COMMAND_STOP_RELEASE_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/stop_release/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_TURN_ID: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/novel_turn/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_INTERCEPT_ID: (COMMAND_REPLAY_SCHEMA,
                                 'docs/experiments/instinct-continuous-command/intercept/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_ALIGNED_ID: (COMMAND_REPLAY_SCHEMA,
                               'docs/experiments/instinct-continuous-command/intercept_aligned/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
                                'docs/experiments/instinct-continuous-command/independent/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_RESERVE_ID: (COMMAND_REPLAY_SCHEMA,
                              'docs/experiments/instinct-continuous-command/intercept_reserve/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_HANDOFF_ID: (COMMAND_REPLAY_SCHEMA,
                              'docs/experiments/instinct-continuous-command/command_handoff/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_COMMITTED_ID: (COMMAND_REPLAY_SCHEMA,
                                'docs/experiments/instinct-continuous-command/intercept_committed/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_PRIORITY_ID: (COMMAND_REPLAY_SCHEMA,
                               'docs/experiments/instinct-continuous-command/intercept_priority/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_OFFSET_STOP_ID: (COMMAND_REPLAY_SCHEMA,
                                  'docs/experiments/instinct-continuous-command/offset_stop/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_NEIGHBOR_HISTORY_ID: (COMMAND_REPLAY_SCHEMA,
                                        'docs/experiments/instinct-continuous-command/neighbor_history/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_STANDOFF_ID: (COMMAND_REPLAY_SCHEMA,
                                'docs/experiments/instinct-continuous-command/confirmed_standoff/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_SETTLED_STOP_ID: (COMMAND_REPLAY_SCHEMA,
                                    'docs/experiments/instinct-continuous-command/settled_offset_stop/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_WALL_TANGENT_ID: (COMMAND_REPLAY_SCHEMA,
                                    'docs/experiments/instinct-continuous-command/wall_tangent/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CONFIRMED_TUBE_ID: (COMMAND_REPLAY_SCHEMA,
                                      'docs/experiments/instinct-continuous-command/confirmed_tube/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_FINITE_EDGE_ID: (COMMAND_REPLAY_SCHEMA,
                                    'docs/experiments/instinct-continuous-command/finite_edge/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_REVALIDATE_ID: (COMMAND_REPLAY_SCHEMA,
                                   'docs/experiments/instinct-continuous-command/independent_second/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_EDGE_CONTINUATION_ID: (COMMAND_REPLAY_SCHEMA,
                                        'docs/experiments/instinct-continuous-command/edge_continuation/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_TANGENT_MAGNITUDE_ID: (COMMAND_REPLAY_SCHEMA,
                                         'docs/experiments/instinct-continuous-command/tangent_magnitude/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_ROTATION_TUBE_ID: (COMMAND_REPLAY_SCHEMA,
                                    'docs/experiments/instinct-continuous-command/rotation_tube/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_VERIFIED_RESUME_ID: (COMMAND_REPLAY_SCHEMA,
                                       'docs/experiments/instinct-continuous-command/verified_resume/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_VERIFIED_ZERO_ID: (COMMAND_REPLAY_SCHEMA,
                                     'docs/experiments/instinct-continuous-command/verified_zero/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_ZERO_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
                                    'docs/experiments/instinct-continuous-command/independent_final/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_INTERCEPT_CHOICE_ID: (COMMAND_REPLAY_SCHEMA,
                                        'docs/experiments/instinct-continuous-command/intercept_choice/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_TRACKING_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
                                        'docs/experiments/instinct-continuous-command/independent_tracking/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_EARLY_DIAGONAL_ID: (COMMAND_REPLAY_SCHEMA,
                                      'docs/experiments/instinct-continuous-command/early_diagonal/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_DIAGONAL_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
                                        'docs/experiments/instinct-continuous-command/independent_diagonal/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_RESERVE_DIRECTION_ID: (COMMAND_REPLAY_SCHEMA,
                                         'docs/experiments/instinct-continuous-command/reserve_direction/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_FALLBACK_ID: (COMMAND_REPLAY_SCHEMA,
                                     'docs/experiments/instinct-continuous-command/crossing_fallback/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_OPENING_STEERING_ID: (COMMAND_REPLAY_SCHEMA,
                                         'docs/experiments/instinct-continuous-command/opening_steering/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_ALONG_INTERCEPT_ID: (COMMAND_REPLAY_SCHEMA,
                                        'docs/experiments/instinct-continuous-command/along_intercept/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_ALONG_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
                                      'docs/experiments/instinct-continuous-command/independent_along/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_FINITE_INTERCEPT_ID: (COMMAND_REPLAY_SCHEMA,
                                         'docs/experiments/instinct-continuous-command/finite_intercept/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_STOP_ID: (COMMAND_REPLAY_SCHEMA,
                                      'docs/experiments/instinct-continuous-command/crossing_stop/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_STOP_CONTINUATION_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/crossing_stop_continuation/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_STOP_SHALLOW_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/crossing_stop_shallow/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_STOP_EARLY_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/crossing_stop_early/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_COMPATIBLE_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/crossing_stop_compatible/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_VERIFIED_HOVER_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/crossing_stop_verified_hover/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_VERIFIED_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_verified_hover/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_RECEDING_HOVER_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/crossing_stop_receding_hover/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_RECEDING_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_receding_hover/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_EXTENDED_RETREAT_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/crossing_extended_retreat/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_SETTLED_REVERSAL_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/crossing_settled_reversal/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_SETTLED_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_settled_reversal/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_REFERENCE_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/crossing_static_reference/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_REFERENCE_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_static_reference/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_SHALLOW_STEERING_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/shallow_steering/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_OBSERVED_HEADING_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/observed_heading/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_OBSERVED_HEADING_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_observed_heading/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_SAMPLED_TURN_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/crossing_sampled_turn/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_SAMPLED_TURN_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_sampled_turn/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_BILATERAL_STOPPING_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/bilateral_stopping/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_BILATERAL_STOPPING_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_bilateral_stopping/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_HOVER_OWN_SIDE_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/hover_escape_own_side/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_HOVER_OWN_SIDE_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_hover_own_side/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_FAR_CROSSING_REFERENCE_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/far_crossing_reference/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_FAR_CROSSING_REFERENCE_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_far_crossing_reference/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_VERIFIED_RETREAT_ONSET_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/verified_retreat_onset/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_VERIFIED_RETREAT_ONSET_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_verified_retreat_onset/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CLIPPED_RETREAT_ONSET_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/clipped_retreat_onset/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CLIPPED_RETREAT_ONSET_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_clipped_retreat_onset/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_TUBE_STOPPING_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/tube_stopping/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_TUBE_STOPPING_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_tube_stopping/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_HOVER_REFERENCE_RESERVE_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/hover_reference_reserve/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_HOVER_REFERENCE_RESERVE_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_hover_reference_reserve/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_RESTART_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/crossing_restart/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_RESTART_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_crossing_restart/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_LOCAL_WALL_STOPPING_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/local_wall_stopping/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_OVERLAPPING_STOP_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/overlapping_stop/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_OVERLAPPING_STOP_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_overlapping_stop/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_BRAKE_RECOVERY_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/brake_recovery/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_BRAKE_TRANSITION_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/brake_transition/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_FRONT_WALL_BRAKING_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/front_wall_braking/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_FACETED_WALL_STOPPING_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/faceted_wall_stopping/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_WALL_EDGE_REPROOF_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/wall_edge_reproof/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_ZERO_CROSSING_WITHDRAWAL_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/zero_crossing_withdrawal/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_RANKED_REVERSAL_STOP_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/ranked_reversal_stop/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_RANKED_REVERSAL_STOP_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_zero_crossing/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_ROOM_TANGENT_MAGNITUDE_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/room_tangent_magnitude/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_COMBINED_CONTROL_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/combined_control/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_RETREAT_ONSET_SIDE_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/retreat_onset_side/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_RETREAT_ONSET_SIDE_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_retreat_onset_side/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_AND_WALL_MOTION_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/crossing_and_wall_motion/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CROSSING_AND_WALL_MOTION_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_crossing_and_wall_motion/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_RETREAT_STOP_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/retreat_stop/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_RETREAT_STOP_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_retreat_stop/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_WALL_MOTION_AND_STOP_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/wall_motion_and_stop/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_WALL_MOTION_AND_STOP_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_wall_motion_and_stop/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_SUPPORTED_WALL_APPROACH_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/supported_wall_approach/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_SUPPORTED_WALL_APPROACH_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_supported_wall_approach/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CURRENT_REQUEST_RECOVERY_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/current_request_recovery/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CURRENT_REQUEST_RECOVERY_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_current_request_recovery/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_COMPOSED_CROSSING_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/composed_crossing_recovery/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_STATIC_RETREAT_HANDOFF_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/static_retreat_handoff/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_STREET_SWEEP_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_street_sweep/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_ACCELERATION_BOUNDED_SWEEP_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/acceleration_bounded_street_sweep/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_NEUTRAL_RETRY_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/neutral_retry/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_OBSERVED_TURN_BRAKING_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/observed_turn_braking/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_FRONT_WALL_HANDOFF_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/front_wall_handoff/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_COLLECTIVE_PROJECTION_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/collective_projection/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_STATIC_TURN_ONSET_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/static_turn_onset/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_CORRIDOR_PROJECTION_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/corridor_projection/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_OBSERVABLE_SWEEP_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_observable_sweep/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_MEASURED_CURVE_DEFLECTION_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/measured_curve_deflection/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_MEASURED_CURVE_SLEW_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/measured_curve_slew/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_MEASURED_CURVE_APPROACH_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/measured_curve_approach/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_RETREAT_CONTINUATION_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/retreat_continuation/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_RETREAT_CONTINUATION_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_retreat_continuation/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_OLDEST_PARTIAL_TURN_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/oldest_partial_turn/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_OLDEST_PARTIAL_TURN_CONFIRM_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/independent_oldest_partial_turn/RESOURCE_LEDGER.json'),
    COMMAND_NOVEL_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/novel/RESOURCE_LEDGER.json'),
    COMMAND_CURVED_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/curved/RESOURCE_LEDGER.json'),
    COMMAND_CORNER_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/corner/RESOURCE_LEDGER.json'),
    COMMAND_CLEARANCE_ID: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/clearance/RESOURCE_LEDGER.json'),
    COMMAND_EARLY_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/early/RESOURCE_LEDGER.json'),
    COMMAND_LONG_OBSERVABLE_SWEEP_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/long/attempt-02/RESOURCE_LEDGER.json'),
    COMMAND_LONG_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/long/RESOURCE_LEDGER.json'),
    COMMAND_RETURN_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/return/RESOURCE_LEDGER.json'),
    COMMAND_ENDURANCE_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/endurance/RESOURCE_LEDGER.json'),
    COMMAND_CRAZY_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/crazy/RESOURCE_LEDGER.json'),
    COMMAND_REGRESSION_ID: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/regression/RESOURCE_LEDGER.json'),
    COMMAND_REGRESSION_RETRY_ID: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/regression/attempt-02/RESOURCE_LEDGER.json'),
    COMMAND_REGRESSION_SUPPORTED_WALL_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/regression/attempt-03/RESOURCE_LEDGER.json'),
    COMMAND_REGRESSION_CORRIDOR_PROJECTION_ID: (COMMAND_REPLAY_SCHEMA,
        'docs/experiments/instinct-continuous-command/regression/attempt-04/RESOURCE_LEDGER.json'),
    COMMAND_GUARD_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/guard/RESOURCE_LEDGER.json'),
    COMMAND_ROUTE_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/routes/RESOURCE_LEDGER.json'),
    COMMAND_REPLAY_PREFIX: (COMMAND_REPLAY_SCHEMA, 'docs/experiments/instinct-continuous-command/RESOURCE_LEDGER.json'),
    PIPE_PROBE_RUN: (PIPE_PROBE_SCHEMA, 'docs/experiments/instinct-post-u1792-two-long/RESOURCE_LEDGER.json'),
    PIPE_SWITCH_RUN: (PIPE_PROBE_SCHEMA, 'docs/experiments/instinct-post-u1792-two-long/rear-switch/RESOURCE_LEDGER.json'),
    PIPE_CLEARANCE_RUN: (PIPE_PROBE_SCHEMA, 'docs/experiments/instinct-post-u1792-two-long/rear-clearance/RESOURCE_LEDGER.json'),
    U1792_STUDY_PREFIX: (U1792_STUDY_SCHEMA, 'docs/experiments/instinct-post-u1792-two-long/study/RESOURCE_LEDGER.json'),
    FRONT_STUDY_PREFIX: (FRONT_STUDY_SCHEMA, 'docs/experiments/instinct-post-u1792-two-long/study-front/RESOURCE_LEDGER.json'),
    NEXT_LONG_PREFIX: (NEXT_LONG_SCHEMA, 'docs/experiments/instinct-post-r4-next-long/RESOURCE_LEDGER.json'),
    THREE_PREFIX: (POST_R4_THREE_SCHEMA, 'docs/experiments/instinct-post-r4-three/RESOURCE_LEDGER.json'),
    f'{GPU_STUDY_PREFIX}R5-RECOVERY-': (GPU_RECOVERY_SCHEMA,
        'docs/experiments/instinct-r4-recovery/RESOURCE_LEDGER.json'),
    f'{GPU_STUDY_PREFIX}R4-LONG-': (GPU_FOLLOWUP_SCHEMA,
        'docs/experiments/instinct-post-r3/long/RESOURCE_LEDGER.json'),
    f'{GPU_STUDY_PREFIX}R4-HOLDOUT-SMOKE': (GPU_FOLLOWUP_SCHEMA,
        'docs/experiments/instinct-post-r3/holdout-smoke/RESOURCE_LEDGER.json'),
    f'{GPU_STUDY_PREFIX}R4-DEV-OPTION': (GPU_FOLLOWUP_SCHEMA,
        'docs/experiments/instinct-post-r3/option-check/RESOURCE_LEDGER.json'),
    f'{GPU_STUDY_PREFIX}R4-DEV-PERSISTENCE-02': (GPU_FOLLOWUP_SCHEMA,
        'docs/experiments/instinct-post-r3/development-02/RESOURCE_LEDGER.json'),
    f'{GPU_STUDY_PREFIX}R4-DEV-PERSISTENCE': (GPU_FOLLOWUP_SCHEMA,
        'docs/experiments/instinct-post-r3/development/RESOURCE_LEDGER.json'),
    THIRD_PREFIX: (THIRD_SCHEMA, 'docs/experiments/third-training/RESOURCE_LEDGER.json'),
    CYCLE_PREFIX: (CYCLE_SCHEMA, 'docs/experiments/instinct-first-cycle/formal/RESOURCE_LEDGER.json'),
    **{f'{NEXT_PREFIX}R{n}-': (NEXT_SCHEMA, f'docs/experiments/instinct-next-two-cycles/round-{n}/RESOURCE_LEDGER.json')
       for n in (2, 3)},
    **{f'{GPU_STUDY_PREFIX}R2-DYNAMIC-PILOT-{attempt:02}':
       (GPU_STUDY_SCHEMA, f'docs/experiments/instinct-gpu-three/pilot-{attempt:02}/RESOURCE_LEDGER.json')
       for attempt in range(1, 5)},
    **{f'{GPU_STUDY_PREFIX}R{n}-DYNAMIC':
       (GPU_STUDY_SCHEMA, f'docs/experiments/instinct-gpu-three/round-{n}-dynamic/RESOURCE_LEDGER.json')
       for n in (2, 3)},
    **{f'{GPU_STUDY_PREFIX}R{n}-VISIBILITY':
       (GPU_STUDY_SCHEMA, f'docs/experiments/instinct-gpu-three/round-{n}-visibility/RESOURCE_LEDGER.json')
       for n in (2, 3)},
    f'{GPU_STUDY_PREFIX}R2-RETRY-02-': (GPU_STUDY_SCHEMA,
        'docs/experiments/instinct-gpu-three/round-2-retry-02/RESOURCE_LEDGER.json'),
    **{f'{GPU_STUDY_PREFIX}R{n}-': (GPU_STUDY_SCHEMA, f'docs/experiments/instinct-gpu-three/round-{n}/RESOURCE_LEDGER.json')
       for n in (1, 2, 3)},
}
THIRD_CAPS = frozenset((
    'allocated_gpu_seconds', 'live_wall_seconds', 'queue_seconds', 'live_starts',
    'simulator_initializations', 'training_transitions', 'nontraining_transitions',
    'training_checkpoint_writes', 'major_scientific_candidates',
    'output_actual_plus_unreleased_reservations_bytes',
))
OUTPUT_CAP = 'output_actual_plus_unreleased_reservations_bytes'


def unlimited(policy):
    if policy is None:
        return False
    authority = json.loads(checked(policy).read_text())
    if (authority.get('schema_version') != 'r6-resource-user-amendment/1'
            or authority.get('user_instruction') != '同时取消所有保护额度和预算限制。'
            or authority.get('policy') != 'UNLIMITED_PROJECT_BUDGET_NO_PROTECTED_ALLOCATION'
            or authority.get('historical_accounting') != 'PRESERVE_ALL_CHARGES_UNKNOWNS_AND_UNRELEASED_RESERVATIONS'):
        raise ValueError('DIRECT_RESOURCE_POLICY')
    return True


def third_policy(pin, *, schema=THIRD_SCHEMA):
    if pin is None:
        raise ValueError('DIRECT_THIRD_RESOURCE_AUTHORITY_REQUIRED')
    policy = json.loads(checked(pin).read_text())
    if (set(policy) != {'schema_version', 'user_authority', 'caps', 'elapsed_seconds'}
            or policy['schema_version'] != schema
            or set(policy['caps']) != THIRD_CAPS
            or type(policy['elapsed_seconds']) is not int or policy['elapsed_seconds'] <= 0
            or not all(type(v) is int and (v >= 0 if key == OUTPUT_CAP else v > 0)
                       for key, v in policy['caps'].items())):
        raise ValueError('DIRECT_THIRD_RESOURCE_POLICY')
    checked(policy['user_authority'])
    return policy


def load_run_ledger(run_id, authority):
    from tools.r6_direct.import_legacy import ROOT, NEW
    from tools.r6_direct.lifecycle import imported_budget

    scope = BOUNDED_SCOPES.get(run_id)
    if scope is None:
        scope = next((value for prefix, value in BOUNDED_SCOPES.items() if run_id.startswith(prefix)), None)
    if scope is None:
        path = NEW / 'RESOURCE_LEDGER.json'
        ledger = json.loads(path.read_text())
        return path, ledger, imported_budget(ledger), None

    schema, relative = scope
    policy = third_policy(authority, schema=schema)
    path = ROOT / relative
    ledger = json.loads(path.read_text())
    zero = dict.fromkeys(THIRD_CAPS, 0)
    if (ledger['resource_policy'] != authority or ledger['caps'] != policy['caps']
            or ledger['historical_charged'] != zero or ledger['new_used'] != zero
            or ledger['protected_confirmation_delivery_repair'] != zero):
        raise ValueError('DIRECT_THIRD_LEDGER_SCOPE')
    return path, ledger, None, policy


def check_elapsed(ledger, policy, reservation_seconds, *, now=None):
    """Reserve against elapsed wall time as well as cumulative GPU time."""
    now = int(time.time()) if now is None else now
    started = ledger.get('first_live_unix', now)
    if type(started) is not int or started > now or started <= 0:
        raise ValueError('DIRECT_THIRD_LEDGER_CLOCK')
    if now + reservation_seconds > started + policy['elapsed_seconds']:
        raise ValueError('DIRECT_THIRD_ELAPSED_LIMIT')
    return started


def third_predecessor(ledger, predecessor):
    """Include direct diagnostics in the same closed-job sequence."""
    attempts = [row for row in ledger['new_reservations'] if row['id'].startswith(tuple(BOUNDED_SCOPES))]
    if attempts and (not attempts[-1].get('receipt') or attempts[-1]['receipt'] != predecessor):
        raise ValueError('DIRECT_THIRD_PREDECESSOR')
