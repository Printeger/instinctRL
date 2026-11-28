"""Continuous command replay on the frozen B/C plant."""
from bisect import bisect_right
from contextlib import ExitStack
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from tools.r6_evidence import checked, file_pin
from .escape_holdout_scene import COMMAND_LAYOUTS, LAYOUTS, command_layout


PREFIX = 'R6-DIRECT-INSTINCT-GPU-COMMAND-'
GUARD_PREFIX = PREFIX + 'GUARD-'
ROUTE_PREFIX = PREFIX + 'ROUTE-'
CRAZY_PREFIX = PREFIX + 'CRAZY-'
ENDURANCE_PREFIX = PREFIX + 'ENDURANCE-'
RETURN_PREFIX = PREFIX + 'RETURN-'
EARLY_PREFIX = PREFIX + 'EARLY-'
HOVER_PREFIX = PREFIX + 'HOVER-'
PASSAGE_PREFIX = PREFIX + 'PASSAGE-'
REARM_PREFIX = PREFIX + 'REARM-'
RANK_PREFIX = PREFIX + 'RANK-'
CURVED_PREFIX = PREFIX + 'CURVED-'
HOVER_PATH_PREFIX = PREFIX + 'HOVERPATH-'
STOP_RELEASE_PREFIX = PREFIX + 'RELEASE-'
NOVEL_PREFIX = PREFIX + 'NOVEL-'
NOVEL_TURN_ID = NOVEL_PREFIX+'02'
NOVEL_INTERCEPT_ID = NOVEL_PREFIX+'03'
NOVEL_ALIGNED_ID = NOVEL_PREFIX+'04'
NOVEL_CONFIRM_ID = NOVEL_PREFIX+'05'
NOVEL_RESERVE_ID = NOVEL_PREFIX+'06'
NOVEL_HANDOFF_ID = NOVEL_PREFIX+'07'
NOVEL_COMMITTED_ID = NOVEL_PREFIX+'08'
NOVEL_PRIORITY_ID = NOVEL_PREFIX+'09'
NOVEL_OFFSET_STOP_ID = NOVEL_PREFIX+'10'
NOVEL_NEIGHBOR_HISTORY_ID = NOVEL_PREFIX+'11'
NOVEL_STANDOFF_ID = NOVEL_PREFIX+'12'
NOVEL_SETTLED_STOP_ID = NOVEL_PREFIX+'13'
NOVEL_WALL_TANGENT_ID = NOVEL_PREFIX+'14'
NOVEL_CONFIRMED_TUBE_ID = NOVEL_PREFIX+'15'
NOVEL_FINITE_EDGE_ID = NOVEL_PREFIX+'16'
NOVEL_REVALIDATE_ID = NOVEL_PREFIX+'17'
NOVEL_EDGE_CONTINUATION_ID = NOVEL_PREFIX+'18'
NOVEL_TANGENT_MAGNITUDE_ID = NOVEL_PREFIX+'19'
NOVEL_ROTATION_TUBE_ID = NOVEL_PREFIX+'20'
NOVEL_VERIFIED_RESUME_ID = NOVEL_PREFIX+'21'
NOVEL_VERIFIED_ZERO_ID = NOVEL_PREFIX+'22'
NOVEL_ZERO_CONFIRM_ID = NOVEL_PREFIX+'23'
NOVEL_INTERCEPT_CHOICE_ID = NOVEL_PREFIX+'24'
NOVEL_TRACKING_CONFIRM_ID = NOVEL_PREFIX+'25'
NOVEL_EARLY_DIAGONAL_ID = NOVEL_PREFIX+'26'
NOVEL_DIAGONAL_CONFIRM_ID = NOVEL_PREFIX+'27'
NOVEL_RESERVE_DIRECTION_ID = NOVEL_PREFIX+'28'
NOVEL_CROSSING_FALLBACK_ID = NOVEL_PREFIX+'29'
NOVEL_OPENING_STEERING_ID = NOVEL_PREFIX+'30'
NOVEL_ALONG_INTERCEPT_ID = NOVEL_PREFIX+'31'
NOVEL_ALONG_CONFIRM_ID = NOVEL_PREFIX+'32'
NOVEL_FINITE_INTERCEPT_ID = NOVEL_PREFIX+'33'
NOVEL_CROSSING_STOP_ID = NOVEL_PREFIX+'34'
NOVEL_CROSSING_STOP_CONTINUATION_ID = NOVEL_PREFIX+'35'
NOVEL_CROSSING_STOP_SHALLOW_ID = NOVEL_PREFIX+'36'
NOVEL_CROSSING_STOP_EARLY_ID = NOVEL_PREFIX+'37'
NOVEL_CROSSING_COMPATIBLE_ID = NOVEL_PREFIX+'38'
NOVEL_CROSSING_VERIFIED_HOVER_ID = NOVEL_PREFIX+'39'
NOVEL_CROSSING_VERIFIED_CONFIRM_ID = NOVEL_PREFIX+'40'
NOVEL_CROSSING_RECEDING_HOVER_ID = NOVEL_PREFIX+'41'
NOVEL_CROSSING_RECEDING_CONFIRM_ID = NOVEL_PREFIX+'42'
NOVEL_CROSSING_EXTENDED_RETREAT_ID = NOVEL_PREFIX+'43'
NOVEL_CROSSING_SETTLED_REVERSAL_ID = NOVEL_PREFIX+'44'
NOVEL_CROSSING_SETTLED_CONFIRM_ID = NOVEL_PREFIX+'45'
NOVEL_CROSSING_REFERENCE_ID = NOVEL_PREFIX+'46'
NOVEL_CROSSING_REFERENCE_CONFIRM_ID = NOVEL_PREFIX+'47'
NOVEL_SHALLOW_STEERING_ID = NOVEL_PREFIX+'48'
NOVEL_OBSERVED_HEADING_ID = NOVEL_PREFIX+'49'
NOVEL_OBSERVED_HEADING_CONFIRM_ID = NOVEL_PREFIX+'50'
NOVEL_CROSSING_SAMPLED_TURN_ID = NOVEL_PREFIX+'51'
NOVEL_CROSSING_SAMPLED_TURN_CONFIRM_ID = NOVEL_PREFIX+'52'
NOVEL_BILATERAL_STOPPING_ID = NOVEL_PREFIX+'53'
NOVEL_BILATERAL_STOPPING_CONFIRM_ID = NOVEL_PREFIX+'54'
NOVEL_HOVER_OWN_SIDE_ID = NOVEL_PREFIX+'55'
NOVEL_HOVER_OWN_SIDE_CONFIRM_ID = NOVEL_PREFIX+'56'
NOVEL_FAR_CROSSING_REFERENCE_ID = NOVEL_PREFIX+'57'
NOVEL_FAR_CROSSING_REFERENCE_CONFIRM_ID = NOVEL_PREFIX+'58'
NOVEL_VERIFIED_RETREAT_ONSET_ID = NOVEL_PREFIX+'59'
NOVEL_VERIFIED_RETREAT_ONSET_CONFIRM_ID = NOVEL_PREFIX+'60'
NOVEL_CLIPPED_RETREAT_ONSET_ID = NOVEL_PREFIX+'61'
NOVEL_CLIPPED_RETREAT_ONSET_CONFIRM_ID = NOVEL_PREFIX+'62'
NOVEL_TUBE_STOPPING_ID = NOVEL_PREFIX+'63'
NOVEL_TUBE_STOPPING_CONFIRM_ID = NOVEL_PREFIX+'64'
NOVEL_HOVER_REFERENCE_RESERVE_ID = NOVEL_PREFIX+'65'
NOVEL_HOVER_REFERENCE_RESERVE_CONFIRM_ID = NOVEL_PREFIX+'66'
NOVEL_CROSSING_RESTART_ID = NOVEL_PREFIX+'67'
NOVEL_CROSSING_RESTART_CONFIRM_ID = NOVEL_PREFIX+'68'
NOVEL_LOCAL_WALL_STOPPING_ID = NOVEL_PREFIX+'69'
NOVEL_OVERLAPPING_STOP_ID = NOVEL_PREFIX+'70'
NOVEL_OVERLAPPING_STOP_CONFIRM_ID = NOVEL_PREFIX+'71'
NOVEL_BRAKE_RECOVERY_ID = NOVEL_PREFIX+'72'
NOVEL_FRONT_WALL_BRAKING_ID = NOVEL_PREFIX+'73'
NOVEL_RETREAT_CONTINUATION_ID = NOVEL_PREFIX+'74'
NOVEL_RETREAT_CONTINUATION_CONFIRM_ID = NOVEL_PREFIX+'75'
NOVEL_BRAKE_TRANSITION_ID = NOVEL_PREFIX+'76'
NOVEL_FACETED_WALL_STOPPING_ID = NOVEL_PREFIX+'77'
NOVEL_OLDEST_PARTIAL_TURN_ID = NOVEL_PREFIX+'78'
NOVEL_OLDEST_PARTIAL_TURN_CONFIRM_ID = NOVEL_PREFIX+'79'
NOVEL_WALL_EDGE_REPROOF_ID = NOVEL_PREFIX+'80'
NOVEL_ZERO_CROSSING_WITHDRAWAL_ID = NOVEL_PREFIX+'81'
NOVEL_RANKED_REVERSAL_STOP_ID = NOVEL_PREFIX+'82'
NOVEL_RANKED_REVERSAL_STOP_CONFIRM_ID = NOVEL_PREFIX+'83'
NOVEL_ROOM_TANGENT_MAGNITUDE_ID = NOVEL_PREFIX+'84'
NOVEL_COMBINED_CONTROL_ID = NOVEL_PREFIX+'85'
NOVEL_RETREAT_ONSET_SIDE_ID = NOVEL_PREFIX+'86'
NOVEL_RETREAT_ONSET_SIDE_CONFIRM_ID = NOVEL_PREFIX+'87'
NOVEL_CROSSING_AND_WALL_MOTION_ID = NOVEL_PREFIX+'88'
NOVEL_CROSSING_AND_WALL_MOTION_CONFIRM_ID = NOVEL_PREFIX+'89'
NOVEL_RETREAT_STOP_ID = NOVEL_PREFIX+'90'
NOVEL_RETREAT_STOP_CONFIRM_ID = NOVEL_PREFIX+'91'
NOVEL_WALL_MOTION_AND_STOP_ID = NOVEL_PREFIX+'92'
NOVEL_WALL_MOTION_AND_STOP_CONFIRM_ID = NOVEL_PREFIX+'93'
NOVEL_SUPPORTED_WALL_APPROACH_ID = NOVEL_PREFIX+'94'
NOVEL_SUPPORTED_WALL_APPROACH_CONFIRM_ID = NOVEL_PREFIX+'95'
NOVEL_CURRENT_REQUEST_RECOVERY_ID = NOVEL_PREFIX+'96'
NOVEL_CURRENT_REQUEST_RECOVERY_CONFIRM_ID = NOVEL_PREFIX+'97'
NOVEL_COMPOSED_CROSSING_ID = NOVEL_PREFIX+'98'
NOVEL_STATIC_RETREAT_HANDOFF_ID = NOVEL_PREFIX+'99'
NOVEL_STREET_SWEEP_ID = NOVEL_PREFIX+'100'
NOVEL_ACCELERATION_BOUNDED_SWEEP_ID = NOVEL_PREFIX+'101'
NOVEL_NEUTRAL_RETRY_ID = NOVEL_PREFIX+'102'
NOVEL_OBSERVED_TURN_BRAKING_ID = NOVEL_PREFIX+'103'
NOVEL_FRONT_WALL_HANDOFF_ID = NOVEL_PREFIX+'104'
NOVEL_COLLECTIVE_PROJECTION_ID = NOVEL_PREFIX+'105'
NOVEL_STATIC_TURN_ONSET_ID = NOVEL_PREFIX+'106'
NOVEL_CORRIDOR_PROJECTION_ID = NOVEL_PREFIX+'107'
RAMPED_STREET_SWEEP_IDS = (NOVEL_ACCELERATION_BOUNDED_SWEEP_ID, NOVEL_OBSERVED_TURN_BRAKING_ID)
STREET_SWEEP_DEVELOPMENT_IDS = (NOVEL_STREET_SWEEP_ID, *RAMPED_STREET_SWEEP_IDS)
MATCHED_ENDURANCE_PREFIX_SECONDS = {NOVEL_COMPOSED_CROSSING_ID: 120, NOVEL_STATIC_RETREAT_HANDOFF_ID: 420,
                                    NOVEL_NEUTRAL_RETRY_ID: 200, NOVEL_FRONT_WALL_HANDOFF_ID: 200,
                                    NOVEL_COLLECTIVE_PROJECTION_ID: 200, NOVEL_STATIC_TURN_ONSET_ID: 120,
                                    NOVEL_CORRIDOR_PROJECTION_ID: 120}
CORNER_PREFIX = PREFIX + 'CORNER-'
LONG_PREFIX = PREFIX + 'LONG-'
PHYSICS_CLEARANCE_PREFIXES = (PASSAGE_PREFIX, REARM_PREFIX, RANK_PREFIX, CURVED_PREFIX, HOVER_PATH_PREFIX, STOP_RELEASE_PREFIX)
ENDURANCE_PREFIXES = (ENDURANCE_PREFIX, HOVER_PREFIX, *PHYSICS_CLEARANCE_PREFIXES)
RETURN_DURATION_MS = 36000
CLEARANCE_ID = EARLY_PREFIX + '02'
REGRESSION_ID = PREFIX + 'R6-01'
REGRESSION_RETRY_ID = PREFIX + 'R6-02'
REGRESSION_SUPPORTED_WALL_ID = PREFIX + 'R6-03'
REGRESSION_IDS = (REGRESSION_ID, REGRESSION_RETRY_ID, REGRESSION_SUPPORTED_WALL_ID)
MODE = 'research-r6-dynamic-assay'
LANES = 32
STEP_MS = 100
CLEARANCE_SHAPE = (21, LANES, 7)
CROSSING_SPEED_M_S = .58
CROSSING_START_M = 2.2
CROSSING_TRAVEL_S = 7.5
# Centre-line arrival, in simulated milliseconds. The two active arms share their path.
CONDITIONS = {'no_crossing': None, 'crossing_at_reversal': 6400, 'crossing_before_reversal': 4400}
MODELS = {
    'b': '841a0a7c7926a14584f257667c5dffe0077129c33458898026ae37405ce9e231',
    'c': '450b8675161f132178eb345f85c3dd6c27dffe9f3de6a33c792c3fabcf6ba172',
}
EXECUTION = 'm4-escape-rear-clearance-p25-v1'
BRAKE_EXECUTION = 'm4-escape-brake-forward-p25-v1'
TUBE_EXECUTION = 'm4-escape-brake-tube-p25-v1'
TUBE_CONFIRM_EXECUTION = 'm4-escape-brake-tube-p25-v2'
RECOVERY_RETREAT_EXECUTION = 'm4-escape-brake-tube-p25-recovery-retreat-v1'
EARLY_RETREAT_EXECUTION = 'm4-escape-brake-tube-p25-early-retreat-v1'
SIDE_HOVER_RETREAT_EXECUTION = 'm4-escape-brake-tube-p25-side-hover-v1'
CLEAR_PASS_EXECUTION = 'm4-escape-brake-tube-p25-clear-pass-v1'
RECOVERY_REARM_EXECUTION = 'm4-escape-brake-tube-p25-recovery-rearm-v1'
RANKED_RETURN_EXECUTION = 'm4-escape-brake-tube-p25-ranked-return-v1'
CURVED_PASS_EXECUTION = 'm4-escape-brake-tube-p25-curved-pass-v1'
HOVER_PATH_EXECUTION = 'm4-escape-brake-tube-p25-hover-path-v1'
STOP_RELEASE_EXECUTION = 'm4-escape-brake-tube-p25-stop-release-v1'
TURN_FIT_EXECUTION = 'm4-escape-brake-tube-p25-turn-fit-v1'
INTERCEPT_EXECUTION = 'm4-escape-brake-tube-p25-intercept-v1'
ALIGNED_INTERCEPT_EXECUTION = 'm4-escape-brake-tube-p25-intercept-v2'
RESERVE_INTERCEPT_EXECUTION = 'm4-escape-brake-tube-p25-intercept-v3'
COMMITTED_INTERCEPT_EXECUTION = 'm4-escape-brake-tube-p25-intercept-v4'
PRIORITY_INTERCEPT_EXECUTION = 'm4-escape-brake-tube-p25-intercept-v5'
COMMAND_HANDOFF_EXECUTION = 'm4-escape-brake-tube-p25-command-handoff-v1'
OFFSET_STOP_EXECUTION = 'm4-escape-brake-tube-p25-offset-stop-v1'
NEIGHBOR_HISTORY_EXECUTION = 'm4-escape-brake-tube-p25-neighbor-history-v1'
CONFIRMED_STANDOFF_EXECUTION = 'm4-escape-brake-tube-p25-confirmed-standoff-v1'
SETTLED_OFFSET_STOP_EXECUTION = 'm4-escape-brake-tube-p25-settled-offset-stop-v1'
WALL_TANGENT_EXECUTION = 'm4-escape-brake-tube-p25-wall-tangent-v1'
CONFIRMED_TUBE_EXECUTION = 'm4-escape-brake-tube-p25-confirmed-inner-tube-v1'
FINITE_EDGE_EXECUTION = 'm4-escape-brake-tube-p25-finite-edge-v1'
EDGE_CONTINUATION_EXECUTION = 'm4-escape-brake-tube-p25-finite-edge-continued-v1'
TANGENT_MAGNITUDE_EXECUTION = 'm4-escape-brake-tube-p25-tangent-magnitude-v1'
ROTATION_TUBE_EXECUTION = 'm4-escape-brake-tube-p25-rotation-aligned-v1'
VERIFIED_RESUME_EXECUTION = 'm4-escape-brake-tube-p25-verified-resume-v1'
VERIFIED_ZERO_EXECUTION = 'm4-escape-brake-tube-p25-verified-zero-v1'
INTERCEPT_CHOICE_EXECUTION = 'm4-escape-brake-tube-p25-intercept-choice-v1'
EARLY_DIAGONAL_EXECUTION = 'm4-escape-brake-tube-p25-early-diagonal-v1'
RESERVE_DIRECTION_EXECUTION = 'm4-escape-brake-tube-p25-reserve-direction-v1'
CROSSING_FALLBACK_EXECUTION = 'm4-escape-brake-tube-p25-crossing-fallback-v1'
OPENING_STEERING_EXECUTION = 'm4-escape-brake-tube-p25-opening-steering-v1'
ALONG_INTERCEPT_EXECUTION = 'm4-escape-brake-tube-p25-along-intercept-v1'
FINITE_INTERCEPT_EXECUTION = 'm4-escape-brake-tube-p25-finite-intercept-v1'
CROSSING_STOP_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v1'
CROSSING_STOP_CONTINUATION_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v2'
CROSSING_STOP_SHALLOW_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v3'
CROSSING_STOP_EARLY_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v4'
CROSSING_COMPATIBLE_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v5'
CROSSING_VERIFIED_HOVER_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v6'
CROSSING_RECEDING_HOVER_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v7'
CROSSING_EXTENDED_RETREAT_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v8'
CROSSING_SETTLED_REVERSAL_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v9'
CROSSING_SETTLED_MOUNT = (0., 0., .05)
CROSSING_REFERENCE_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v10'
SHALLOW_STEERING_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v11'
OBSERVED_HEADING_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v12'
CROSSING_SAMPLED_TURN_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v13'
BILATERAL_STOPPING_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v14'
HOVER_OWN_SIDE_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v15'
FAR_CROSSING_REFERENCE_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v16'
VERIFIED_RETREAT_ONSET_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v17'
CLIPPED_RETREAT_ONSET_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v18'
TUBE_STOPPING_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v19'
TUBE_STOPPING_TRACE_FIELDS = ('tube_stopping_hold_s','tube_stopping_target_b','tube_baseline_velocity_b',
    'tube_baseline_beta','tube_gap_m','tube_required_gap_m','tube_cleared_limiting_beams')
HOVER_REFERENCE_RESERVE_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v20'
CROSSING_RESTART_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v21'
LOCAL_WALL_STOPPING_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v22'
LOCAL_WALL_STOPPING_TRACE_FIELDS = tuple(field.replace('tube_', 'local_wall_', 1) for field in TUBE_STOPPING_TRACE_FIELDS)
OVERLAPPING_STOP_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v23'
BRAKE_RECOVERY_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v24'
BRAKE_TRANSITION_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v27'
COMBINED_CONTROL_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v34'
CROSSING_AND_WALL_MOTION_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v36'
WALL_MOTION_AND_STOP_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v38'
SUPPORTED_WALL_APPROACH_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v39'
CURRENT_REQUEST_RECOVERY_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v40'
COMPOSED_CROSSING_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v41'
STATIC_RETREAT_HANDOFF_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v43'
GRAVITY_ALIGNED_PROTECTION_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v44'
RELEASE_STOP_CERTIFICATES_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v45'
STEADY_HANDOFF_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v46'
TANGENT_RESPONSE_HEADROOM_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v47'
NEUTRAL_RETRY_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v48'
OBSERVED_TURN_BRAKING_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v49'
FRONT_WALL_HANDOFF_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v50'
COLLECTIVE_PROJECTION_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v51'
STATIC_TURN_ONSET_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v52'
CORRIDOR_PROJECTION_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v53'
STATIC_TURN_ONSET_EXECUTIONS = (STATIC_TURN_ONSET_EXECUTION, CORRIDOR_PROJECTION_EXECUTION)
COLLECTIVE_PROJECTION_EXECUTIONS = (COLLECTIVE_PROJECTION_EXECUTION, CORRIDOR_PROJECTION_EXECUTION)
FRONT_WALL_HANDOFF_EXECUTIONS = (FRONT_WALL_HANDOFF_EXECUTION, COLLECTIVE_PROJECTION_EXECUTION,
                                *STATIC_TURN_ONSET_EXECUTIONS)
OBSERVED_TURN_BRAKING_EXECUTIONS = (OBSERVED_TURN_BRAKING_EXECUTION, *FRONT_WALL_HANDOFF_EXECUTIONS)
NEUTRAL_RETRY_EXECUTIONS = (NEUTRAL_RETRY_EXECUTION, *OBSERVED_TURN_BRAKING_EXECUTIONS)
TANGENT_RESPONSE_HEADROOM_EXECUTIONS = (TANGENT_RESPONSE_HEADROOM_EXECUTION, *NEUTRAL_RETRY_EXECUTIONS)
STEADY_HANDOFF_EXECUTIONS = (STEADY_HANDOFF_EXECUTION, *TANGENT_RESPONSE_HEADROOM_EXECUTIONS)
RELEASE_STOP_CERTIFICATES_EXECUTIONS = (RELEASE_STOP_CERTIFICATES_EXECUTION, *STEADY_HANDOFF_EXECUTIONS)
GRAVITY_ALIGNED_PROTECTION_EXECUTIONS = (GRAVITY_ALIGNED_PROTECTION_EXECUTION, *RELEASE_STOP_CERTIFICATES_EXECUTIONS)
STATIC_RETREAT_HANDOFF_EXECUTIONS = (STATIC_RETREAT_HANDOFF_EXECUTION, *GRAVITY_ALIGNED_PROTECTION_EXECUTIONS)
COMPOSED_CROSSING_EXECUTIONS = (COMPOSED_CROSSING_EXECUTION, *STATIC_RETREAT_HANDOFF_EXECUTIONS)
CURRENT_REQUEST_RECOVERY_EXECUTIONS = (CURRENT_REQUEST_RECOVERY_EXECUTION, *COMPOSED_CROSSING_EXECUTIONS)
SUPPORTED_WALL_EXECUTIONS = (SUPPORTED_WALL_APPROACH_EXECUTION, *CURRENT_REQUEST_RECOVERY_EXECUTIONS)
COMPOSED_MOTION_EXECUTIONS = (COMBINED_CONTROL_EXECUTION, CROSSING_AND_WALL_MOTION_EXECUTION,
                              WALL_MOTION_AND_STOP_EXECUTION, *SUPPORTED_WALL_EXECUTIONS)
BRAKE_TRANSITION_EXECUTIONS = (BRAKE_TRANSITION_EXECUTION, *COMPOSED_MOTION_EXECUTIONS)
BRAKE_RECOVERY_EXECUTIONS = (BRAKE_RECOVERY_EXECUTION, *BRAKE_TRANSITION_EXECUTIONS)
FRONT_WALL_BRAKING_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v25'
FACETED_WALL_STOPPING_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v28'
WALL_EDGE_REPROOF_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v30'
FACETED_WALL_STOPPING_EXECUTIONS = (FACETED_WALL_STOPPING_EXECUTION, *COMPOSED_MOTION_EXECUTIONS)
WALL_EDGE_REPROOF_EXECUTIONS = (WALL_EDGE_REPROOF_EXECUTION, *COMPOSED_MOTION_EXECUTIONS)
FRONT_WALL_BRAKING_EXECUTIONS = (FRONT_WALL_BRAKING_EXECUTION, FACETED_WALL_STOPPING_EXECUTION, *WALL_EDGE_REPROOF_EXECUTIONS)
FRONT_WALL_BRAKING_TRACE_FIELDS = ('front_wall_brake_verified', 'front_wall_brake_previous_b',
    'front_wall_brake_gap_m', 'front_wall_brake_required_m', 'front_wall_brake_attributed_beams')
LOCAL_WALL_STOPPING_EXECUTIONS = (LOCAL_WALL_STOPPING_EXECUTION, *FRONT_WALL_BRAKING_EXECUTIONS)
RETREAT_CONTINUATION_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v26'
OLDEST_PARTIAL_TURN_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v29'
ZERO_CROSSING_WITHDRAWAL_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v31'
RANKED_REVERSAL_STOP_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v32'
RETREAT_ONSET_SIDE_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v35'
RETREAT_STOP_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v37'
RETREAT_STOP_EXECUTIONS = (RETREAT_STOP_EXECUTION, WALL_MOTION_AND_STOP_EXECUTION, *SUPPORTED_WALL_EXECUTIONS)
RETREAT_ONSET_SIDE_EXECUTIONS = (RETREAT_ONSET_SIDE_EXECUTION, CROSSING_AND_WALL_MOTION_EXECUTION,
                                *RETREAT_STOP_EXECUTIONS)
RANKED_REVERSAL_STOP_EXECUTIONS = (RANKED_REVERSAL_STOP_EXECUTION, COMBINED_CONTROL_EXECUTION, *RETREAT_ONSET_SIDE_EXECUTIONS)
ZERO_CROSSING_WITHDRAWAL_EXECUTIONS = (ZERO_CROSSING_WITHDRAWAL_EXECUTION, *RANKED_REVERSAL_STOP_EXECUTIONS)
ROOM_TANGENT_MAGNITUDE_EXECUTION = 'm4-escape-brake-tube-p25-crossing-stop-v33'
TANGENT_MAGNITUDE_EXECUTIONS = (TANGENT_MAGNITUDE_EXECUTION, ROOM_TANGENT_MAGNITUDE_EXECUTION,
                                CROSSING_AND_WALL_MOTION_EXECUTION, WALL_MOTION_AND_STOP_EXECUTION,
                                *SUPPORTED_WALL_EXECUTIONS)
OLDEST_PARTIAL_TURN_EXECUTIONS = (OLDEST_PARTIAL_TURN_EXECUTION, ROOM_TANGENT_MAGNITUDE_EXECUTION,
                                  *ZERO_CROSSING_WITHDRAWAL_EXECUTIONS)
RETREAT_CONTINUATION_EXECUTIONS = (RETREAT_CONTINUATION_EXECUTION, *OLDEST_PARTIAL_TURN_EXECUTIONS)
OVERLAPPING_STOP_EXECUTIONS = (OVERLAPPING_STOP_EXECUTION, *RETREAT_CONTINUATION_EXECUTIONS)
CROSSING_RESTART_EXECUTIONS = (CROSSING_RESTART_EXECUTION, *BRAKE_RECOVERY_EXECUTIONS, *OVERLAPPING_STOP_EXECUTIONS,
                               *LOCAL_WALL_STOPPING_EXECUTIONS)
HOVER_REFERENCE_RESERVE_EXECUTIONS = (HOVER_REFERENCE_RESERVE_EXECUTION, *CROSSING_RESTART_EXECUTIONS)
CLIPPED_RETREAT_ONSET_EXECUTIONS = (CLIPPED_RETREAT_ONSET_EXECUTION, *HOVER_REFERENCE_RESERVE_EXECUTIONS)
VERIFIED_RETREAT_ONSET_EXECUTIONS = (VERIFIED_RETREAT_ONSET_EXECUTION, *CLIPPED_RETREAT_ONSET_EXECUTIONS)
FAR_CROSSING_REFERENCE_EXECUTIONS = (FAR_CROSSING_REFERENCE_EXECUTION, TUBE_STOPPING_EXECUTION, *VERIFIED_RETREAT_ONSET_EXECUTIONS)
HOVER_OWN_SIDE_EXECUTIONS = (HOVER_OWN_SIDE_EXECUTION, *FAR_CROSSING_REFERENCE_EXECUTIONS)
SAMPLED_TURN_EXECUTIONS = (CROSSING_SAMPLED_TURN_EXECUTION, BILATERAL_STOPPING_EXECUTION, *HOVER_OWN_SIDE_EXECUTIONS)
OBSERVED_HEADING_EXECUTIONS = (OBSERVED_HEADING_EXECUTION, *SAMPLED_TURN_EXECUTIONS)
SHALLOW_STEERING_EXECUTIONS = (SHALLOW_STEERING_EXECUTION, *OBSERVED_HEADING_EXECUTIONS)
CROSSING_REFERENCE_EXECUTIONS = (CROSSING_REFERENCE_EXECUTION, *SHALLOW_STEERING_EXECUTIONS)
CROSSING_SETTLED_EXECUTIONS = (CROSSING_SETTLED_REVERSAL_EXECUTION, *CROSSING_REFERENCE_EXECUTIONS)
CROSSING_EXTENDED_RETREAT_EXECUTIONS = (CROSSING_EXTENDED_RETREAT_EXECUTION, *CROSSING_SETTLED_EXECUTIONS)
CROSSING_RECEDING_HOVER_EXECUTIONS = (CROSSING_RECEDING_HOVER_EXECUTION, *CROSSING_EXTENDED_RETREAT_EXECUTIONS)
CROSSING_VERIFIED_HOVER_EXECUTIONS = (CROSSING_VERIFIED_HOVER_EXECUTION, *CROSSING_RECEDING_HOVER_EXECUTIONS)
CROSSING_COMPATIBLE_EXECUTIONS = (CROSSING_COMPATIBLE_EXECUTION, *CROSSING_VERIFIED_HOVER_EXECUTIONS)
CROSSING_EARLY_EXECUTIONS = (CROSSING_STOP_EARLY_EXECUTION, *CROSSING_COMPATIBLE_EXECUTIONS)
CROSSING_SHALLOW_EXECUTIONS = (CROSSING_STOP_SHALLOW_EXECUTION, *CROSSING_EARLY_EXECUTIONS)
CROSSING_CONTINUATION_EXECUTIONS = (CROSSING_STOP_CONTINUATION_EXECUTION, *CROSSING_SHALLOW_EXECUTIONS)
CROSSING_STOP_EXECUTIONS = (CROSSING_STOP_EXECUTION, *CROSSING_CONTINUATION_EXECUTIONS)
FINITE_INTERCEPT_EXECUTIONS = (FINITE_INTERCEPT_EXECUTION, *CROSSING_STOP_EXECUTIONS)
ALONG_INTERCEPT_EXECUTIONS = (ALONG_INTERCEPT_EXECUTION, *FINITE_INTERCEPT_EXECUTIONS)
OPENING_STEERING_EXECUTIONS = (OPENING_STEERING_EXECUTION, *ALONG_INTERCEPT_EXECUTIONS)
CROSSING_FALLBACK_EXECUTIONS = (CROSSING_FALLBACK_EXECUTION, *OPENING_STEERING_EXECUTIONS)
RESERVE_DIRECTION_EXECUTIONS = (RESERVE_DIRECTION_EXECUTION, *CROSSING_FALLBACK_EXECUTIONS)
EARLY_DIAGONAL_EXECUTIONS = (EARLY_DIAGONAL_EXECUTION, *RESERVE_DIRECTION_EXECUTIONS)
INTERCEPT_CHOICE_EXECUTIONS = (INTERCEPT_CHOICE_EXECUTION, *EARLY_DIAGONAL_EXECUTIONS)
VERIFIED_ZERO_EXECUTIONS = (VERIFIED_ZERO_EXECUTION, *INTERCEPT_CHOICE_EXECUTIONS)
ROTATION_TUBE_EXECUTIONS = (ROTATION_TUBE_EXECUTION, VERIFIED_RESUME_EXECUTION, *VERIFIED_ZERO_EXECUTIONS)
FINITE_EDGE_EXECUTIONS = (FINITE_EDGE_EXECUTION, EDGE_CONTINUATION_EXECUTION,
                          TANGENT_MAGNITUDE_EXECUTION, *ROTATION_TUBE_EXECUTIONS)
CONFIRMED_TUBE_EXECUTIONS = (CONFIRMED_TUBE_EXECUTION, *FINITE_EDGE_EXECUTIONS)
WALL_TANGENT_EXECUTIONS = (WALL_TANGENT_EXECUTION, *CONFIRMED_TUBE_EXECUTIONS)
SETTLED_OFFSET_STOP_EXECUTIONS = (SETTLED_OFFSET_STOP_EXECUTION, *WALL_TANGENT_EXECUTIONS)
NEIGHBOR_HISTORY_EXECUTIONS = (NEIGHBOR_HISTORY_EXECUTION, CONFIRMED_STANDOFF_EXECUTION, *SETTLED_OFFSET_STOP_EXECUTIONS)
OFFSET_STOP_EXECUTIONS = (OFFSET_STOP_EXECUTION, *NEIGHBOR_HISTORY_EXECUTIONS)
COMMAND_HANDOFF_EXECUTIONS = (COMMAND_HANDOFF_EXECUTION, *OFFSET_STOP_EXECUTIONS)
PRIORITY_INTERCEPT_EXECUTIONS = (PRIORITY_INTERCEPT_EXECUTION, *COMMAND_HANDOFF_EXECUTIONS)
COMMITTED_INTERCEPT_EXECUTIONS = (COMMITTED_INTERCEPT_EXECUTION, *PRIORITY_INTERCEPT_EXECUTIONS)
INTERCEPT_EXECUTIONS = (INTERCEPT_EXECUTION, ALIGNED_INTERCEPT_EXECUTION, RESERVE_INTERCEPT_EXECUTION,
                        *COMMITTED_INTERCEPT_EXECUTIONS)
STOP_RELEASE_EXECUTIONS = (STOP_RELEASE_EXECUTION, TURN_FIT_EXECUTION, *INTERCEPT_EXECUTIONS)
RANK_EXECUTIONS = (RANKED_RETURN_EXECUTION, CURVED_PASS_EXECUTION, HOVER_PATH_EXECUTION, *STOP_RELEASE_EXECUTIONS)
REARM_EXECUTIONS = (RECOVERY_REARM_EXECUTION, *RANK_EXECUTIONS)
CLEAR_PASS_EXECUTIONS = (CLEAR_PASS_EXECUTION, *REARM_EXECUTIONS)
EXECUTION_PAIRS = {
    GUARD_PREFIX: (EXECUTION, BRAKE_EXECUTION),
    CRAZY_PREFIX: (BRAKE_EXECUTION, TUBE_EXECUTION),
    ENDURANCE_PREFIX: (TUBE_EXECUTION, TUBE_CONFIRM_EXECUTION),
    RETURN_PREFIX: (TUBE_CONFIRM_EXECUTION, RECOVERY_RETREAT_EXECUTION),
    EARLY_PREFIX: (TUBE_CONFIRM_EXECUTION, EARLY_RETREAT_EXECUTION),
    HOVER_PREFIX: (EARLY_RETREAT_EXECUTION, SIDE_HOVER_RETREAT_EXECUTION),
    PASSAGE_PREFIX: (EARLY_RETREAT_EXECUTION, CLEAR_PASS_EXECUTION),
    REARM_PREFIX: (CLEAR_PASS_EXECUTION, RECOVERY_REARM_EXECUTION),
    RANK_PREFIX: (RECOVERY_REARM_EXECUTION, RANKED_RETURN_EXECUTION),
    CURVED_PREFIX: (RANKED_RETURN_EXECUTION, CURVED_PASS_EXECUTION),
    HOVER_PATH_PREFIX: (RANKED_RETURN_EXECUTION, HOVER_PATH_EXECUTION),
    STOP_RELEASE_PREFIX: (HOVER_PATH_EXECUTION, STOP_RELEASE_EXECUTION),
    CORNER_PREFIX: (EARLY_RETREAT_EXECUTION, EARLY_RETREAT_EXECUTION),
}
CRAZY_SCHEMA = 'instinctrl-crazy-command-generator/1'
CRAZY_MODES = ('normal', 'aggressive', 'obstacleward', 'oscillation', 'stop')
PLANNED_POSITION_LIMITS = 'layout-envelope-v1'
LAYOUT_COMMAND_BOUNDS = {
    'room_corner': ((-1.5, 8.9), (-2.3, 2.3)),
    'trunk_aisle': ((-3.4, 6.4), (-2.1, 2.1)),
    'street_gap': ((-1.5, 4.4), (-2.0, 2.0)),
    'corridor': ((-6., 8.), (-1.2, 1.2)),
    'open_tube': ((-6., 8.), (-1.4, 1.4)),
}


def matches(run_id):
    return (run_id in (*STREET_SWEEP_DEVELOPMENT_IDS, NOVEL_NEUTRAL_RETRY_ID, NOVEL_FRONT_WALL_HANDOFF_ID,
                       NOVEL_COLLECTIVE_PROJECTION_ID, NOVEL_STATIC_TURN_ONSET_ID, NOVEL_CORRIDOR_PROJECTION_ID)
            or isinstance(run_id, str)
            and re.fullmatch(PREFIX + r'(?:PAIR|GUARD|ROUTE|CRAZY|ENDURANCE|RETURN|EARLY|HOVER|PASSAGE|REARM|RANK|CURVED|HOVERPATH|RELEASE|CORNER|NOVEL|LONG)-[0-9]{2}', run_id) is not None)


def route_layouts(run_id):
    return run_id.startswith((ROUTE_PREFIX, CRAZY_PREFIX, RETURN_PREFIX, EARLY_PREFIX,
                              CORNER_PREFIX, NOVEL_PREFIX, LONG_PREFIX, *ENDURANCE_PREFIXES))


def arm_execution(params, label):
    return params.get('candidate_execution', params['execution']) if label == 'candidate' else params['execution']


def compress_trace(run_id):
    return run_id.startswith(LONG_PREFIX) or run_id in (NOVEL_CURRENT_REQUEST_RECOVERY_ID, NOVEL_CURRENT_REQUEST_RECOVERY_CONFIRM_ID,
                                                       *STREET_SWEEP_DEVELOPMENT_IDS, *MATCHED_ENDURANCE_PREFIX_SECONDS,
                                                       NOVEL_INTERCEPT_CHOICE_ID, NOVEL_TRACKING_CONFIRM_ID,
                                                       NOVEL_EARLY_DIAGONAL_ID, NOVEL_DIAGONAL_CONFIRM_ID,
                                                       NOVEL_RESERVE_DIRECTION_ID, NOVEL_CROSSING_FALLBACK_ID,
                                                       NOVEL_OPENING_STEERING_ID, NOVEL_ALONG_INTERCEPT_ID,
                                                       NOVEL_ALONG_CONFIRM_ID, NOVEL_FINITE_INTERCEPT_ID,
                                                       NOVEL_CROSSING_STOP_ID, NOVEL_CROSSING_STOP_CONTINUATION_ID,
                                                       NOVEL_CROSSING_STOP_SHALLOW_ID, NOVEL_CROSSING_STOP_EARLY_ID,
                                                       NOVEL_CROSSING_COMPATIBLE_ID, NOVEL_CROSSING_VERIFIED_HOVER_ID,
                                                       NOVEL_CROSSING_VERIFIED_CONFIRM_ID, NOVEL_CROSSING_RECEDING_HOVER_ID,
                                                       NOVEL_CROSSING_RECEDING_CONFIRM_ID,
                                                       NOVEL_CROSSING_EXTENDED_RETREAT_ID, NOVEL_CROSSING_SETTLED_REVERSAL_ID,
                                                       NOVEL_CROSSING_SETTLED_CONFIRM_ID, NOVEL_CROSSING_REFERENCE_ID,
                                                       NOVEL_CROSSING_REFERENCE_CONFIRM_ID, NOVEL_SHALLOW_STEERING_ID,
                                                       NOVEL_OBSERVED_HEADING_ID, NOVEL_OBSERVED_HEADING_CONFIRM_ID,
                                                       NOVEL_CROSSING_SAMPLED_TURN_ID, NOVEL_CROSSING_SAMPLED_TURN_CONFIRM_ID,
                                                       NOVEL_BILATERAL_STOPPING_ID, NOVEL_BILATERAL_STOPPING_CONFIRM_ID,
                                                       NOVEL_HOVER_OWN_SIDE_ID, NOVEL_HOVER_OWN_SIDE_CONFIRM_ID,
                                                       NOVEL_FAR_CROSSING_REFERENCE_ID, NOVEL_FAR_CROSSING_REFERENCE_CONFIRM_ID,
                                                       NOVEL_VERIFIED_RETREAT_ONSET_ID, NOVEL_VERIFIED_RETREAT_ONSET_CONFIRM_ID,
                                                       NOVEL_CLIPPED_RETREAT_ONSET_ID, NOVEL_CLIPPED_RETREAT_ONSET_CONFIRM_ID,
                                                       NOVEL_TUBE_STOPPING_ID, NOVEL_TUBE_STOPPING_CONFIRM_ID,
                                                       NOVEL_HOVER_REFERENCE_RESERVE_ID, NOVEL_HOVER_REFERENCE_RESERVE_CONFIRM_ID,
                                                       NOVEL_CROSSING_RESTART_ID, NOVEL_CROSSING_RESTART_CONFIRM_ID,
                                                       NOVEL_LOCAL_WALL_STOPPING_ID, NOVEL_OVERLAPPING_STOP_ID,
                                                       NOVEL_OVERLAPPING_STOP_CONFIRM_ID, NOVEL_BRAKE_RECOVERY_ID, NOVEL_FRONT_WALL_BRAKING_ID,
                                                       NOVEL_RETREAT_CONTINUATION_ID, NOVEL_RETREAT_CONTINUATION_CONFIRM_ID,
                                                       NOVEL_BRAKE_TRANSITION_ID, NOVEL_FACETED_WALL_STOPPING_ID,
                                                       NOVEL_OLDEST_PARTIAL_TURN_ID, NOVEL_OLDEST_PARTIAL_TURN_CONFIRM_ID,
                                                       NOVEL_WALL_EDGE_REPROOF_ID, NOVEL_ZERO_CROSSING_WITHDRAWAL_ID,
                                                       NOVEL_RANKED_REVERSAL_STOP_ID, NOVEL_RANKED_REVERSAL_STOP_CONFIRM_ID,
                                                       NOVEL_ROOM_TANGENT_MAGNITUDE_ID, NOVEL_COMBINED_CONTROL_ID,
                                                       NOVEL_RETREAT_ONSET_SIDE_ID, NOVEL_RETREAT_ONSET_SIDE_CONFIRM_ID,
                                                       NOVEL_CROSSING_AND_WALL_MOTION_ID, NOVEL_CROSSING_AND_WALL_MOTION_CONFIRM_ID,
                                                       NOVEL_RETREAT_STOP_ID, NOVEL_RETREAT_STOP_CONFIRM_ID,
                                                       NOVEL_WALL_MOTION_AND_STOP_ID, NOVEL_WALL_MOTION_AND_STOP_CONFIRM_ID)


def trace_path(output, label, condition, *, run_id):
    suffix = 'jsonl.gz' if compress_trace(run_id) else 'jsonl'
    return output/f'{label}-{condition}.{suffix}'


def open_trace(path):
    return gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else path.open(encoding='utf-8')


def encode_trace_row(row, *, compressed):
    data = (json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n').encode()
    # Each completed strategy step remains independently recoverable after interruption.
    return gzip.compress(data, compresslevel=1, mtime=0) if compressed else data


def validate_stream(stream):
    generated = stream.get('source') == 'crazy_command_generator'
    if (set(stream) != {'schema_version', 'source', 'duration_ms', 'events'}
            or stream['schema_version'] != 'instinctrl-world-command-stream/1'
            or (not generated and stream['source'] != 'scripted_development')
            or type(stream['duration_ms']) is not int or stream['duration_ms'] % STEP_MS
            or not 8000 <= stream['duration_ms'] <= (3_600_000 if generated else 60000)
            or not isinstance(stream['events'], list) or not stream['events']):
        raise ValueError('COMMAND_REPLAY_STREAM')
    last = -1
    for event in stream['events']:
        fields = {'t_ms', 'velocity_w_m_s'}
        if isinstance(event, dict) and generated and 'mode' in event:
            if event['mode'] not in CRAZY_MODES:
                raise ValueError('COMMAND_REPLAY_GENERATOR_MODE')
            fields.add('mode')
        if not isinstance(event, dict) or set(event) != fields:
            raise ValueError('COMMAND_REPLAY_EVENT')
        t = event['t_ms']
        command = event['velocity_w_m_s']
        if (type(t) is not int or not last < t < stream['duration_ms']
                or not isinstance(command, list) or len(command) != 3
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in command)
                or command[2] != 0 or math.hypot(*command[:2]) > .30):
            raise ValueError('COMMAND_REPLAY_EVENT')
        last = t
    if stream['events'][0]['t_ms'] != 0:
        raise ValueError('COMMAND_REPLAY_INITIAL_EVENT')
    return stream


def load_stream(pin):
    return validate_stream(json.loads(checked(pin).read_bytes()))


def lane_seed_for(seed, lane, layout):
    digest = sha256(f'{CRAZY_SCHEMA}/{seed}/{lane}/{layout}'.encode()).digest()
    return int.from_bytes(digest[:4], 'big') % 2**31


def layout_stream(duration_ms, events, *, generated):
    return validate_stream(dict(schema_version='instinctrl-world-command-stream/1',
                    source='crazy_command_generator' if generated else 'scripted_development',
                    duration_ms=duration_ms, events=events))


def generator_prefix(generator):
    fields = {'schema_version', 'seed', 'prefix', 'prefix_duration_ms'}
    if generator.get('planned_position_limits') == PLANNED_POSITION_LIMITS:
        fields.add('planned_position_limits')
    if (set(generator) != fields
            or generator['schema_version'] != CRAZY_SCHEMA or type(generator['seed']) is not int
            or not 0 <= generator['seed'] < 2**31 or type(generator['prefix_duration_ms']) is not int):
        raise ValueError('COMMAND_REPLAY_GENERATOR_METADATA')
    template = json.loads(checked(generator['prefix']).read_bytes())
    if (set(template) != {'schema_version', 'duration_ms', 'layouts'}
            or template['duration_ms'] != generator['prefix_duration_ms']):
        raise ValueError('COMMAND_REPLAY_GENERATOR_PREFIX')
    return template


def check_generated(stream, prefix, prefix_duration, *, bounds=None):
    extension = stream['events'][len(prefix):]
    if (stream['events'][:len(prefix)] != prefix or not extension or extension[0]['t_ms'] != prefix_duration
            or {event.get('mode') for event in extension} != set(CRAZY_MODES)):
        raise ValueError('COMMAND_REPLAY_GENERATOR_COVERAGE')
    if bounds is not None:
        point = [0., 0.]
        end_times = [event['t_ms'] for event in stream['events'][1:]]+[stream['duration_ms']]
        for event, end in zip(stream['events'], end_times, strict=True):
            duration = (end-event['t_ms'])/1000.
            for axis, (lower, upper) in enumerate(bounds):
                point[axis] += duration*event['velocity_w_m_s'][axis]
                if not lower-1.e-6 <= point[axis] <= upper+1.e-6:
                    raise ValueError('COMMAND_REPLAY_GENERATOR_POSITION_LIMIT')


def route_streams(value):
    fields = {'schema_version', 'duration_ms', 'layouts'}
    generated = 'generator' in value
    if generated:
        fields.add('generator')
    if (set(value) != fields
            or value['schema_version'] != 'instinctrl-command-route-streams/1'
            or not isinstance(value['layouts'], dict) or set(value['layouts']) != set(COMMAND_LAYOUTS)):
        raise ValueError('COMMAND_REPLAY_ROUTES')
    streams = {layout: layout_stream(value['duration_ms'], events, generated=generated)
               for layout, events in value['layouts'].items()}
    if generated:
        template = generator_prefix(value['generator'])
        for layout, stream in streams.items():
            bounds = LAYOUT_COMMAND_BOUNDS[layout] if 'planned_position_limits' in value['generator'] else None
            check_generated(stream, template['layouts'][layout], template['duration_ms'], bounds=bounds)
    return streams


def load_route_streams(pin):
    return route_streams(json.loads(checked(pin).read_bytes()))


def load_lane_streams(pin, *, require_independent=False):
    value = json.loads(checked(pin).read_bytes())
    layouts = [command_layout(lane) for lane in range(LANES)]
    if value.get('schema_version') != 'instinctrl-command-lane-streams/1':
        if require_independent:
            raise ValueError('COMMAND_REPLAY_INDEPENDENT_LANES_REQUIRED')
        streams = route_streams(value)
        return layouts, [streams[name] for name in layouts]
    if set(value) != {'schema_version', 'duration_ms', 'generator', 'lanes'} or len(value['lanes']) != LANES:
        raise ValueError('COMMAND_REPLAY_LANE_STREAMS')
    generator = value['generator']
    template = generator_prefix(generator)
    streams = []
    for lane, (layout, row) in enumerate(zip(layouts, value['lanes'], strict=True)):
        if (set(row) != {'lane', 'layout', 'seed', 'events'} or row['lane'] != lane or row['layout'] != layout
                or row['seed'] != lane_seed_for(generator['seed'], lane, layout)):
            raise ValueError('COMMAND_REPLAY_LANE_SEED')
        stream = layout_stream(value['duration_ms'], row['events'], generated=True)
        bounds = LAYOUT_COMMAND_BOUNDS[layout] if 'planned_position_limits' in generator else None
        check_generated(stream, template['layouts'][layout], template['duration_ms'], bounds=bounds)
        streams.append(stream)
    if len({row['seed'] for row in value['lanes']}) != LANES:
        raise ValueError('COMMAND_REPLAY_NONINDEPENDENT_SEED')
    return layouts, streams


def conditions(params):
    if params['experiment_id'].startswith(NOVEL_PREFIX):
        key = 'street_sweep' if params['experiment_id'] in STREET_SWEEP_DEVELOPMENT_IDS else 'street_center_ms'
        return {'layout_routes': params[key]}
    if params['experiment_id'].startswith(LONG_PREFIX):
        protocol = json.loads(checked(params['protocol']).read_bytes())
        key = 'street_sweep' if 'street_sweep' in protocol else 'street_center_ms'
        return {'layout_routes': protocol[key]}
    if route_layouts(params['experiment_id']):
        return {'layout_routes': CONDITIONS['crossing_at_reversal']}
    return CONDITIONS


def event_schedule(params, label=None):
    """Keep single-stream pair rows stable and expand route inputs by lane."""
    if route_layouts(params['experiment_id']):
        stream_pin = params.get('candidate_stream', params['stream']) if label == 'candidate' else params['stream']
        layouts, by_lane = load_lane_streams(stream_pin, require_independent=params['experiment_id'].startswith((NOVEL_PREFIX, LONG_PREFIX)))
        limit = RETURN_DURATION_MS//STEP_MS+1 if params['experiment_id'].startswith((RETURN_PREFIX, EARLY_PREFIX)) else None
        if params['experiment_id'] in MATCHED_ENDURANCE_PREFIX_SECONDS:
            limit = MATCHED_ENDURANCE_PREFIX_SECONDS[params['experiment_id']]*1000//STEP_MS+1
        indices = list(zip(*(sampled_events(stream)[:limit] for stream in by_lane), strict=True))
        events = [[stream['events'][index] for stream, index in zip(by_lane, row, strict=True)] for row in indices]
        return layouts, [list(row) for row in indices], events
    stream = load_stream(params['stream'])
    indices = sampled_events(stream)
    return [LAYOUTS[lane//8] for lane in range(LANES)], indices, [stream['events'][i] for i in indices]


def sampled_events(stream):
    """Sample the latest event at each boundary, including the final observation."""
    events = stream['events']
    times = [event['t_ms'] for event in events]
    return [bisect_right(times, t) - 1 for t in range(0, stream['duration_ms'] + 1, STEP_MS)]


def clearance_radii(pin):
    envelope = json.loads(checked(pin).read_bytes())
    checked(envelope['asset'])
    if (envelope['schema_version'] != 'instinctrl-collider-envelope-study/1'
            or envelope['asset_meters_per_unit'] != 1.0 or not envelope['shapes']):
        raise ValueError('COMMAND_REPLAY_CLEARANCE_ENVELOPE')
    radii = {}
    for shape in envelope['shapes']:
        name = shape['body'].rsplit('/', 1)[-1]
        radius = shape['link_bound_radius_m']
        if not name or type(radius) not in (int, float) or not math.isfinite(radius) or not 0. < radius < .5:
            raise ValueError('COMMAND_REPLAY_CLEARANCE_RADIUS')
        radii[name] = max(radii.get(name, 0.), radius)
    return radii


def check_corner_pair(params):
    baseline = load_route_streams(params['stream'])
    retimed = load_route_streams(params['candidate_stream'])
    for name in COMMAND_LAYOUTS:
        original, revised = baseline[name], retimed[name]
        original_tail = [event for event in original['events'] if 'mode' in event]
        revised_tail = [event for event in revised['events'] if 'mode' in event]
        if (original['source'] != 'crazy_command_generator' or original['duration_ms'] != revised['duration_ms']
                or original_tail != revised_tail or ((original != revised) != (name == 'room_corner'))):
            raise ValueError('COMMAND_REPLAY_CORNER_INPUT_PAIR')


def valid_crossing_centers(centers):
    return (isinstance(centers, list) and len(centers) == 8
            and all(type(t) is int and 4000 <= t <= 20000 for t in centers) and len(set(centers)) == 8)


def novel_contract(params):
    sweep = params['experiment_id'] in STREET_SWEEP_DEVELOPMENT_IDS
    crossing_field = 'street_sweep' if sweep else 'street_center_ms'
    fields = {'experiment_id', 'training_seed', 'seed', 'models', 'execution', 'stream', 'clearance_envelope',
              crossing_field, 'pulse_acceleration_m_s2', 'wall_seconds', 'output_cap_bytes'}
    comparison = {NOVEL_TURN_ID: (STOP_RELEASE_EXECUTION, TURN_FIT_EXECUTION),
                  NOVEL_INTERCEPT_ID: (TURN_FIT_EXECUTION, INTERCEPT_EXECUTION),
                  NOVEL_ALIGNED_ID: (INTERCEPT_EXECUTION, ALIGNED_INTERCEPT_EXECUTION),
                  NOVEL_RESERVE_ID: (ALIGNED_INTERCEPT_EXECUTION, RESERVE_INTERCEPT_EXECUTION),
                  NOVEL_COMMITTED_ID: (RESERVE_INTERCEPT_EXECUTION, COMMITTED_INTERCEPT_EXECUTION),
                  NOVEL_PRIORITY_ID: (COMMITTED_INTERCEPT_EXECUTION, PRIORITY_INTERCEPT_EXECUTION),
                  NOVEL_HANDOFF_ID: (PRIORITY_INTERCEPT_EXECUTION, COMMAND_HANDOFF_EXECUTION),
                  NOVEL_OFFSET_STOP_ID: (COMMAND_HANDOFF_EXECUTION, OFFSET_STOP_EXECUTION),
                  NOVEL_NEIGHBOR_HISTORY_ID: (OFFSET_STOP_EXECUTION, NEIGHBOR_HISTORY_EXECUTION),
                  NOVEL_STANDOFF_ID: (NEIGHBOR_HISTORY_EXECUTION, CONFIRMED_STANDOFF_EXECUTION),
                  NOVEL_SETTLED_STOP_ID: (NEIGHBOR_HISTORY_EXECUTION, SETTLED_OFFSET_STOP_EXECUTION),
                  NOVEL_WALL_TANGENT_ID: (SETTLED_OFFSET_STOP_EXECUTION, WALL_TANGENT_EXECUTION),
                  NOVEL_CONFIRMED_TUBE_ID: (WALL_TANGENT_EXECUTION, CONFIRMED_TUBE_EXECUTION),
                  NOVEL_FINITE_EDGE_ID: (CONFIRMED_TUBE_EXECUTION, FINITE_EDGE_EXECUTION),
                  NOVEL_EDGE_CONTINUATION_ID: (FINITE_EDGE_EXECUTION, EDGE_CONTINUATION_EXECUTION),
                  NOVEL_TANGENT_MAGNITUDE_ID: (FINITE_EDGE_EXECUTION, TANGENT_MAGNITUDE_EXECUTION),
                  NOVEL_ROTATION_TUBE_ID: (FINITE_EDGE_EXECUTION, ROTATION_TUBE_EXECUTION)}.get(params['experiment_id'])
    paired = comparison is not None
    prefix_seconds = MATCHED_ENDURANCE_PREFIX_SECONDS.get(params['experiment_id'])
    labels = {'control', 'candidate'} if paired else {'candidate'}
    execution = (HOVER_PATH_EXECUTION, STOP_RELEASE_EXECUTION)
    if paired:
        execution = (comparison[0],)
    elif params['experiment_id'] in (NOVEL_CONFIRM_ID, NOVEL_REVALIDATE_ID):
        execution = (ALIGNED_INTERCEPT_EXECUTION, RESERVE_INTERCEPT_EXECUTION, *COMMITTED_INTERCEPT_EXECUTIONS)
    elif params['experiment_id'] == NOVEL_VERIFIED_RESUME_ID:
        execution = (VERIFIED_RESUME_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_VERIFIED_ZERO_ID, NOVEL_ZERO_CONFIRM_ID):
        execution = (VERIFIED_ZERO_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_INTERCEPT_CHOICE_ID, NOVEL_TRACKING_CONFIRM_ID):
        execution = (INTERCEPT_CHOICE_EXECUTION,)
    elif params['experiment_id'] == NOVEL_EARLY_DIAGONAL_ID:
        execution = (EARLY_DIAGONAL_EXECUTION,)
    elif params['experiment_id'] == NOVEL_DIAGONAL_CONFIRM_ID:
        execution = (EARLY_DIAGONAL_EXECUTION, OPENING_STEERING_EXECUTION)
    elif params['experiment_id'] == NOVEL_RESERVE_DIRECTION_ID:
        execution = (RESERVE_DIRECTION_EXECUTION,)
    elif params['experiment_id'] == NOVEL_CROSSING_FALLBACK_ID:
        execution = (CROSSING_FALLBACK_EXECUTION,)
    elif params['experiment_id'] == NOVEL_OPENING_STEERING_ID:
        execution = (OPENING_STEERING_EXECUTION,)
    elif params['experiment_id'] == NOVEL_ALONG_INTERCEPT_ID:
        execution = (ALONG_INTERCEPT_EXECUTION,)
    elif params['experiment_id'] == NOVEL_FINITE_INTERCEPT_ID:
        execution = (FINITE_INTERCEPT_EXECUTION,)
    elif params['experiment_id'] == NOVEL_CROSSING_STOP_ID:
        execution = (CROSSING_STOP_EXECUTION,)
    elif params['experiment_id'] == NOVEL_CROSSING_STOP_CONTINUATION_ID:
        execution = (CROSSING_STOP_CONTINUATION_EXECUTION,)
    elif params['experiment_id'] == NOVEL_CROSSING_STOP_SHALLOW_ID:
        execution = (CROSSING_STOP_SHALLOW_EXECUTION,)
    elif params['experiment_id'] == NOVEL_CROSSING_STOP_EARLY_ID:
        execution = (CROSSING_STOP_EARLY_EXECUTION,)
    elif params['experiment_id'] == NOVEL_CROSSING_COMPATIBLE_ID:
        execution = (CROSSING_COMPATIBLE_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_CROSSING_VERIFIED_HOVER_ID, NOVEL_CROSSING_VERIFIED_CONFIRM_ID):
        execution = (CROSSING_VERIFIED_HOVER_EXECUTION,)
    elif params['experiment_id'] == NOVEL_CROSSING_RECEDING_HOVER_ID:
        execution = (CROSSING_RECEDING_HOVER_EXECUTION,)
    elif params['experiment_id'] == NOVEL_CROSSING_EXTENDED_RETREAT_ID:
        execution = (CROSSING_EXTENDED_RETREAT_EXECUTION,)
    elif params['experiment_id'] == NOVEL_CROSSING_RECEDING_CONFIRM_ID:
        execution = (CROSSING_RECEDING_HOVER_EXECUTION, CROSSING_EXTENDED_RETREAT_EXECUTION)
    elif params['experiment_id'] in (NOVEL_CROSSING_SETTLED_REVERSAL_ID, NOVEL_CROSSING_SETTLED_CONFIRM_ID):
        execution = (CROSSING_SETTLED_REVERSAL_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_CROSSING_REFERENCE_ID, NOVEL_CROSSING_REFERENCE_CONFIRM_ID):
        execution = (CROSSING_REFERENCE_EXECUTION,)
    elif params['experiment_id'] == NOVEL_SHALLOW_STEERING_ID:
        execution = (SHALLOW_STEERING_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_OBSERVED_HEADING_ID, NOVEL_OBSERVED_HEADING_CONFIRM_ID):
        execution = (OBSERVED_HEADING_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_CROSSING_SAMPLED_TURN_ID, NOVEL_CROSSING_SAMPLED_TURN_CONFIRM_ID):
        execution = (CROSSING_SAMPLED_TURN_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_BILATERAL_STOPPING_ID, NOVEL_BILATERAL_STOPPING_CONFIRM_ID):
        execution = (BILATERAL_STOPPING_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_HOVER_OWN_SIDE_ID, NOVEL_HOVER_OWN_SIDE_CONFIRM_ID):
        execution = (HOVER_OWN_SIDE_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_FAR_CROSSING_REFERENCE_ID, NOVEL_FAR_CROSSING_REFERENCE_CONFIRM_ID):
        execution = (FAR_CROSSING_REFERENCE_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_VERIFIED_RETREAT_ONSET_ID, NOVEL_VERIFIED_RETREAT_ONSET_CONFIRM_ID):
        execution = (VERIFIED_RETREAT_ONSET_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_CLIPPED_RETREAT_ONSET_ID, NOVEL_CLIPPED_RETREAT_ONSET_CONFIRM_ID):
        execution = (CLIPPED_RETREAT_ONSET_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_TUBE_STOPPING_ID, NOVEL_TUBE_STOPPING_CONFIRM_ID):
        execution = (TUBE_STOPPING_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_HOVER_REFERENCE_RESERVE_ID, NOVEL_HOVER_REFERENCE_RESERVE_CONFIRM_ID):
        execution = (HOVER_REFERENCE_RESERVE_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_CROSSING_RESTART_ID, NOVEL_CROSSING_RESTART_CONFIRM_ID):
        execution = (CROSSING_RESTART_EXECUTION,)
    elif params['experiment_id'] == NOVEL_LOCAL_WALL_STOPPING_ID:
        execution = (LOCAL_WALL_STOPPING_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_OVERLAPPING_STOP_ID, NOVEL_OVERLAPPING_STOP_CONFIRM_ID):
        execution = (OVERLAPPING_STOP_EXECUTION,)
    elif params['experiment_id'] == NOVEL_BRAKE_RECOVERY_ID:
        execution = (BRAKE_RECOVERY_EXECUTION,)
    elif params['experiment_id'] == NOVEL_BRAKE_TRANSITION_ID:
        execution = (BRAKE_TRANSITION_EXECUTION,)
    elif params['experiment_id'] == NOVEL_FRONT_WALL_BRAKING_ID:
        execution = (FRONT_WALL_BRAKING_EXECUTION,)
    elif params['experiment_id'] == NOVEL_FACETED_WALL_STOPPING_ID:
        execution = (FACETED_WALL_STOPPING_EXECUTION,)
    elif params['experiment_id'] == NOVEL_WALL_EDGE_REPROOF_ID:
        execution = (WALL_EDGE_REPROOF_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_RETREAT_CONTINUATION_ID, NOVEL_RETREAT_CONTINUATION_CONFIRM_ID):
        execution = (RETREAT_CONTINUATION_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_OLDEST_PARTIAL_TURN_ID, NOVEL_OLDEST_PARTIAL_TURN_CONFIRM_ID):
        execution = (OLDEST_PARTIAL_TURN_EXECUTION,)
    elif params['experiment_id'] == NOVEL_ZERO_CROSSING_WITHDRAWAL_ID:
        execution = (ZERO_CROSSING_WITHDRAWAL_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_RANKED_REVERSAL_STOP_ID, NOVEL_RANKED_REVERSAL_STOP_CONFIRM_ID):
        execution = (RANKED_REVERSAL_STOP_EXECUTION,)
    elif params['experiment_id'] == NOVEL_ROOM_TANGENT_MAGNITUDE_ID:
        execution = (ROOM_TANGENT_MAGNITUDE_EXECUTION,)
    elif params['experiment_id'] == NOVEL_COMBINED_CONTROL_ID:
        execution = (COMBINED_CONTROL_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_RETREAT_ONSET_SIDE_ID, NOVEL_RETREAT_ONSET_SIDE_CONFIRM_ID):
        execution = (RETREAT_ONSET_SIDE_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_CROSSING_AND_WALL_MOTION_ID, NOVEL_CROSSING_AND_WALL_MOTION_CONFIRM_ID):
        execution = (CROSSING_AND_WALL_MOTION_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_RETREAT_STOP_ID, NOVEL_RETREAT_STOP_CONFIRM_ID):
        execution = (RETREAT_STOP_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_WALL_MOTION_AND_STOP_ID, NOVEL_WALL_MOTION_AND_STOP_CONFIRM_ID):
        execution = (WALL_MOTION_AND_STOP_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_SUPPORTED_WALL_APPROACH_ID, NOVEL_SUPPORTED_WALL_APPROACH_CONFIRM_ID):
        execution = (SUPPORTED_WALL_APPROACH_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_CURRENT_REQUEST_RECOVERY_ID, NOVEL_CURRENT_REQUEST_RECOVERY_CONFIRM_ID):
        execution = (CURRENT_REQUEST_RECOVERY_EXECUTION,)
    elif params['experiment_id'] == NOVEL_COMPOSED_CROSSING_ID:
        execution = (COMPOSED_CROSSING_EXECUTION,)
    elif params['experiment_id'] == NOVEL_OBSERVED_TURN_BRAKING_ID:
        execution = (OBSERVED_TURN_BRAKING_EXECUTION,)
    elif params['experiment_id'] == NOVEL_FRONT_WALL_HANDOFF_ID:
        execution = (FRONT_WALL_HANDOFF_EXECUTION,)
    elif params['experiment_id'] == NOVEL_COLLECTIVE_PROJECTION_ID:
        execution = (COLLECTIVE_PROJECTION_EXECUTION,)
    elif params['experiment_id'] == NOVEL_STATIC_TURN_ONSET_ID:
        execution = (STATIC_TURN_ONSET_EXECUTION,)
    elif params['experiment_id'] == NOVEL_CORRIDOR_PROJECTION_ID:
        execution = (CORRIDOR_PROJECTION_EXECUTION,)
    elif params['experiment_id'] in (NOVEL_STATIC_RETREAT_HANDOFF_ID, *STREET_SWEEP_DEVELOPMENT_IDS):
        execution = (RELEASE_STOP_CERTIFICATES_EXECUTION,)
    elif params['experiment_id'] == NOVEL_NEUTRAL_RETRY_ID:
        execution = (NEUTRAL_RETRY_EXECUTION,)
    elif params['experiment_id'] == NOVEL_ALONG_CONFIRM_ID:
        execution = ALONG_INTERCEPT_EXECUTIONS
    wall = 1800 if paired else 1200
    if params['experiment_id'] == NOVEL_STATIC_RETREAT_HANDOFF_ID:
        wall = 2100
    if paired:
        fields.add('candidate_execution')
    if (set(params) != fields or not matches(params['experiment_id']) or params['training_seed'] != 20260908
            or type(params['seed']) is not int or not 0 <= params['seed'] < 2**31 or params['seed'] in (167149835, 185193671)
            or params['execution'] not in execution or (paired and params['candidate_execution'] != comparison[1])
            or set(params['models']) != labels or any(pin['sha256'] != MODELS['b'] for pin in params['models'].values())
            or (paired and params['models']['candidate'] != params['models']['control'])
            or params['wall_seconds'] != wall or params['output_cap_bytes'] != 512*2**20
            or params['pulse_acceleration_m_s2'] != (1. if prefix_seconds else .5)):
        raise ValueError('COMMAND_REPLAY_NOVEL_PARAMETERS')
    crossing = params[crossing_field]
    ramped = params['experiment_id'] in RAMPED_STREET_SWEEP_IDS
    if not (crossing == authored_street_sweep(params['seed'], ramped=ramped) if sweep else valid_crossing_centers(crossing)):
        raise ValueError('COMMAND_REPLAY_NOVEL_CROSSINGS')
    checked(params['models']['candidate'])
    clearance_radii(params['clearance_envelope'])
    _, streams = load_lane_streams(params['stream'], require_independent=True)
    if any(stream['duration_ms'] != (1_800_000 if prefix_seconds else 120_000) for stream in streams):
        raise ValueError('COMMAND_REPLAY_NOVEL_DURATION')
    transitions = LANES*((prefix_seconds or 120)*1000//STEP_MS)*len(labels)
    return dict(schema_version='r6-local-runtime-contract/1', execution_mode='local-direct', operation='dynamic',
                mode=MODE, parameters=params, budgets=dict(wall_seconds=wall, live_process_starts=1,
                    simulator_initializations=1, initialization_ticks=10, training_transitions=0,
                    nontraining_transitions=transitions, optimizer_updates=0, checkpoint_writes=0,
                    explicit_resets=len(labels), automatic_resets=transitions))


def authored_street_sweep(seed, *, ramped=False):
    from instinctlab.observations.moving_sphere import SWEEP_ACCELERATION_RAMP_S, street_sweep_phase
    from .robustness_scenarios import reset_spec

    block = dict(id='continuous-replay-independent', task='pulse_hold')
    phases = []
    for lane in range(LANES):
        if command_layout(lane) == 'street_gap':
            spec = reset_spec(block, seed=seed, split='development', lane=lane, mount_position_m=(0., 0., .05))
            phases.append(street_sweep_phase(seed, lane, spec['root_position_local_m'][0]))
    if ramped:
        return dict(schema_version='instinctrl-independent-street-sweep/2', phase_m=phases,
                    ramp_s=SWEEP_ACCELERATION_RAMP_S)
    return dict(schema_version='instinctrl-independent-street-sweep/1', phase_m=phases)


def long_contract(params):
    fields = {'experiment_id', 'training_seed', 'seed', 'models', 'execution', 'stream', 'protocol',
              'clearance_envelope', 'wall_seconds', 'output_cap_bytes'}
    if (set(params) != fields or not matches(params['experiment_id']) or params['training_seed'] != 20260908
            or type(params['seed']) is not int or not 0 <= params['seed'] < 2**31 or params['seed'] in (167149835, 185193671)
            or params['wall_seconds'] != 7200 or params['output_cap_bytes'] != 2**31
            or params['execution'] not in (HOVER_PATH_EXECUTION, *STOP_RELEASE_EXECUTIONS)
            or set(params['models']) != {'candidate'} or params['models']['candidate']['sha256'] != MODELS['b']):
        raise ValueError('COMMAND_REPLAY_LONG_PARAMETERS')
    checked(params['models']['candidate'])
    clearance_radii(params['clearance_envelope'])
    protocol = json.loads(checked(params['protocol']).read_bytes())
    versions = ('instinctrl-command-long-protocol/1', 'instinctrl-command-long-protocol/2',
                'instinctrl-command-long-protocol/3')
    schema = protocol.get('schema_version')
    sweep = schema in versions[1:]
    ramped = schema == versions[2]
    crossing_field = 'street_sweep' if sweep else 'street_center_ms'
    scenario = protocol.get(crossing_field)
    if (set(protocol) != {'schema_version', 'experiment_id', 'scene_seed', 'stream', 'execution', 'checkpoint',
                          'progress_opportunities', 'acceptance_gates', crossing_field, 'pulse_acceleration_m_s2'}
            or schema not in versions
            or protocol['experiment_id'] != params['experiment_id'] or protocol['scene_seed'] != params['seed']
            or protocol['stream'] != params['stream'] or protocol['execution'] != params['execution']
            or protocol['checkpoint'] != params['models']['candidate']
            or not (scenario == authored_street_sweep(params['seed'], ramped=ramped) if sweep else valid_crossing_centers(scenario))
            or type(protocol['pulse_acceleration_m_s2']) not in (int, float)
            or protocol['pulse_acceleration_m_s2'] not in (.5, 1.)):
        raise ValueError('COMMAND_REPLAY_LONG_FREEZE')
    opportunities = json.loads(checked(protocol['progress_opportunities']).read_bytes())
    if (opportunities['schema_version'] != 'instinctrl-frozen-progress-opportunities/1'
            or opportunities['run_id'] != params['experiment_id']
            or set(opportunities) != {'schema_version', 'run_id', 'windows', 'stop_windows'}):
        raise ValueError('COMMAND_REPLAY_LONG_PROGRESS_PLAN')
    gates = json.loads(checked(protocol['acceptance_gates']).read_bytes())
    limits = {'minimum_clearance_m', 'maximum_tracking_rmse_m_s', 'minimum_progress_fraction',
              'maximum_stop_s', 'maximum_stop_drift_m', 'maximum_intervention_s', 'maximum_recovery_s',
              'maximum_slow_feasible_s'}
    if (set(gates) != {'schema_version', 'by_layout'} or gates['schema_version'] != 'instinctrl-command-acceptance-gates/1'
            or set(gates['by_layout']) != set(COMMAND_LAYOUTS)):
        raise ValueError('COMMAND_REPLAY_LONG_GATES')
    counts = {'minimum_hazard_events', 'minimum_recovery_opportunities'}
    for gate in gates['by_layout'].values():
        if (set(gate) != limits | counts or any(type(gate[name]) is not int or gate[name] <= 0 for name in counts)
                or not 0 < gate['minimum_progress_fraction'] <= 1
                or any(type(gate[name]) not in (int, float) or not math.isfinite(gate[name]) or gate[name] <= 0
                       for name in limits)):
            raise ValueError('COMMAND_REPLAY_LONG_LAYOUT_GATES')
    layouts, streams = load_lane_streams(params['stream'], require_independent=True)
    if any(stream['duration_ms'] != 1_800_000 for stream in streams):
        raise ValueError('COMMAND_REPLAY_LONG_DURATION')
    stream_meta = json.loads(checked(params['stream']).read_bytes())['generator']
    if stream_meta.get('planned_position_limits') != PLANNED_POSITION_LIMITS:
        raise ValueError('COMMAND_REPLAY_LONG_POSITION_LIMITS')
    verify_long_opportunities(opportunities, layouts, streams)
    transitions = LANES*18000
    return dict(schema_version='r6-local-runtime-contract/1', execution_mode='local-direct', operation='dynamic',
                mode=MODE, parameters=params, budgets=dict(wall_seconds=7200, live_process_starts=1,
                    simulator_initializations=1, initialization_ticks=10, training_transitions=0,
                    nontraining_transitions=transitions, optimizer_updates=0, checkpoint_writes=0,
                    explicit_resets=1, automatic_resets=transitions))


def verify_long_opportunities(plan, layouts, streams):
    """Bind each opportunity to the frozen input before a simulator starts."""
    scheduled = [[math.hypot(*stream['events'][i]['velocity_w_m_s'][:2])
                  for i in sampled_events(stream)[:-1]] for stream in streams]
    horizon = len(scheduled[0])
    progress, stops, identities = set(), set(), set()
    for section in ('windows', 'stop_windows'):
        for row in plan[section]:
            start, end = row['start_step'], row['end_step']
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= horizon
                    or not isinstance(row['id'], str) or not row['id'] or row['id'] in identities):
                raise ValueError('COMMAND_REPLAY_LONG_OPPORTUNITY_WINDOW')
            identities.add(row['id'])
            lane = row.get('lane')
            if 'lane' in row and (type(lane) is not int or not 0 <= lane < len(layouts)):
                raise ValueError('COMMAND_REPLAY_LONG_OPPORTUNITY_LANE')
            if section == 'windows':
                expected = {'id', 'layout', 'start_step', 'end_step'}
                if set(row) not in (expected, expected | {'lane'}):
                    raise ValueError('COMMAND_REPLAY_LONG_PROGRESS_PLAN')
                lanes = [i for i, layout in enumerate(layouts)
                         if layout == row['layout'] and lane in (None, i)]
                if not lanes or any(not any(speed > .03 for speed in scheduled[i][start:end]) for i in lanes):
                    raise ValueError('COMMAND_REPLAY_LONG_PROGRESS_WITHOUT_INPUT')
                progress.update(lanes)
            else:
                if (set(row) != {'id', 'lane', 'start_step', 'end_step'} or start == 0 or end-start < 20
                        or scheduled[lane][start-1] <= .03 or any(scheduled[lane][start:end])):
                    raise ValueError('COMMAND_REPLAY_LONG_STOP_WITHOUT_RELEASE')
                stops.add(lane)
    if progress != stops or stops != set(range(len(layouts))):
        raise ValueError('COMMAND_REPLAY_LONG_MISSING_OPPORTUNITIES')


def contract(params):
    run_id = params['experiment_id']
    if run_id.startswith(NOVEL_PREFIX):
        return novel_contract(params)
    if run_id.startswith(LONG_PREFIX):
        return long_contract(params)
    guard = run_id.startswith(GUARD_PREFIX)
    crazy = run_id.startswith(CRAZY_PREFIX)
    endurance = run_id.startswith(ENDURANCE_PREFIXES)
    returns = run_id.startswith((RETURN_PREFIX, EARLY_PREFIX))
    routes = route_layouts(run_id)
    selected_pair = next((pair for prefix, pair in EXECUTION_PAIRS.items() if run_id.startswith(prefix)), None)
    control_execution, candidate_execution = selected_pair or (EXECUTION, BRAKE_EXECUTION)
    paired_weights = selected_pair is not None
    expected_models = MODELS
    if paired_weights:
        expected_models = {'control': MODELS['b'], 'candidate': MODELS['b']}
    elif routes:
        expected_models = dict(MODELS, candidate=MODELS['b'])
    fields = {'experiment_id', 'training_seed', 'seed', 'models', 'execution', 'stream', 'wall_seconds', 'output_cap_bytes'}
    if guard or routes:
        fields.add('candidate_execution')
    captures_clearance = run_id == CLEARANCE_ID or run_id.startswith(PHYSICS_CLEARANCE_PREFIXES)
    if captures_clearance:
        fields.add('clearance_envelope')
    if run_id.startswith(CORNER_PREFIX):
        fields.add('candidate_stream')
    if (set(params) != fields or not matches(params['experiment_id']) or params['training_seed'] != 20260908
            or ((guard or routes) and params['candidate_execution'] != candidate_execution)
            or type(params['seed']) is not int
            or not 0 <= params['seed'] < 2**31 or params['execution'] != control_execution
            or not isinstance(params['models'], dict) or set(params['models']) != set(expected_models)
            or type(params['wall_seconds']) is not int or not 120 <= params['wall_seconds'] <= 1800
            or type(params['output_cap_bytes']) is not int or not 256*2**20 <= params['output_cap_bytes'] <= 2**30):
        raise ValueError('COMMAND_REPLAY_PARAMETERS')
    if captures_clearance:
        clearance_radii(params['clearance_envelope'])
    if run_id.startswith(CORNER_PREFIX):
        check_corner_pair(params)
    for label, checkpoint in params['models'].items():
        if checkpoint['sha256'] != expected_models[label]:
            raise ValueError('COMMAND_REPLAY_FROZEN_CHECKPOINT')
        checked(checkpoint)
    if paired_weights and params['models']['control'] != params['models']['candidate']:
        raise ValueError('COMMAND_REPLAY_GUARD_SAME_WEIGHTS')
    if routes and not paired_weights and params['models']['b'] != params['models']['candidate']:
        raise ValueError('COMMAND_REPLAY_ROUTES_SAME_WEIGHTS')
    stream = load_lane_streams(params['stream'])[1][16] if routes else load_stream(params['stream'])
    duration = stream['duration_ms']
    valid_duration = 101000 <= duration <= 125000 if endurance or returns else duration <= 90000
    if not valid_duration or ((crazy or endurance or returns) and stream['source'] != 'crazy_command_generator'):
        raise ValueError('COMMAND_REPLAY_DEVELOPMENT_DURATION')
    events = {event['t_ms']: event['velocity_w_m_s'] for event in stream['events']}
    reversal = CONDITIONS['crossing_at_reversal']
    previous = next(event['velocity_w_m_s'] for event in reversed(stream['events']) if event['t_ms'] < reversal)
    if events.get(reversal, [0.])[0] >= 0 or previous[0] <= 0 or stream['duration_ms'] <= reversal + 4000:
        raise ValueError('COMMAND_REPLAY_PAIRED_REVERSAL')
    trials = len(params['models']) * len(conditions(params))
    transitions = trials * LANES * ((RETURN_DURATION_MS if returns else duration)//STEP_MS)
    return dict(schema_version='r6-local-runtime-contract/1', execution_mode='local-direct',
                operation='dynamic', mode=MODE, parameters=params,
                budgets=dict(wall_seconds=params['wall_seconds'], live_process_starts=1,
                             simulator_initializations=1, initialization_ticks=10, training_transitions=0,
                             nontraining_transitions=transitions, optimizer_updates=0, checkpoint_writes=0,
                             explicit_resets=trials, automatic_resets=transitions))


def verify_supported_wall_pipelines(env, *, aligned=False):
    from instinctlab.control.local_wall_stopping import GravityAlignedWallEdgeReproofPipeline, SupportedWallEdgeReproofPipeline

    expected = GravityAlignedWallEdgeReproofPipeline if aligned else SupportedWallEdgeReproofPipeline
    if {type(pipeline) for pipeline in env._control_pipelines} != {expected}:
        raise ValueError('COMMAND_SUPPORTED_WALL_PIPELINE')


def verify_current_request_recovery(env):
    if not (env.escape_option.retry_command_handoff and env.escape_option.verify_outward_brake):
        raise ValueError('COMMAND_CURRENT_REQUEST_RECOVERY')


def run(backend, params):
    import torch
    from instinctlab.control.escape_option import METHOD_FIELD
    from instinctlab.control.lee_velocity_controller import rotate_world_to_body_wxyz
    from instinctlab.training.checkpoint_h1 import checkpoint_actor
    from instinctlab.training.networks import NetworkConfig
    from training_workloads.scale_003.persistence import atomic_json_replace, reserve_output_bytes
    from .lidar_calibration import record as record_mount

    default_schedule = event_schedule(params)
    layouts, event_ids, _ = default_schedule
    env, counters = backend.environment, backend.counters
    budgets = backend.runtime_contract['budgets']
    routes = route_layouts(params['experiment_id'])
    obstacle = env._h1_moving_sphere
    axis = env._command_b.new_zeros((LANES, 3))
    axis[:, 0] = 1.
    street = torch.tensor([layout == 'street_gap' for layout in layouts], device=env.device)
    steps = len(event_ids)-1
    if not math.isclose(env.step_dt, STEP_MS/1000., abs_tol=1e-8):
        raise ValueError('COMMAND_REPLAY_CLOCK')
    if env.max_episode_length <= steps:
        raise ValueError('COMMAND_REPLAY_EPISODE_HORIZON')
    block = dict(id='continuous-replay-pair', task='stop', speed_m_s=.25,
                 pulse_acceleration_m_s2=0., sensor_to_wall_distance_m=None, steps=steps)
    novel = params['experiment_id'].startswith(NOVEL_PREFIX)
    long_run = params['experiment_id'].startswith(LONG_PREFIX)
    compressed = compress_trace(params['experiment_id'])
    independently_timed = novel or long_run
    street_sweep = None
    if independently_timed:
        settings = json.loads(checked(params['protocol']).read_bytes()) if long_run else params
        street_sweep = settings.get('street_sweep')
        block.update(id='continuous-replay-independent', task='pulse_hold',
                     pulse_acceleration_m_s2=settings['pulse_acceleration_m_s2'])
    output = Path(backend.output)
    endurance = params['experiment_id'].startswith(ENDURANCE_PREFIXES)
    returns = params['experiment_id'].startswith((RETURN_PREFIX, EARLY_PREFIX))
    timed_crossings = endurance or returns or independently_timed or params['experiment_id'].startswith(CORNER_PREFIX)
    env.capture_ics_limiter = timed_crossings or params['experiment_id'].startswith(CRAZY_PREFIX)
    capture = 'clearance_envelope' in params
    if capture:
        if env.cfg.decimation+1 != CLEARANCE_SHAPE[0]:
            raise ValueError('COMMAND_REPLAY_CLEARANCE_CLOCK')
        env.enable_clearance_capture(clearance_radii(params['clearance_envelope']))

    def save_row(handle, row):
        data = encode_trace_row(row, compressed=compressed)
        reserve_output_bytes(Path(handle.name), len(data))
        handle.write(data)

    results = {}
    for label, checkpoint in params['models'].items():
        _, event_ids, events = (event_schedule(params, label) if label == 'candidate' and 'candidate_stream' in params
                                else default_schedule)
        if routes:
            commands = env._command_b.new_tensor([[event['velocity_w_m_s'] for event in row] for row in events])
        else:
            commands = env._command_b.new_tensor([event['velocity_w_m_s'] for event in events])
            commands = commands[:, None, :].expand(-1, LANES, -1)
        payload = torch.load(checked(checkpoint), map_location='cpu', weights_only=True)
        if (NetworkConfig(**payload['network_config']) != NetworkConfig()
                or payload['method_hashes'].get(METHOD_FIELD) != EXECUTION):
            raise ValueError('COMMAND_REPLAY_EXECUTION_BINDING')
        actor = checkpoint_actor(payload).to(env.device)
        actor.load_state_dict(payload['actor'], strict=True)
        actor.eval()
        del payload
        execution = arm_execution(params, label)
        env.set_escape_option(True, version=execution)
        env.set_h1_ics_clearance_margin(.15)
        model = dict(checkpoint=checkpoint, execution=execution, lidar_mount=record_mount(env, output, label), trials={})
        if timed_crossings:
            configuration = {(p.config.ics.motion_tube_radius_m, p.config.ics.motion_tube_confirm_closure,
                              p.config.ics.motion_tube_neighbor_history, p.config.ics.confirmed_standoff_m,
                              p.config.ics.confirmed_tube_radius_m, p.config.ics.confirmed_tube_align_rotation)
                             for p in env._control_pipelines}
            confirm = execution in (TUBE_CONFIRM_EXECUTION, RECOVERY_RETREAT_EXECUTION, EARLY_RETREAT_EXECUTION,
                                    SIDE_HOVER_RETREAT_EXECUTION, *CLEAR_PASS_EXECUTIONS)
            neighbors = execution in NEIGHBOR_HISTORY_EXECUTIONS
            standoff = .45 if execution == CONFIRMED_STANDOFF_EXECUTION else 0.
            inner_tube = .41 if execution in CONFIRMED_TUBE_EXECUTIONS else 0.
            settled_stop = execution in SETTLED_OFFSET_STOP_EXECUTIONS
            tangent = execution in WALL_TANGENT_EXECUTIONS
            detour = execution in FINITE_EDGE_EXECUTIONS
            edge_continuation = execution == EDGE_CONTINUATION_EXECUTION
            tangent_magnitude = execution in TANGENT_MAGNITUDE_EXECUTIONS
            aligns_rotation = execution in ROTATION_TUBE_EXECUTIONS
            resumes = execution == VERIFIED_RESUME_EXECUTION
            zero_stop = execution in VERIFIED_ZERO_EXECUTIONS
            intercept_choice = execution in INTERCEPT_CHOICE_EXECUTIONS
            early_diagonal = execution in EARLY_DIAGONAL_EXECUTIONS
            reserve_direction = execution in RESERVE_DIRECTION_EXECUTIONS
            crossing_fallback = execution in CROSSING_FALLBACK_EXECUTIONS
            opening_steering = execution in OPENING_STEERING_EXECUTIONS
            along_intercept = execution in ALONG_INTERCEPT_EXECUTIONS
            finite_intercept = execution in FINITE_INTERCEPT_EXECUTIONS
            crossing_stop = execution in CROSSING_STOP_EXECUTIONS
            crossing_stop_continuation = execution in CROSSING_CONTINUATION_EXECUTIONS
            crossing_stop_shallow = execution in CROSSING_SHALLOW_EXECUTIONS
            crossing_stop_early = execution in CROSSING_EARLY_EXECUTIONS
            crossing_stop_compatible = execution in CROSSING_COMPATIBLE_EXECUTIONS
            crossing_stop_verified_hover = execution in CROSSING_VERIFIED_HOVER_EXECUTIONS
            crossing_stop_receding_hover = execution in CROSSING_RECEDING_HOVER_EXECUTIONS
            crossing_retreat_extended = execution in CROSSING_EXTENDED_RETREAT_EXECUTIONS
            settled_mount = CROSSING_SETTLED_MOUNT if execution in CROSSING_SETTLED_EXECUTIONS else None
            crossing_reference = execution in CROSSING_REFERENCE_EXECUTIONS
            shallow_steering = execution in SHALLOW_STEERING_EXECUTIONS
            observed_heading = execution in OBSERVED_HEADING_EXECUTIONS
            sampled_turn = execution in SAMPLED_TURN_EXECUTIONS
            bilateral_stopping = execution == BILATERAL_STOPPING_EXECUTION
            hover_own_side = execution in HOVER_OWN_SIDE_EXECUTIONS
            far_crossing_reference = execution in FAR_CROSSING_REFERENCE_EXECUTIONS
            verified_retreat_onset = execution in VERIFIED_RETREAT_ONSET_EXECUTIONS
            clipped_retreat_only = execution in CLIPPED_RETREAT_ONSET_EXECUTIONS
            tube_stopping = execution == TUBE_STOPPING_EXECUTION
            hover_reference_reserve = execution in HOVER_REFERENCE_RESERVE_EXECUTIONS
            crossing_restart = execution in CROSSING_RESTART_EXECUTIONS
            local_wall_stopping = execution in LOCAL_WALL_STOPPING_EXECUTIONS
            front_wall_braking = execution in FRONT_WALL_BRAKING_EXECUTIONS
            faceted_wall_stopping = execution in FACETED_WALL_STOPPING_EXECUTIONS
            wall_edge_reproof = execution in WALL_EDGE_REPROOF_EXECUTIONS
            overlapping_stop = execution in OVERLAPPING_STOP_EXECUTIONS
            retreat_continuation = execution in RETREAT_CONTINUATION_EXECUTIONS
            oldest_partial_turn = execution in OLDEST_PARTIAL_TURN_EXECUTIONS
            zero_crossing_withdrawal = execution in ZERO_CROSSING_WITHDRAWAL_EXECUTIONS
            ranked_reversal_stop = execution in RANKED_REVERSAL_STOP_EXECUTIONS
            retreat_onset_side = execution in RETREAT_ONSET_SIDE_EXECUTIONS
            retreat_stop = execution in RETREAT_STOP_EXECUTIONS
            brake_recovery = execution in BRAKE_RECOVERY_EXECUTIONS
            brake_transition = execution in BRAKE_TRANSITION_EXECUTIONS
            if (configuration != {(.60, confirm, neighbors, standoff, inner_tube, aligns_rotation)}
                    or env.escape_option.settled_offset_stop != settled_stop or env.escape_option.wall_tangent != tangent
                    or env.escape_option.finite_edge != detour or env.escape_option.edge_continuation != edge_continuation
                    or env.escape_option.tangent_magnitude != tangent_magnitude
                    or env.escape_option.verified_resume != resumes or env.escape_option.verified_zero != zero_stop
                    or env.escape_option.intercept_choice != intercept_choice
                    or env.escape_option.receding_stop != intercept_choice
                    or env.escape_option.early_diagonal != early_diagonal
                    or env.escape_option.reserve_direction != reserve_direction
                    or env.escape_option.zero_recheck != crossing_fallback
                    or env.escape_option.reverse_diagonal != crossing_fallback
                    or env.escape_option.opening_steering != opening_steering
                    or env.escape_option.along_intercept != along_intercept
                    or env.escape_option.finite_intercept != finite_intercept
                    or env.escape_option.crossing_stop != crossing_stop
                    or env.escape_option.crossing_stop_continuation != crossing_stop_continuation
                    or env.escape_option.crossing_stop_shallow != crossing_stop_shallow
                    or env.escape_option.crossing_stop_early != crossing_stop_early
                    or env.escape_option.crossing_stop_compatible != crossing_stop_compatible
                    or env.escape_option.crossing_stop_verified_hover != crossing_stop_verified_hover
                    or env.escape_option.crossing_stop_receding_hover != crossing_stop_receding_hover
                    or env.escape_option.crossing_retreat_own_side != crossing_stop_receding_hover
                    or env.escape_option.crossing_retreat_extended != crossing_retreat_extended
                    or env.escape_option.settled_crossing_mount != settled_mount
                    or env.escape_option.static_scan_stop != crossing_reference
                    or env.escape_option.crossing_reference != crossing_reference
                    or env.escape_option.shallow_steering != shallow_steering
                    or env.escape_option.observed_heading_steering != observed_heading
                    or env.escape_option.tracked_resume != observed_heading
                    or env.escape_option.crossing_turn_between_samples != sampled_turn
                    or env.bilateral_stopping != bilateral_stopping
                    or env.escape_option.hover_escape_own_side != hover_own_side
                    or env.escape_option.far_crossing_reference != far_crossing_reference
                    or env.escape_option.verified_retreat_onset != verified_retreat_onset
                    or env.escape_option.retain_only_clipped_retreat != clipped_retreat_only
                    or env.tube_stopping != tube_stopping
                    or env.escape_option.verify_hover_reference_reserve != hover_reference_reserve
                    or env.escape_option.verify_crossing_restart != crossing_restart
                    or env.local_wall_stopping != local_wall_stopping
                    or env.escape_option.verify_overlapping_stop != overlapping_stop
                    or env.escape_option.verify_brake_recovery != brake_recovery
                    or env.escape_option.verify_brake_transition != brake_transition
                    or env.front_wall_braking != front_wall_braking
                    or env.faceted_wall_stopping != faceted_wall_stopping
                    or env.wall_edge_reproof != wall_edge_reproof
                    or env.escape_option.verify_retreat_continuation != retreat_continuation
                    or env.escape_option.verify_oldest_partial_turn != oldest_partial_turn
                    or env.escape_option.verify_zero_crossing_withdrawal != zero_crossing_withdrawal
                    or env.escape_option.verify_ranked_reversal_stop != ranked_reversal_stop
                    or env.escape_option.verify_retreat_onset_side != retreat_onset_side
                    or env.escape_option.verify_retreat_stop != retreat_stop):
                raise ValueError('COMMAND_REPLAY_ICS_PROFILE')
            model['ics_profile'] = dict(radius_m=.60, confirms_closure=confirm)
            if neighbors:
                model['ics_profile']['uses_neighbor_history'] = True
            if standoff:
                model['ics_profile']['confirmed_standoff_m'] = standoff
            if inner_tube:
                model['ics_profile']['confirmed_tube_radius_m'] = inner_tube
            if aligns_rotation:
                model['ics_profile']['confirmed_tube_align_rotation'] = True
            if settled_stop:
                model['settled_offset_stop'] = True
            if tangent:
                model['wall_tangent'] = True
            if detour:
                model['finite_edge'] = True
            if edge_continuation:
                model['edge_continuation'] = True
            if tangent_magnitude:
                model['tangent_magnitude'] = True
            if resumes:
                model['verified_resume'] = True
            if zero_stop:
                model['verified_zero'] = True
            if intercept_choice:
                model.update(intercept_choice=True, receding_stop=True)
            if early_diagonal:
                model['early_diagonal'] = True
            if reserve_direction:
                model['reserve_direction'] = True
            if crossing_fallback:
                model.update(zero_recheck=True, reverse_diagonal=True)
            if opening_steering:
                model['opening_steering'] = True
            if along_intercept:
                model['along_intercept'] = True
            if finite_intercept:
                model['finite_intercept'] = True
            if crossing_stop:
                model['crossing_stop'] = True
            if crossing_stop_continuation:
                model['crossing_stop_continuation'] = True
            if crossing_stop_shallow:
                model['crossing_stop_shallow'] = True
            if crossing_stop_early:
                model['crossing_stop_early'] = True
            if crossing_stop_compatible:
                model['crossing_stop_compatible'] = True
            if crossing_stop_verified_hover:
                model['crossing_stop_verified_hover'] = True
            if crossing_stop_receding_hover:
                model.update(crossing_stop_receding_hover=True, crossing_retreat_own_side=True)
            if crossing_retreat_extended:
                model['crossing_retreat_extended'] = True
            if settled_mount is not None:
                model['settled_crossing_mount_m'] = list(settled_mount)
            if crossing_reference:
                model.update(static_scan_stop=True, crossing_reference=True)
            if shallow_steering:
                model['shallow_steering'] = True
            if observed_heading:
                model.update(observed_heading_steering=True, tracked_resume=True)
            if sampled_turn:
                model['crossing_turn_between_samples'] = True
            if bilateral_stopping:
                model['bilateral_stopping'] = True
            if hover_own_side:
                model['hover_escape_own_side'] = True
            if far_crossing_reference:
                model['far_crossing_reference'] = True
            if verified_retreat_onset:
                model['verified_retreat_onset'] = True
            if clipped_retreat_only:
                model['retain_only_clipped_retreat'] = True
            if tube_stopping:
                model['tube_stopping'] = True
            if hover_reference_reserve:
                model['verify_hover_reference_reserve'] = True
            if crossing_restart:
                model['verify_crossing_restart'] = True
            if local_wall_stopping:
                model['local_wall_stopping'] = True
            if overlapping_stop:
                model['verify_overlapping_stop'] = True
            if brake_recovery:
                model['verify_brake_recovery'] = True
            if brake_transition:
                model['verify_brake_transition'] = True
            if front_wall_braking:
                model['front_wall_braking'] = True
            if faceted_wall_stopping:
                model['faceted_wall_stopping'] = True
            if wall_edge_reproof:
                model['wall_edge_reproof'] = True
            if retreat_continuation:
                model['verify_retreat_continuation'] = True
            if oldest_partial_turn:
                model['verify_oldest_partial_turn'] = True
            if zero_crossing_withdrawal:
                model['verify_zero_crossing_withdrawal'] = True
            if ranked_reversal_stop:
                model['verify_ranked_reversal_stop'] = True
            if retreat_onset_side:
                model['verify_retreat_onset_side'] = True
            if retreat_stop:
                model['verify_retreat_stop'] = True
        for name, crossing in conditions(params).items():
            pulse = []

            def capture_force(row):
                if independently_timed and row['active']:
                    pulse.append(dict(block_tick=row['block_tick'], sim_tick=row['sim_tick'],
                                      force_n=row['additional_world_n'].sum(dim=1).cpu().tolist(),
                                      mass_kg=row['mass_kg'].sum(dim=1).cpu().tolist()))

            env.configure_r6_physics(block=block, seed=params['seed'], split='development',
                                     reset_sink=lambda row: None, force_sink=capture_force,
                                     expected_mass_kg=1.181517923)
            sensor_origins = env.scene.env_origins + axis.new_tensor(
                [spec['sensor_position_local_m'] for spec in env._r6_physics['specs']])
            modes = torch.full((LANES,), 7, device=env.device, dtype=torch.long)
            delay = None
            phase = None
            if crossing is not None:
                ids = obstacle.indices
                modes[street] = 2 + ids[street].remainder(2)
                if street_sweep is not None:
                    delay = axis.new_zeros(LANES)
                    phase = axis.new_zeros((LANES, 2))
                    phase[street] = axis.new_tensor(street_sweep['phase_m'])
                elif independently_timed:
                    delay = axis.new_zeros(LANES)
                    delay[street] = axis.new_tensor(crossing)/1000.-CROSSING_START_M/CROSSING_SPEED_M_S
                else:
                    delay = crossing/1000. - CROSSING_START_M/CROSSING_SPEED_M_S
            obstacle.configure(sensor_origins, axis, modes, departure_delay_s=delay,
                               half_period_s=CROSSING_TRAVEL_S if timed_crossings else None,
                               exact_clock=independently_timed, sweep_phase_m=phase,
                               sweep_ramp_s=street_sweep.get('ramp_s', 0.) if street_sweep is not None else 0.)
            obstacle.end_travel_s[street & obstacle.active] = CROSSING_TRAVEL_S
            env.set_h1_initial_command(torch.zeros_like(env._command_b), 0)
            env.stage_r6_world_command(commands[0])
            backend.resource_check('before_command_replay_reset', counters.explicit_resets+1)
            observation, _ = backend._preserving_reset(seed=params['seed'])
            trace = trace_path(output, label, name, run_id=params['experiment_id'])
            dense_path = output/f'{label}-{name}-clearance.f32'
            # Stream completed transitions to disk. A failed run still leaves its prefix intact.
            with ExitStack() as files:
                handle = files.enter_context(trace.open('xb'))
                dense = files.enter_context(dense_path.open('xb')) if capture else None
                initial = dict(kind='initial', mover_start_w=obstacle.start.cpu().tolist(),
                               path_reference_w=sensor_origins.cpu().tolist(),
                               scene_origins_w=env.scene.env_origins.cpu().tolist(),
                               root_state_w=env._robot.data.root_state_w.cpu().tolist(),
                               departure_delay_s=delay.cpu().tolist() if independently_timed else delay,
                               layouts=layouts)
                if street_sweep is not None:
                    initial['street_sweep'] = street_sweep
                save_row(handle, initial)
                for step in range(steps):
                    backend.resource_check('before_command_replay_step', step)
                    counters.require_step_capacity(kind='nontraining', lanes=LANES,
                                                   maximum=budgets['nontraining_transitions'])
                    counters.require_reset_capacity(automatic=True, lanes=LANES,
                                                    maximum=budgets['automatic_resets'])
                    policy = observation['policy']
                    attitude = env._robot.data.root_state_w[:, 3:7].clone()
                    body = rotate_world_to_body_wxyz(attitude, commands[step])
                    actor_state = policy['state_vec'].reshape(LANES, 4, 13)
                    if (not torch.allclose(actor_state[:, -1, 6:9], body, atol=1e-6, rtol=0)
                            or not torch.equal(actor_state[:, -1, 6:9], env._command_b)):
                        raise RuntimeError(f'COMMAND_REPLAY_ACTOR_COMMAND {label}/{name}/{step}')
                    before = obstacle.latest
                    grid = policy['lidar_grid'].reshape(LANES, 4, 3, -1)
                    witness = grid[obstacle.indices, :, :, before['witness_beam']]
                    valid_count = grid[:, -1, 1].sum(dim=1)
                    env.stage_r6_world_command(commands[step+1])
                    env.prepare_h1_step(global_policy_step=step, command_b=env._command_b,
                                        next_command_b=rotate_world_to_body_wxyz(attitude, commands[step+1]))
                    with torch.no_grad():
                        action = actor(policy).mean_action
                    counters.attempt_step()
                    observation, _, terminated, truncated, extras = env.step(action)
                    counters.complete_step(LANES, kind='nontraining')
                    terminal = (terminated | truncated).reshape(-1)
                    counters.reserve_reset(automatic=True, lanes=int(terminal.sum()),
                                           maximum=budgets['automatic_resets'])
                    pre = extras['pre_reset']
                    task, control, option = pre['task'], pre['control'], pre['escape_option']
                    post = obstacle.pre_reset
                    state = pre['root_state_w']
                    tensors = dict(
                        attitude_wxyz=attitude, actor_state=actor_state, action=action,
                        actor_witness=witness, valid_beams=valid_count, mover_beam=before['witness_beam'],
                        mover_winning_beams=before['winning_beams'], mover_witness_range_m=before['witness_range_m'],
                        mover_position_w=before['sphere_position_w'], mover_position_after_w=post['sphere_position_w'],
                        mover_velocity_w=before['sphere_velocity_w'], v_corr_b=control['v_corr_b'],
                        v_gov_b=control['v_gov_b'], v_final_b=control['v_final_b'], v_final_w=control['v_final_w'],
                        ics_beta=control['ics_metrics']['beta'].reshape(-1),
                        ics_emergency=control['ics_metrics']['emergency'].reshape(-1),
                        ics_active_beams=control['ics_metrics']['active_beam_count'].reshape(-1),
                        selected_action=option['applied_action'], escape_active=option['active'],
                        escape_recovering=option['recovery'], escape_redirected=option['redirected'],
                        root_state_after_w=state, ray_minimum_m=task['exact_clearance_m'].reshape(-1),
                        collision=task['termination']['masks']['collision'].reshape(-1),
                        termination_reason=task['termination']['reason_code'].reshape(-1), reset=terminal,
                        episode_length=pre['episode_length'])
                    if 'collective_projection_peak_m_s2' in pre:
                        tensors['collective_projection_peak_m_s2'] = pre['collective_projection_peak_m_s2']
                    if 'corridor_projection_width_m' in pre:
                        tensors['corridor_projection_width_m'] = pre['corridor_projection_width_m']
                    if 'braking' in option:
                        tensors['escape_braking'] = option['braking']
                    if 'retreating' in option:
                        tensors['recovery_retreating'] = option['retreating']
                    if 'holding' in option:
                        tensors['escape_clear_pass_hold'] = option['holding']
                    if 'rearm_steps' in option:
                        tensors['recovery_rearm_steps'] = option['rearm_steps']
                    if 'ranked' in option:
                        tensors['escape_ranked_direction'] = option['ranked']
                    if 'released' in option:
                        tensors['escape_stop_release'] = option['released']
                    if 'intercepted' in option:
                        tensors['escape_intercepted'] = option['intercepted']
                    if 'intercept_continued' in option:
                        tensors['escape_intercept_continued'] = option['intercept_continued']
                    if 'handoff' in option:
                        tensors['escape_command_handoff'] = option['handoff']
                    if 'steady_handoff_witness' in option:
                        tensors['escape_steady_handoff_witness'] = option['steady_handoff_witness']
                        tensors['escape_steady_handoff_used'] = option['steady_handoff_used']
                    if 'neutral_retry_floor' in option:
                        tensors['escape_neutral_retry_floor'] = option['neutral_retry_floor']
                    if 'observed_turn_brake_floor' in option:
                        tensors['escape_observed_turn_brake_floor'] = option['observed_turn_brake_floor']
                    if 'front_wall_handoff_margin' in option:
                        tensors['escape_front_wall_handoff_margin'] = option['front_wall_handoff_margin']
                    if 'offset_stop' in option:
                        tensors['escape_offset_stop'] = option['offset_stop']
                    if 'wall_tangent' in option:
                        tensors['escape_wall_tangent'] = option['wall_tangent']
                    if 'finite_edge' in option:
                        tensors['escape_finite_edge'] = option['finite_edge']
                    if 'shallow_steering' in option:
                        tensors['escape_shallow_steering'] = option['shallow_steering']
                    if 'observed_heading' in option:
                        tensors['escape_observed_heading'] = option['observed_heading']
                    if 'tracked_resume' in option:
                        tensors['escape_tracked_resume'] = option['tracked_resume']
                    if 'settled_crossing' in option:
                        tensors['escape_settled_crossing'] = option['settled_crossing']
                    if 'bilateral_stopping' in option:
                        tensors['bilateral_stopping'] = option['bilateral_stopping']
                    if 'oldest_partial_turn_verified' in option:
                        tensors['escape_oldest_partial_turn_verified'] = option['oldest_partial_turn_verified']
                    if 'zero_crossing_mode' in option:
                        tensors['escape_zero_crossing_mode'] = option['zero_crossing_mode']
                        tensors['escape_zero_crossing_floor'] = option['zero_crossing_floor']
                    if 'ranked_reversal_verified' in option:
                        tensors['escape_ranked_reversal_verified'] = option['ranked_reversal_verified']
                    if 'retreat_onset_side_floor' in option:
                        tensors['escape_retreat_onset_side_floor'] = option['retreat_onset_side_floor']
                    if 'retreat_stop_floor' in option:
                        tensors['escape_retreat_stop_floor'] = option['retreat_stop_floor']
                    if 'composed_crossing_floor' in option:
                        tensors['escape_composed_crossing_floor'] = option['composed_crossing_floor']
                    if 'static_retreat_witness' in option:
                        tensors['escape_static_retreat_witness'] = option['static_retreat_witness']
                    if 'hover_own_side' in option:
                        tensors['escape_hover_own_side'] = option['hover_own_side']
                    if 'far_crossing_refined' in option:
                        tensors['escape_far_crossing_refined'] = option['far_crossing_refined']
                    if 'retreat_request_kept' in option:
                        tensors['escape_retreat_request_kept'] = option['retreat_request_kept']
                    if 'static_turn_floor' in option:
                        tensors['escape_static_turn_floor'] = option['static_turn_floor']
                    if 'crossing_restart_verified' in option:
                        tensors['escape_crossing_restart_verified'] = option['crossing_restart_verified']
                        tensors['escape_crossing_restart_previous_b'] = option['crossing_restart_previous_b']
                    if 'overlapping_stop_verified' in option:
                        tensors['escape_overlapping_stop_verified'] = option['overlapping_stop_verified']
                    if 'brake_recovery_verified' in option:
                        tensors['escape_brake_recovery_verified'] = option['brake_recovery_verified']
                        tensors['escape_brake_recovery_bounds'] = option['brake_recovery_bounds']
                    if 'retreat_continuation_verified' in option:
                        tensors['escape_retreat_continuation_verified'] = option['retreat_continuation_verified']
                        tensors['escape_retreat_continuation_floor'] = option['retreat_continuation_floor']
                    if 'tube_stopping_hold_s' in option:
                        tensors.update({field:option[field] for field in TUBE_STOPPING_TRACE_FIELDS})
                    if 'local_wall_stopping_hold_s' in option:
                        tensors.update({field:option[field] for field in LOCAL_WALL_STOPPING_TRACE_FIELDS})
                    if 'faceted_wall_stopping_used' in option:
                        tensors['faceted_wall_stopping_used'] = option['faceted_wall_stopping_used']
                    if 'front_wall_brake_verified' in option:
                        tensors.update({field:option[field] for field in FRONT_WALL_BRAKING_TRACE_FIELDS})
                    if 'wall_edge_reproof' in option:
                        tensors['wall_edge_reproof'] = option['wall_edge_reproof']
                    if 'edge_continued' in option:
                        tensors['escape_edge_continued'] = option['edge_continued']
                    if 'verified_resume' in option:
                        tensors['escape_verified_resume'] = option['verified_resume']
                    if 'opening_steered' in option:
                        tensors['escape_opening_steered'] = option['opening_steered']
                    if 'ics_limiter' in pre:
                        tensors.update(pre['ics_limiter'])
                    save_row(handle, dict(kind='transition', step=step, t_ms=step*STEP_MS,
                                          event_index=event_ids[step], raw_event=events[step],
                                          physics_tick=int(before['tick']), after_tick=int(post['tick']),
                                          contact_samples=post['contact_samples'],
                                          **{key: value.detach().cpu().tolist() for key, value in tensors.items()}))
                    if dense is not None:
                        samples = pre['clearance_samples'].cpu().numpy().astype('<f4', copy=False)
                        data = samples.tobytes()
                        reserve_output_bytes(dense_path, len(data))
                        dense.write(data)
                        dense.flush()
                    handle.flush()
            completion = env.finish_r6_physics()
            obstacle.finish()
            model['trials'][name] = dict(trace=file_pin(trace), physics=completion)
            if capture:
                model['trials'][name]['clearance'] = file_pin(dense_path)
            if independently_timed:
                filename = f'{label}-{name}-pulse.json'
                atomic_json_replace(output, filename, pulse)
                model['trials'][name]['pulse'] = file_pin(output/filename)
        if execution in SUPPORTED_WALL_EXECUTIONS:
            verify_supported_wall_pipelines(env, aligned=execution in GRAVITY_ALIGNED_PROTECTION_EXECUTIONS)
            model['supported_wall_approach'] = True
        if execution in CURRENT_REQUEST_RECOVERY_EXECUTIONS:
            verify_current_request_recovery(env)
            model.update(retry_command_handoff=True, verify_outward_brake=True)
        if execution in COMPOSED_CROSSING_EXECUTIONS:
            if not env.escape_option.verify_composed_crossing:
                raise ValueError('COMMAND_COMPOSED_CROSSING_PROOF')
            model['verify_composed_crossing'] = True
        if execution in STATIC_RETREAT_HANDOFF_EXECUTIONS:
            if not (env.escape_option.verify_recent_handoff and env.escape_option.verify_static_retreat_handoff):
                raise ValueError('COMMAND_STATIC_RETREAT_HANDOFF_PROOF')
            model.update(verify_recent_handoff=True, verify_static_retreat_handoff=True)
        if execution in GRAVITY_ALIGNED_PROTECTION_EXECUTIONS:
            if not env.escape_option.align_planar_protection:
                raise ValueError('COMMAND_PLANAR_PROTECTION_PROOF')
            model['align_planar_protection'] = True
        if execution in RELEASE_STOP_CERTIFICATES_EXECUTIONS:
            if not env.escape_option.verify_release_stop:
                raise ValueError('COMMAND_RELEASE_STOP_PROOF')
            model['verify_release_stop'] = True
        if execution in STEADY_HANDOFF_EXECUTIONS:
            if not env.escape_option.verify_steady_handoff:
                raise ValueError('COMMAND_STEADY_HANDOFF_PROOF')
            model['verify_steady_handoff'] = True
        if execution in TANGENT_RESPONSE_HEADROOM_EXECUTIONS:
            if not env.escape_option.tangent_response_headroom:
                raise ValueError('COMMAND_TANGENT_HEADROOM_PROOF')
            model['tangent_response_headroom'] = .605
        if execution in NEUTRAL_RETRY_EXECUTIONS:
            if not env.escape_option.verify_neutral_retry:
                raise ValueError('COMMAND_NEUTRAL_RETRY_PROOF')
            model['verify_neutral_retry'] = True
        if execution in OBSERVED_TURN_BRAKING_EXECUTIONS:
            if not env.escape_option.observed_turn_braking:
                raise ValueError('COMMAND_OBSERVED_TURN_BRAKING_PROOF')
            model['observed_turn_braking'] = True
        if execution in FRONT_WALL_HANDOFF_EXECUTIONS:
            if env.escape_option_version != execution:
                raise ValueError('COMMAND_FRONT_WALL_HANDOFF_PROOF')
            model['front_wall_handoff'] = True
        if execution in COLLECTIVE_PROJECTION_EXECUTIONS:
            from instinctlab.control.collective_projection import CollectiveProjectionAdapter

            if not isinstance(env._wrench_adapter, CollectiveProjectionAdapter):
                raise ValueError('COMMAND_COLLECTIVE_PROJECTION_PROOF')
            model['collective_projection'] = True
        if execution in STATIC_TURN_ONSET_EXECUTIONS:
            if not env.escape_option.verify_static_turn:
                raise ValueError('COMMAND_STATIC_TURN_ONSET_PROOF')
            model['static_turn_onset'] = True
        if execution == CORRIDOR_PROJECTION_EXECUTION:
            if env._corridor_projection is None:
                raise ValueError('COMMAND_CORRIDOR_PROJECTION_PROOF')
            model['corridor_projection'] = True
        results[label] = model
    atomic_json_replace(output, 'command_replay.json', dict(models=results, counters=counters.snapshot()))
