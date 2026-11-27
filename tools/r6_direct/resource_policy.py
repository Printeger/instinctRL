"""Explicit new-input resource authorization; historical accounting stays intact."""
import json
import time

from tools.r6_evidence import checked
from training_workloads.unified_h1.command_replay import (
    GUARD_PREFIX as COMMAND_GUARD_PREFIX, PREFIX as COMMAND_REPLAY_PREFIX, ROUTE_PREFIX as COMMAND_ROUTE_PREFIX,
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
