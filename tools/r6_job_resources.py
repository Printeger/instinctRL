"""Control-plane charges; full frozen workload validation remains mandatory."""
from tools.r6_recovery import RecoveryError

MODES = ('research-r6-evaluation', 'research-r6-training', 'research-r6-candidate-evaluation')
DISK = 'output_actual_plus_unreleased_reservations_bytes'


def resources(spec, auth, *, prepaid=False):
    mode = spec.get('h1', {}).get('mode')
    if spec.get('backend') != 'real' or mode not in MODES:
        raise RecoveryError('R6_JOB_RESOURCES_MODE')
    result = dict(live_starts=1, simulator_initializations=1,
        live_wall_seconds=spec['resources']['wall_seconds']+180,
        allocated_gpu_seconds=spec['resources']['wall_seconds']+180)
    result[DISK] = 0 if prepaid else auth['parameters']['output_cap_bytes']
    if mode != MODES[1]:
        if mode == MODES[2]:
            budget = auth.get('budgets', {})
            transitions = budget.get('nontraining_transitions')
            if (auth.get('mode') != mode or type(transitions) is not int or transitions <= 0
                    or spec['resources']['transitions'] != transitions
                    or any(type(budget.get(k)) is not int or budget[k] != 0
                           for k in ('training_transitions', 'optimizer_updates', 'checkpoint_writes'))):
                raise RecoveryError('R6_JOB_RESOURCES_CANDIDATE_BUDGET')
        result['nontraining_transitions'] = spec['resources']['transitions']
    else:
        budget = auth['budgets']
        transitions = budget.get('training_transitions')
        checkpoints = budget.get('checkpoint_writes')
        if (auth.get('mode') != mode or type(transitions) is not int or transitions <= 0
                or type(checkpoints) is not int or checkpoints <= 0
                or type(budget.get('nontraining_transitions')) is not int
                or budget['nontraining_transitions'] != 0
                or spec['resources']['transitions'] != transitions):
            raise RecoveryError('R6_JOB_RESOURCES_TRAINING_BUDGET')
        result.update(training_transitions=transitions, training_checkpoint_writes=checkpoints)
    return result
