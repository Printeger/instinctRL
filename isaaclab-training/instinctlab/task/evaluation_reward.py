"""Explicit unavailable reward for audited, non-learning R6 startup only."""
from .rewards_h1 import MeasurementCoverageUnavailable


def validate_evaluation_contract(contract):
    if contract is None:
        return False
    if (contract.get('execution_mode') != 'local-direct'
            or contract.get('operation') != 'baseline'
            or contract.get('mode') != 'research-r6-evaluation'
            or contract.get('budgets', {}).get('training_transitions') != 0):
        raise ValueError('unavailable reward requires explicit nontraining baseline contract')
    return True


def compute_evaluation_reward(compute, *, context):
    """None is unavailable, never a reward or risk estimate for learning.

    The normal computation always runs first. The exception remains mandatory
    for received-but-insufficient data and for every training/original-suite run.
    """
    try:
        return compute()
    except MeasurementCoverageUnavailable:
        if not context or not all(context.get(key) is True for key in (
                'nontraining_baseline', 'first_delayed_observation',
                'all_lanes_reset', 'unavailable_all_lanes',
                'delivered_all_invalid', 'producer_coverage_sufficient',
                'not_dropped', 'audit_written')):
            raise
        return None
