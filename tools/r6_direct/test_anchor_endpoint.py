"""A reset or missing trace must never turn into a smaller anchor endpoint."""
import pytest

from tools.r6_direct.anchor_endpoint import LANES, LAST_STEP, STOP_STEP, endpoint


def records(*, reset_at=None, missing_at=None):
    for step in range(STOP_STEP - 1, LAST_STEP + 1):
        if step == missing_at:
            continue
        active = int(step >= STOP_STEP)
        sample = dict(command_w_m_s=[1 - active, 0, 0], episode_id=0,
                      missing=False, collision=False, height_violation=False)
        yield dict(kind='step_complete', payload=dict(
            policy_step=step, lanes=[sample] * LANES, reset_mask=[False] * LANES,
            control=dict(anchor_metrics=dict(
                active=[[active]] * LANES,
                reset_code=[[int(step == reset_at)]] + [[0]] * (LANES - 1),
                usable_fraction=[[0.5 * active]] * LANES,
                mean_absolute_error_m=[[0.025 * active]] * LANES,
            )),
        ))


def test_endpoint_uses_full_stop_windows_and_retains_ineligible_lane():
    assert [row['anchor_mae_mm'] for row in endpoint(records())] == [25.0] * LANES
    rows = endpoint(records(reset_at=STOP_STEP + 1))
    assert rows[0]['anchor_mae_mm'] is None
    assert rows[0]['invalid_steps'] == [STOP_STEP + 1]
    assert rows[1]['anchor_mae_mm'] == 25.0


def test_endpoint_rejects_missing_sample():
    with pytest.raises(ValueError, match='missing'):
        endpoint(records(missing_at=STOP_STEP + 2))
