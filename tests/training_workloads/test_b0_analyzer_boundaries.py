"""Unit mechanics boundaries; existing exact-coverage validator tests stay separate."""
import copy
from unittest.mock import patch

import pytest
from training_workloads.unified_h1 import b0_causal_analyzer as analyzer
from training_workloads.unified_h1.b0_enhanced import EvidenceError
from test_b0_enhanced_stage1 import row


def test_reset_jump_is_not_momentum_force():
    before=row(); after=copy.deepcopy(before)
    after['reset']['mask']=True
    after['bodies'][0]['world_linear_velocity_m_s']=[100.,0.,0.]
    # Serialized reset momentum is deliberately discontinuous.
    assert analyzer.derive_mechanics(before,after)['momentum_wrench'] is None


def test_same_episode_is_differenced_but_another_lane_or_case_is_not():
    before=row(); after=copy.deepcopy(before)
    assert analyzer.derive_mechanics(before,after)['momentum_wrench']['force_n']==(0.,0.,0.)
    for field in ('episode_id','lane'):
        other=copy.deepcopy(after);other[field]+=1
        assert analyzer.derive_mechanics(before,other)['momentum_wrench'] is None
    other=copy.deepcopy(after);other['case']='T'
    assert analyzer.derive_mechanics(before,other)['momentum_wrench'] is None


def test_boundary_counter_maps_to_following_post_reset_episode():
    boundary=row();boundary['reset']['mask']=True
    following=row();following['episode_id']=1
    assert analyzer.segment_key(boundary)==analyzer.segment_key(following)
    assert analyzer.derive_mechanics(boundary,following)['momentum_wrench'] is not None


def test_root_acceleration_disagreement_is_not_com_closure_failure():
    before=row(); after=copy.deepcopy(before)
    after['response']['acceleration_world_m_s2']=[10.,0.,0.]
    result=analyzer.derive_mechanics(before,after)
    assert result['com_acceleration_world_m_s2']==(0.,0.,0.)
    assert result['root_vs_com_force_difference_n']==20.


def test_changed_articulated_mass_rejected():
    before=row(); after=copy.deepcopy(before);after['bodies'][0]['mass_kg']+=1
    with pytest.raises(EvidenceError,match='mass changed'):
        analyzer.derive_mechanics(before,after)


def test_segment_endpoints_and_opposite_lanes_not_pooled_into_causal_verdict():
    rows=[]
    for lane,direction in ((0,1),(1,-1)):
        for step in range(3):
            sample=row();sample['lane']=lane
            sample['response']['velocity_world_m_s']=[direction,0.,0.]
            sample['response']['position_world_m']=[direction*step,0.,0.]
            sample['reset']['mask']=step==0
            sample['episode_id']=0 if step==0 else 1
            rows.append(sample)
    # This fixture isolates analyzer logic; it does NOT claim full live coverage.
    with patch.object(analyzer,'TraceValidator') as validator:
        validator.return_value.finalize.return_value={'status':'UNIT_FIXTURE'}
        result=analyzer.analyze_raw(rows)
    segments=result['response_segments']
    assert len(segments)==2
    assert segments[0]['root_position_end_m']==(2.,0.,0.)
    assert segments[1]['root_position_end_m']==(-2.,0.,0.)
    assert segments[0]['root_velocity_mean_m_s']==(1.,0.,0.)
    assert segments[1]['root_velocity_mean_m_s']==(-1.,0.,0.)
    assert result['hypothesis_scores']['reset_state']==1.
    assert result['outcome']=='DIAGNOSIS_INCONCLUSIVE' and result['winner'] is None
    assert result['mechanically_closed'] is None
