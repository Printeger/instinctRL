from tools.r6_direct.study_report import compare


def model(pair, error, *, tracking=0.03, safety=True):
    def endpoint(offset):
        return dict(block={'example': True},seed=12,lane_scores=[
            dict(anchor_mae_mm=error+offset,mean_usable_fraction=0.75) for _ in range(32)
        ])
    return dict(pair=pair,safety_pass=safety,tracking_rmse_mps={'S-delay-drop':tracking},
                endpoints={'S-delay-drop':endpoint(0),'S-clean-low':endpoint(10)})


def test_mean_threshold_reproducibility_and_tracking_are_separate():
    freeze=dict(threshold_mm=5,clean_increase_mm=5,tracking_rmse_increase_mps=.01,
                coverage_drop=.02,bootstrap_seed=19,bootstrap_draws=64)
    reference=[model(i,20) for i in (1,2,3)]
    trial=[model(i,14) for i in (1,2,3)]
    result=compare(freeze,trial,reference)
    assert result['mean_delta_mm']==-6 and result['conditional_95_ci_mm']==[-6,-6]
    assert result['meets_5mm'] and result['guardrails_pass']
    trial[0]=model(1,14,tracking=.05)
    assert not compare(freeze,trial,reference)['guardrails_pass']
    trial[0]=model(1,21)
    assert not compare(freeze,trial,reference)['meets_5mm']
