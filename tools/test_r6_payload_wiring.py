"""Exercise the real entry up to quota installation without any backend launch."""
from unittest.mock import Mock
import pytest
from training_workloads.unified_h1 import result_volume
from training_workloads.unified_h1.live_adapter import run_main_sequence


@pytest.mark.parametrize('mode,native',[
    ('research-r6-evaluation',True),('research-r6-evaluation',False),
    ('research-formal-training',True)])
def test_entry_installs_exact_payload_only_for_bound_r6(tmp_path,monkeypatch,mode,native):
    attempt=tmp_path/'attempt';results=attempt/'results';results.mkdir(parents=True)
    cap=6*1024**3;payload=cap-result_volume.MAXIMUM-result_volume.CONTROL_ALLOWANCE-96*1024**2
    params={'output_cap_bytes':cap}
    if native:params.update(native_volume={'fixture':'native'},result_volume={'fixture':'result'})
    native_check=Mock();volume_check=Mock(return_value={'payload_cap_bytes':payload})
    monkeypatch.setattr('training_workloads.unified_h1.native_volume.validate',native_check)
    monkeypatch.setattr(result_volume,'validate',volume_check)
    installed=Mock(side_effect=RuntimeError('CPU boundary before backend'))
    monkeypatch.setattr('training_workloads.scale_003.persistence.install_output_write_quota',installed)
    backend=Mock(side_effect=AssertionError('backend must not be created'))
    with pytest.raises(RuntimeError,match='CPU boundary before backend'):
        run_main_sequence(authorization={'mode':mode,'parameters':params},attempt=attempt,backend_factory=backend)
    expected=payload if mode=='research-r6-evaluation' and native else cap-3*1024**3
    installed.assert_called_once_with(results/mode,expected);backend.assert_not_called()
    if mode=='research-r6-evaluation' and native:
        volume_check.assert_called_once_with(params['result_volume'],output_cap_bytes=cap,
            expected_target=results,minimum_free_bytes=256*1024**2)
        native_check.assert_called_once_with(params['native_volume'])
    else:volume_check.assert_not_called();native_check.assert_not_called()


def test_bad_native_binding_cannot_enable_lower_deduction(monkeypatch):
    native=Mock(side_effect=ValueError('NATIVE_BINDING_REJECTED'))
    monkeypatch.setattr('training_workloads.unified_h1.native_volume.validate',native)
    check=Mock();monkeypatch.setattr(result_volume,'validate',check)
    with pytest.raises(ValueError,match='NATIVE_BINDING_REJECTED'):
        result_volume.workload_write_cap(dict(output_cap_bytes=6*1024**3,native_volume={},result_volume={}))
    check.assert_not_called()
