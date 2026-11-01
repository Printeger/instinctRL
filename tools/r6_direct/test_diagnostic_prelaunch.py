import json
import pytest
from tools.r6_direct.cuda_diagnostic import prelaunch, validated_bounds


def test_diagnostic_bounds_preserve_legacy_and_allow_measured_validation_time():
    for child,total in [(30,240),(180,390)]:
        bounds=dict(child_seconds=child,gpu_and_wall_reservation_seconds=total,
                    queue_seconds=180,syscall_log_bytes=64*1024**2)
        assert validated_bounds({'bounds':bounds})==bounds


@pytest.mark.parametrize('key,value',[
    ('child_seconds',True),('child_seconds',float('inf')),('child_seconds',0),
    ('child_seconds',-1),('child_seconds',31),
    ('gpu_and_wall_reservation_seconds',239),('queue_seconds',181),
    ('syscall_log_bytes',128*1024**2)])
def test_invalid_diagnostic_bounds_rejected(key,value):
    bounds=dict(child_seconds=30,gpu_and_wall_reservation_seconds=240,
                queue_seconds=180,syscall_log_bytes=64*1024**2)
    bounds[key]=value
    with pytest.raises(ValueError):validated_bounds({'bounds':bounds})


def test_failed_admission_gets_terminal_receipt_without_child(tmp_path,monkeypatch):
    from training_workloads.unified_h1 import config
    monkeypatch.setattr(config,'resource_binding_pins',lambda: {})
    def reject(*args,**kwargs):raise MemoryError('resource memory limit')
    with pytest.raises(MemoryError):
        prelaunch(tmp_path,tmp_path/'gate',{'run_id':'CPU'}, {'sha256':'a'*64},reject)
    receipt=json.loads((tmp_path/'receipt.json').read_text())
    assert receipt['status']=='CLOSED' and receipt['phase']=='PRELAUNCH'
    assert receipt['remaining']==receipt['owned_processes']==[]
    assert not receipt['cuda_child_started'] and receipt['exit_code']==1
    assert receipt['training_transitions']==receipt['optimizer_updates']==0


def test_successful_admission_does_not_close_running_job(tmp_path,monkeypatch):
    from training_workloads.unified_h1 import config
    monkeypatch.setattr(config,'resource_binding_pins',lambda: {})
    assert prelaunch(tmp_path,tmp_path/'gate',{'run_id':'CPU'},{'sha256':'a'*64},lambda *a,**k:'pass')=='pass'
    assert not (tmp_path/'receipt.json').exists()
