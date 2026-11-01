import json
import pytest
from tools.r6_direct.cuda_diagnostic import prelaunch


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
