"""Local locking must work without a legacy controller file or database."""
import copy
import fcntl
import json
import pytest
from tools.r6_direct import lifecycle
from tools.r6_evidence import file_pin


def test_local_lock_without_legacy_state_and_legacy_writer_interlock(tmp_path,monkeypatch):
    monkeypatch.setattr(lifecycle,'stop_paths',lambda:[])
    new=tmp_path/'new';new.mkdir();old=tmp_path/'absent'
    with lifecycle.writer(new,old):
        with pytest.raises(BlockingIOError):
            with lifecycle.writer(new,old):pass
    (old/'batch').mkdir(parents=True)
    with (old/'batch/writer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            with lifecycle.writer(new,old):pass
    with lifecycle.writer(new,old):pass
    monkeypatch.setattr(lifecycle,'stop_paths',lambda:['STOP'])
    with pytest.raises(RuntimeError,match='STOP'):
        with lifecycle.writer(new,old):pass


def test_imported_budget_survives_missing_legacy_files_but_not_tampering(tmp_path):
    imported=dict(caps={'x':100},historical_charged={'x':30},protected_confirmation_delivery_repair={'x':20},
        protected_model_sessions=8,sources=[{'path':'/unavailable/archived/state.json','sha256':'0'*64}])
    path=tmp_path/'import.json';path.write_text(json.dumps(imported))
    ledger=dict(imported,legacy_import=file_pin(path),new_used={'x':0},new_reservations=[])
    assert lifecycle.imported_budget(ledger)==imported
    wrong=copy.deepcopy(ledger);wrong['historical_charged']['x']=0
    with pytest.raises(ValueError,match='BUDGET_IMPORT_DRIFT'):lifecycle.imported_budget(wrong)
    path.write_text('{}')
    with pytest.raises(RuntimeError):lifecycle.imported_budget(ledger)
