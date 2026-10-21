"""No paid CLI calls: sandbox source-read preflight and pre-intent rejection."""
import hashlib,sys
from pathlib import Path
import pytest
from tools.r6_recovery import file_pin,RecoveryError
from tools import r6_review_access as access,r6_initial_baseline as initial


def test_reads_source_through_sandbox_subcommand(tmp_path):
    p=tmp_path/'source';p.write_text('fixture')
    exe=tmp_path/'cli';exe.write_text('#!'+sys.executable+'\nimport sys,hashlib\nfrom pathlib import Path\nassert sys.argv[1:6]==["sandbox","-P",":read-only","--","/usr/bin/sha256sum"]\np=Path(sys.argv[-1]);print(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p))\n');exe.chmod(0o700)
    assert access.check(file_pin(exe),file_pin(p))['model_calls']==0


@pytest.mark.parametrize('output',['','wrong','0'*64+'  fixture'])
def test_bad_read_never_passes(tmp_path,output):
    p=tmp_path/'source';p.write_text('fixture');exe=tmp_path/'cli';exe.write_text('#!'+sys.executable+'\nprint('+repr(output)+')\n');exe.chmod(0o700)
    with pytest.raises(RecoveryError,match='LOCAL_READ_UNAVAILABLE'):access.check(file_pin(exe),file_pin(p))


def test_failed_local_read_precedes_any_model_reservation(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock
    from tools.r6_recovery import Journal
    from tools import r6_decision as decision
    b=json.loads((Path(__file__).parent/'fixtures/r6_budget.json').read_text());b['goal_id']='R6-PREFLIGHT-CPU-CHECK'
    j=Journal(tmp_path/'batch')
    with j.writer():j.initialize(b)
    exe=tmp_path/'cli';exe.write_text('#!/bin/sh\necho fixture-local-sandbox-unavailable >&2\nexit 1\n');exe.chmod(0o700)
    pin=file_pin(exe);monkeypatch.setattr(decision,'CLI',exe);monkeypatch.setattr(decision,'CLI_SHA',pin['sha256'])
    source=tmp_path/'plan';source.write_text('{}')
    monkeypatch.setattr(initial,'first_plan',lambda *a:({}, {}, 'J-fixture'))
    intent=Mock(side_effect=AssertionError('must not reserve a model'));monkeypatch.setattr(j,'intent',intent)
    before=j.read()
    with j.writer(),pytest.raises(RecoveryError,match='LOCAL_READ_UNAVAILABLE_NO_MODEL_STARTED'):
        initial.register_review(j,file_pin(source),SimpleNamespace(root=tmp_path),executable=pin,free_disk_bytes=500*1024**3)
    assert j.read()==before;intent.assert_not_called()
