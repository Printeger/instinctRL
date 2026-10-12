from types import SimpleNamespace
import pytest
from rlflow.isolation import cli_version, IsolationError, PINNED_VERSION


def test_historical_contract_retained_and_explicit_new_candidate():
    assert PINNED_VERSION == '0.153.4'
    for version in ('0.153.4','0.161.0'):
        assert cli_version(SimpleNamespace(returncode=0,stdout=f'codex-cli {version}\n')) == version


@pytest.mark.parametrize('output,code', [('codex-cli 0.162.1',0), ('codex-cli 0.161.0-extra',0), ('error 0.161.0',0), ('codex-cli 0.161.0 extra',0), ('codex-cli 0.161.0',1)])
def test_unknown_or_malformed_cli_fails_closed(output,code):
    with pytest.raises(IsolationError):
        cli_version(SimpleNamespace(returncode=code,stdout=output))


def test_snapshot_dispatcher_requires_exact_reviewed_bytes(tmp_path):
    from rlflow import experiments
    path=tmp_path/'dispatcher.py'
    path.write_bytes(experiments.DISPATCHER.read_bytes())
    value={'pins':{'dispatcher':experiments.file_pin(path)}}
    experiments.check_pins(value)
    path.write_text('different dispatcher')
    value['pins']['dispatcher']=experiments.file_pin(path)
    with pytest.raises(Exception,match='reviewed H1 dispatcher'):
        experiments.check_pins(value)
