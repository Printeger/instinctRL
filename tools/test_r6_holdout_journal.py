import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools import r6_holdout as h
from tools.r6_recovery import Journal, file_pin, RecoveryError
from tools.r6_budget import totals


@pytest.fixture
def setup(tmp_path, monkeypatch):
    def write(name, value):
        p = tmp_path / name
        p.write_text(json.dumps(value))
        return file_pin(p)
    baseline = write('baseline', {'fixture': True})
    monkeypatch.setattr(h, 'BASELINE_SHA', baseline['sha256'])
    freeze = write('freeze.json', dict(schema_version='r6-final-freeze/1',
        branch='BASELINE_SUFFICIENT', protocol=write('protocol.json', {'frozen': True}),
        recipe=write('recipe.json', {}), selection_rule=write('selection.json', {}),
        models={'baseline': baseline}))
    review = write('review.json', dict(decision='PASS_R6_FINAL_FREEZE', must_fix=[],
        reviewer='independent-fixture', freeze_sha256=freeze['sha256']))
    j = Journal(tmp_path / 'batch')
    with j.writer():
        j.initialize(json.loads((Path(__file__).parent/'fixtures/r6_budget.json').read_text()))
    return j, dict(freeze=freeze, review=review), write


def prepare(j, request):
    with j.writer():
        return h.prepare(j, request, free_disk_bytes=500*1024**3)


def test_single_draw_recovery_and_repeated_release(setup, monkeypatch):
    j, request, write = setup
    draw = Mock(return_value='12'*32);monkeypatch.setattr(h.secrets, 'token_hex', draw)
    key, public = prepare(j, request)
    assert 'nonce' not in public
    restarted = Journal(j.directory)
    assert prepare(restarted, request) == (key, public)
    assert draw.call_count == 1
    assert totals(j.read()['budget'])[h.DISK] == 65536
    pub = write('published.json', public)
    with restarted.writer():
        first = h.release(restarted, key, pub)
        assert h.release(restarted, key, pub) == first
    assert 'nonce' not in j.read()['events'][key]


def test_freeze_drift_and_unreviewed_freeze_never_draw(setup, monkeypatch):
    j, request, write = setup
    draw = Mock();monkeypatch.setattr(h.secrets, 'token_hex', draw)
    Path(request['freeze']['path']).write_text('{}')
    with pytest.raises(RecoveryError): prepare(j, request)
    draw.assert_not_called()
    assert not j.read()['events']


def test_stop_blocks_entropy_and_release(setup, monkeypatch):
    j, request, write = setup
    (j.directory/'STOP').touch()
    with pytest.raises(RecoveryError): prepare(j, request)
    (j.directory/'STOP').unlink()  # Isolated fixture only.
    key, record = prepare(j, request); pub = write('published.json', record)
    (j.directory/'STOP').touch()
    with j.writer(), pytest.raises(RecoveryError): h.release(j, key, pub)


def test_crash_after_claim_does_not_redraw(setup, monkeypatch):
    j, request, write = setup
    draw = Mock(side_effect=RuntimeError('fixture entropy failure'))
    monkeypatch.setattr(h.secrets, 'token_hex', draw)
    with pytest.raises(RuntimeError): prepare(j, request)
    assert len(j.read()['events']) == 1
    # File is empty: preserve ambiguity, do not treat as permission to regenerate.
    with pytest.raises((ValueError, RecoveryError)): prepare(j, request)
    assert draw.call_count == 1


def test_publication_and_private_tampering_rejected(setup):
    j, request, write = setup
    key, record = prepare(j, request)
    bad = write('bad.json', dict(record, nonce_commitment='ab'*32))
    with j.writer(), pytest.raises(RecoveryError): h.release(j, key, bad)
    private = Path(j.read()['events'][key]['private_pin']['path'])
    private.write_text('{}')
    with pytest.raises(RecoveryError): prepare(j, request)


def test_other_freeze_cannot_create_second_holdout(setup):
    j, request, write = setup
    prepare(j, request)
    value = json.loads(Path(request['freeze']['path']).read_text())
    value['recipe'] = write('recipe2.json', {'changed': True})
    freeze = write('freeze2.json', value)
    review = write('review2.json', dict(decision='PASS_R6_FINAL_FREEZE', must_fix=[],
        reviewer='independent-fixture', freeze_sha256=freeze['sha256']))
    with pytest.raises(RecoveryError, match='ALREADY_BOUND'):
        prepare(j, dict(freeze=freeze, review=review))
    assert len(j.read()['events']) == 1
