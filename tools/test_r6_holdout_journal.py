import json
import subprocess
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
    freeze = write('freeze.json', dict(schema_version='r6-final-freeze/2',
        branch='BASELINE_SUFFICIENT', protocol=write('protocol.json', {'frozen': True}),
        recipe=write('recipe.json', {}), selection_rule=write('selection.json', {}),
        models={'baseline': baseline},evidence={'fixture':True}))
    review = write('review.json', dict(decision='PASS_R6_FINAL_FREEZE', must_fix=[],
        reviewer='independent-fixture', freeze_sha256=freeze['sha256']))
    # Review-worker lifecycle is tested separately; these cases isolate holdout persistence.
    monkeypatch.setattr(h, "verify_model_review", Mock(return_value="fixture-review"))
    # Final model provenance is tested separately. This fixture isolates entropy lifecycle.
    monkeypatch.setattr(h.final_models,'validate',Mock(side_effect=lambda j,f:f))
    monkeypatch.setattr(h, "ROOT", tmp_path)
    j = Journal(tmp_path / 'batch')
    with j.writer():
        j.initialize(json.loads((Path(__file__).parent/'fixtures/r6_budget.json').read_text()))
    return j, dict(freeze=freeze, review=review), write


def publish(j, pin):
    root = j.directory.parent
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()
    git("init", "-q")
    git("add", str(Path(pin["path"]).relative_to(root)))
    git("-c", "user.name=CPU Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Publish fixture commitment")
    return git("rev-parse", "HEAD")


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
    commit = publish(j, pub)
    with restarted.writer():
        first = h.release(restarted, key, pub, publication_commit=commit)
        assert h.release(restarted, key, pub, publication_commit=commit) == first
    assert 'nonce' not in j.read()['events'][key]


def test_freeze_drift_and_unreviewed_freeze_never_draw(setup, monkeypatch):
    j, request, write = setup
    draw = Mock();monkeypatch.setattr(h.secrets, 'token_hex', draw)
    Path(request['freeze']['path']).write_text('{}')
    with pytest.raises(RecoveryError): prepare(j, request)
    draw.assert_not_called()
    assert not j.read()['events']


def test_incomplete_model_provenance_rejected_before_reservation_or_entropy(setup,monkeypatch):
    j,request,_=setup;draw=Mock();monkeypatch.setattr(h.secrets,'token_hex',draw)
    monkeypatch.setattr(h.final_models,'validate',Mock(side_effect=RecoveryError('FINAL_MODELS_INCOMPLETE')))
    before=j.read()
    with pytest.raises(RecoveryError,match='FINAL_MODELS_INCOMPLETE'):prepare(j,request)
    assert j.read()==before;draw.assert_not_called()


def test_old_pin_only_freeze_cannot_authorize_entropy(setup,monkeypatch):
    j,request,write=setup;freeze=json.loads(h.checked(request['freeze']).read_text())
    freeze['schema_version']='r6-final-freeze/1';freeze.pop('evidence')
    request['freeze']=write('legacy-freeze.json',freeze)
    draw=Mock();monkeypatch.setattr(h.secrets,'token_hex',draw)
    with pytest.raises(RecoveryError,match='FREEZE_FIELDS'):prepare(j,request)
    draw.assert_not_called();assert not j.read()['events']


def test_stop_blocks_entropy_and_release(setup, monkeypatch):
    j, request, write = setup
    (j.directory/'STOP').touch()
    with pytest.raises(RecoveryError): prepare(j, request)
    (j.directory/'STOP').unlink()  # Isolated fixture only.
    key, record = prepare(j, request); pub = write('published.json', record)
    (j.directory/'STOP').touch()
    with j.writer(), pytest.raises(RecoveryError): h.release(j, key, pub, publication_commit="0"*40)


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
    with j.writer(), pytest.raises(RecoveryError): h.release(j, key, bad, publication_commit="0"*40)
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


def test_uncommitted_publication_cannot_release(setup):
    j, request, write = setup
    key, record = prepare(j, request)
    prior = publish(j, write('unrelated.json', {}))
    pub = write('published.json', record)
    with j.writer(), pytest.raises(RecoveryError, match='PUBLICATION_GIT'):
        h.release(j, key, pub, publication_commit=prior)
    assert 'release' not in j.read()['events'][key]


def test_changed_worktree_publication_does_not_match_committed_blob(setup):
    j, request, write = setup
    key, record = prepare(j, request)
    commit = publish(j, write('published.json', {}))
    pub = write('published.json', record)
    with j.writer(), pytest.raises(RecoveryError, match='CONTENT_DRIFT'):
        h.release(j, key, pub, publication_commit=commit)
    assert 'release' not in j.read()['events'][key]


def test_release_replay_cannot_rebind_to_later_commit(setup):
    j, request, write = setup
    key, record = prepare(j, request)
    pub = write('published.json', record); first = publish(j, pub)
    with j.writer(): h.release(j, key, pub, publication_commit=first)
    second = publish(j, write('later.json', {}))
    with j.writer(), pytest.raises(RecoveryError, match='RELEASE_DRIFT'):
        h.release(j, key, pub, publication_commit=second)
