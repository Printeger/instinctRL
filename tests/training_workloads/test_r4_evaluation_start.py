"""Exercise formal entry -> real authority checker -> real lifecycle on CPU.

Repository identity is a fixture here; separate validate-only checks the actual
frozen candidate. Only simulator construction and rollout execution are replaced.
"""
import copy
import json
import sys
import types
from pathlib import Path

import pytest

from training_workloads.unified_h1 import authorization as authmod
from training_workloads.unified_h1 import formal_unified_h1 as entry
from training_workloads.unified_h1 import live_adapter as live
from training_workloads.unified_h1.budget import required_mode_budget


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    attempt = tmp_path / 'attempt'
    training = attempt / 'results/r4-training'
    training.mkdir(parents=True)
    (training / 'process_start.json').write_text('{}')
    (training / 'status.json').write_text('{"status":"PASS"}')
    checkpoints = []
    for update in (0, 64, 128, 192, 256):
        path = training / f'u{update}.pt'
        path.write_bytes(b'CPU checkpoint fixture')
        checkpoints.append({'update':update, 'path':str(path), 'sha256':authmod.sha256_file(path)})
    identity = dict(product_commit='a'*40, product_tree='b'*40, workload_commit='c'*40, workload_tree='d'*40)
    config = {'resolved_identity':identity}
    monkeypatch.setattr(entry, 'load_config', lambda:config)
    monkeypatch.setattr(authmod, 'load_config', lambda:config)
    parameters = {'stage':'R4', 'checkpoints':checkpoints}
    auth = dict(schema_version='unified-h1-r3-authorization.v2', status='AUTHORIZED',
                order_id='UNIFIED-H1-R4', method_sha256=authmod.METHOD_SHA256,
                config_sha256=authmod.sha256_file(authmod.CONFIG_PATH), **identity,
                attempt=str(attempt), mode='evaluation', parameters=parameters,
                budgets=required_mode_budget('evaluation', parameters))
    path = tmp_path / 'authority.json'
    path.write_text(json.dumps(auth))
    calls = []
    class Resource:
        h1_initialization_physics_ticks = 0
        h1_initialization_tick_source = 'CPU'
        def close(self, *, wait_for_replicator=None, skip_cleanup=None): calls.append('close')
    class Backend:
        def resource_check(self, *args): pass
        def preserve(self, status, outcome=None):
            live._persist(attempt/'results/evaluation', self.counters, status, self.initialization_measurement, outcome)
    def factory(**kw):
        calls.append('factory')
        backend = Backend(); backend.counters = kw['counters']; backend.environment = Resource()
        kw['resources'].update(environment=backend.environment, app=Resource())
        return backend
    backend_module = types.ModuleType('training_workloads.unified_h1.backend')
    backend_module.create_live_backend = factory
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    sequence = types.ModuleType('training_workloads.unified_h1.r3_sequence')
    def evaluate(backend, *, checkpoints): calls.append(('evaluation', [r['update'] for r in checkpoints]))
    def forbidden(*args, **kwargs): raise AssertionError('training must not execute')
    sequence.execute_evaluation_phase = evaluate
    sequence.execute_controlled_failure = sequence.execute_r3_main = sequence.execute_training_phase = forbidden
    monkeypatch.setitem(sys.modules, sequence.__name__, sequence)
    argv = ['--authorization', str(path), '--attempt', str(attempt), '--mode', 'evaluation']
    return attempt, path, auth, argv, calls


def test_formal_entry_evaluates_after_training_and_refuses_second_start(prepared):
    attempt, _, _, argv, calls = prepared
    original = (attempt/'results/r4-training/status.json').read_bytes()
    assert entry.main(argv) == 0
    assert calls == ['factory', ('evaluation', [0,64,128,192,256]), 'close', 'close']
    status = json.loads((attempt/'results/evaluation/status.json').read_text())
    assert status['status'] == 'PASS'
    assert status['counters']['live_process_starts'] == 1
    assert status['counters']['training_transitions'] == 0
    assert (attempt/'results/r4-training/status.json').read_bytes() == original
    with pytest.raises(ValueError, match='mode output already exists'): entry.main(argv)
    assert calls.count('factory') == 1


@pytest.mark.parametrize('case', ['fail', 'pending', 'missing', 'other-start', 'symlink', 'dangling-output'])
def test_predecessor_and_duplicate_fail_closed_before_backend(prepared, case):
    attempt, _, _, argv, calls = prepared
    status = attempt/'results/r4-training/status.json'
    if case in {'fail','pending'}: status.write_text(json.dumps({'status':case.upper()}))
    elif case == 'missing': status.unlink()
    elif case == 'other-start':
        other = attempt/'results/main'; other.mkdir(); (other/'process_start.json').write_text('{}')
    elif case == 'symlink':
        status.rename(status.with_name('saved.json')); status.symlink_to('saved.json')
    else: (attempt/'results/evaluation').symlink_to(attempt/'missing')
    with pytest.raises((RuntimeError, ValueError)): entry.main(argv)
    assert not calls


@pytest.mark.parametrize('case', ['checkpoint', 'budget', 'order', 'stage'])
def test_real_authorization_still_checks_hashes_and_budget(prepared, case):
    _, path, original, argv, calls = prepared
    auth = copy.deepcopy(original)
    if case == 'checkpoint': auth['parameters']['checkpoints'][0]['sha256'] = '0'*64
    elif case == 'budget': auth['budgets']['live_process_starts'] = 2
    elif case == 'order': auth['order_id'] = 'UNIFIED-H1-R3'
    else: auth['parameters']['stage'] = 'R5-512'
    path.write_text(json.dumps(auth))
    with pytest.raises(PermissionError): entry.main(argv)
    assert not calls


def test_explicit_checkpoint_origin_must_keep_product_identity(prepared):
    _, path, auth, argv, calls = prepared
    origin = {key: auth[key] for key in ('product_commit','product_tree','workload_commit','workload_tree','config_sha256')}
    origin.update(workload_commit='e'*40, workload_tree='f'*40)
    auth['parameters']['checkpoint_source_identity'] = origin
    path.write_text(json.dumps(auth))
    assert entry.main(argv) == 0
    assert calls.count('factory') == 1


@pytest.mark.parametrize('change', ['product', 'missing', 'malformed'])
def test_checkpoint_origin_drift_is_rejected(prepared, change):
    _, path, auth, argv, calls = prepared
    origin = {key: auth[key] for key in ('product_commit','product_tree','workload_commit','workload_tree','config_sha256')}
    if change == 'product': origin['product_commit'] = 'f'*40
    elif change == 'missing': del origin['workload_tree']
    else: origin['workload_tree'] = 'invalid'
    auth['parameters']['checkpoint_source_identity'] = origin
    path.write_text(json.dumps(auth))
    with pytest.raises(PermissionError): entry.main(argv)
    assert not calls


def test_evaluation_gate_does_not_overwrite_training_gate(tmp_path, monkeypatch):
    from training_workloads.unified_h1 import backend, config
    attempt = tmp_path/'attempt'
    old_gate = attempt/'results/prelaunch'; old_gate.mkdir(parents=True)
    (old_gate/'trace').write_bytes(b'training trace')
    (attempt/'results/evaluation').mkdir()
    monkeypatch.setattr(backend, 'load_config', lambda:{'execution':{'isaacsim_asset_root':'/unused'}})
    monkeypatch.setattr(config, 'validate_prelaunch_environment', lambda _:None)
    calls = []
    def stop_before_gpu(output, **kwargs):
        calls.append(output)
        raise RuntimeError('CPU test boundary before GPU')
    monkeypatch.setattr(backend, 'start_after_coexistence_gate', stop_before_gpu)
    with pytest.raises(RuntimeError, match='CPU test boundary'):
        backend.create_live_backend(authorization={'mode':'evaluation','authorization_sha256':'a'*64}, attempt=attempt)
    assert calls == [attempt/'results/evaluation/prelaunch']
    assert (old_gate/'trace').read_bytes() == b'training trace'


@pytest.mark.parametrize('label,update', [('init0000',0),('u0064',64),('u0128',128),('u0192',192),('u0256',256)])
def test_real_checkpoints_pass_provenance_before_gpu_rng_boundary(monkeypatch, label, update):
    from training_workloads.unified_h1 import backend
    root = Path('/home/mint/rl_dev/instinctRL_coordination')
    old_auth = json.loads((root/'commander/authorizations/UNIFIED-H1-R4_D0001_TRAINING_AUTHORIZATION.json').read_text())
    origin = {k:old_auth[k] for k in ('product_commit','product_tree','workload_commit','workload_tree','config_sha256')}
    config = json.loads(authmod.CONFIG_PATH.read_text())
    config['resolved_identity'] = {**origin,'workload_commit':'e'*40,'workload_tree':'f'*40}
    monkeypatch.setattr(backend, 'load_config', lambda:config)
    instance = object.__new__(backend.UnifiedH1LiveBackend)
    instance.environment = types.SimpleNamespace(device='cpu')
    instance.authorization = {'parameters':{'checkpoint_source_identity':origin}}
    checkpoint = root/f'attempts/development/UNIFIED-H1-R4/UNIFIED-H1-R4-D0001/results/r4-training/checkpoints/{label}.pt'
    from instinctlab.training.checkpoint import CheckpointError
    # These are GPU checkpoints. The unchanged strict loader deliberately stops
    # CPU restoration at CUDA RNG handling, after method/source/tensor checks.
    # Do not strip RNG or mutate frozen checkpoints merely to make a test pass.
    with pytest.raises(CheckpointError, match='CPU trainer cannot restore CUDA RNG state'):
        instance._load_evaluation_checkpoint(checkpoint)
    instance.authorization['parameters']['checkpoint_source_identity'] = config['resolved_identity']
    with pytest.raises(CheckpointError, match='method/source identity mismatch'):
        instance._load_evaluation_checkpoint(checkpoint)
