"""Real backend preservation/cleanup on CPU; no simulator or training launch."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from training_workloads.unified_h1.backend import UnifiedH1LiveBackend
from training_workloads.unified_h1.budget import ExecutionCounters
from training_workloads.unified_h1.lossless_evidence import EvidencePool, ChunkReader
from training_workloads.unified_h1.robustness_training_storage import R6LearningLog
from training_workloads.unified_h1.robustness_training_storage import create_training_evidence
from training_workloads.scale_003.lifecycle import run_lifecycle


@pytest.fixture
def backend(tmp_path):
    b=UnifiedH1LiveBackend.__new__(UnifiedH1LiveBackend)
    b.output=tmp_path
    b.counters=ExecutionCounters()
    b.initialization_measurement={}
    b.environment=SimpleNamespace(close=Mock())
    b.app=SimpleNamespace(close=Mock())
    pool=EvidencePool(tmp_path/'compressed',run_id='CPU',maximum_bytes=1024**2,
                      maximum_raw_file_bytes=1024**2)
    b.journal=R6LearningLog(tmp_path/'logical',pool=pool)
    b.journal.append('updates.jsonl',{'update':1})
    return b


def read(b):
    return ChunkReader(b.output/'compressed/learning_log/updates.jsonl.r6z',
                      run_id='CPU',logical_name='learning_log/updates.jsonl',
                      maximum_raw_bytes=1024**2)


def test_actual_backend_seals_once_before_app_close(backend):
    b=backend
    with patch.object(b.journal,'finalize_right_censored',wraps=b.journal.finalize_right_censored) as finalize:
        b.preserve('PASS_PENDING_CLOSE')
        b.preserve('PASS_PENDING_CLOSE')
        finalize.assert_called_once()
    with read(b) as stream:assert json.loads(stream.read())=={'update':1}
    b.close_bounded()
    b.preserve('PASS')
    assert b.journal.pool.closed and not b.journal.pool.failed
    b.environment.close.assert_called_once();b.app.close.assert_called_once()


def test_cleanup_without_preserve_aborts_unsealed_stream(backend):
    b=backend;b.close_bounded()
    assert b.journal.closed and b.journal.pool.closed
    with pytest.raises(ValueError,match='INCOMPLETE'):read(b)
    b.environment.close.assert_called_once();b.app.close.assert_called_once()


def test_original_training_error_survives_fsync_failure(backend):
    b=backend;primary=RuntimeError('training failed');outcomes=[]
    def operation():raise primary
    def failure(outcome):
        outcomes.append(outcome);b.preserve('FAIL',outcome)
    with patch('training_workloads.unified_h1.lossless_evidence.os.fsync',side_effect=OSError('disk failed')):
        code=run_lifecycle(operation,[('backend',b.close_bounded)],lambda _:None,failure,
            lambda _,outcome:b.preserve('FAIL_PENDING_CLOSE',outcome))
    assert code==1 and outcomes[0].primary_error is primary
    assert any(str(row.error)=='disk failed' for row in outcomes[0].cleanup_failures)
    assert b.journal.pool.closed and b.journal.pool.failed
    assert all(w.stream.closed for w in b.journal.pool.writers.values())
    b.environment.close.assert_called_once();b.app.close.assert_called_once()


def factory(root,**overrides):
    args=dict(run_id='CPU',payload_bytes=2*1024**2,compressed_bytes=1024**2,
              plain_bytes=1024**2,maximum_raw_file_bytes=1024**2,starting_update=640)
    args.update(overrides)
    return create_training_evidence(root,**args)


def test_factory_shares_pool_and_backend_seals_diagnostic_and_log(backend):
    import torch
    b=backend;b.journal.abort_evidence()
    journal,diagnostic=factory(b.output/'factory')
    b.journal=journal
    assert journal.pool is diagnostic.pool
    journal.append('updates.jsonl',{'update':641})
    diagnostic._save_tensor({'x':torch.arange(8)},diagnostic.output/'input.pt')
    b.preserve('PASS_PENDING_CLOSE');b.close_bounded()
    for name in ('learning_log/updates.jsonl','pilot_diagnostic/input.pt'):
        with ChunkReader(journal.pool.root/(name+'.r6z'),run_id='CPU',logical_name=name,
                         maximum_raw_bytes=1024**2) as reader:
            assert reader.read()
    assert journal.pool.closed and not journal.pool.failed


@pytest.mark.parametrize('override',[
    {'compressed_bytes':1024**2+1}, {'plain_bytes':0}, {'payload_bytes':True},
    {'starting_update':-1}])
def test_factory_rejects_bad_capacity_before_creating_files(tmp_path,override):
    with pytest.raises(ValueError,match='R6_TRAINING_STORAGE'):factory(tmp_path,**override)
    assert not list(tmp_path.iterdir())


def test_factory_partial_construction_preserves_original_error(tmp_path):
    pools=[]
    original=RuntimeError('diagnostic constructor failed')
    def fail(*args,**kwargs):
        pools.append(kwargs['pool']);raise original
    with patch('training_workloads.unified_h1.robustness_training_storage.R6FirstUpdateDiagnostic',side_effect=fail):
        with pytest.raises(RuntimeError) as caught:factory(tmp_path)
    assert caught.value is original and pools[0].closed and pools[0].failed
    with pytest.raises(FileExistsError):factory(tmp_path)


def test_finalization_error_is_not_retried_or_certified(backend):
    b=backend
    with patch.object(b.journal,'finalize_right_censored',side_effect=OSError('terminal rows failed')) as finalize:
        for _ in range(2):
            with pytest.raises(OSError,match='terminal rows failed'):b.preserve('PASS_PENDING_CLOSE')
        finalize.assert_called_once()
    with pytest.raises(ValueError,match='INCOMPLETE'):read(b)
    b.close_bounded();assert b.journal.pool.failed


def test_early_log_close_does_not_certify_pool(backend):
    b=backend;b.journal.close()
    with pytest.raises(ValueError,match='CLOSED_BEFORE_FINALIZATION'):b.preserve('PASS_PENDING_CLOSE')
    assert b.journal.pool.failed and b.journal.pool.closed


def test_legacy_log_keeps_existing_preservation_path(backend):
    b=backend
    from training_workloads.unified_h1.logging import LearningLog
    b.journal.abort_evidence()
    b.journal=LearningLog(b.output/'legacy')
    with patch.object(b.journal,'finalize_right_censored',wraps=b.journal.finalize_right_censored) as finalize:
        b.preserve('PASS_PENDING_CLOSE');b.close_bounded();finalize.assert_called_once()


@pytest.mark.parametrize('episode_write_fails',[False,True])
def test_training_loop_retains_primary_when_quota_prevents_failure_logs(backend,episode_write_fails):
    import torch
    from training_workloads.unified_h1.driver import UnifiedH1TrainingLoop
    b=backend;primary=RuntimeError('original rollout failure')
    actor=torch.nn.Linear(1,1);critic=torch.nn.Linear(1,1)
    trainer=SimpleNamespace(update_counter=0,
        actor_optimizer=torch.optim.Adam(actor.parameters()),
        critic_optimizer=torch.optim.Adam(critic.parameters()))
    driver=SimpleNamespace(counters=b.counters,observation={},evidence_buffer=[])
    loop=UnifiedH1TrainingLoop(driver,trainer,b.journal,update_limit=1)
    def rollout(*args):
        # After the initial durable attempt record, exhaust the real shared
        # pool so both remaining failure-record writes fail independently.
        b.journal.pool.maximum=b.journal.pool.used
        raise primary
    episode_effect=OSError('episode write failed') if episode_write_fails else None
    with patch('instinctlab.training.rollout_h1.collect_h1_rollout',side_effect=rollout), \
         patch.object(b.journal,'record_partial_episodes',side_effect=episode_effect) as episodes, \
         patch.object(b.journal,'record_partial_rollout_boundary',wraps=b.journal.record_partial_rollout_boundary) as boundary, \
         patch.object(b.journal,'record_partial_failure',wraps=b.journal.record_partial_failure) as failure:
        with pytest.raises(RuntimeError) as caught:loop.run_update()
    assert caught.value is primary
    assert len(primary.__notes__)==2+int(episode_write_fails)
    episodes.assert_called_once();boundary.assert_called_once();failure.assert_called_once()
    assert b.counters.attempted_updates==1 and b.counters.completed_updates==0
    assert b.counters.actor_optimizer_steps==b.counters.critic_optimizer_steps==0
    b.close_bounded()
    assert b.journal.pool.failed and b.journal.pool.closed
    assert all(w.stream.closed for w in b.journal.pool.writers.values())
