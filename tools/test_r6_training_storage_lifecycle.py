"""Real backend preservation/cleanup on CPU; no simulator or training launch."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from training_workloads.unified_h1.backend import UnifiedH1LiveBackend
from training_workloads.unified_h1.budget import ExecutionCounters
from training_workloads.unified_h1.lossless_evidence import EvidencePool, ChunkReader
from training_workloads.unified_h1.robustness_training_storage import R6LearningLog
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
