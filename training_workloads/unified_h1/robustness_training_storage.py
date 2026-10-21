"""Opt-in R6 evidence adapters; no training dispatcher or gate changes."""
import json
from pathlib import Path
import torch
from .logging import LearningLog,_jsonable
from .pilot_diagnostic import FirstUpdateDiagnostic


def create_training_evidence(output, *, run_id, payload_bytes, compressed_bytes,
                             plain_bytes, maximum_raw_file_bytes, starting_update,
                             initial_start_types=None):
    """Build both observers on one pool within an already admitted payload.

    This factory grants no training authority. The prospective R6 training
    contract must supply the exact payload and allowances before calling it.
    Plain checkpoint/metadata writes still use the installed global write cap.
    """
    limits=(payload_bytes,compressed_bytes,plain_bytes,maximum_raw_file_bytes)
    if any(type(v) is not int or v<=0 for v in limits):
        raise ValueError('R6_TRAINING_STORAGE_LIMITS')
    if compressed_bytes+plain_bytes>payload_bytes:
        raise ValueError('R6_TRAINING_STORAGE_PAYLOAD')
    if type(starting_update) is not int or starting_update<0:
        raise ValueError('R6_TRAINING_STORAGE_START')
    from .lossless_evidence import EvidencePool
    output=Path(output)
    pool=EvidencePool(output/'compressed_evidence',run_id=run_id,
        maximum_bytes=compressed_bytes,maximum_raw_file_bytes=maximum_raw_file_bytes)
    try:
        journal=R6LearningLog(output/'learning_log',pool=pool,
            initial_start_types=initial_start_types)
        diagnostic=R6FirstUpdateDiagnostic(output/'pilot_diagnostic',pool=pool,
            enabled=True,starting_update=starting_update)
    except BaseException as error:
        try:pool.abort()
        except BaseException as cleanup:error.add_note('Evidence construction cleanup failed: '+repr(cleanup))
        raise
    return journal,diagnostic


class R6LearningLog(LearningLog):
    def __init__(self,root,*,pool,logical_prefix="learning_log",fixed_reset_types=None,**kwargs):
        path=Path(logical_prefix)
        if not logical_prefix or path.is_absolute() or '..' in path.parts or path.as_posix()!=logical_prefix:
            raise ValueError('R6_LOG_PREFIX')
        if fixed_reset_types is not None:
            if kwargs.get('initial_start_types')!=fixed_reset_types:raise ValueError('R6_LOG_RESET_TYPES')
        self.logical_prefix=logical_prefix;self.fixed_reset_types=fixed_reset_types
        super().__init__(root,**kwargs)
        self.pool=pool;self.compressed_streams={};self.closed=False
        self.finalization_error=None
        self.evidence_finalized=False

    def next_reset_start_type(self,lane,driver):
        if self.fixed_reset_types is not None:return self.fixed_reset_types[lane]
        return super().next_reset_start_type(lane,driver)

    def finalize_evidence(self):
        """Write terminal rows once, then seal the shared diagnostic/log pool.

        A complete stream certifies bytes, not successful training. Failed runs
        may retain complete evidence; an I/O failure leaves an incomplete prefix.
        """
        if self.finalization_error is not None:raise self.finalization_error
        if self.evidence_finalized:return
        try:
            if self.closed:raise ValueError('R6_LOG_CLOSED_BEFORE_FINALIZATION')
            self.finalize_right_censored()
            self.pool.close()
            self.evidence_finalized=True
        except BaseException as error:
            self.finalization_error=error
            try:self.pool.abort()
            except BaseException as cleanup:error.add_note('Evidence abort failed: '+repr(cleanup))
            raise
        finally:self.closed=True

    def abort_evidence(self):
        """Fallback before simulator close if preservation did not seal evidence."""
        if self.pool.closed:return
        self.closed=True
        self.pool.abort()

    def append(self,relative,record):
        if self.closed:raise ValueError('R6_LOG_CLOSED')
        name=self.logical_prefix+'/'+relative
        # Validate each request even when a stream has already been opened.
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or path.as_posix()!=relative:raise ValueError('R6_LOG_PATH')
        data=(json.dumps(_jsonable(dict(record)),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        if len(data)>64*1024**2:raise ValueError('R6_LOG_RECORD_BOUND')
        if relative not in self.compressed_streams:self.compressed_streams[relative]=self.pool.open(name)
        stream=self.compressed_streams[relative]
        try:stream.write(data);stream.flush()
        except BaseException:stream.abort();raise

    def close(self):
        self.closed=True
        errors=[]
        for stream in self.compressed_streams.values():
            try:stream.close()
            except BaseException as error:errors.append(error)
        if errors:raise errors[0]


class R6FirstUpdateDiagnostic(FirstUpdateDiagnostic):
    def __init__(self,output,*,pool,enabled=True,starting_update=0):
        super().__init__(output,enabled=enabled,starting_update=starting_update)
        self.pool=pool

    def _save_tensor(self,value,path):
        path=Path(path)
        relative=path.relative_to(self.output)
        stream=self.pool.open('pilot_diagnostic/'+relative.as_posix())
        try:
            torch.save(value,stream);stream.close()
        except BaseException:stream.abort();raise
