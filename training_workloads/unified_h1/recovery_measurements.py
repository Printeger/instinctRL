"""Lossless R6 training frames retained inside the supervised evidence pool."""

from .range_archive import ARRAY_KEYS, PRIMING_KEYS, encode


RUN_PREFIX = 'R6-DIRECT-INSTINCT-GPU-R5-RECOVERY-'
STREAM = 'measurement.bin'


def sink(journal):
    stream = journal.pool.open(f'{journal.logical_prefix}/{STREAM}')
    sequence = 0
    previous = None

    def append(index, frame):
        nonlocal sequence, previous
        if index != sequence:
            raise ValueError('R6_RECOVERY_MEASUREMENT_SEQUENCE')
        arrays = {name: frame[name].detach().cpu().numpy().copy()
                  for name in (*ARRAY_KEYS, *PRIMING_KEYS) if name in frame}
        data = encode(dict(frame, **arrays), sequence=sequence, run_id=journal.pool.run_id,
                      block_id=journal.logical_prefix, previous=previous)
        try:
            stream.write(data)
            stream.flush()
        except BaseException:
            stream.abort()
            raise
        sequence += 1
        previous = arrays

    return append
