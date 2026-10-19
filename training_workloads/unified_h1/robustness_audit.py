"""Bounded lossless R6 reset/physics/motion records, independent per-record zlib."""
from pathlib import Path
import hashlib
import json
import os
import struct
import zlib
from .logging import _jsonable
from training_workloads.scale_003.persistence import reserve_output_bytes

HEADER=struct.Struct('<II32s')
MAX_RECORD=8*1024**2


class AuditLog:
    def __init__(self,path,*,run_id,block_id,maximum_bytes):
        if type(maximum_bytes)is not int or maximum_bytes<=0:raise ValueError('AUDIT_QUOTA')
        self.path=Path(path);self.run_id=run_id;self.block_id=block_id;self.maximum=maximum_bytes
        self.written=0;self.records=0;self.failed=False;self.stream=self.path.open('xb')
        fd=os.open(self.path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def append(self,kind,payload):
        if self.failed:raise ValueError('AUDIT_PREVIOUS_FAILURE')
        raw=json.dumps(dict(schema_version='r6-audit/1',run_id=self.run_id,block_id=self.block_id,
            sequence=self.records,kind=kind,payload=_jsonable(payload)),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if len(raw)>MAX_RECORD:raise ValueError('AUDIT_RECORD_BOUND')
        compressed=zlib.compress(raw,3)
        record=HEADER.pack(len(compressed),len(raw),hashlib.sha256(raw).digest())+compressed
        if self.written+len(record)>self.maximum:raise ValueError('AUDIT_QUOTA_BEFORE_WRITE')
        reserve_output_bytes(self.path,len(record))
        try:
            self.stream.write(record);self.stream.flush();os.fsync(self.stream.fileno())
        except BaseException:self.failed=True;raise
        self.written+=len(record);self.records+=1
    def close(self):self.stream.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def read_audit(path,*,run_id,block_id,expected_records):
    if type(expected_records)is not int or expected_records<=0:raise ValueError('AUDIT_EXPECTED_COUNT')
    with Path(path).open('rb') as stream:
        for sequence in range(expected_records):
            prefix=stream.read(HEADER.size)
            if len(prefix)!=HEADER.size:raise ValueError('AUDIT_MISSING_RECORD')
            size,raw_size,digest=HEADER.unpack(prefix)
            if not 0<size<=MAX_RECORD+65536 or not 0<raw_size<=MAX_RECORD:raise ValueError('AUDIT_RECORD_BOUND')
            compressed=stream.read(size)
            if len(compressed)!=size:raise ValueError('AUDIT_TRUNCATED')
            decoder=zlib.decompressobj();raw=decoder.decompress(compressed,raw_size+1)
            if len(raw)!=raw_size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError('AUDIT_DECOMPRESSION_BOUND')
            if hashlib.sha256(raw).digest()!=digest:raise ValueError('AUDIT_HASH')
            row=json.loads(raw)
            if (row['schema_version'],row['run_id'],row['block_id'],row['sequence'])!=('r6-audit/1',run_id,block_id,sequence):
                raise ValueError('AUDIT_IDENTITY')
            yield row
        if stream.read(1):raise ValueError('AUDIT_EXTRA_RECORDS')
