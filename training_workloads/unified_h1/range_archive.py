"""Lossless, quota-before-write R6 measurement-frame archive; no pickle.

Each frame is independently compressed, hashed and fsynced. An interrupted tail
is an error and is preserved. Reader requires the externally expected frame count;
an EOF alone never proves a completed evaluation. This does not replace raw motion.
"""
from pathlib import Path
import hashlib
import json
import os
import struct
import zlib
import numpy as np
from training_workloads.scale_003.persistence import reserve_output_bytes

ARRAY_KEYS = ('clean_ranges', 'clean_validity', 'delivered_ranges', 'delivered_validity',
              'acquisition_time', 'delivery_time', 'frame_age', 'drop', 'frame_available')
MAX_FRAME = 64*1024**2
MAX_HEADER = 64*1024


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()


def encode(frame, *, sequence, run_id, block_id):
    if set(frame) != {*ARRAY_KEYS,'noise_rng_identity'}:raise ValueError('FRAME_FIELDS')
    if type(sequence) is not int or sequence<0 or not run_id or not block_id:raise ValueError('FRAME_IDENTITY')
    identities=frame['noise_rng_identity']
    if not isinstance(identities,list) or not all(isinstance(x,str) and len(x)==64 for x in identities):
        raise ValueError('RNG_IDENTITY')
    fields=[]; parts=[]; prior={}; raw_total=0;offset=0; originals={}
    for name in ARRAY_KEYS:
        value=frame[name]
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        a=np.asarray(value)
        if a.dtype.str not in ('<f4','<f8','|b1') or a.ndim!=2:raise ValueError('ARRAY_TYPE')
        if a.shape[0]!=len(identities) or a.shape[0]<=0 or a.shape[1]<=0:raise ValueError('ARRAY_SHAPE')
        boolean=name in ('clean_validity','delivered_validity','drop','frame_available')
        if (a.dtype.kind=='b')!=boolean:raise ValueError('FIELD_DTYPE')
        if name not in ARRAY_KEYS[:4] and a.shape[1]!=1:raise ValueError('TIMESTAMP_SHAPE')
        if name in ARRAY_KEYS[1:4] and a.shape!=originals['clean_ranges'].shape:raise ValueError('BEAM_SHAPE')
        if a.dtype.kind=='f' and not np.isfinite(a).all():raise ValueError('NONFINITE_FRAME')
        a=np.ascontiguousarray(a);raw=a.tobytes();raw_total+=len(raw)
        if raw_total>MAX_FRAME:raise ValueError('FRAME_TOO_LARGE')
        key=(a.dtype.str,a.shape,raw)
        item=dict(name=name,dtype=a.dtype.str,shape=list(a.shape),sha256=digest(raw))
        if key in prior:item['reference']=prior[key]
        else:
            # Byte planes make equal exponent/high-mantissa bytes adjacent;
            # inverse transpose restores the original bytes, including -0.
            shuffled=a.view(np.uint8).reshape(-1,a.dtype.itemsize).T.copy().tobytes()
            compressed=zlib.compress(shuffled,level=3)
            base={'delivered_ranges':'clean_ranges','delivered_validity':'clean_validity'}.get(name)
            if base in originals and originals[base].dtype==a.dtype and originals[base].shape==a.shape:
                xor=np.bitwise_xor(a.view(np.uint8),originals[base].view(np.uint8))
                alternative=zlib.compress(xor.reshape(-1,a.dtype.itemsize).T.copy().tobytes(),level=3)
                if len(alternative)<len(compressed):
                    compressed=alternative;item['xor_base']=base
            item.update(offset=offset,length=len(compressed),raw_bytes=len(raw))
            parts.append(compressed);offset+=len(compressed);prior[key]=name
        fields.append(item)
        originals[name]=a
    payload=b''.join(parts)
    header=canonical(dict(schema_version='r6-range-frame/1',sequence=sequence,run_id=run_id,
                          block_id=block_id,noise_rng_identity=identities,fields=fields,
                          payload_bytes=len(payload),payload_sha256=digest(payload)))
    if len(header)>MAX_HEADER:raise ValueError('HEADER_TOO_LARGE')
    return struct.pack('<I',len(header))+header+payload


def decode(header,payload):
    if header['schema_version']!='r6-range-frame/1' or digest(payload)!=header['payload_sha256']:
        raise ValueError('FRAME_INTEGRITY')
    if [f['name'] for f in header['fields']]!=list(ARRAY_KEYS):raise ValueError('FRAME_FIELDS')
    result={};total=0;offset=0
    for item in header['fields']:
        dtype=np.dtype(item['dtype']);shape=item['shape']
        if dtype.str not in ('<f4','<f8','|b1') or len(shape)!=2 or any(type(n)is not int or n<=0 for n in shape):
            raise ValueError('ARRAY_DESCRIPTOR')
        size=shape[0]*shape[1]*dtype.itemsize;total+=size
        if total>MAX_FRAME:raise ValueError('FRAME_TOO_LARGE')
        if 'reference' in item:
            a=result[item['reference']].copy()
            if a.dtype!=dtype or list(a.shape)!=shape:raise ValueError('REFERENCE_TYPE')
        else:
            length=item['length']
            if type(length)is not int or length<0 or item['offset']!=offset or item['raw_bytes']!=size:
                raise ValueError('PAYLOAD_LAYOUT')
            compressor=zlib.decompressobj()
            raw=compressor.decompress(payload[offset:offset+length],size+1)
            if len(raw)!=size or not compressor.eof or compressor.unused_data or compressor.unconsumed_tail:
                raise ValueError('DECOMPRESSION_BOUND')
            restored=np.frombuffer(raw,dtype=np.uint8).reshape(dtype.itemsize,-1).T.copy()
            if 'xor_base' in item:
                base=result[item['xor_base']]
                if base.dtype!=dtype or list(base.shape)!=shape:raise ValueError('XOR_BASE_TYPE')
                restored=np.bitwise_xor(restored,base.view(np.uint8).reshape(-1,dtype.itemsize))
            a=np.frombuffer(restored.tobytes(),dtype=dtype).reshape(shape).copy();offset+=length
        if digest(a.tobytes())!=item['sha256']:raise ValueError('ARRAY_HASH')
        result[item['name']]=a
    if offset!=len(payload):raise ValueError('UNCONSUMED_PAYLOAD')
    result['noise_rng_identity']=header['noise_rng_identity']
    return result


class RangeArchive:
    def __init__(self,path,*,run_id,block_id,maximum_bytes):
        if type(maximum_bytes)is not int or maximum_bytes<=0:raise ValueError('QUOTA')
        self.path=Path(path);self.run_id=run_id;self.block_id=block_id
        self.maximum=maximum_bytes;self.written=0;self.frames=0
        self.stream=self.path.open('xb')
        self.failed=False
        fd=os.open(self.path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)

    def append(self,frame):
        if self.failed:raise ValueError('PREVIOUS_WRITE_FAILED_PRESERVE_PARTIAL_ARCHIVE')
        record=encode(frame,sequence=self.frames,run_id=self.run_id,block_id=self.block_id)
        if self.written+len(record)>self.maximum:raise ValueError('ARCHIVE_QUOTA_BEFORE_WRITE')
        reserve_output_bytes(self.path,len(record))
        try:
            self.stream.write(record);self.stream.flush();os.fsync(self.stream.fileno())
        except BaseException:
            self.failed=True
            raise
        self.written+=len(record);self.frames+=1

    def close(self):self.stream.close()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def read_archive(path,*,run_id,block_id,expected_frames):
    if type(expected_frames)is not int or expected_frames<=0:raise ValueError('EXPECTED_COUNT_REQUIRED')
    with Path(path).open('rb') as stream:
        for sequence in range(expected_frames):
            prefix=stream.read(4)
            if len(prefix)!=4:raise ValueError('MISSING_OR_PARTIAL_FRAME')
            length=struct.unpack('<I',prefix)[0]
            if not 0<length<=MAX_HEADER:raise ValueError('HEADER_BOUND')
            raw=stream.read(length)
            if len(raw)!=length:raise ValueError('PARTIAL_HEADER')
            header=json.loads(raw)
            if header['run_id']!=run_id or header['block_id']!=block_id or header['sequence']!=sequence:
                raise ValueError('FRAME_IDENTITY')
            size=header['payload_bytes']
            if type(size)is not int or not 0<=size<=MAX_FRAME+MAX_HEADER:raise ValueError('PAYLOAD_BOUND')
            payload=stream.read(size)
            if len(payload)!=size:raise ValueError('PARTIAL_PAYLOAD')
            yield decode(header,payload)
        if stream.read(1):raise ValueError('EXTRA_FRAME_OR_TAIL')
