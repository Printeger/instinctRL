"""Versioned lossless encoding of full force audit tensors; never pickle.

The AuditLog envelope still hashes, compresses and bounds each independent
record. Decoding restores the same JSON-compatible values as legacy records.
"""
import json
import struct
import numpy as np
from .logging import _jsonable

MAGIC=b'R6F1\x00'
SHAPES={'mass_kg':(32,5),'body_quaternion_wxyz':(32,5,4),
    **{k:(32,5,3) for k in ('rotor_forces_local_n','rotor_torques_local_nm',
        'additional_world_n','additional_local_n','staged_forces_local_n','staged_torques_local_nm')}}
MAX_HEADER=65536


def encode_force(row):
    payload=row['payload'];descriptor=[];parts=[];offset=0
    for name,shape in SHAPES.items():
        value=payload[name]
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        array=np.asarray(value)
        if array.shape!=shape or array.dtype.str not in ('<f4','<f8') or not np.isfinite(array).all():
            raise ValueError('FORCE_ARCHIVE_TENSOR')
        if payload['dtype']!='torch.'+array.dtype.name:raise ValueError('FORCE_ARCHIVE_DTYPE')
        raw=np.ascontiguousarray(array).tobytes()
        descriptor.append(dict(name=name,dtype=array.dtype.str,shape=list(shape),offset=offset,bytes=len(raw)))
        parts.append(raw);offset+=len(raw)
    metadata={**row,'payload':{k:_jsonable(v) for k,v in payload.items() if k not in SHAPES}}
    header=json.dumps(dict(schema_version='r6-force-binary/1',row=metadata,arrays=descriptor),
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(header)>MAX_HEADER:raise ValueError('FORCE_ARCHIVE_HEADER_BOUND')
    return MAGIC+struct.pack('<I',len(header))+header+b''.join(parts)


def decode_force(raw):
    if not raw.startswith(MAGIC) or len(raw)<len(MAGIC)+4:raise ValueError('FORCE_ARCHIVE_PREFIX')
    size=struct.unpack('<I',raw[len(MAGIC):len(MAGIC)+4])[0]
    if not 0<size<=MAX_HEADER or len(raw)<len(MAGIC)+4+size:raise ValueError('FORCE_ARCHIVE_HEADER_BOUND')
    header=json.loads(raw[len(MAGIC)+4:len(MAGIC)+4+size]);data=raw[len(MAGIC)+4+size:]
    if header['schema_version']!='r6-force-binary/1' or [r['name'] for r in header['arrays']]!=list(SHAPES):
        raise ValueError('FORCE_ARCHIVE_FIELDS')
    row=header['row']
    if row.get('kind')!='force' or any(k in row['payload'] for k in SHAPES):raise ValueError('FORCE_ARCHIVE_METADATA')
    offset=0
    for item in header['arrays']:
        name=item['name'];dtype=np.dtype(item['dtype']);shape=SHAPES[name]
        if dtype.str not in ('<f4','<f8') or item['shape']!=list(shape):raise ValueError('FORCE_ARCHIVE_DESCRIPTOR')
        count=int(np.prod(shape));length=count*dtype.itemsize
        if item['offset']!=offset or item['bytes']!=length or offset+length>len(data):raise ValueError('FORCE_ARCHIVE_LAYOUT')
        array=np.frombuffer(data[offset:offset+length],dtype=dtype).reshape(shape)
        if not np.isfinite(array).all() or row['payload']['dtype']!='torch.'+dtype.name:raise ValueError('FORCE_ARCHIVE_DTYPE')
        row['payload'][name]=array.tolist();offset+=length
    if offset!=len(data):raise ValueError('FORCE_ARCHIVE_TRAILING_BYTES')
    return row
