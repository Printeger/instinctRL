"""Read-only lossless stream compression measurement of historical R5 files.

No torch.load/pickle execution, no originals modified, no archive files retained.
This does not implement or approve a new training logging format.
"""
import hashlib
import json
from pathlib import Path
import time
import zlib

ROOT=Path(__file__).resolve().parents[1]


def measure(path):
    path=Path(path);before=path.stat();encoder=zlib.compressobj(3);decoder=zlib.decompressobj()
    original=hashlib.sha256();restored=hashlib.sha256();raw_size=0;compressed_size=0;decoded_size=0
    start=time.monotonic()
    with path.open('rb') as stream:
        for raw in iter(lambda:stream.read(1024**2),b''):
            original.update(raw);raw_size+=len(raw)
            compressed=encoder.compress(raw);compressed_size+=len(compressed)
            decoded=decoder.decompress(compressed);restored.update(decoded);decoded_size+=len(decoded)
    tail=encoder.flush();compressed_size+=len(tail)
    decoded=decoder.decompress(tail)+decoder.flush();restored.update(decoded);decoded_size+=len(decoded)
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise RuntimeError('original changed')
    if not decoder.eof or decoder.unused_data or original.digest()!=restored.digest() or raw_size!=decoded_size:
        raise RuntimeError('compression roundtrip mismatch')
    return dict(path=str(path.resolve()),sha256=original.hexdigest(),original_bytes=raw_size,
        compressed_bytes=compressed_size,exact_byte_roundtrip=True,seconds=time.monotonic()-start)


def run():
    source=ROOT/'docs/experiments/r6-robustness-auto/TRAINING_STORAGE_CPU.json'
    inventory=json.loads(source.read_text());rows=[]
    for row in inventory['files']:
        if row['bytes']>=1024**2:
            result=measure(row['path'])
            if result['original_bytes']!=row['bytes']:raise RuntimeError('historical size drift')
            result['relative']=row['relative'];rows.append(result)
    measured=sum(r['original_bytes'] for r in rows)
    unchanged=inventory['total_bytes']-measured
    return dict(status='READ_ONLY_HISTORICAL_BYTE_COMPRESSION_NOT_R6_LOGGING_APPROVAL',capacity_pass=False,
        codec='zlib level3 streaming; original bytes recover exactly',files=rows,
        original_total_bytes=inventory['total_bytes'],unmeasured_small_files_bytes=unchanged,
        projected_lossless_total_bytes=sum(r['compressed_bytes'] for r in rows)+unchanged,
        source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        tool=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        limitations='Actual oldR5 bytes; R6 noise/fields can change ratio. Not a worst-case bound. No compressed files or new training writer created. No pickle loading, GPU or simulator.')


if __name__=='__main__':print(json.dumps(run(),indent=2))
