"""Measure the actual chunk writer/reader on old bytes without loading pickle."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from training_workloads.unified_h1.lossless_evidence import EvidencePool,ChunkReader,CHUNK

ROOT=Path(__file__).resolve().parents[1]


def run():
    source=ROOT/'docs/experiments/r6-robustness-auto/TRAINING_STORAGE_CPU.json'
    inventory=json.loads(source.read_text());rows=[]
    if shutil.disk_usage(tempfile.gettempdir()).free<32*1024**3:raise RuntimeError('30GiB free floor plus2GiB temporary pool')
    with tempfile.TemporaryDirectory(prefix='r6-training-store-') as directory:
        pool=EvidencePool(Path(directory)/'evidence',run_id='CPU-HISTORICAL-BYTES',maximum_bytes=2*1024**3,maximum_raw_file_bytes=2*1024**3)
        for row in inventory['files']:
            if row['bytes']<1024**2:continue
            path=Path(row['path']);before=path.stat();digest=hashlib.sha256();raw_bytes=0;records=0;maximum_line=0
            writer=pool.open(row['relative']);start=time.monotonic()
            with path.open('rb') as stream:
                while True:
                    raw=stream.readline(64*1024**2+1) if path.suffix=='.jsonl' else stream.read(CHUNK)
                    if not raw:break
                    if len(raw)>64*1024**2:raise ValueError('actual historical record exceeds proposed64MiB cap')
                    if path.suffix=='.jsonl' and not raw.endswith(b'\n'):raise ValueError('historical partial JSONL row')
                    writer.write(raw);digest.update(raw);raw_bytes+=len(raw)
                    if path.suffix=='.jsonl':writer.flush();records+=1;maximum_line=max(maximum_line,len(raw))
            writer.close();write_seconds=time.monotonic()-start;start=time.monotonic()
            with ChunkReader(writer.path,run_id=pool.run_id,logical_name=row['relative'],maximum_raw_bytes=2*1024**3) as reader:
                if reader.raw_sha256!=digest.hexdigest() or reader.size!=raw_bytes:raise ValueError('actual codec raw mismatch')
            after=path.stat()
            if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) or raw_bytes!=row['bytes']:
                raise ValueError('historical input changed')
            rows.append(dict(path=str(path),relative=row['relative'],sha256=digest.hexdigest(),raw_bytes=raw_bytes,
                compressed_bytes=writer.path.stat().st_size,chunks=writer.chunks,jsonl_records=records,
                maximum_jsonl_record_bytes=maximum_line,write_fsync_seconds=write_seconds,
                verified_read_seconds=time.monotonic()-start,exact_bytes=True))
        charged=pool.used
        if charged!=sum(r['compressed_bytes'] for r in rows):raise ValueError('pool accounting')
    small=inventory['total_bytes']-sum(r['raw_bytes'] for r in rows)
    def pin(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(status='ACTUAL_CPU_CODEC_HISTORICAL_BYTES_NOT_R6_TRAINING_ACCEPTANCE',capacity_pass=False,
        files=rows,measured_compressed_bytes=charged,small_uncompressed_bytes=small,
        projected_total_bytes=charged+small,original_total_bytes=inventory['total_bytes'],
        source=pin(source),code=[pin(ROOT/p) for p in ['tools/measure_r6_training_store.py','training_workloads/unified_h1/lossless_evidence.py']],
        limitations='Actual original R5 bytes and new chunk codec, including per-JSONL-record flush/fsync. Temporary archives deleted only after exact verification. No original modification, no pickle loading, no R6 training. Future noisy fields may compress differently; no capacity acceptance.')


if __name__=='__main__':print(json.dumps(run(),indent=2))
