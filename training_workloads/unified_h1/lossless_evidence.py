"""Bounded, seekable, lossless R6 evidence streams; no pickle interpretation.

Independent <=1MiB chunks preserve arbitrary writer bytes. A mandatory footer
distinguishes completion from a crashed prefix. Never resumes an existing file.
"""
import bisect
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import zlib
from training_workloads.scale_003.persistence import reserve_output_bytes

MAGIC=b'R6Z1\n'
HEADER=struct.Struct('<II32s')
CHUNK=1024**2
MAX_CHUNKS=131072


class EvidencePool:
    def __init__(self,root,*,run_id,maximum_bytes,maximum_raw_file_bytes):
        if not isinstance(run_id,str) or not run_id or len(run_id)>100:raise ValueError('EVIDENCE_RUN_ID')
        if any(type(v) is not int or v<=0 for v in (maximum_bytes,maximum_raw_file_bytes)) or maximum_raw_file_bytes>16*1024**3:
            raise ValueError('EVIDENCE_QUOTA')
        self.root=Path(root).absolute()
        if self.root.resolve()!=self.root:raise ValueError('EVIDENCE_PATH_ALIAS')
        self.root.mkdir(parents=True,exist_ok=False)
        self.run_id=run_id;self.maximum=maximum_bytes;self.raw_maximum=maximum_raw_file_bytes
        self.used=0;self.writers={};self.failed=False

    def reserve(self,path,size):
        if self.failed or self.used+size>self.maximum:
            self.failed=True;raise ValueError('EVIDENCE_OUTPUT_QUOTA_OR_FAILURE')
        self.used+=size
        try:reserve_output_bytes(path,size)
        except BaseException:self.failed=True;raise

    def open(self,logical_name):
        p=Path(logical_name)
        if not logical_name or p.is_absolute() or '..' in p.parts or p.as_posix()!=logical_name:
            raise ValueError('EVIDENCE_LOGICAL_PATH')
        if logical_name in self.writers:raise ValueError('EVIDENCE_ALREADY_OPENED')
        if self.failed:raise ValueError('EVIDENCE_POOL_FAILED')
        path=self.root/(logical_name+'.r6z')
        if path.resolve()!=path or self.root not in path.parents:raise ValueError('EVIDENCE_PATH_ALIAS')
        path.parent.mkdir(parents=True,exist_ok=True)
        writer=ChunkWriter(self,path,logical_name);self.writers[logical_name]=writer
        return writer


class ChunkWriter:
    def __init__(self,pool,path,name):
        self.pool=pool;self.path=path;self.name=name;self.raw_bytes=0;self.chunks=0
        self.digest=hashlib.sha256();self.pending=bytearray();self.failed=False;self.complete=False
        metadata=json.dumps(dict(run_id=pool.run_id,logical_name=name),sort_keys=True,separators=(',',':')).encode()
        if len(metadata)>4096:raise ValueError('EVIDENCE_METADATA_BOUND')
        self.stream=path.open('xb')
        try:self._emit(MAGIC+struct.pack('<I',len(metadata))+metadata)
        except BaseException:self.stream.close();raise

    def _emit(self,data):
        if self.failed:raise ValueError('EVIDENCE_PREVIOUS_FAILURE')
        try:
            self.pool.reserve(self.path,len(data))
            if self.stream.write(data)!=len(data):raise OSError('EVIDENCE_SHORT_WRITE')
        except BaseException:self.failed=True;self.pool.failed=True;raise

    def _chunk(self):
        if not self.pending:return
        if self.chunks>=MAX_CHUNKS:
            self.failed=True;self.pool.failed=True;raise ValueError('EVIDENCE_CHUNK_COUNT')
        raw=bytes(self.pending);compressed=zlib.compress(raw,3)
        self._emit(HEADER.pack(len(raw),len(compressed),hashlib.sha256(raw).digest())+compressed)
        self.pending.clear();self.chunks+=1

    def write(self,data):
        if self.stream.closed or self.complete or self.failed or self.pool.failed:raise ValueError('EVIDENCE_WRITER_CLOSED_OR_FAILED')
        data=memoryview(data).cast('B');size=len(data)
        if self.raw_bytes+size>self.pool.raw_maximum:
            self.failed=True;self.pool.failed=True;raise ValueError('EVIDENCE_RAW_QUOTA')
        self.raw_bytes+=size;self.digest.update(data)
        offset=0
        while offset<size:
            count=min(CHUNK-len(self.pending),size-offset);self.pending.extend(data[offset:offset+count]);offset+=count
            if len(self.pending)==CHUNK:self._chunk()
        return size

    def tell(self):return self.raw_bytes

    def flush(self):
        if self.failed:raise ValueError('EVIDENCE_PREVIOUS_FAILURE')
        self._chunk();self.stream.flush();os.fsync(self.stream.fileno())

    def close(self):
        if self.stream.closed:return
        try:
            self.flush();self._emit(HEADER.pack(0,0,self.digest.digest()))
            self.stream.flush();os.fsync(self.stream.fileno());self.complete=True
            directory=os.open(self.path.parent,os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
        finally:self.stream.close()

    def abort(self):
        # Preserve every emitted byte, but never write a success footer.
        self.failed=True;self.pool.failed=True;self.stream.close()


class ChunkReader(io.RawIOBase):
    def __init__(self,path,*,run_id,logical_name,maximum_raw_bytes):
        super().__init__()
        if type(maximum_raw_bytes) is not int or not 0<maximum_raw_bytes<=16*1024**3:raise ValueError('EVIDENCE_READ_BOUND')
        self.stream=Path(path).open('rb');self.entries=[];self.starts=[];self.size=0;self.position=0;self.cache=None
        try:
            if self.stream.read(len(MAGIC))!=MAGIC:raise ValueError('EVIDENCE_MAGIC')
            prefix=self.stream.read(4)
            if len(prefix)!=4:raise ValueError('EVIDENCE_METADATA')
            length=struct.unpack('<I',prefix)[0]
            if not 0<length<=4096:raise ValueError('EVIDENCE_METADATA_BOUND')
            metadata=json.loads(self.stream.read(length))
            if metadata!=dict(run_id=run_id,logical_name=logical_name):raise ValueError('EVIDENCE_IDENTITY')
            digest=hashlib.sha256()
            while True:
                header=self.stream.read(HEADER.size)
                if len(header)!=HEADER.size:raise ValueError('EVIDENCE_INCOMPLETE')
                raw_size,compressed_size,expected=HEADER.unpack(header)
                if raw_size==0:
                    if compressed_size!=0 or digest.digest()!=expected or self.stream.read(1):raise ValueError('EVIDENCE_FOOTER')
                    self.raw_sha256=digest.hexdigest()
                    break
                if not 0<raw_size<=CHUNK or not 0<compressed_size<=CHUNK+65536 or self.size+raw_size>maximum_raw_bytes or len(self.entries)>=MAX_CHUNKS:
                    raise ValueError('EVIDENCE_CHUNK_BOUND')
                entry=(self.stream.tell(),compressed_size,raw_size,expected)
                raw=self._decode(entry);digest.update(raw)
                self.starts.append(self.size);self.entries.append(entry);self.size+=raw_size
        except BaseException:self.stream.close();raise

    def _decode(self,entry):
        offset,size,raw_size,expected=entry;self.stream.seek(offset);compressed=self.stream.read(size)
        decoder=zlib.decompressobj();raw=decoder.decompress(compressed,raw_size+1)
        if len(raw)!=raw_size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or hashlib.sha256(raw).digest()!=expected:
            raise ValueError('EVIDENCE_CHUNK_INTEGRITY')
        return raw

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=io.SEEK_SET):
        if whence not in (io.SEEK_SET,io.SEEK_CUR,io.SEEK_END):raise ValueError('EVIDENCE_SEEK_WHENCE')
        position=offset+({io.SEEK_SET:0,io.SEEK_CUR:self.position,io.SEEK_END:self.size}[whence])
        if position<0:raise ValueError('EVIDENCE_NEGATIVE_SEEK')
        self.position=position;return position

    def read(self,size=-1):
        if self.closed:raise ValueError('closed evidence reader')
        remaining=max(0,self.size-self.position)
        count=remaining if size is None or size<0 else min(size,remaining)
        result=bytearray()
        while count:
            i=bisect.bisect_right(self.starts,self.position)-1
            if self.cache is None or self.cache[0]!=i:self.cache=(i,self._decode(self.entries[i]))
            start=self.position-self.starts[i];n=min(count,len(self.cache[1])-start)
            result.extend(self.cache[1][start:start+n]);self.position+=n;count-=n
        return bytes(result)

    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)

    def close(self):
        if hasattr(self,'stream'):self.stream.close()
        super().close()
