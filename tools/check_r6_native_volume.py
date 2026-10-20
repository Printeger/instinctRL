"""Bounded CPU-QUOTA-01 aggregate ENOSPC check. Never deletes or reuses files."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
VOLUME=Path('/var/lib/instinctrl-r6-native/CPU-QUOTA-01')
TARGET=VOLUME/'output'
SIZE=1610612736
CHUNK=1024**2
PER_FILE=64*CHUNK


def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def inspect():
    record=json.loads((VOLUME/'volume.json').read_text())
    assert record['target']==str(TARGET) and record['maximum_backing_bytes']==SIZE
    assert record['implementation_sha256']==pin(ROOT/'tools/provision_r6_native_volume.py')['sha256']
    assert TARGET.is_mount() and not TARGET.is_symlink()
    mount=json.loads(subprocess.check_output(['/usr/bin/findmnt','-J','-T',str(TARGET),
        '-o','TARGET,SOURCE,FSTYPE,OPTIONS'],text=True))['filesystems'][0]
    assert mount['target']==str(TARGET) and mount['fstype']=='ext4'
    assert {'rw','nodev','nosuid','noexec'}<=set(mount['options'].split(','))
    device=Path(mount['source']).name
    assert device.startswith('loop') and device[4:].isdigit()
    backing=Path('/sys/class/block')/device/'loop/backing_file'
    assert backing.read_text().strip()==record['image']
    stat=os.statvfs(TARGET)
    assert stat.f_blocks*stat.f_frsize<=SIZE
    assert TARGET.stat().st_uid==os.getuid()
    return dict(volume=pin(VOLUME/'volume.json'),mount=mount,backing_file=backing.read_text().strip(),
        filesystem_total_bytes=stat.f_blocks*stat.f_frsize,
        available_bytes=stat.f_bavail*stat.f_frsize)


def verify(result_path):
    result=json.loads(result_path.read_text());current=inspect()
    assert result['source']==pin(Path(__file__).resolve())
    assert current['volume']==result['before']['volume']
    for entry in result['files']:
        path=Path(entry['path'])
        assert path.parent==TARGET/'aggregate-test' and path.stat().st_size==entry['bytes']
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while data:=stream.read(CHUNK):digest.update(data)
        assert digest.hexdigest()==entry['sha256']
    # Fresh process attempts to extend the final existing file, without deleting
    # anything or relying on the original writer's cached exception.
    last=Path(result['files'][-1]['path']);before=last.stat().st_size
    fd=os.open(last,os.O_WRONLY|os.O_APPEND)
    try:
        try:os.write(fd,b'x'*CHUNK);os.fsync(fd)
        except OSError as error:
            assert error.errno==errno.ENOSPC
        else:raise AssertionError('FULL_VOLUME_ACCEPTED_MORE_DATA')
    finally:os.close(fd)
    assert last.stat().st_size==before
    return dict(status='FRESH_PROCESS_HASH_AND_ENOSPC_PASS',files_verified=len(result['files']),
                bytes_verified=sum(e['bytes'] for e in result['files']),mount=current,
                remount_or_reboot_tested=False)


def fill(result_path):
    before=inspect();directory=TARGET/'aggregate-test'
    if result_path.exists():raise FileExistsError('Result must be new')
    directory.mkdir()  # Existing artifacts are never reset or silently reused.
    chunk=bytes(range(256))*(CHUNK//256)
    files=[];total=0;failure=None
    for index in range(SIZE//PER_FILE+1):
        path=directory/f'{index:03d}.bin';written=0;digest=hashlib.sha256()
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        try:
            while written<PER_FILE:
                if total+CHUNK>SIZE:raise RuntimeError('STOP_AT_ABSOLUTE_TEST_WRITE_CEILING')
                try:
                    n=os.write(fd,chunk);os.fsync(fd)
                except OSError as error:
                    if error.errno!=errno.ENOSPC:raise
                    failure=dict(errno=error.errno,name='ENOSPC',file=str(path));break
                if n<=0:raise RuntimeError('NO_WRITE_PROGRESS')
                digest.update(chunk[:n]);written+=n;total+=n
        finally:os.close(fd)
        # Hash actual file contents after a potential delayed-allocation failure.
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(dict(path=str(path),bytes=path.stat().st_size,sha256=actual))
        if failure:break
    assert failure and len(files)>1
    assert all(e['bytes']<=PER_FILE for e in files)
    assert sum(e['bytes'] for e in files)<=SIZE
    result=dict(status='MULTIFILE_AGGREGATE_ENOSPC_PASS',before=before,after=inspect(),
        files=files,failure=failure,source=pin(Path(__file__).resolve()),
        maximum_individual_file_bytes=PER_FILE,write_ceiling_bytes=SIZE,
        capacity_pass=False,native_path_containment_verified=False,remount_tested=False,
        gpu_queries=0,simulator_initializations=0,files_deleted=0)
    with result_path.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inspect','fill','verify'])
    parser.add_argument('--result',type=Path)
    args=parser.parse_args()
    if args.action!='inspect' and args.result is None:parser.error('--result required')
    output=inspect() if args.action=='inspect' else (fill(args.result) if args.action=='fill' else verify(args.result))
    print(json.dumps(output,indent=2))
