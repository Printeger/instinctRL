"""One-shot CPU proof of multi-file and descendant aggregate result quota.

Requires an administrator-created CPU-QUOTA-02 16MiB result volume. Preserves
all originals, never cleans/reuses it, imports no simulator or GPU modules.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'coordination/research/attempts/R6-CPU-RESULT-QUOTA-02/results'
MANIFEST=Path('/var/lib/instinctrl-r6-results/CPU-QUOTA-02/volume.json')
OUTPUT=ROOT/'docs/experiments/r6-robustness-auto/RESULT_VOLUME_CPU02'
SIZE=16*1024**2
CHUNK=1024**2


def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def originals():
    return [dict(**pin(p),bytes=p.stat().st_size) for p in sorted(TARGET.glob('cpu-multifile-*.bin'))]


def restrict():
    sys.path.insert(0,str(ROOT))
    from training_workloads.unified_h1.write_boundary import prepare,enforce
    enforce(prepare([TARGET],['/dev/null']))


def fill():
    restrict();total=0;denied=None
    # Bypass every cooperative workload writer: ordinary open/write syscalls.
    for i in range(9):
        path=TARGET/f'cpu-multifile-{i:02d}.bin'
        fd=-1
        try:
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            for _ in range(2):
                chunk=b'x'*CHUNK;offset=0
                while offset<len(chunk):
                    count=os.write(fd,chunk[offset:]);assert count>0
                    offset+=count;total+=count
            os.fsync(fd)
        except OSError as error:
            if error.errno not in (errno.ENOSPC,errno.EDQUOT):raise
            denied=error.errno
        finally:
            if fd>=0:os.close(fd)
        if denied is not None:break
    assert denied is not None and 2*CHUNK<total<=SIZE
    # The exec child inherits the kernel write policy, no new policy applied.
    child=subprocess.run([sys.executable,'-I','-B',str(Path(__file__).resolve()),'append'],
                         capture_output=True,text=True,timeout=20)
    assert child.returncode==0,child.stderr
    print(json.dumps(dict(bytes_written=total,aggregate_errno=denied,descendant=json.loads(child.stdout))))


def append():
    paths=sorted(TARGET.glob('cpu-multifile-*.bin'));assert paths
    fd=os.open(paths[-1],os.O_WRONLY|os.O_APPEND)
    try:
        try:os.write(fd,b'y'*CHUNK)
        except OSError as error:
            assert error.errno in (errno.ENOSPC,errno.EDQUOT)
            print(json.dumps(dict(append_errno=error.errno,requested_bytes=CHUNK)));return
        raise AssertionError('Exhausted volume accepted descendant append')
    finally:os.close(fd)


def main():
    OUTPUT.mkdir(exist_ok=False)
    report=dict(status='STARTED',capacity_pass=False,gpu_queries=0,live_starts=0)
    try:
        sys.path.insert(0,str(ROOT))
        from training_workloads.unified_h1.result_volume import validate,MAXIMUM,CONTROL_ALLOWANCE
        p=pin(MANIFEST);record=json.loads(MANIFEST.read_text())
        assert record['volume_id']=='CPU-QUOTA-02' and record['maximum_backing_bytes']==SIZE
        assert record['implementation_sha256']==pin(ROOT/'tools/provision_r6_result_volume.py')['sha256']
        binding=validate(p,output_cap_bytes=MAXIMUM+CONTROL_ALLOWANCE+SIZE,
            expected_target=TARGET,minimum_free_bytes=0,allow_cpu_fixture=True)
        assert not originals()
        report.update(manifest=p,binding=binding,tool=pin(Path(__file__).resolve()))
        child=subprocess.run([sys.executable,'-I','-B',str(Path(__file__).resolve()),'fill'],
                             capture_output=True,text=True,timeout=60)
        (OUTPUT/'fill.stdout').write_text(child.stdout);(OUTPUT/'fill.stderr').write_text(child.stderr)
        assert child.returncode==0,child.stderr
        report['fill']=json.loads(child.stdout);before=originals()
        assert len(before)>1 and all(item['bytes']<=2*CHUNK for item in before)
        # A separate interpreter also sees exhaustion; preserve all original hashes.
        fresh=subprocess.run([sys.executable,'-I','-B',str(Path(__file__).resolve()),'append'],
                             capture_output=True,text=True,timeout=20)
        (OUTPUT/'reopen.stdout').write_text(fresh.stdout);(OUTPUT/'reopen.stderr').write_text(fresh.stderr)
        assert fresh.returncode==0,fresh.stderr
        assert originals()==before
        after=validate(p,output_cap_bytes=MAXIMUM+CONTROL_ALLOWANCE+SIZE,
            expected_target=TARGET,minimum_free_bytes=0,allow_cpu_fixture=True)
        assert all(after[k]==binding[k] for k in ('mount_id','device','target','maximum_bytes'))
        report.update(status='PASS_CPU_MULTI_FILE_DESCENDANT_AND_REOPEN',originals=before,
            reopen=json.loads(fresh.stdout),backing_bytes=SIZE,
            limitations=['16MiB CPU ext4 fixture, not production result or native volume acceptance',
                         '1MiB append granularity, not proof every last allocatable byte is consumed',
                         'No remount/reboot or Kit/CUDA execution'])
    except BaseException as error:
        report.update(status='FAIL_PRESERVED',error=repr(error),originals=originals());raise
    finally:
        (OUTPUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_FSIZE,(4*CHUNK,4*CHUNK))
    mode=sys.argv[1] if len(sys.argv)>1 else 'run'
    {'run':main,'fill':fill,'append':append}[mode]()
