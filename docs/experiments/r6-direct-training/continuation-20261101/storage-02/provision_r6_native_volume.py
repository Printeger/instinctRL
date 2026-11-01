"""Administrator-only creation of one persistent, size-bounded R6 output volume.

Default prints a plan. --apply requires root and a new identity; it never formats
an existing image, changes fstab, installs anything or touches frozen runtimes.
This bounds files INSIDE the mount, not other native write paths.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess

BASE=Path('/var/lib/instinctrl-r6-native')
SIZE=1610612736
FLOOR=30*1024**3


def plan(volume_id):
    if not re.fullmatch(r'(?:J-[0-9a-f]{16}|CPU-QUOTA-[0-9]{2})',volume_id):
        raise ValueError('EXACT_VOLUME_ID_REQUIRED')
    owner=pwd.getpwnam('mint')
    directory=BASE/volume_id;image=directory/'native.ext4';target=directory/'output'
    return dict(volume_id=volume_id,image=str(image),target=str(target),
        maximum_backing_bytes=SIZE,owner_uid=owner.pw_uid,owner_gid=owner.pw_gid,
        commands=[['/usr/bin/fallocate','--length',str(SIZE),str(image)],
                  ['/usr/sbin/mkfs.ext4','-q','-F','-m','0',str(image)],
                  ['/usr/bin/mount','-t','ext4','-o','loop,nodev,nosuid,noexec',str(image),str(target)]],
        limitations=['Only writes inside the mounted output directory are bounded',
            'CPU exhaustion, remount persistence and native-path containment not yet verified',
            'Each identity retains its image; no deletion, reuse, refund or automatic startup mount',
            'Account image plus metadata before administrator applies this plan'])


def secure_directory(path):
    info=path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid!=0 or info.st_mode & 0o022:
        raise RuntimeError('ROOT_OWNED_NON_WRITABLE_PARENT_REQUIRED: '+str(path))


def apply(spec):
    if os.geteuid()!=0:raise PermissionError('Administrator execution required; no automatic sudo')
    # Ancestors and provisioning lock cannot be replaced by the output owner.
    for parent in (Path('/var'),Path('/var/lib')):secure_directory(parent)
    if not BASE.exists():BASE.mkdir(mode=0o755)
    secure_directory(BASE)
    fd=os.open(BASE/'.provision.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        directory=Path(spec['image']).parent
        if directory.exists() or directory.is_symlink():raise FileExistsError('Existing identity preserved: '+str(directory))
        if shutil.disk_usage(BASE).free<SIZE+FLOOR+65536:raise RuntimeError('30_GIB_FREE_FLOOR')
        directory.mkdir(mode=0o755)
        image=Path(spec['image']);target=Path(spec['target'])
        # Unmounted target is inaccessible to mint, preventing silent writes to
        # the host filesystem if the mount is absent.
        target.mkdir(mode=0o000)
        with image.open('xb') as stream:stream.truncate(SIZE)
        os.chmod(image,0o600)
        (directory/'request.json').write_text(json.dumps(spec,indent=2)+'\n')
        for command in spec['commands']:
            subprocess.run(command,check=True,timeout=120,stdin=subprocess.DEVNULL)
        if not os.path.ismount(target):raise RuntimeError('EXPECTED_MOUNT_MISSING')
        if image.stat().st_size!=SIZE:raise RuntimeError('BACKING_SIZE_DRIFT')
        # Inventory must be able to enumerate the whole volume, including ext4's
        # system directory. Keep it root-owned and not writable by the workload.
        recovery=target/'lost+found'
        recovery_info=recovery.lstat()
        if not stat.S_ISDIR(recovery_info.st_mode) or recovery_info.st_uid!=0:
            raise RuntimeError('EXT4_RECOVERY_DIRECTORY_IDENTITY')
        os.chmod(recovery,0o555)
        os.chown(target,spec['owner_uid'],spec['owner_gid']);os.chmod(target,0o700)
        info=os.statvfs(target)
        if info.f_blocks*info.f_frsize>SIZE:raise RuntimeError('FILESYSTEM_EXCEEDS_IMAGE')
        result=dict(spec,status='MOUNT_CREATED_NOT_ACCEPTED',
            filesystem_total_bytes=info.f_blocks*info.f_frsize,
            filesystem_available_bytes=info.f_bavail*info.f_frsize,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        with (directory/'volume.json').open('x') as stream:
            json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('volume_id');parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();spec=plan(args.volume_id)
    print(json.dumps(apply(spec) if args.apply else spec,indent=2))


if __name__=='__main__':main()
