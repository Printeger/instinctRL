"""Administrator creation of a NEW fixed-layout result volume at an exact R6 attempt.

Standalone successor: preserve the accepted CPU02 provisioning tool unchanged.

Default is a plan only. Caller must reserve the full backing image first. No
reuse, cleanup, runtime changes, fstab changes or automatic sudo.
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

BASE=Path('/var/lib/instinctrl-r6-results')
ATTEMPTS=Path('/home/mint/rl_dev/instinctRL-dev/coordination/research/attempts')
FLOOR=30*1024**3


def plan(volume_id,target,size,minimum_free_bytes,minimum_free_inodes):
    target=Path(target)
    if not re.fullmatch(r'(?:J-[0-9a-f]{16}|CPU-QUOTA-[0-9]{2})',volume_id):raise ValueError('EXACT_RESULT_VOLUME_ID')
    if (not target.is_absolute() or target.resolve()!=target or target.name!='results'
        or target.parent.parent!=ATTEMPTS or not re.fullmatch(r'R6-[A-Za-z0-9_-]+',target.parent.name)):
        raise ValueError('EXACT_R6_RESULT_TARGET')
    if type(size)is not int or not 16*1024**2<=size<=128*1024**3 or size%4096:
        raise ValueError('ALIGNED_RESULT_CAP_REQUIRED')
    if (type(minimum_free_bytes)is not int or minimum_free_bytes<=0
        or size-minimum_free_bytes!=96*1024**2 or size<128*1024**2):
        raise ValueError('FIXED_RESULT_PAYLOAD_AND_96MIB_OVERHEAD_REQUIRED')
    if type(minimum_free_inodes)is not int or not 1<=minimum_free_inodes<=size//262144:
        raise ValueError('FIXED_RESULT_INODE_REQUIREMENT')
    owner=pwd.getpwnam('mint');image=BASE/volume_id/'results.ext4'
    return dict(volume_id=volume_id,target=str(target),image=str(image),maximum_backing_bytes=size,
        owner_uid=owner.pw_uid,owner_gid=owner.pw_gid,layout='fixed-results-v1',
        minimum_free_bytes=minimum_free_bytes,minimum_free_inodes=minimum_free_inodes,
        commands=[['/usr/bin/fallocate','--length',str(size),str(image)],
            ['/usr/sbin/mkfs.ext4','-q','-F','-m','0','-b','4096','-I','256',
             '-i','262144','-O','^resize_inode','-J','size=32',str(image)],
            ['/usr/bin/mount','-t','ext4','-o','loop,nodev,nosuid,noexec',str(image),str(target)]])


def secure(path):
    s=path.lstat()
    if not stat.S_ISDIR(s.st_mode) or s.st_uid!=0 or s.st_mode & 0o022:
        raise ValueError('ROOT_OWNED_VOLUME_PARENT_REQUIRED')


def capacity_fields(usage,spec):
    measurements=dict(filesystem_total_bytes=usage.f_blocks*usage.f_frsize,
        filesystem_available_bytes=usage.f_bavail*usage.f_frsize,
        filesystem_inodes=usage.f_files,filesystem_available_inodes=usage.f_favail)
    enough=(measurements['filesystem_available_bytes']>=spec['minimum_free_bytes']
            and measurements['filesystem_available_inodes']>=spec['minimum_free_inodes'])
    return dict(**measurements,status='MOUNT_CREATED_NOT_ACCEPTED' if enough else 'MOUNT_CAPACITY_FAILED_PRESERVED')


def apply(spec):
    if os.geteuid()!=0:raise PermissionError('Administrator execution required; no automatic sudo')
    if spec!=plan(spec['volume_id'],spec['target'],spec['maximum_backing_bytes'],
                  spec['minimum_free_bytes'],spec['minimum_free_inodes']):raise ValueError('PLAN_DRIFT')
    for p in (Path('/var'),Path('/var/lib')):secure(p)
    if not BASE.exists():BASE.mkdir(mode=0o755)
    secure(BASE)
    fd=os.open(BASE/'.provision.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        image=Path(spec['image']);directory=image.parent;target=Path(spec['target']);size=spec['maximum_backing_bytes']
        if directory.exists() or directory.is_symlink():raise FileExistsError('Existing result identity preserved')
        if (target.is_symlink() or not target.is_dir() or target.is_mount()
            or target.stat().st_uid!=spec['owner_uid'] or list(target.iterdir())):
            raise ValueError('FRESH_EMPTY_OWNED_RESULT_TARGET_REQUIRED')
        if shutil.disk_usage(BASE).free<size+FLOOR+65536:raise ValueError('30_GIB_FREE_FLOOR')
        directory.mkdir(mode=0o755)
        # Hold the parent and target by descriptors: never chmod/chown a
        # caller-writable path after validation, and mount onto the held inode.
        parent_fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        target_fd=-1;mounted_fd=-1
        try:
            if Path(os.readlink(f'/proc/self/fd/{parent_fd}'))!=target.parent:
                raise ValueError('RESULT_PARENT_CHANGED')
            target_fd=os.open('results',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent_fd)
            if os.listdir(target_fd) or os.fstat(target_fd).st_uid!=spec['owner_uid']:
                raise ValueError('RESULT_TARGET_CHANGED')
            os.fchown(target_fd,0,0);os.fchmod(target_fd,0o000)
            if os.listdir(target_fd):raise ValueError('RESULT_TARGET_NOT_EMPTY_AFTER_LOCKDOWN')
            with image.open('xb') as stream:stream.truncate(size)
            os.chmod(image,0o600)
            (directory/'request.json').write_text(json.dumps(spec,indent=2)+'\n')
            for command in spec['commands'][:-1]:
                subprocess.run(command,check=True,timeout=120,stdin=subprocess.DEVNULL)
            mount_command=[*spec['commands'][-1][:-1],f'/proc/self/fd/{target_fd}']
            subprocess.run(mount_command,check=True,timeout=120,stdin=subprocess.DEVNULL,pass_fds=(target_fd,))
            mounted_fd=os.open('results',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent_fd)
            if (Path(os.readlink(f'/proc/self/fd/{mounted_fd}'))!=target or not target.is_mount()
                or os.fstat(mounted_fd).st_dev==os.fstat(target_fd).st_dev or image.stat().st_size!=size):
                raise ValueError('RESULT_MOUNT_IDENTITY')
            recovery_fd=os.open('lost+found',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=mounted_fd)
            try:
                if os.fstat(recovery_fd).st_uid!=0:raise ValueError('RESULT_RECOVERY_DIRECTORY')
                os.fchmod(recovery_fd,0o555)
            finally:os.close(recovery_fd)
            os.fchown(mounted_fd,spec['owner_uid'],spec['owner_gid']);os.fchmod(mounted_fd,0o700)
            usage=os.fstatvfs(mounted_fd)
            if usage.f_blocks*usage.f_frsize>size:raise ValueError('RESULT_FILESYSTEM_SIZE')
            result=dict(spec,**capacity_fields(usage,spec),mount_target_via_held_descriptor=True,
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            with (directory/'volume.json').open('x') as stream:
                json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            if result['status']=='MOUNT_CAPACITY_FAILED_PRESERVED':raise ValueError('MOUNT_CAPACITY_FAILED_PRESERVED')
            return result
        finally:
            for descriptor in (mounted_fd,target_fd,parent_fd):
                if descriptor>=0:os.close(descriptor)



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('volume_id');parser.add_argument('target',type=Path);parser.add_argument('bytes',type=int)
    parser.add_argument('--minimum-free-bytes',type=int,required=True)
    parser.add_argument('--minimum-free-inodes',type=int,required=True)
    parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    spec=plan(args.volume_id,args.target,args.bytes,args.minimum_free_bytes,args.minimum_free_inodes)
    print(json.dumps(apply(spec) if args.apply else spec,indent=2))
