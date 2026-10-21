"""Inherited, fail-closed Linux write restrictions for R6 native jobs.

Read/execute permissions and GPU ioctls are unchanged. Only declared output
roots and explicitly named character devices receive filesystem write access.
Apply in the held child after closing handshake descriptors, before exec.
"""
import ctypes
import os
from pathlib import Path
import platform
import stat

WRITE=(1<<15)-1 & ~(1|4|8)
FILE_WRITE=2|(1<<14)


def check(result,operation):
    if result<0:raise OSError(ctypes.get_errno(),'R6 write boundary: '+operation)
    return result


def prepare(roots,devices=()):
    if platform.machine()!='x86_64':raise ValueError('R6 write boundary requires tested x86_64 ABI')
    libc=ctypes.CDLL(None,use_errno=True)
    abi=check(libc.syscall(444,0,0,1),'query ABI')
    if abi<3:raise ValueError('R6 write boundary requires truncate support')
    entries=[]
    for raw in roots:
        p=Path(raw)
        if not p.is_absolute() or p.resolve()!=p or p==Path('/') or not p.is_dir():
            raise ValueError('R6 exact output root required')
        # Output roots cannot contain each other; caller binds exact job paths.
        if any(p.is_relative_to(Path(old)) or Path(old).is_relative_to(p) for old,_ in entries):
            raise ValueError('R6 overlapping output roots')
        entries.append((str(p),WRITE & ~((1<<6)|(1<<11))))
    for raw in devices:
        p=Path(raw)
        if not p.is_absolute() or p.resolve()!=p or not stat.S_ISCHR(p.stat().st_mode):
            raise ValueError('R6 character device required')
        entries.append((str(p),FILE_WRITE))
    if not entries:raise ValueError('R6 write boundary has no outputs')
    return dict(abi=abi,entries=entries)


def enforce(policy):
    # Pre-opened descriptors bypass Landlock. Only the bounded standard streams
    # survive into the executable; the dispatcher has already closed handshakes.
    for name in os.listdir('/proc/self/fd'):
        number=int(name)
        if number>2:
            try:os.close(number)
            except OSError:pass
    libc=ctypes.CDLL(None,use_errno=True)
    class Ruleset(ctypes.Structure):_fields_=[('handled_access_fs',ctypes.c_uint64)]
    class Beneath(ctypes.Structure):
        _pack_=1
        _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int)]
    attr=Ruleset(WRITE)
    fd=check(libc.syscall(444,ctypes.byref(attr),ctypes.sizeof(attr),0),'create ruleset')
    try:
        for path,rights in policy['entries']:
            parent=os.open(path,os.O_PATH|os.O_CLOEXEC|os.O_NOFOLLOW)
            try:
                rule=Beneath(rights,parent)
                check(libc.syscall(445,fd,1,ctypes.byref(rule),0),'add path')
            finally:os.close(parent)
        check(libc.prctl(38,1,0,0,0),'no_new_privs')
        check(libc.syscall(446,fd,0),'restrict self')
    finally:os.close(fd)
    # Landlock does not mediate metadata changes or process injection. Deny
    # these globally; no R6 workload requires changing existing file metadata.
    sec=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    sec.seccomp_init.argtypes=[ctypes.c_uint32];sec.seccomp_init.restype=ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];sec.seccomp_syscall_resolve_name.restype=ctypes.c_int
    sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    sec.seccomp_load.argtypes=[ctypes.c_void_p];sec.seccomp_release.argtypes=[ctypes.c_void_p]
    context=sec.seccomp_init(0x7fff0000)
    if not context:raise RuntimeError('R6 seccomp context unavailable')
    try:
        for name in ('chmod','fchmod','fchmodat','fchmodat2','chown','lchown','fchown','fchownat',
                     'utime','utimes','futimesat','utimensat','mount','umount2','pivot_root',
                     'unshare','setns','ptrace','process_vm_writev','pidfd_getfd','open_by_handle_at',
                     'setxattr','lsetxattr','fsetxattr','removexattr','lremovexattr','fremovexattr',
                     'io_uring_setup','io_uring_enter','io_uring_register'):
            number=sec.seccomp_syscall_resolve_name(name.encode())
            if number>=0 and sec.seccomp_rule_add(context,0x50000|1,number,0)!=0:
                raise RuntimeError('R6 seccomp rule failed: '+name)
        if sec.seccomp_load(context)!=0:raise RuntimeError('R6 seccomp load failed')
    finally:sec.seccomp_release(context)
