"""Linux model-worker child ownership, including orphaned CLI descendants."""
import ctypes
import os
from pathlib import Path
import signal
import time
from rlflow.jobs import identity, owns


def become_subreaper():
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER')


def descendants(parent):
    parents={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            parents[int(path.name)]=int(fields[1])
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    found=set();front={parent}
    while front:
        front={pid for pid,ppid in parents.items() if ppid in front and pid not in found}
        found.update(front)
    return found


def cleanup(proc,*,timeout=15):
    """Subreaper owns adopted children; signals require exact live identities.

    Reap CLI only through Popen, preserving its native exit status. Other
    descendants are reaped only if this worker is their actual parent.
    """
    started=time.monotonic();empty=0;observed={};sent=set()
    while time.monotonic()-started<timeout:
        proc.poll()
        pids=descendants(os.getpid())
        live=[]
        for pid in pids:
            ident=identity(pid)
            if ident:
                observed[(pid,ident['starttime'])]=ident;live.append(ident)
            elif pid!=proc.pid:
                try:os.waitpid(pid,os.WNOHANG)
                except ChildProcessError:pass
        for ident in live:
            if owns(ident):
                try:os.kill(ident['pid'],signal.SIGKILL if time.monotonic()-started>3 else signal.SIGTERM)
                except ProcessLookupError:pass
                sent.add((ident['pid'],ident['starttime']))
        if not live and proc.poll() is not None:
            empty+=1
            if empty>=2:
                proc.wait()
                return {'status':'VERIFIED_NO_OWNED_DESCENDANTS','observed_identities':list(observed.values()),
                        'signalled_process_count':len(sent),'stable_empty_scans':empty}
        else:empty=0
        time.sleep(.05)
    raise RuntimeError('OWNED_DESCENDANT_CLEANUP_INCOMPLETE')
