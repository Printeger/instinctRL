"""Model-free local-read check before charging an initial baseline review."""
import subprocess
from tools.r6_recovery import checked,RecoveryError


def check(executable,source_pin):
    exe=checked(executable);source=checked(source_pin)
    argv=[str(exe),'sandbox','-P',':read-only','--','/usr/bin/sha256sum','--',str(source)]
    try:
        result=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=15)
    except subprocess.TimeoutExpired as error:
        raise RecoveryError('REVIEW_LOCAL_READ_TIMEOUT_NO_MODEL_STARTED') from error
    lines=result.stdout.strip().splitlines()
    if result.returncode!=0 or len(lines)!=1 or lines[0].split(maxsplit=1)[0]!=source_pin['sha256']:
        detail=result.stderr[:1024].strip()
        raise RecoveryError('REVIEW_LOCAL_READ_UNAVAILABLE_NO_MODEL_STARTED: '+detail)
    return dict(status='LOCAL_READONLY_SOURCE_READ_VERIFIED',source=source_pin,model_calls=0)
