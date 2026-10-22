"""Read-only validation of an exact administrator-created native output volume.

No mounting, permission issuance, runtime import or capacity acceptance. Test
volumes are explicitly excluded from production; checks apply to this mount now.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

BASE=Path('/var/lib/instinctrl-r6-native')
MAXIMUM=1610612736


def require(condition, reason):
    if not condition:raise ValueError('R6_NATIVE_VOLUME_'+reason)


def validate(pin, *, minimum_free_bytes=256*1024**2, allow_cpu_fixture=False, host_root_uid=0):
    return validate_volume(pin,base=BASE,maximum=MAXIMUM,image_name='native.ext4',
                           minimum_free_bytes=minimum_free_bytes,allow_cpu_fixture=allow_cpu_fixture,host_root_uid=host_root_uid)


def validate_volume(pin, *, base, maximum, image_name, expected_target=None,
                    minimum_free_bytes=256*1024**2, allow_cpu_fixture=False, host_root_uid=0):
    if host_root_uid != 0:
        mapping=[tuple(map(int,row.split())) for row in Path('/proc/self/uid_map').read_text().splitlines()]
        require(mapping==[(1000,1000,1)] and os.getuid()==1000
                and type(host_root_uid)is int
                and host_root_uid==int(Path('/proc/sys/kernel/overflowuid').read_text()),'HOST_ROOT_UID_MAPPING')
    require(type(minimum_free_bytes)is int and 0<=minimum_free_bytes<=maximum,'FREE_REQUIREMENT')
    require(isinstance(pin,dict) and set(pin)=={'path','sha256'},'PIN')
    manifest=Path(pin['path'])
    require(manifest.is_absolute() and manifest.resolve()==manifest and manifest.name=='volume.json'
            and manifest.parent.parent==base,'MANIFEST_PATH')
    require(hashlib.sha256(manifest.read_bytes()).hexdigest()==pin['sha256'],'MANIFEST_HASH')
    record=json.loads(manifest.read_text());name=manifest.parent.name
    require(bool(re.fullmatch(r'J-[0-9a-f]{16}',name)) or
            (allow_cpu_fixture and bool(re.fullmatch(r'CPU-QUOTA-[0-9]{2}',name))),'PRODUCTION_IDENTITY')
    target=manifest.parent/'output' if expected_target is None else Path(expected_target)
    image=manifest.parent/image_name
    require(record['volume_id']==name and record['image']==str(image)
            and record['target']==str(target) and record['maximum_backing_bytes']==maximum,'IDENTITY')
    for parent in (base.parent,base,manifest.parent):
        info=parent.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid==host_root_uid and not info.st_mode & 0o022,'PARENT_OWNERSHIP')
    for path in (manifest,image):
        info=path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid==host_root_uid and not info.st_mode & 0o022,'FILE_OWNERSHIP')
    require(image.stat().st_size==maximum,'BACKING_SIZE')
    require(target.resolve()==target and target.is_mount(),'MOUNT_ABSENT')
    info=target.stat()
    require(info.st_uid==record['owner_uid']==os.getuid() and stat.S_IMODE(info.st_mode)==0o700,'OUTPUT_OWNER')
    if not allow_cpu_fixture:
        recovery=(target/'lost+found').lstat()
        require(stat.S_ISDIR(recovery.st_mode) and recovery.st_uid==host_root_uid
                and stat.S_IMODE(recovery.st_mode)==0o555,'RECOVERY_DIRECTORY_INVENTORY_ACCESS')
    matches=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1);fields=left.split();fs=right.split()
        if fields[4]==str(target):matches.append((fields,fs))
    require(len(matches)==1,'MOUNT_IDENTITY')
    fields,fs=matches[0]
    require(fs[0]=='ext4' and {'rw','nosuid','nodev','noexec'}<=set(fields[5].split(',')),'MOUNT_OPTIONS')
    device=Path(fs[1]).name
    require(bool(re.fullmatch(r'loop[0-9]+',device)),'LOOP_DEVICE')
    require((Path('/sys/class/block')/device/'loop/backing_file').read_text().strip()==str(image),'LOOP_BACKING')
    usage=os.statvfs(target)
    require(usage.f_blocks*usage.f_frsize<=maximum,'FILESYSTEM_SIZE')
    require(usage.f_bavail*usage.f_frsize>=minimum_free_bytes,'FREE_SPACE')
    return dict(target=str(target),manifest=pin,mount_id=fields[0],device=fs[1],
                maximum_bytes=maximum,free_bytes=usage.f_bavail*usage.f_frsize)


def settings(binding):
    root=Path(binding['target'])
    return {'/crashreporter/dumpDir':str(root/'crash'),
            '/rtx/shaderDb/shaderCachePath':str(root/'shader'),
            '/rtx/shaderDb/driverShaderCachePath':str(root/'driver_shader'),
            '/rtx-transient/resourcemanager/localTextureCachePath':str(root/'texture'),
            # Installed extension search folders remain supplied by the frozen
            # experience. Never fetch missing dependencies into this output.
            '/app/extensions/registryEnabled':False,
            '/app/extensions/registryCacheFull':str(root/'extension_install'),
            '/app/extensions/registryCache':str(root/'extension_local'),
            '/app/userConfigPath':str(root/'user.config.json'),
            '/structuredLog/logDirectory':str(root/'telemetry'),
            '/telemetry/log/file':str(root/'telemetry/transmitter.log'),
            '/telemetry/testLogFile':str(root/'telemetry/transmitter-test.log'),
            '/telemetry/cloudLogEndpoint':(root/'telemetry/cloud.log').as_uri(),
            '/exts/omni.kit.pipapi/envPath':str(root/'pip_env'),
            '/exts/omni.kit.pipapi/allowOnlineIndex':False,
            '/exts/omni.kit.pipapi/tryUpgradePipOnFirstUse':False,
            '/exts/omni.kit.registry.nucleus/cachePath':str(root/'extension_cache'),
            '/exts/omni.kit.registry.nucleus/cachePrune/enabled':False,
            '/exts/omni.kit.registry.nucleus/cachePrune/runOnStartup':False}


def require_capture(binding,capture):
    target=Path(binding['target']);path=Path(capture['path'])
    require(path.is_absolute() and path.resolve()==path and path.is_relative_to(target),'CAPTURE_OUTSIDE_VOLUME')
    require(path.is_file() and path.stat().st_dev==target.stat().st_dev,'CAPTURE_DEVICE')
