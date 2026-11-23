"""Bind newly reserved volumes without inheriting old storage acceptance."""
import json
import os
import re
import shutil
import stat
from pathlib import Path
from tools.r6_evidence import checked


SYSTEM_ROOT = Path.home() / 'instinctrl-runs'
SYSTEM_FLOOR = 30 * 1024**3
CONTROL_ALLOWANCE = 256 * 1024**2
SYSTEM_STORAGE_SCHEMA = 'instinctrl-system-storage/1'


def system_paths(plan, *, diagnostic=False):
    run_id = plan['run_id']
    if not re.fullmatch(r'R6-DIRECT-(THIRD|INSTINCT-CYCLE1|INSTINCT-NEXT-R[23])-[A-Z0-9-]+', run_id):
        raise ValueError('DIRECT_SYSTEM_RUN_ID')
    root = SYSTEM_ROOT / run_id
    if (Path(plan['output_root']) != root / 'control'
            or (not diagnostic and Path(plan['attempt_dir']) != root)
            or SYSTEM_ROOT.resolve() != SYSTEM_ROOT or root.resolve() != root):
        raise ValueError('DIRECT_SYSTEM_PATH')
    info = SYSTEM_ROOT.stat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o022 or info.st_dev != Path.home().stat().st_dev):
        raise ValueError('DIRECT_SYSTEM_ROOT')
    if root.exists():
        info = root.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('DIRECT_SYSTEM_ATTEMPT')
    return root


def system_native(plan, *, diagnostic=False):
    if (plan.get('system_results') is not True or plan.get('system_native') is not True
            or 'native_volume' in plan or 'result_volume' in plan):
        raise ValueError('DIRECT_ORDINARY_SYSTEM_STORAGE_REQUIRED')
    root = system_paths(plan, diagnostic=diagnostic)
    native = root / 'native'
    if Path(plan['native_dir']) != native or native.resolve() != native:
        raise ValueError('DIRECT_SYSTEM_NATIVE_PATH')
    system_device = Path('/').stat().st_dev
    mount_targets = {row.partition(' - ')[0].split()[4]
                     for row in Path('/proc/self/mountinfo').read_text().splitlines()}
    for path in (SYSTEM_ROOT, root, native):
        mount_name = re.sub(r'[\t\n \\]', lambda match: f'\\{ord(match[0]):03o}', str(path))
        if path.exists() and (mount_name in mount_targets or path.stat().st_dev != system_device):
            raise ValueError('DIRECT_SYSTEM_NATIVE_DEVICE')
    if native.exists():
        info = native.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('DIRECT_SYSTEM_NATIVE_PERMISSIONS')
    return native


def system_result(plan, contract, *, before_launch=False):
    root = system_paths(plan)
    cap = plan['result_payload_cap_bytes']
    if type(cap) is not int or not 256 * 1024**2 <= cap <= contract['parameters']['output_cap_bytes']:
        raise ValueError('DIRECT_SYSTEM_PAYLOAD')
    free = shutil.disk_usage(root.parent).free
    if free < SYSTEM_FLOOR + (cap + CONTROL_ALLOWANCE if before_launch else 0):
        raise ValueError('DIRECT_SYSTEM_DISK_FLOOR')
    if before_launch and os.statvfs(root.parent).f_favail < 4096:
        raise ValueError('DIRECT_SYSTEM_INODES')
    return dict(target=str(root.parent), payload_cap_bytes=cap, free_bytes=free)


def evidence_bytes(*directories):
    """Account retained bytes conservatively, including sparse/logical files."""
    total = 0
    paths = {Path(directory) for directory in directories}
    for directory in paths:
        if any(parent in paths for parent in directory.parents):
            continue
        for path in (directory, *directory.rglob('*')):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode):
                total += max(info.st_size, info.st_blocks * 512)
    return total


def fresh_binding(plan, ledger, contract, *, third=False):
    identity=plan['storage_reservation_id']
    rows=[r for r in ledger['new_reservations'] if r['id']==identity]
    if len(rows)!=1:raise ValueError('DIRECT_STORAGE_RESERVATION')
    row=rows[0]
    if row['status']!='UNRELEASED':raise ValueError('DIRECT_STORAGE_RESERVATION_STATUS')
    spec=json.loads(checked(row['storage_plan']).read_text())
    if plan.get('system_results'):
        from tools.r6_direct.resource_policy import OUTPUT_CAP
        if (not third or spec.get('root') != str(SYSTEM_ROOT)
                or row['resources'] != {OUTPUT_CAP: 0} or Path(plan['attempt_dir']).exists()):
            raise ValueError('DIRECT_SYSTEM_STORAGE_BINDING')
        if plan.get('system_native') is True:
            if spec.get('schema_version') != SYSTEM_STORAGE_SCHEMA or spec.get('new_volume_allocation_bytes') != 0:
                raise ValueError('DIRECT_SYSTEM_STORAGE_BINDING')
            system_native(plan)
        elif (spec.get('schema_version') != 'instinctrl-third-system-storage/1'
                or plan['native_volume'] not in spec['native_manifests']):
            raise ValueError('DIRECT_SYSTEM_STORAGE_BINDING')
        system_result(plan,contract,before_launch=True)
        return identity
    reuse=spec.get('schema_version')=='instinctrl-third-storage-reuse/1'
    if reuse:
        from tools.r6_direct.resource_policy import OUTPUT_CAP, THIRD_PREFIX
        if (not third or not plan['run_id'].startswith(THIRD_PREFIX)
                or spec.get('new_volume_allocation_bytes')!=0
                or row['resources']!={OUTPUT_CAP:0}
                or any(type(plan.get(key)) is not int or plan[key]<=0
                       for key in ('result_payload_cap_bytes','native_minimum_free_bytes'))):
            raise ValueError('DIRECT_STORAGE_REUSE_SCOPE')
    total=0
    for kind,destination in (('native','native_dir'),('result','attempt_dir')):
        actual=json.loads(checked(plan[kind+'_volume']).read_text())
        expected=spec[kind]
        if reuse and expected['manifest']!=plan[kind+'_volume']:
            raise ValueError('DIRECT_STORAGE_REUSE_MANIFEST')
        for key in ('volume_id','target','image','maximum_backing_bytes','owner_uid','owner_gid'):
            if actual[key]!=expected[key]:raise ValueError('DIRECT_STORAGE_MANIFEST:'+key)
        if Path(plan[destination])!=Path(expected['target'])/plan['run_id']:
            raise ValueError('DIRECT_STORAGE_DESTINATION')
        total+=expected['maximum_backing_bytes']
    if not reuse and row['resources']['output_actual_plus_unreleased_reservations_bytes']<total:
        raise ValueError('DIRECT_STORAGE_UNACCOUNTED')
    # Control metadata allowance is distinct from the two backing images.
    if contract['parameters']['output_cap_bytes']!=total+64*1024**2:
        raise ValueError('DIRECT_STORAGE_CAP')
    return identity
