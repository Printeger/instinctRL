"""Read-only inventory of exact closed-job originals; no archive copies/refunds."""
import hashlib
import os
from pathlib import Path
import stat

from rlflow.store import atomic, canonical, sha
from tools.r6_budget import observe
from tools.r6_job_handoff import close_terminal_job, read, exact_job, cleanup_evidence, TERMINAL
from tools.r6_recovery import RecoveryError, checked, file_pin

DISK = 'output_actual_plus_unreleased_reservations_bytes'
MAX_FILES = 100000
MAX_MANIFEST_BYTES = 64 * 1024**2


def signature(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def inventory(roots, *, same_device_roots=()):
    """Hash all regular files, checking directory/file stability and special files.

    Hard links count once per pathname conservatively. Missing roots are explicit,
    never silently presented as complete evidence. Roots must not overlap.
    """
    resolved = {}
    for name, root in roots.items():
        root = Path(root)
        if not root.is_absolute() or root.is_symlink():
            raise RecoveryError('INVENTORY_ROOT_PATH')
        if root.resolve() != root:
            raise RecoveryError('INVENTORY_ROOT_ALIAS')
        resolved[name] = root
    paths = list(resolved.values())
    if any(a == b or a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[i+1:]):
        raise RecoveryError('INVENTORY_ROOT_OVERLAP')
    files = []; directories = []; checked_files = []; missing = []
    for name, root in sorted(resolved.items()):
        if not root.exists():
            missing.append(name)
            continue
        stack = [root]
        device=root.stat().st_dev
        while stack:
            directory = stack.pop(); before = directory.lstat()
            if not stat.S_ISDIR(before.st_mode):
                raise RecoveryError('INVENTORY_DIRECTORY_TYPE')
            directories.append((directory, signature(before)))
            for path in sorted(directory.iterdir()):
                info = path.lstat()
                if name in same_device_roots and info.st_dev!=device:
                    raise RecoveryError('INVENTORY_CROSS_DEVICE')
                if stat.S_ISDIR(info.st_mode):
                    stack.append(path)
                elif stat.S_ISREG(info.st_mode):
                    if len(files) >= MAX_FILES:
                        raise RecoveryError('INVENTORY_FILE_COUNT_BOUND')
                    digest = hashlib.sha256()
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, 'rb') as stream:
                        opened = os.fstat(stream.fileno())
                        if signature(opened) != signature(info):
                            raise RecoveryError('INVENTORY_FILE_CHANGED')
                        for block in iter(lambda: stream.read(1024**2), b''):
                            digest.update(block)
                        if signature(os.fstat(stream.fileno())) != signature(info):
                            raise RecoveryError('INVENTORY_FILE_CHANGED')
                    checked_files.append((path, signature(info)))
                    files.append(dict(category=name, relative=str(path.relative_to(root)),
                        path=str(path), bytes=info.st_size, sha256=digest.hexdigest()))
                else:
                    raise RecoveryError('INVENTORY_SYMLINK_OR_SPECIAL_FILE')
                # Covers replacement/deletion after reading, not just changes to
                # an open inode that was unlinked while being hashed.
                if signature(path.lstat()) != signature(info):
                    raise RecoveryError('INVENTORY_FILE_CHANGED')
    for path, before in checked_files:
        if signature(path.lstat()) != before:
            raise RecoveryError('INVENTORY_FILE_CHANGED')
    for directory, before in directories:
        if signature(directory.lstat()) != before:
            raise RecoveryError('INVENTORY_DIRECTORY_CHANGED')
    if any(resolved[name].exists() for name in missing):
        raise RecoveryError('INVENTORY_MISSING_ROOT_APPEARED')
    files.sort(key=lambda row:(row['category'],row['relative']))
    return dict(roots={k:str(v) for k,v in resolved.items()},files=files,missing_roots=missing,
        total_original_bytes=sum(row['bytes'] for row in files),
        total_archive_copy_bytes=0,hardlink_accounting='Each pathname charged; no deduplication refund')


def record_inventory(journal, key, store):
    """Attach exact output observations atomically to the closed-job journal."""
    journal.require_writer()
    state = journal.read(); entry = state['events'][key]
    if state.get('current_job') not in (None, entry['identity']['job_id']):
        raise RecoveryError('OTHER_JOB_HANDOFF_ACTIVE')
    if entry.get('inventory'):
        # Rechecking originals must not rewind a later analysis/decision phase,
        # including when drift is detected part way through the verification.
        entry = journal.canonical_job(key,state=state)
        closure = read(checked(entry['closure']))
        spec = read(checked(entry['request']['spec']))
        job = exact_job(store.replay(),entry,spec)
        if job['status'] not in TERMINAL:raise RecoveryError('JOB_NOT_TERMINAL')
        pins,receipt = cleanup_evidence(store.root/'jobs'/job['id'],job)
        expected = dict(event_key=key,request_sha256=sha(entry['request']),job_id=job['id'],
            spec_sha256=sha(spec),evidence=pins,native_status=receipt['status'],
            native_exit_code=receipt.get('exit_code'),reservation_retained=entry['resources'],
            h1_evidence=job.get('h1_evidence'),scientific_acceptance=False)
        if (entry['identity']['input_sha256']!=sha(entry['request'])
                or any(closure.get(k)!=v for k,v in expected.items())):
            raise RecoveryError('JOB_CLOSURE_DRIFT')
    else:
        closure = close_terminal_job(journal, key, store)
    state = journal.read(); entry = state['events'][key]
    spec = read(checked(entry['request']['spec']))
    attempt = Path(spec['h1']['attempt'])
    # The exact submitted spec owns this attempt. Do not accept caller-selected
    # roots that could omit failed outputs or scan unrelated historical runs.
    expected_parent = store.root.resolve()/'attempts'
    if attempt.parent != expected_parent or not attempt.name or attempt.is_symlink():
        raise RecoveryError('INVENTORY_EXACT_ATTEMPT_ROOT')
    roots={'job':store.root.resolve()/'jobs'/closure['job_id'], 'attempt':attempt}
    native=None;binding=None
    if spec.get('h1',{}).get('mode')=='research-r6-evaluation':
        authorization=read(checked(spec['pins']['authorization']))
        native_pin=authorization['parameters'].get('native_volume')
        if native_pin is not None:
            from training_workloads.unified_h1.native_volume import validate
            binding=validate(native_pin,minimum_free_bytes=0)
            roots['native']=Path(binding['target'])
            manifest=checked(native_pin)
            request=manifest.parent/'request.json'
            if request.is_symlink() or not request.is_file() or request.stat().st_size>65536:
                raise RecoveryError('INVENTORY_NATIVE_REQUEST')
            native=dict(manifest=native_pin,provision_request=file_pin(request),
                target=binding['target'],backing_bytes=binding['maximum_bytes'],
                metadata_bytes=manifest.stat().st_size+request.stat().st_size)
    value = inventory(roots,same_device_roots=('native',) if native else ())
    if native is not None:
        after=validate(native['manifest'],minimum_free_bytes=0)
        if any(after[k]!=binding[k] for k in ('mount_id','device','target','maximum_bytes')):
            raise RecoveryError('INVENTORY_NATIVE_MOUNT_CHANGED')
        checked(native['provision_request'])
        value['native_volume']=native
        # Files inside the image are hashed originals, but charging both their
        # logical bytes and the image would double-count the same storage.
        value['total_charged_output_bytes']=(sum(row['bytes'] for row in value['files'] if row['category']!='native')
            +native['backing_bytes']+native['metadata_bytes'])
    result = dict(schema_version='r6-job-inventory/1',event_key=key,job_id=closure['job_id'],
        closure=entry['closure'],spec=entry['request']['spec'],**value,
        output_inventory_complete=not value['missing_roots'],scientific_acceptance=False,
        native_status=closure['native_status'],native_exit_code=closure['native_exit_code'],
        metadata_accounting=('Native file contents inventoried; fixed backing image plus provisioning metadata charged once. Controller metadata separately reserved'
            if native is not None else 'Original job/attempt bytes only; controller metadata uses separate reserved capacity'))
    data = canonical(result)+b'\n'
    if len(data) > MAX_MANIFEST_BYTES:
        raise RecoveryError('INVENTORY_MANIFEST_BOUND')
    path = journal.directory/'jobs'/key/'inventory.json'
    if entry.get('inventory'):
        if read(checked(entry['inventory'])) != result:
            raise RecoveryError('INVENTORY_REPLAY_DRIFT')
        return result
    elif path.exists():
        if read(path) != result:
            raise RecoveryError('INVENTORY_REPLAY_DRIFT')
    else:
        atomic(path,data)
    pin = file_pin(path)
    charged_bytes=result.get('total_charged_output_bytes',result['total_original_bytes'])
    state['budget'] = observe(state['budget'],event_key=entry['identity'],
        resources={DISK:charged_bytes},evidence=pin)
    entry['inventory'] = pin
    violations = list(closure['budget_violations'])
    if charged_bytes > entry['resources'].get(DISK, 0):
        violations.append('RESERVATION_EXCEEDED_'+DISK)
    entry['inventory_violations'] = violations
    if state['budget'].get('observed_cap_excess') or violations:
        state['status'] = 'JOB_BOUND_EXCEEDED'
    elif entry.get('stop_seen'):
        state['status'] = 'STOPPED_AFTER_OWNED_CLEANUP'
    elif entry.get('deadline_seen'):
        state['status'] = 'JOB_BOUND_EXCEEDED'
    elif not result['output_inventory_complete']:
        state['status'] = 'JOB_EVIDENCE_INCOMPLETE'
    else:
        state['status'] = 'INVENTORIED_PENDING_POSTRUN_AND_ANALYSIS'
    journal.save(state)
    return result
