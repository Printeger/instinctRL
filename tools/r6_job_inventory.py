"""Read-only inventory of exact closed-job originals; no archive copies/refunds."""
import hashlib
import os
from pathlib import Path
import stat

from rlflow.store import atomic, canonical
from tools.r6_budget import observe
from tools.r6_job_handoff import close_terminal_job, read
from tools.r6_recovery import RecoveryError, checked, file_pin

DISK = 'output_actual_plus_unreleased_reservations_bytes'
MAX_FILES = 100000
MAX_MANIFEST_BYTES = 64 * 1024**2


def signature(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def inventory(roots):
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
        while stack:
            directory = stack.pop(); before = directory.lstat()
            if not stat.S_ISDIR(before.st_mode):
                raise RecoveryError('INVENTORY_DIRECTORY_TYPE')
            directories.append((directory, signature(before)))
            for path in sorted(directory.iterdir()):
                info = path.lstat()
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
    closure = close_terminal_job(journal, key, store)
    state = journal.read(); entry = state['events'][key]
    spec = read(checked(entry['request']['spec']))
    attempt = Path(spec['h1']['attempt'])
    # The exact submitted spec owns this attempt. Do not accept caller-selected
    # roots that could omit failed outputs or scan unrelated historical runs.
    expected_parent = store.root.resolve()/'attempts'
    if attempt.parent != expected_parent or not attempt.name or attempt.is_symlink():
        raise RecoveryError('INVENTORY_EXACT_ATTEMPT_ROOT')
    value = inventory({'job':store.root.resolve()/'jobs'/closure['job_id'], 'attempt':attempt})
    result = dict(schema_version='r6-job-inventory/1',event_key=key,job_id=closure['job_id'],
        closure=entry['closure'],spec=entry['request']['spec'],**value,
        output_inventory_complete=not value['missing_roots'],scientific_acceptance=False,
        native_status=closure['native_status'],native_exit_code=closure['native_exit_code'],
        metadata_accounting='Original job/attempt bytes only; controller metadata uses separate reserved capacity')
    data = canonical(result)+b'\n'
    if len(data) > MAX_MANIFEST_BYTES:
        raise RecoveryError('INVENTORY_MANIFEST_BOUND')
    path = journal.directory/'jobs'/key/'inventory.json'
    if entry.get('inventory'):
        if read(checked(entry['inventory'])) != result:
            raise RecoveryError('INVENTORY_REPLAY_DRIFT')
    elif path.exists():
        if read(path) != result:
            raise RecoveryError('INVENTORY_REPLAY_DRIFT')
    else:
        atomic(path,data)
    pin = file_pin(path)
    state['budget'] = observe(state['budget'],event_key=entry['identity'],
        resources={DISK:result['total_original_bytes']},evidence=pin)
    entry['inventory'] = pin
    violations = list(closure['budget_violations'])
    if result['total_original_bytes'] > entry['resources'].get(DISK, 0):
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
