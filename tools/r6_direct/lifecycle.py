"""Local writer ownership; archived scheduler state is not a launch condition."""
from contextlib import ExitStack, contextmanager
import fcntl
from tools.r6_direct.import_legacy import NEW, OLD, process_identity, stop_paths


@contextmanager
def writer(directory=NEW, legacy_directory=OLD):
    if stop_paths():
        raise RuntimeError('STOP')
    with ExitStack() as stack:
        lock = stack.enter_context((directory/'writer.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Interlock only: an old writer cannot run concurrently. No old database,
        # status, review, chat session or controller decision is consulted.
        legacy_lock = legacy_directory/'batch/writer.lock'
        if legacy_lock.exists():
            old = stack.enter_context(legacy_lock.open('r'))
            fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stop_paths():
            raise RuntimeError('STOP')
        yield


def imported_budget(ledger):
    from tools.r6_evidence import checked
    import json
    imported = json.loads(checked(ledger['legacy_import']).read_text())
    for key in ('caps', 'historical_charged', 'protected_confirmation_delivery_repair', 'protected_model_sessions'):
        # new caps require explicit new authorization, never a silent import reset.
        if ledger[key] != imported[key]:
            raise ValueError('DIRECT_BUDGET_IMPORT_DRIFT:' + key)
    old = imported.get('legacy_process_check', {}).get('recorded')
    if old and (actual := process_identity(old['pid'])) and all(old[k] == v for k, v in actual.items()):
        raise RuntimeError('DIRECT_OLD_PROCESS_ACTIVE')
    return imported
