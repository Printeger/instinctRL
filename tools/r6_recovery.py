"""R6 recovery journal, reusing rlflow atomic persistence and process identity.

This is not a launcher. The controller must hold this lock before calling any
mutation. An intent without authoritative completion never authorizes a retry.
"""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os

from rlflow.jobs import identity, owns
from rlflow.store import atomic, canonical, sha
from tools.r6_budget import reserve


class RecoveryError(RuntimeError):
    pass


def file_pin(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def checked(pin):
    path = Path(pin['path'])
    if path.is_symlink() or file_pin(path) != pin:
        raise RecoveryError('PIN_DRIFT')
    return path


class Journal:
    def __init__(self, directory, stop_paths=()):
        self.directory = Path(directory)
        self.path = self.directory / 'state.json'
        self.stop_paths = [self.directory / 'STOP', *map(Path, stop_paths)]
        self._lock = None
        self._owner_pid = None

    @contextmanager
    def writer(self):
        if self._lock is not None:
            raise RecoveryError('NESTED_WRITER')
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / 'writer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock, self._owner_pid = lock, os.getpid()
            try:
                yield self
            finally:
                self._lock = self._owner_pid = None
                fcntl.flock(lock, fcntl.LOCK_UN)

    def require_writer(self):
        if self._lock is None or self._owner_pid != os.getpid():
            raise RecoveryError('SINGLE_WRITER_REQUIRED')

    def stopped(self):
        return [str(p) for p in self.stop_paths if p.exists()]

    def guard(self):
        self.require_writer()
        if self.stopped():
            raise RecoveryError('STOP_NEW_ACTIONS')

    def read(self):
        return json.loads(self.path.read_text())

    def save(self, state):
        self.require_writer()
        atomic(self.path, canonical(state) + b'\n')

    def initialize(self, budget):
        self.guard()
        if self.path.exists():
            raise RecoveryError('USE_EXISTING_STATE')
        self.save({'schema_version': 'r6-recovery/1', 'goal_id': budget['goal_id'],
                   'controller_identity': identity(os.getpid()), 'budget': budget,
                   'events': {}, 'duplicate_events_rejected': 0,
                   'status': 'CPU_PREPARE', 'current_job': None})

    def intent(self, event, *, kind, request, resources, purpose, free_disk_bytes):
        """Persist request and reservation in one transaction before side effects.

        Returns (key, fresh). False means inspect existing work, never launch.
        Input hash covers the full request, including executable/prompt/schema.
        """
        self.guard()
        if kind not in {'job', 'model', 'settle', 'postrun', 'cpu'}:
            raise RecoveryError('UNKNOWN_EVENT_KIND')
        if event.get('input_sha256') != sha(request):
            raise RecoveryError('REQUEST_IDENTITY_MISMATCH')
        state = self.read()
        key = sha(event)
        entry = {'identity': event, 'kind': kind, 'request': request,
                 'resources': resources, 'purpose': purpose}
        if key in state['events']:
            previous = state['events'][key]
            if any(previous[k] != v for k, v in entry.items()):
                raise RecoveryError('EVENT_REPLAY_DRIFT')
            state['duplicate_events_rejected'] += 1
            self.save(state)
            return key, False
        state['budget'] = reserve(state['budget'], event_key=event, resources=resources,
                                  purpose=purpose, free_disk_bytes=free_disk_bytes)
        state['events'][key] = {**entry, 'status': 'INTENT', 'process': None,
                                'receipt': None}
        self.save(state)
        return key, True

    def bind_process(self, key, process):
        self.require_writer()
        if not owns(process):
            raise RecoveryError('PROCESS_IDENTITY_NOT_LIVE')
        state = self.read()
        entry = state['events'][key]
        if entry['status'] != 'INTENT' or entry['process'] is not None:
            raise RecoveryError('PROCESS_ALREADY_BOUND')
        entry.update(process=process, status='RUNNING')
        self.save(state)

    def observe(self, key, *, receipt_path=None, exact_job=None):
        """Read-only recovery decision based on a receipt or a live exact handle.

        STOP never suppresses observation/cleanup of an existing owned job.
        A callback result must include the exact event job ID and request hash.
        A job receipt is not settlement; returned COMPLETE only closes this event.
        """
        entry = self.read()['events'][key]
        if entry['receipt']:
            checked(entry['receipt'])
            return {'action': 'ALREADY_CONSUMED', 'receipt': entry['receipt']}
        if receipt_path is not None and Path(receipt_path).exists():
            try:
                raw = Path(receipt_path).read_bytes()
                receipt = json.loads(raw)
            except (ValueError, UnicodeError):
                return {'action': 'RECONCILE', 'reason': 'PARTIAL_RECEIPT_NO_RETRY'}
            if not isinstance(receipt, dict):
                raise RecoveryError('INVALID_RECEIPT_OBJECT')
            if receipt.get('event_key') != key or receipt.get('request_sha256') != sha(entry['request']):
                raise RecoveryError('RECEIPT_IDENTITY_MISMATCH')
            if type(receipt.get('exit_code')) is not int:
                raise RecoveryError('INVALID_RECEIPT_EXIT_CODE')
            if entry['process'] is not None and receipt.get('process') != entry['process']:
                raise RecoveryError('RECEIPT_PROCESS_MISMATCH')
            if owns(entry['process']) or owns(receipt.get('process')):
                return {'action': 'WAIT_OWNED_EXIT'}
            return {'action': 'ACCEPT_RECEIPT', 'receipt': {
                'path': str(Path(receipt_path).resolve()),
                'sha256': hashlib.sha256(raw).hexdigest()}}
        if exact_job is not None:
            expected = entry['identity']
            if exact_job.get('job_id') != expected['job_id'] or exact_job.get('input_sha256') != expected['input_sha256']:
                raise RecoveryError('EXACT_JOB_IDENTITY_MISMATCH')
            if type(exact_job.get('terminal')) is not bool:
                raise RecoveryError('INVALID_JOB_TERMINAL_FLAG')
            if exact_job['terminal']:
                return {'action': 'CLOSE_AND_SETTLE', 'job_id': expected['job_id']}
            if owns(exact_job.get('process')):
                return {'action': 'CANCEL_OWNED' if self.stopped() else 'WAIT_EXACT_JOB',
                        'job_id': expected['job_id']}
            return {'action': 'RECONCILE', 'reason': 'JOB_HANDLE_MISSING_NO_RETRY'}
        if owns(entry['process']):
            return {'action': 'CANCEL_OWNED' if self.stopped() else 'WAIT_OWNED_WORKER'}
        return {'action': 'RECONCILE', 'reason': 'INTENT_WITHOUT_LIVE_HANDLE_OR_RECEIPT'}

    def consume(self, key, receipt_path):
        self.require_writer()
        observed = self.observe(key, receipt_path=receipt_path)
        if observed['action'] == 'ALREADY_CONSUMED':
            return False
        if observed['action'] != 'ACCEPT_RECEIPT':
            raise RecoveryError('RECEIPT_NOT_READY: ' + observed['action'])
        checked(observed['receipt'])
        state = self.read()
        entry = state['events'][key]
        entry.update(receipt=observed['receipt'], status='RECEIPT_RECORDED')
        # Even nonzero exits retain the full reservation. Decision/schema and
        # evidence acceptance are separate gates in the eventual controller.
        self.save(state)
        return True
