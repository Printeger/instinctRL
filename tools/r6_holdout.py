"""Single-use final realization journal. No Store permission or job dispatch.

Private entropy is outside committed state; commitment publication precedes
release. Missing/partial private files are never silently regenerated.
"""
import json
import os
import secrets
import hashlib
import re
import subprocess
from pathlib import Path

from rlflow.store import sha
from tools.r6_recovery import checked, file_pin, RecoveryError
from tools.r6_postrun import stop_paths
from tools.r6_job_handoff import no_model_workers
from tools.r6_budget import admissible
from training_workloads.unified_h1.robustness_holdout import commitment, reveal
from training_workloads.unified_h1.robustness_contract import BASELINE_SHA

ROLE = 'final-holdout'
DISK = 'output_actual_plus_unreleased_reservations_bytes'
ROOT = Path(__file__).resolve().parents[1]


def require(value, reason):
    if not value:
        raise RecoveryError('HOLDOUT_' + reason)


def validate_entry(state, key):
    entry = state['events'][key]
    expected = dict(goal_id=state['goal_id'], event_id='FINAL-HOLDOUT-01', job_id='FINAL-BATCH',
                    input_sha256=sha(entry['request']), role=ROLE)
    require(entry['identity'] == expected and sha(expected) == key and entry['kind'] == 'cpu'
            and entry['resources'] == {DISK:65536} and entry['purpose'] == 'confirmation_delivery',
            'EVENT_IDENTITY')
    rows = [r for r in state['budget']['reservations'] if r['identity'] == expected]
    require(len(rows) == 1 and rows[0]['resources'] == entry['resources']
            and rows[0]['purpose'] == entry['purpose'], 'RESERVATION')
    admissible(state['budget'])
    return entry


def validate_freeze(request):
    require(set(request) == {'freeze', 'review'}, 'REQUEST_FIELDS')
    freeze = json.loads(checked(request['freeze']).read_text())
    require(set(freeze) == {'schema_version', 'branch', 'protocol', 'recipe', 'selection_rule', 'models'}, 'FREEZE_FIELDS')
    require(freeze['schema_version'] == 'r6-final-freeze/1', 'FREEZE_SCHEMA')
    protocol = json.loads(checked(freeze['protocol']).read_text())
    require(protocol.get('frozen') is True, 'PROTOCOL_NOT_FROZEN')
    checked(freeze['recipe']); checked(freeze['selection_rule'])
    models = freeze['models']
    expected = {'baseline'} if freeze['branch'] == 'BASELINE_SUFFICIENT' else {'baseline', 'seed1', 'seed2', 'seed3'}
    require(freeze['branch'] in {'BASELINE_SUFFICIENT', 'THREE_CONTINUATIONS'} and set(models) == expected, 'MODEL_SET')
    for pin in models.values():
        checked(pin)
    require(models['baseline']['sha256'] == BASELINE_SHA, 'BASELINE_IDENTITY')
    review = json.loads(checked(request['review']).read_text())
    require(review.get('decision') == 'PASS_R6_FINAL_FREEZE' and review.get('must_fix') == []
            and review.get('reviewer') not in (None, '', 'root')
            and review.get('freeze_sha256') == request['freeze']['sha256'], 'INDEPENDENT_FREEZE_REVIEW')
    return freeze


def prepare(journal, request, *, free_disk_bytes):
    journal.guard(); require(not stop_paths(journal), 'STOP'); no_model_workers(journal)
    freeze = validate_freeze(request)
    state = journal.read()
    event = dict(goal_id=state['goal_id'], event_id='FINAL-HOLDOUT-01', job_id='FINAL-BATCH',
                 input_sha256=sha(request), role=ROLE)
    key = sha(event)
    owners = [k for k, e in state['events'].items() if e['identity']['role'] == ROLE]
    require(not owners or owners == [key], 'ALREADY_BOUND_TO_OTHER_FREEZE')
    if not owners:
        journal.intent(event, kind='cpu', request=request, resources={DISK:65536},
                       purpose='confirmation_delivery', free_disk_bytes=free_disk_bytes)
    state = journal.read(); entry = validate_entry(state, key)
    require(entry['request'] == request and entry['identity'] == event, 'EVENT_DRIFT')
    private = journal.directory / ('holdout-private-' + key + '.json')
    if 'entropy_claim' not in entry:
        # Persist claim before opening the secret file. An interrupted claim
        # without a complete file is ambiguous and must not draw again.
        require(not private.exists() and not private.is_symlink(), 'UNCLAIMED_SECRET')
        entry['entropy_claim'] = True
        journal.save(state)
        fd = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(dict(request_sha256=sha(request), nonce=secrets.token_hex(32)), stream)
            stream.flush(); os.fsync(stream.fileno())
        fd = os.open(journal.directory, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    require(private.is_file() and not private.is_symlink() and private.stat().st_mode & 0o077 == 0,
            'SECRET_MISSING_OR_PERMISSIONS')
    secret = json.loads(private.read_text())
    require(set(secret) == {'request_sha256', 'nonce'} and secret['request_sha256'] == sha(request), 'SECRET_BINDING')
    record = commitment(secret['nonce'], protocol_sha256=freeze['protocol']['sha256'],
                        freeze_sha256=request['freeze']['sha256'])
    state = journal.read(); entry = state['events'][key]
    if 'commitment' in entry:
        require(entry['commitment'] == record and entry['private_pin'] == file_pin(private), 'COMMITMENT_DRIFT')
    else:
        entry.update(commitment=record, private_pin=file_pin(private), status='COMMITTED_NOT_RELEASED')
        journal.save(state)
    return key, record


def verify_publication(pin, commit):
    require(isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit), 'EXACT_PUBLICATION_COMMIT')
    path = checked(pin)
    require(path.is_relative_to(ROOT), 'PUBLICATION_OUTSIDE_REPOSITORY')
    relative = path.relative_to(ROOT).as_posix()
    def git(*args):
        result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, timeout=30)
        require(result.returncode == 0, 'PUBLICATION_GIT_IDENTITY')
        return result.stdout
    git('merge-base', '--is-ancestor', commit, 'HEAD')
    blob = git('show', commit + ':' + relative)
    require(hashlib.sha256(blob).hexdigest() == pin['sha256'], 'PUBLICATION_CONTENT_DRIFT')


def release(journal, key, published_commitment, *, publication_commit):
    journal.guard(); require(not stop_paths(journal), 'STOP'); no_model_workers(journal)
    state = journal.read(); entry = validate_entry(state, key)
    validate_freeze(entry['request'])
    public = json.loads(checked(published_commitment).read_text())
    require(public == entry.get('commitment'), 'PUBLICATION_BINDING')
    verify_publication(published_commitment, publication_commit)
    private = checked(entry['private_pin'])
    require(private.stat().st_mode & 0o077 == 0, 'SECRET_PERMISSIONS')
    secret = json.loads(private.read_text())
    require(secret['request_sha256'] == sha(entry['request']), 'SECRET_BINDING')
    value = reveal(public, secret['nonce'])
    binding = dict(publication=published_commitment, publication_commit=publication_commit,
                   realization_sha256=sha(value))
    if 'release' in entry:
        require(entry['release'] == binding, 'RELEASE_DRIFT')
    else:
        entry.update(release=binding, status='RELEASED_FINAL_INPUT_ONLY')
        journal.save(state)
    return value
