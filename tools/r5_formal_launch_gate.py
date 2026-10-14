"""Specific final-package gate and conservative reserve transfer for R5.

No approval, submit, process, GPU query or file mutation occurs in this module.
The existing Commander must separately validate its Store approval and pins.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

BASE = Path('docs/experiments/r5-formal-readiness-auto')
GOAL = 'R5-FORMAL-READINESS-AUTO-01'
FORMAL_MODES = {'research-formal-training', 'research-formal-evaluation'}
FIELDS = {'training_transitions': 'training_transitions',
          'nontraining_transitions': 'nontraining_transitions',
          'optimizer_updates': 'optimizer_updates',
          'training_checkpoint_writes': 'checkpoint_writes'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def checked_path(pin, expected):
    expected = expected.resolve()
    if (not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}
            or pin['path'] != str(expected) or expected.is_symlink()
            or not expected.is_file() or digest(expected) != pin['sha256']):
        raise PermissionError('formal package pin mismatch: ' + str(expected))
    return expected


def pinned(pin, expected):
    return read(checked_path(pin, expected))


def require_formal_start(root, specpath, approval_id, *, explicit_start=False, accepted_evidence=None):
    """A later explicit start requires the exact independently accepted package."""
    if not explicit_start:
        raise PermissionError('Formal submit requires an explicit later --start-formal action')
    root = Path(root).resolve()
    base = root / BASE
    ready_path = base / 'formal_r5_readiness.json'
    if not ready_path.is_file() or ready_path.is_symlink():
        raise PermissionError('Final formal READY artifact is absent')
    ready = read(ready_path)
    if (ready.get('status') != 'READY_FOR_FORMAL_R5_TRAINING'
            or ready.get('goal_id') != GOAL or ready.get('formal_training_submitted_at_readiness') is not False):
        raise PermissionError('Final formal READY identity mismatch')
    manifest_path = base / 'formal_r5_launch_manifest.json'
    review_path = base / 'reviews/final-readiness.json'
    permission_path = base / 'formal_conditional_permission.json'
    manifest = pinned(ready['manifest'], manifest_path)
    checked_path(manifest['commander'], root / 'tools/r5_formal_commander.py')
    checked_path(manifest['launch_gate'], root / 'tools/r5_formal_launch_gate.py')
    review = pinned(ready['independent_review'], review_path)
    permission = pinned(ready['conditional_permission'], permission_path)
    if (manifest.get('goal_id') != GOAL
            or manifest.get('status') != 'FROZEN_CONDITIONAL_FORMAL_FIRST_SEGMENT'
            or review.get('verdict') != 'PASS' or review.get('must_fix') != []
            or review.get('identities', {}).get('formal_manifest_sha256') != digest(manifest_path)
            or permission.get('goal_id') != GOAL
            or permission.get('status') != 'CONDITIONAL_APPROVED_AT_READINESS'):
        raise PermissionError('Formal final review/manifest/permission not accepted')
    if (permission['manifest'] != ready['manifest']
            or permission['independent_review'] != ready['independent_review']
            or permission['master_authority'] != manifest['master_authority']):
        raise PermissionError('Formal package cross-binding mismatch')
    pinned(manifest['master_authority'], base / 'MASTER_AUTHORITY.json')
    specpath = Path(specpath).resolve()
    spec = read(specpath)
    if not isinstance(spec, dict) or not {'id', 'h1', 'pins', 'argv', 'cwd'} <= spec.keys():
        raise PermissionError('Formal spec structure invalid')
    expected = [e for e in manifest['entries'] if e['spec_id'] == spec['id']]
    if len(expected) != 1:
        raise PermissionError('Formal spec absent from reviewed manifest')
    expected = expected[0]
    matches = [entry for entry in permission['entries'] if entry['spec']['path'] == str(specpath)]
    generated = spec['h1']['mode'] == 'research-formal-evaluation'
    if generated:
        if (matches or spec['id'] not in permission.get('allowed_generated_evaluation_spec_ids', [])
                or expected.get('generated_checkpoint_inputs') is not True):
            raise PermissionError('Formal generated evaluation is not conditionally permitted')
        # The Store approval is checked independently by the Commander. Here
        # validate the frozen template and exact checkpoint acceptance condition.
        entry = {'spec': {'path': str(specpath), 'sha256': digest(specpath)},
                 'approval_id': approval_id, 'authorization': spec['pins']['authorization']}
    else:
        if len(matches) != 1:
            raise PermissionError('Formal exact permission entry absent or duplicated')
        entry = matches[0]
    spec = pinned(entry['spec'], specpath)
    if spec['h1']['mode'] not in FORMAL_MODES or entry['approval_id'] != approval_id:
        raise PermissionError('Formal mode or exact approval mismatch')
    authorization = spec['pins']['authorization']
    if (authorization != entry['authorization']
            or (not generated and authorization['sha256'] != expected['expected_authorization_sha256'])
            or spec['argv'] != expected['argv'] or spec['cwd'] != expected['cwd']
            or spec['h1'] != expected['h1']):
        raise PermissionError('Formal exact input differs from reviewed manifest')
    auth = pinned(authorization, Path(authorization['path']))
    checkpoint_acceptance_sha256 = None
    if generated:
        template = pinned(expected['authorization_template'], Path(expected['authorization_template']['path']))
        rows = auth['parameters']['checkpoints']
        proposed = template['parameters']['checkpoints']
        if (len(rows) != len(proposed) or not rows
                or any(set(a) != {'path', 'sha256', 'update'} or set(b) != set(a)
                       or b['sha256'] is not None or a['path'] != b['path'] or a['update'] != b['update']
                       for a, b in zip(rows, proposed))):
            raise PermissionError('Formal generated checkpoint identities differ from frozen template')
        filled = deepcopy(template)
        filled['parameters']['checkpoints'] = deepcopy(rows)
        if filled != auth:
            raise PermissionError('Formal generated authorization changed other frozen fields')
        for row in rows:
            path = Path(row['path'])
            if path.is_symlink() or not path.is_file() or digest(path) != row['sha256']:
                raise PermissionError('Formal generated checkpoint file drift')
        for condition in spec.get('conditions', []):
            evidence = (accepted_evidence or {}).get(condition['evidence_id'], {})
            if not evidence.get('accepted') or evidence.get('sha256') != condition['sha256']:
                continue
            review = pinned({'path': evidence['path'], 'sha256': evidence['sha256']}, Path(evidence['path']))
            accepted = review.get('accepted_evaluation_checkpoints', [])
            if (review.get('verdict') == 'PASS' and review.get('must_fix') == []
                    and all(any(all(ck.get(k) == row[k] for k in ['path', 'sha256', 'update'])
                                for ck in accepted) for row in rows)):
                checkpoint_acceptance_sha256 = evidence['sha256']
                break
        if checkpoint_acceptance_sha256 is None:
            raise PermissionError('Generated formal checkpoints lack exact accepted training evidence')
    return {'manifest_sha256': digest(manifest_path), 'review_sha256': digest(review_path),
            'permission_sha256': digest(permission_path), 'approval_id': approval_id,
            'generated_checkpoint_acceptance_sha256': checkpoint_acceptance_sha256}


def transfer_formal_reserve(ledger, budget, output_cap):
    """Return a checked copy; later admission persists transfer with entry charge.

    Previously charged entries and historical reservations are never refunded.
    The caller still checks master totals, disk and all ordinary release gates.
    """
    result = deepcopy(ledger)
    formal = result['reservations']['formal_initial_segment']
    stage = result['stage_allocation']
    amounts = {key: budget[source] for key, source in FIELDS.items()}
    amounts.update(live_starts=1, simulator_initialization_reservations=1,
                   actor_optimizer_steps=budget['optimizer_updates'] * 8,
                   critic_optimizer_steps=budget['optimizer_updates'] * 8,
                   allocated_gpu_seconds=budget['wall_seconds'] + 10,
                   wall_seconds_including_cleanup=budget['wall_seconds'] + 10,
                   output_peak_and_reservations_bytes=output_cap)
    old_updates = stage['optimizer_updates']
    for key, amount in amounts.items():
        if (type(amount) is not int or amount < 0 or key not in formal
                or formal[key] < amount):
            raise PermissionError('Insufficient protected formal reserve: ' + key)
        formal[key] -= amount
        prior = stage.get(key, old_updates * 8 if key in ['actor_optimizer_steps', 'critic_optimizer_steps'] else 0)
        stage[key] = prior + amount
        if stage[key] > result['caps'][key]:
            raise PermissionError('Formal stage transfer exceeds master cap: ' + key)
    return result
