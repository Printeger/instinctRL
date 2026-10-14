"""Narrow origin binding for read-only transition checks after adapter changes.

This declaration is not an approval. Exact independent review and Commander
conditions still authorize an entry. No legacy evaluation rule is relaxed.
"""
from pathlib import Path
import hashlib
import json
import subprocess

IDENTITY_FIELDS = {'product_commit', 'product_tree', 'workload_commit',
                   'workload_tree', 'config_sha256'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pinned_json(pin):
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise PermissionError('compatibility pin schema mismatch')
    path = Path(pin['path'])
    if (not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file()
            or digest(path) != pin['sha256']):
        raise PermissionError('compatibility input identity drift')
    return json.loads(path.read_text())


def validate_source_identity(identity):
    if not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS:
        raise PermissionError('checkpoint origin requires exact five source fields')
    for key, value in identity.items():
        size = 64 if key == 'config_sha256' else 40
        if not isinstance(value, str) or len(value) != size or any(c not in '0123456789abcdef' for c in value):
            raise PermissionError('checkpoint origin source identity malformed')
    return identity


def _validate_compatibility(pin, origin, config, scope):
    """Require identical critical product files and exact original/new identity.

    Only workloads may change for this read-only mode. A future serializer or
    learning-method change needs its own reviewed compatibility contract.
    """
    validate_source_identity(origin)
    execution = validate_source_identity(config['resolved_identity'])
    record = pinned_json(pin)
    fields = {'schema_version', 'scope', 'origin_source_identity',
              'execution_source_identity', 'origin_config', 'execution_config',
              'unchanged_product_sha256'}
    if (set(record) != fields or record['schema_version'] != 'instinctrl-checkpoint-compatibility/1'
            or record['scope'] != scope):
        raise PermissionError('read-only transition compatibility scope invalid')
    if record['origin_source_identity'] != origin or record['execution_source_identity'] != execution:
        raise PermissionError('compatibility origin/execution identity mismatch')
    original_config = pinned_json(record['origin_config'])
    current_config = pinned_json(record['execution_config'])
    if record['origin_config']['sha256'] != origin['config_sha256']:
        raise PermissionError('compatibility original config mismatch')
    if (record['execution_config']['sha256'] != execution['config_sha256']
            or current_config != {k: v for k, v in config.items() if k != 'resolved_identity'}):
        raise PermissionError('compatibility execution config mismatch')
    for key in ['version', 'proposal_sha256', 'hashes']:
        if original_config['method'][key] != config['method'][key]:
            raise PermissionError('compatibility cannot change scientific method')
    original_method = Path(original_config['method']['proposal_path'])
    if (not original_method.is_absolute() or original_method.is_symlink()
            or not original_method.is_file()
            or digest(original_method) != original_config['method']['proposal_sha256']):
        raise PermissionError('compatibility original method document drift')
    if original_config['versions'] != config['versions'] or original_config['actor'] != config['actor']:
        raise PermissionError('compatibility cannot change actor or component versions')
    product = original_config['product']['source_sha256']
    if product != config['product']['source_sha256'] or product != record['unchanged_product_sha256']:
        raise PermissionError('compatibility cannot change any critical product source')
    # New execution has already passed load_config's full manifest checks.
    # Independently recheck the immutable original checkout, including workload
    # files, so a new path cannot inherit an old receipt's identity.
    for section in ['product', 'workload']:
        root = Path(original_config[section]['path'])
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise PermissionError('compatibility original checkout invalid')
        for suffix, revision in [('commit', 'HEAD'), ('tree', 'HEAD^{tree}')]:
            actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', revision], text=True).strip()
            if actual != origin[section + '_' + suffix]:
                raise PermissionError('compatibility original checkout identity drift')
        for relative, expected in original_config[section]['source_sha256'].items():
            path = root / relative
            if (Path(relative).is_absolute() or '..' in Path(relative).parts
                    or path.is_symlink() or not path.is_file() or digest(path) != expected):
                raise PermissionError('compatibility original source drift')
    return dict(origin)


def validate_transition_compatibility(pin, origin, config):
    return _validate_compatibility(pin, origin, config, 'READ_ONLY_TRANSITION_NO_PRODUCT_CHANGE')


def validate_formal_compatibility(pin, origin, config):
    # This binding only establishes a strict checkpoint read across unchanged
    # product implementations. The formal contract and independent exact
    # Commander authorization separately admit future optimizer effects.
    return _validate_compatibility(pin, origin, config, 'FORMAL_WARM_START_IDENTICAL_PRODUCT')
