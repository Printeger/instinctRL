from copy import deepcopy
import json
import subprocess

import pytest

from training_workloads.unified_h1.checkpoint_compatibility import (
    digest, validate_transition_compatibility,
)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()


@pytest.fixture
def binding(tmp_path):
    configs, identities = [], []
    for name in ['old', 'new']:
        root = tmp_path / name
        root.mkdir()
        git(root, 'init', '-q')
        (root / 'actor.py').write_text('same actor data source\n')
        (root / 'adapter.py').write_text(name + ' adapter\n')
        (root / 'contract.md').write_text('same scientific method\n')
        git(root, 'add', '.')
        git(root, '-c', 'user.name=CPU fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', name)
        config = {'method': {'version': 'same', 'proposal_sha256': digest(root / 'contract.md'),
                            'proposal_path': str(root / 'contract.md'), 'hashes': {'actor': 'same'}},
                  'versions': {'reward': 'same'}, 'actor': {'keys': ['sensor']},
                  'product': {'path': str(root), 'source_sha256': {'actor.py': digest(root / 'actor.py')}},
                  'workload': {'path': str(root), 'source_sha256': {'adapter.py': digest(root / 'adapter.py')}}}
        path = tmp_path / (name + '-config.json')
        write(path, config)
        identity = {kind + '_' + field: git(root, 'rev-parse', revision)
                    for kind in ['product', 'workload']
                    for field, revision in [('commit', 'HEAD'), ('tree', 'HEAD^{tree}') ]}
        identity['config_sha256'] = digest(path)
        configs.append((path, config))
        identities.append(identity)
    record = {'schema_version': 'instinctrl-checkpoint-compatibility/1',
              'scope': 'READ_ONLY_TRANSITION_NO_PRODUCT_CHANGE',
              'origin_source_identity': deepcopy(identities[0]),
              'execution_source_identity': deepcopy(identities[1]),
              'origin_config': {'path': str(configs[0][0]), 'sha256': identities[0]['config_sha256']},
              'execution_config': {'path': str(configs[1][0]), 'sha256': identities[1]['config_sha256']},
              'unchanged_product_sha256': configs[0][1]['product']['source_sha256']}
    path = tmp_path / 'binding.json'
    write(path, record)
    execution = {**configs[1][1], 'resolved_identity': identities[1]}
    return path, record, identities[0], execution


def check(binding):
    path, _, origin, execution = binding
    return validate_transition_compatibility({'path': str(path), 'sha256': digest(path)}, origin, execution)


def test_adapter_only_binding_retains_exact_original_identity(binding):
    assert check(binding) == binding[2]


def test_new_path_cannot_inherit_other_checkpoint_source(binding):
    binding[2]['product_commit'] = '0' * 40
    with pytest.raises(PermissionError, match='origin/execution identity'):
        check(binding)


def test_critical_actor_change_is_not_adapter_compatibility(binding):
    path, record, origin, execution = binding
    execution['product']['source_sha256']['actor.py'] = '1' * 64
    current = execution.copy()
    current.pop('resolved_identity')
    currentpath = path.parent / 'new-config.json'
    write(currentpath, current)
    execution['resolved_identity']['config_sha256'] = digest(currentpath)
    record['execution_source_identity'] = deepcopy(execution['resolved_identity'])
    record['execution_config']['sha256'] = digest(currentpath)
    write(path, record)
    with pytest.raises(PermissionError, match='critical product source'):
        check(binding)


def test_original_source_is_rechecked_not_just_documented(binding):
    root = binding[0].parent / 'old'
    (root / 'adapter.py').write_text('changed after acceptance\n')
    with pytest.raises(PermissionError, match='original source drift'):
        check(binding)


def test_changed_record_pin_is_rejected(binding):
    path, _, origin, execution = binding
    pin = {'path': str(path), 'sha256': digest(path)}
    path.write_text(path.read_text() + '\n')
    with pytest.raises(PermissionError, match='input identity drift'):
        validate_transition_compatibility(pin, origin, execution)
