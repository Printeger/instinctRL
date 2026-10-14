"""Recovery entries must not silently substitute another scientific run."""
import json

import pytest

from tools import analyze_r5_formal_research as analysis


@pytest.fixture
def pair(tmp_path, monkeypatch):
    base = tmp_path / 'scientific-parent'
    base.mkdir()
    card = {'experiment_id': 'PARENT', 'primary_node': 128,
            'predefined_late_nodes': [96, 128],
            'profiles': {g: {'group': g, 'training_seed': 19} for g in ['R0', 'R1']}}
    cardpath = base / 'EXPERIMENT_CARD.json'
    cardpath.write_text(json.dumps(card))
    directories = {'R0': tmp_path / 'fresh-recovery', 'R1': base}
    directories['R0'].mkdir()
    for g, directory in directories.items():
        params = {**card['profiles'][g], 'experiment_card': {
            'path': str(cardpath), 'sha256': analysis.digest(cardpath)}}
        checks = [{'path': str(directory / f'u{u}.pt'), 'sha256': g + str(u), 'update': u}
                  for u in [96, 128]]
        common = {'group': g, 'source_commit': 'same-source',
                  'binding': {'source_commit': 'same-source', 'job_id': g}}
        train = {**common, 'actual_parameters': params,
                 'scientific_training_coverage_complete': True,
                 'checkpoints': [{'label': f"u{c['update']}", **c} for c in checks]}
        evaluation = {**common, 'actual_parameters': {**params, 'checkpoints': checks},
                      'coverage_complete': True, 'nodes': {'96': {}, '128': {}},
                      'all_original_gates_pass': True}
        for kind, value in [('training', train), ('evaluation', evaluation)]:
            (directory / f'{g}-{kind}-analysis.json').write_text(json.dumps(value))
    monkeypatch.setattr(analysis, 'compare_inputs', lambda *a, **kw: {'checked_elsewhere': True})
    return base, directories


def mutate(directories, group, kind, change):
    path = directories[group] / f'{group}-{kind}-analysis.json'
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_fresh_recovery_keeps_actual_provenance(pair):
    base, directories = pair
    result = analysis.paired_results(base, directories)
    assert result['training_analyses']['R0'].startswith(str(directories['R0']))
    assert result['groups']['R0']['binding']['job_id'] == 'R0'


def test_rejects_eval_from_original_failed_or_other_training(pair):
    base, directories = pair
    mutate(directories, 'R0', 'evaluation',
           lambda v: v['actual_parameters']['checkpoints'][0].update(path='/old-failed/u96.pt'))
    with pytest.raises(AssertionError, match='this actual training entry'):
        analysis.paired_results(base, directories)


def test_rejects_different_scientific_seed(pair):
    base, directories = pair
    mutate(directories, 'R0', 'training', lambda v: v['actual_parameters'].update(training_seed=20))
    with pytest.raises(AssertionError, match='frozen scientific card'):
        analysis.paired_results(base, directories)


def test_rejects_cross_version_pair(pair):
    base, directories = pair
    for kind in ['training', 'evaluation']:
        def change(v):
            v['source_commit'] = v['binding']['source_commit'] = 'other-source'
        mutate(directories, 'R1', kind, change)
    with pytest.raises(AssertionError, match='source mismatch'):
        analysis.paired_results(base, directories)


def test_rejects_missing_predefined_node(pair):
    base, directories = pair
    mutate(directories, 'R1', 'evaluation', lambda v: v['nodes'].pop('96'))
    with pytest.raises(AssertionError):
        analysis.paired_results(base, directories)
