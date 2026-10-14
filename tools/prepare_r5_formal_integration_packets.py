"""Prepare the already specified R5 integration/formal inputs; never release.

The immutable science snapshot and previously committed formal templates are
reused. This only binds the compatible control release and the G6 proposal.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/experiments/r5-formal-readiness-auto'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    return {'path': str(path), 'sha256': sha(path)}


def save(path, value):
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    from rlflow.releases import verify, files
    from training_workloads.unified_h1.research_contract import validate_parameters, required_budget
    integration = BASE / 'experiments/R5-MEAN10-INTEGRATION-01'
    cardpath = integration / 'EXPERIMENT_CARD.json'
    card = read(cardpath)
    source = card['source']
    snapshot = Path(source['snapshot'])
    configpath = snapshot / 'training_workloads/unified_h1/candidate_config.json'
    config = read(configpath)
    assert sha(configpath) == source['config_sha256']
    for section in ['product', 'workload']:
        for relative, expected in config[section]['source_sha256'].items():
            assert sha(Path(config[section]['path']) / relative) == expected
    control = read(BASE / 'CONTROL_INTEGRATION_CPU.json')['control_release']
    release = verify(Path(control['path']))
    assert {name: sha(path) for name, path in files(ROOT / 'tools/rlflow').items()} == release['files']
    baseline = {key: source[key] for key in ['snapshot', 'snapshot_commit', 'snapshot_tree',
                'config_sha256', 'snapshot_diff_sha256', 'read_only']}
    baseline.update(tool_release_id=release['id'], tool_release_path=control['path'])
    parent = read(BASE / 'experiments/R5-MEAN10-DEV-02/launch_inputs.proposed.json')
    old = next(row for row in parent['entries'] if row['group'] == 'R1' and row['mode'] == 'research-evaluation')
    spec, auth = deepcopy(old['experiment_proposal']), deepcopy(old['authorization_proposal'])
    mode = 'research-transition'
    attempt = ROOT / 'coordination/research/attempts' / f"{card['experiment_id']}-R1-RESEARCH-TRANSITION-{source['snapshot_commit'][:12]}"
    assert not attempt.exists()
    parameters = {**deepcopy(card['profiles']['R1']), 'experiment_card': pin(cardpath),
                  'checkpoints': [deepcopy(card['primary_checkpoint'])]}
    validate_parameters(mode, parameters)
    budget = required_budget(mode, parameters)
    auth.update(order_id=attempt.name, attempt=str(attempt), mode=mode, budgets=budget,
                parameters=parameters, **source['source_identity'])
    authpath = attempt / 'commands/AUTHORIZATION.json'
    spec.update(id=attempt.name, cwd=str(snapshot), seed=parameters['evaluation_seed'],
                argv=[config['execution']['python_path'], '-B', '-m',
                      'training_workloads.unified_h1.formal_unified_h1', '--authorization',
                      str(authpath), '--attempt', str(attempt), '--mode', mode],
                resources={'starts': 1, 'wall_seconds': 910, 'gpu_seconds': 910, 'transitions': 3584})
    for name, relative in [('dispatcher', 'training_workloads/unified_h1/local_dispatch.py'),
                           ('workload', 'training_workloads/unified_h1/formal_unified_h1.py'),
                           ('product', 'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py')]:
        spec['pins'][name] = pin(snapshot / relative)
    spec['pins']['authorization'] = {'path': str(authpath), 'sha256': None}
    spec['h1'].update(workload_root=str(snapshot), attempt=str(attempt), mode=mode,
                      config_sha256=source['config_sha256'])
    row = {'group': 'R1', 'mode': mode, 'experiment_proposal': spec,
           'authorization_proposal': auth, 'missing_bindings':
           ['actualG4accepted', 'independent exact G6/G7 prelive', 'delegated exact release']}
    save(integration / 'inputs/R1-transition-AUTH.proposed.json', auth)
    common = {key: deepcopy(parent[key]) for key in ['schema_version', 'master_authority',
                                                     'first_rollout_gate', 'stop_paths']}
    for directory, rows in [(integration, [row]),
                            (BASE / 'experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01', None)]:
        if rows is None:
            rows = []
            for group in ['R0', 'R1']:
                for formal_mode in ['research-formal-training', 'research-formal-evaluation']:
                    authfile = directory / f'inputs/{group}-{formal_mode}-AUTH.proposed.json'
                    specfile = directory / f'inputs/{group}-{formal_mode}-spec.proposed.json'
                    proposal = read(specfile)
                    item = {'group': group, 'mode': formal_mode,
                            'experiment_proposal': proposal, 'authorization_proposal': read(authfile),
                            'missing_bindings': ['finalG1-G9review', 'conditional permission; no submit in thisGoal']}
                    if formal_mode == 'research-formal-training':
                        item['expected_authorization'] = {'path': str(authfile.relative_to(directory)),
                                                           'sha256': sha(authfile)}
                        assert proposal['pins']['authorization']['sha256'] == sha(authfile)
                    rows.append(item)
        packet = {**deepcopy(common), 'status': 'PREPARED_NOT_QUALIFIED_OR_AUTHORIZED',
                  'experiment_id': directory.name, 'baseline': baseline,
                  'experiment_card': pin(directory / 'EXPERIMENT_CARD.json'), 'entries': rows,
                  'generator_sha256': sha(Path(__file__))}
        save(directory / 'launch_inputs.proposed.json', packet)
        print(json.dumps({'experiment_id': directory.name,
                          'packet_sha256': sha(directory / 'launch_inputs.proposed.json')}))


if __name__ == '__main__':
    main()
