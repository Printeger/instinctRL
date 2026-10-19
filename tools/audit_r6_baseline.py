"""Read-only CPU audit of the exact R5 inputs selected for R6."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'docs/experiments/r6-robustness-auto'


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': digest}


def verify(path, expected):
    result = pin(path)
    if result['sha256'] != expected:
        raise ValueError('BASELINE_DRIFT: ' + str(path))
    return result


def audit():
    binding = json.loads((BASE/'BASELINE_BINDINGS.json').read_text())
    snap = Path(binding['snapshot']['snapshot'])
    commit = subprocess.check_output(['git', '-C', str(snap), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != binding['source_identity']['product_commit']:
        raise ValueError('BASELINE_COMMIT_DRIFT')
    cfgpin = verify(snap/'training_workloads/unified_h1/candidate_config.json',
                    binding['source_identity']['config_sha256'])
    cfg = json.loads(Path(cfgpin['path']).read_text())
    sources = {}
    for category in ('product', 'workload'):
        for relative, expected in cfg[category]['source_sha256'].items():
            sources[relative] = verify(snap/relative, expected)
    method = verify(cfg['method']['proposal_path'], cfg['method']['proposal_sha256'])
    checkpoint = verify(binding['selected_model']['path'], binding['selected_model']['sha256'])
    runtime = verify(cfg['execution']['python_path'], cfg['execution']['python_sha256'])
    asset_module_path = snap/'training_workloads/unified_h1/asset_closure.py'
    spec = importlib.util.spec_from_file_location('r6_original_asset_audit', asset_module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ground = module.validate_asset_closure(Path(cfg['execution']['isaacsim_asset_root']),
                                           cfg['execution']['isaacsim_asset_manifest_sha256'])
    asset_dir = snap/'isaaclab-training/instinctlab/assets/taslab_uav'
    provenance = json.loads((asset_dir/'PROVENANCE.json').read_text())
    assets = [verify(asset_dir/a['target'], a['sha256']) for a in provenance['artifacts']]
    for record in assets:
        relative = str(Path(record['path']).relative_to(snap))
        blob = subprocess.check_output(['git', 'show', commit+':'+relative], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != record['sha256']:
            raise ValueError('ASSET_COMMIT_DRIFT: '+relative)
    return dict(schema_version='r6-baseline-cpu-audit/1', status='PASS',
                scope='Checkpoint, original config, enumerated source hashes, method, runtime executable, '
                      'ground closure and three provenance-bound UAV assets; no live equivalence claim.',
                auditor=pin(__file__), binding=pin(BASE/'BASELINE_BINDINGS.json'),
                source_commit=commit, config=cfgpin, checkpoint=checkpoint, method=method,
                runtime_executable=runtime, sources=sources, ground_assets=ground,
                uav_provenance=pin(asset_dir/'PROVENANCE.json'), uav_assets=assets,
                gpu_queries=0, simulator_initializations=0)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
