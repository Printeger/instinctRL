"""Frozen direct-run source manifest and CPU-only source/asset verification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tools.r6_direct.import_legacy import pin

ROOT=Path(__file__).resolve().parents[2]
# Pre-existing R6 training integration. The added paths are conditioned on a
# curriculum, absent from baseline evaluation. Additional exact deltas below
# implement the independently reviewed evaluation-only delayed-start handling
# and PPO rejection guard (EVALUATION_STARTUP_REPAIR_REVIEW.json).
PRODUCT_DELTAS = {
    'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py': ('a3c06c94865c5211d1d438abb35629071eb6171e3a4ab57fe64cd23d04508e2a', '997376697fc30d650cc69c126892272bf4ebe9a494be948edc03a367c8b3a3da'),
    'isaaclab-training/instinctlab/observations/robustness_adapter.py': ('3d41a275babc165c0dc75b3cba85ad45abe5d680bf79f50bae95ca57f56eb463', 'ae24396f8abc37d96edddcce5a97b9e9e2676231368d85ef2dea84d76160060b'),
    'isaaclab-training/instinctlab/training/rollout_h1.py': ('99368e65552960680e2fbf860fdf726ec51ab3671c420719cdec90b6992cdabc', 'e59034c1c63b76a54a9c779418b61c26f0a4c597cb247da33cf8aab9d58c74d6'),
    'isaaclab-training/instinctlab/observations/robustness_physics.py': (
        '19b66208bd99e93853ffc2ce4246382dac6c2e36262ed11dd688b4c7ae429863',
        '33741b457a879086e6261e3a887b58bcc7f63e799b108751d41ae49bff0fee4c'),
}


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def checked(value):
    path=Path(value['path'])
    if not path.is_absolute() or path.is_symlink() or pin(path)!=value:
        raise ValueError('DIRECT_INPUT_PIN_DRIFT')
    return path


def training_identity(source):
    """Current detached sources, retaining the pinned scientific configuration."""
    return dict(config_sha256=source['scientific_config']['sha256'],
                product_commit=source['source_commit'],product_tree=source['source_tree'],
                workload_commit=source['source_commit'],workload_tree=source['source_tree'])


def manifest():
    """Call only after committing implementation. Does not create a snapshot."""
    folders=('training_workloads','isaaclab-training/instinctlab','tools/r6_direct',
             'evaluation_workloads/eval_002','tools/r6_budget.py')
    files=git('ls-files',*folders).splitlines()
    files=[name for name in files if Path(name).suffix in ('.py','.json','.yaml','.yml','.usd','.usda','.urdf')]
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    dirty=git('diff','HEAD','--',*files)
    if dirty:raise ValueError('DIRECT_SOURCE_NOT_COMMITTED')
    compatibility=ROOT/'docs/experiments/r6-robustness-auto/BASELINE_COMPATIBILITY_ACCEPTED_02.json'
    record=json.loads(compatibility.read_text())
    science=record['execution_config']
    config=json.loads(checked(science).read_text())
    # Existing accepted product integration must remain byte-identical. New
    # controller/entry paths have their own source identity, not an old receipt.
    for name,digest in config['product']['source_sha256'].items():
        if sources.get(name)!=digest and PRODUCT_DELTAS.get(name)!=(digest,sources.get(name)):
            raise ValueError('DIRECT_PRODUCT_METHOD_CHANGED:'+name)
    return dict(schema_version='r6-direct-source/1',source_commit=git('rev-parse','HEAD'),
                source_tree=git('rev-parse','HEAD^{tree}'),sources=sources,
                scientific_config=science,execution=config['execution'],
                actor=config['actor'],training=config['training'],method=config['method'],
                product_deltas=PRODUCT_DELTAS)


def validate(value, *, frozen=True):
    if value['schema_version']!='r6-direct-source/1':raise ValueError('DIRECT_SOURCE_SCHEMA')
    if git('rev-parse','HEAD')!=value['source_commit'] or git('rev-parse','HEAD^{tree}')!=value['source_tree']:
        raise ValueError('DIRECT_SOURCE_COMMIT')
    if frozen and git('branch','--show-current'):raise ValueError('DIRECT_RUN_REQUIRES_DETACHED_SNAPSHOT')
    for name,digest in value['sources'].items():
        path=ROOT/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
            raise ValueError('DIRECT_SOURCE_PATH')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('DIRECT_SOURCE_DRIFT:'+name)
    cfg=json.loads(checked(value['scientific_config']).read_text())
    for key in ('execution','actor','training','method'):
        if value[key]!=cfg[key]:raise ValueError('DIRECT_SCIENTIFIC_CONFIG_DRIFT:'+key)
    for name,digest in cfg['product']['source_sha256'].items():
        if value['sources'].get(name)!=digest and PRODUCT_DELTAS.get(name)!=(digest,value['sources'].get(name)):
            raise ValueError('DIRECT_PRODUCT_METHOD_DRIFT:'+name)
    executable=Path(sys.executable).resolve()
    execution=value['execution']
    if executable!=Path(execution['python_path']).resolve() or hashlib.sha256(executable.read_bytes()).hexdigest()!=execution['python_sha256']:
        raise ValueError('DIRECT_INTERPRETER')
    from training_workloads.unified_h1.asset_closure import validate_asset_closure
    validate_asset_closure(Path(execution['isaacsim_asset_root']),execution['isaacsim_asset_manifest_sha256'])
    from training_workloads.unified_h1.config import resource_binding_pins
    resource_binding_pins()
    audit_modules()
    return value


def audit_modules():
    forbidden=[name for name in sys.modules if name=='rlflow' or name.startswith('rlflow.')]
    forbidden_paths=[p for p in sys.path if '/tools/rlflow' in str(p)]
    if forbidden or forbidden_paths:raise RuntimeError('DIRECT_FORBIDDEN_DEPENDENCY')
    for name,module in tuple(sys.modules.items()):
        if name.startswith(('training_workloads.','instinctlab.','tools.r6_direct.')):
            filename=getattr(module,'__file__',None)
            if filename and not Path(filename).resolve().is_relative_to(ROOT):
                raise RuntimeError('DIRECT_MODULE_ORIGIN:'+name)
    return dict(loaded_modules=sorted(sys.modules),forbidden_modules=forbidden,
                forbidden_pythonpath=forbidden_paths,execution_mode='local-direct')
