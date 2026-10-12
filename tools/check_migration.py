"""Read-only checks of the consolidated candidate; never authorizes execution."""
from __future__ import annotations
import argparse, ast, hashlib, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='903afc5abd0fbf82297527e5790c2cd40827b0b2'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    required=('isaaclab-training','training_workloads','evaluation_workloads','tools/rlflow','coordination','docs/history')
    for path in required:
        if not (ROOT/path).is_dir():raise ValueError('missing source directory: '+path)
    if (ROOT/'product_snapshot').exists():raise ValueError('duplicate product_snapshot')
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only','-z',BASE],cwd=ROOT,text=True).split('\0')[:-1]
    missing=[p for p in tracked if not (ROOT/p).exists()]
    if missing:raise ValueError('lost original tracked files: '+str(missing))
    config=json.loads((ROOT/'training_workloads/unified_h1/candidate_config.json').read_text())
    for key in ('product','workload'):
        if config[key]['path']!=str(ROOT) or config[key]['branch']!='main':raise ValueError('stale current repository identity')
        for relative,expected in config[key]['source_sha256'].items():
            path=ROOT/relative
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT) or digest(path)!=expected:raise ValueError('source hash drift: '+relative)
    if any(config['runtime'].values()):raise ValueError('migration config grants runtime budget')
    method=Path(config['method']['proposal_path'])
    if not method.is_relative_to(ROOT) or digest(method)!=config['method']['proposal_sha256']:raise ValueError('method binding drift')
    asset=Path(config['execution']['isaacsim_asset_root'])/'ASSET_MANIFEST.json'
    if digest(asset)!=config['execution']['isaacsim_asset_manifest_sha256']:raise ValueError('asset manifest drift')
    from training_workloads.unified_h1.asset_closure import validate_asset_closure
    closure=validate_asset_closure(asset.parent,digest(asset))
    product=ROOT/'isaaclab-training/instinctlab/evaluation/raycaster_observability.py'
    base_bytes=subprocess.check_output(['git','show',BASE+':'+str(product.relative_to(ROOT))],cwd=ROOT)
    if product.read_bytes()!=base_bytes:raise ValueError('R3 numerical repair regressed')
    python_count=0
    for directory in ('isaaclab-training','training_workloads','evaluation_workloads','training_preparation','evaluation_tooling','tests','tools/rlflow'):
        for path in (ROOT/directory).rglob('*.py'):
            if '__pycache__' not in path.parts:ast.parse(path.read_text(),filename=str(path));python_count+=1
    state=json.loads((ROOT/'coordination/PROJECT_STATE.json').read_text())
    if state['new_live_execution_authorized'] or state['new_training_authorized']:raise ValueError('state grants unapproved runtime')
    result={'status':'PASS','base_product_commit':BASE,'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),'checked_python_files':python_count,'preserved_baseline_files':len(tracked),'source_manifest_files':sum(len(config[k]['source_sha256']) for k in ('product','workload')),'asset_files':len(closure['opened_files']),'scientific_acceptance':'NOT_GRANTED','live_authorized':False,'training_authorized':False}
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
