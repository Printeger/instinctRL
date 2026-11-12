import json,shutil,stat,os
from pathlib import Path
from tools.r6_direct.import_legacy import pin,stop_paths
from tools.r6_direct.lifecycle import writer
from tools.r6_direct.run import reserve
from tools.r6_direct.io import atomic
from training_workloads.unified_h1.extension_inputs import tree_identity
b=Path('/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training');e=b/'continuation-20261101/delivery/render-08';dest=Path('/home/mint/.local/share/instinctrl-r6/extension-bindings/standard-names-01');old=json.loads((Path('/home/mint/rl_dev/instinctRL-dev')/'docs/experiments/r6-robustness-auto/EXTENSION_INPUTS_CANDIDATE_02.json').read_text());plan=json.loads((e/'EXTENSION_BINDING_CANDIDATE_PLAN.json').read_text());inventory=json.loads((e/'EXTENSION_BINDING_PREFLIGHT_INVENTORY.json').read_text())
assert not stop_paths() and not dest.exists() and not inventory['unresolved_cross_extension_links']
assert shutil.disk_usage(dest.parents[3]).free > old['tree']['bytes']+30*1024**3
root=Path(old['tree']['root']);assert tree_identity(root)==old['tree']
for r in plan['mapping']:
 if r['renamed']:assert (Path(r['source'])/'config/extension.toml').is_file()
with writer():
 p=b/'RESOURCE_LEDGER.json';l=json.loads(p.read_text());reserve(l,'DIRECT-EXTENSION-BINDING-STANDARD-01',{'output_actual_plus_unreleased_reservations_bytes':old['tree']['bytes']},purpose='confirmation_delivery',resource_policy=pin(b/'continuation-20261101/RESOURCE_POLICY_USER_AMENDMENT.json'));atomic(p,l)
dest.mkdir(parents=True);rows=[]
for r in plan['mapping']:
 assert not stop_paths()
 src=Path(r['source']);target=dest/r['target_name']
 if src.is_dir():
  before=tree_identity(src);shutil.copytree(src,target,symlinks=True);after=tree_identity(target);assert all(before[k]==after[k] for k in ('sha256','files','bytes'))
  for d,ds,fs in os.walk(src):
   for name in ds+fs:
    a=Path(d)/name;z=target/a.relative_to(src);assert stat.S_IMODE(a.lstat().st_mode)==stat.S_IMODE(z.lstat().st_mode)
    if a.is_file() and not a.is_symlink():assert (a.stat().st_dev,a.stat().st_ino)!=(z.stat().st_dev,z.stat().st_ino)
 else:
  before=pin(src);shutil.copy2(src,target);after=pin(target);assert before['sha256']==after['sha256']
 rows.append(dict(mapping=r,original=before,candidate=after))
assert tree_identity(root)==old['tree'];newtree=tree_identity(dest);assert newtree['files']==old['tree']['files'] and newtree['bytes']==old['tree']['bytes']
atomic(e/'EXTENSION_BINDING_MATERIALIZED.json',dict(status='BYTE_IDENTICAL_CONTENT_NEW_PATHS_NOT_LIVE_VALIDATED',original=old['tree'],candidate=newtree,entries=rows,metadata_files=3,directories=269,renames=182,config_parse_exception_preserved=inventory['config_parse_errors'],rustdesk_touched=False))
atomic(e/'EXTENSION_INPUTS_STANDARD_01.json',dict(schema_version='r6-extension-inputs/2',tree=newtree,user_config=old['user_config']))
print('MATERIALIZED_AND_VERIFIED',newtree,flush=True)
