import json
from pathlib import Path
from tools.r6_direct.import_legacy import pin,process_identity,stop_paths
from tools.r6_evidence import checked
b=Path('/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training');e=b/'continuation-20261101/delivery';ledger=json.loads((b/'RESOURCE_LEDGER.json').read_text());runs=[];owners=[];modules=[]
for row in ledger['new_reservations']:
 if 'receipt' not in row:continue
 receipt=json.loads(checked(row['receipt']).read_text());assert receipt['status']=='CLOSED' and not receipt['remaining']
 identities=[receipt['parent'],*receipt['owned_processes']];assert all(process_identity(o['pid'])!=o for o in identities);owners.extend(identities)
 runs.append(dict(id=row['id'],receipt=row['receipt'],status=row['status'],settlement=row.get('settlement'),exit_code=receipt['exit_code'],error=receipt['error'],identities_checked=len(identities)))
 p=Path(row['receipt']['path']).parent/'RUN_INPUTS.json'
 if p.exists():
  record=json.loads(p.read_text());plan=record.get('plan',record);q=Path(plan['attempt_dir'])/'CPU_MODULES.json' if 'attempt_dir' in plan else None
  if q and q.exists():
   v=json.loads(q.read_text());assert not v['forbidden_modules'] and not v['forbidden_pythonpath'];modules.append(dict(run_id=row['id'],modules=pin(q),execution_mode=v['execution_mode'],loaded_module_count=len(v['loaded_modules'])))
totals={k:ledger['historical_charged'][k]+ledger['new_used'][k]+sum(r['resources'].get(k,0) for r in ledger['new_reservations']) for k in ledger['caps']}
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
save(e/'FINAL_RESOURCE_AND_CLEANUP_AUDIT.json',dict(status='SELF_CHECKED_PENDING_INDEPENDENT_REVIEW',ledger=pin(b/'RESOURCE_LEDGER.json'),resource_policy=pin(b/'continuation-20261101/RESOURCE_POLICY_USER_AMENDMENT.json'),conservative_used_plus_unreleased_reservations=totals,reservations=ledger['new_reservations'],closed_runs=runs,owner_observations=len(owners),all_recorded_closed_run_owners_gone=True,stops_present=stop_paths(),limitations=['Totals are accounting charges plus unreleased reservations, not actual workload counters','Unknown platform usage preserved','Coverage is receipts attached to current direct ledger, not unrelated system processes'],rustdesk_touched=False))
save(b/'NO_RLFLOW_AUDIT.json',dict(status='PASS_RECORDED_MODULE_SNAPSHOT_SCOPE',prior_cpu_audit=pin(b/'NO_RLFLOW_CPU_AUDIT_20261101.json'),module_snapshots=modules,implementation_guard=pin(b.parents[2]/'tools/r6_direct/dependencies.py') if False else pin(Path('/home/mint/rl_dev/instinctRL-dev/tools/r6_direct/dependencies.py')),video_analysis=pin(e/'render-09/analysis-02/binding.json'),limitations=['Snapshots do not prove complete lifetime import history','Direct entry guard and reviewed source supplement snapshots','No Store/RPC state required for execution; old documents remain historical']))
save(e/'MODEL_USAGE_CONTINUATION_AUDIT.json',dict(status='KNOWN_HISTORY_PRESERVED_CURRENT_PLATFORM_USAGE_UNKNOWN',historical_audit=pin(b/'MODEL_USAGE_AUDIT.json'),ledger=pin(b/'RESOURCE_LEDGER.json'),model_charge_settlements=ledger.get('model_charge_settlements'),continuation_usage_unknown=ledger.get('continuation_usage_unknown'),current_platform_sessions='UNKNOWN',current_platform_tokens='UNKNOWN',independent_reviewer='/root/baseline_scientific_review',waiting='Local jobs waited through ordinary Python supervisors/tool sessions; reviewer calls and interactive goal continuations are not assumed zero-cost',policy='User canceled cumulative caps/protected allocations; unknown usage and unreleased charges retained; no invented exact total',limitations=['Historical239/506 are scoped historical records, not current global total','Platform continuation model usage is unavailable to project ledger']))
print('AUDITS',len(runs),'runs',len(modules),'module snapshots',len(owners),'owner observations')
