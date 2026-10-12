"""Prepare one exact read-only R5 baseline and drafts. Does not authorize or launch."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import argparse

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/diagnostics/r5-readiness/inputs'
from training_workloads.unified_h1.b0_enhanced import EXPERIMENT_ID as IDENTITY


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git','--no-pager',*args],cwd=root,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null'))
def write(path,value):
    if path.exists():raise ValueError('preserve existing input: '+str(path))
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config-origin',default='docs/diagnostics/r5-readiness/inputs/baseline_snapshot_config.json')
    parser.add_argument('--prefix',default='baseline')
    args=parser.parse_args()
    if not args.prefix.replace('_','').isalnum():raise ValueError('invalid output prefix')
    output=lambda name:OUT/(args.prefix+'_'+name+'.json')
    from rlflow.policy import stopped
    if stopped(ROOT/'coordination/research'):raise RuntimeError('STOP forbids new preparation')
    ledger=json.loads((ROOT/'docs/diagnostics/r5-readiness/goal_budget.json').read_text())
    remaining={key:ledger['caps'][key]-ledger['used'][key]-ledger.get('reserved',{}).get(key,0)
        for key in ('live_starts','simulator_initializations','nontraining_transitions','wall_seconds','gpu_seconds','output_bytes')}
    if any(remaining[key]<amount for key,amount in
        {'live_starts':1,'simulator_initializations':1,'nontraining_transitions':20160,'wall_seconds':910,'gpu_seconds':910,'output_bytes':16*1024**3}.items()):
        raise RuntimeError('current goal remaining envelope cannot fit candidate')
    if any(output(name).exists() for name in ('handoff','authorization.expected','experiment_spec.draft','input_manifest')):
        raise RuntimeError('input outputs already exist; preserve them')
    commit=git(ROOT,'rev-parse','HEAD').decode().strip()
    config_origin=args.config_origin
    config_bytes=git(ROOT,'show',commit+':'+config_origin)
    config=json.loads(config_bytes);snapshot=Path(config['workload']['path'])
    if config['product']['path']!=str(snapshot) or config['method']['proposal_path']!=str(snapshot/'coordination/contracts/R1_UNIFIED_HOVER_TRAINING_PLAN.md'):
        raise ValueError('snapshot product/workload/method paths must all bind the new snapshot before creation')
    if snapshot.exists():raise ValueError('snapshot already exists; inspect instead of repeating preparation')
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','-c','core.hooksPath=/dev/null','clone','--no-hardlinks','--local','--single-branch','--branch','main',str(ROOT),str(snapshot)],check=True)
    if git(snapshot,'rev-parse','HEAD').decode().strip()!=commit:raise ValueError('source changed during clone')
    (snapshot/'training_workloads/unified_h1/candidate_config.json').write_bytes(config_bytes)
    tree=git(snapshot,'rev-parse','HEAD^{tree}').decode().strip()
    from training_workloads.unified_h1.b0_handoff import _candidate_diff
    diff=hashlib.sha256(_candidate_diff(snapshot,commit)).hexdigest()
    attempt=ROOT/'coordination/research/attempts'/IDENTITY
    if attempt.exists() and any(p.is_file() or p.is_symlink() for p in attempt.rglob('*')):
        raise ValueError('attempt already has inputs or evidence; never reuse consumed attempt')
    (attempt/'commands').mkdir(parents=True,exist_ok=True);(attempt/'results').mkdir(exist_ok=True)
    auth_path=attempt/'commands/AUTHORIZATION.json';handoff_path=output('handoff')
    execution=config['execution'];runtime=Path(execution['python_path']);asset_manifest=Path(execution['isaacsim_asset_root'])/'ASSET_MANIFEST.json'
    source={'product_repository':str(snapshot),'product_commit':commit,'product_tree':tree,
            'workload_repository':str(snapshot),'workload_base_commit':commit,'workload_base_tree':tree,
            'runtime_python':str(runtime),'runtime_python_sha256':sha(runtime),
            'asset_manifest':str(asset_manifest),'asset_manifest_sha256':sha(asset_manifest)}
    argv=[str(runtime),'-B','-m','training_workloads.unified_h1.formal_unified_h1','--authorization',str(auth_path),'--attempt',str(attempt),'--mode','enhanced-b0-diagnostic']
    from training_workloads.unified_h1.budget import required_mode_budget
    from training_workloads.unified_h1.config import METHOD_SHA256
    mode_budget=required_mode_budget('enhanced-b0-diagnostic',{'handoff':str(handoff_path)})
    handoff={'schema_version':'instinctrl-enhanced-b0-handoff/2','status':'VALIDATE_ONLY_NOT_AUTHORIZED',
        'experiment_id':IDENTITY,'argv':argv,'authorization':str(auth_path),'attempt':str(attempt),
        'checkpoint':None,'source_identity':source,
        'candidate':{'repository':str(snapshot),'base_commit':commit,'commit':commit,'diff_sha256':diff},
        'frozen':{'seed':20260909,'cases':['Z','T','S'],'case_policy_steps':{'Z':200,'T':100,'S':330},'lanes':32,'decimation':20,'target_velocity_b_m_s':[0.,0.,0.],
                 'initial_state':'frozen-R4-B0','curriculum':'frozen-Z/T/S','control_frequency_hz':200,'policy_frequency_hz':10,'safety_standards':'unchanged-R4-B0'},
        'budgets':{'live_process_starts':1,'simulator_initializations':1,'nontraining_transitions':20160,'physics_rows':403200,'training_transitions':0,'optimizer_updates':0,'checkpoint_writes':0,'p0_plr':0,'policy_evaluations':0,'automatic_retries':0,'wall_seconds':910,'gpu_allocation_seconds':910,'diagnostic_bytes':16*1024**3},
        'output_manifest':{'raw_physics_jsonl':'enhanced_b0_physics.jsonl','raw_manifest':'enhanced_b0_manifest.json','causal_analysis':'enhanced_b0_causal_analysis.json','measurement_failure':'enhanced_b0_measurement_failure.json','budget_ledger':'budget_ledger.json','status':'status.json','cleanup_evidence':'cleanup_evidence.json'},
        'cleanup':{'environment_close':True,'app_close':True,'no_retry_on_failure':True}}
    write(handoff_path,handoff)
    expected_auth={'schema_version':'unified-h1-r3-authorization.v2','status':'AUTHORIZED','order_id':IDENTITY,
        'method_sha256':METHOD_SHA256,'config_sha256':hashlib.sha256(config_bytes).hexdigest(),
        'product_commit':commit,'product_tree':tree,'workload_commit':commit,'workload_tree':tree,
        'attempt':str(attempt),'mode':'enhanced-b0-diagnostic','budgets':mode_budget,'parameters':{'handoff':str(handoff_path)}}
    write(output('authorization.expected'),expected_auth)
    pin=lambda p:{'path':str(p),'sha256':sha(p)}
    authorization_pin={'path':str(auth_path),'sha256':sha(output('authorization.expected'))}
    spec={'id':IDENTITY,'stage_id':'R5-READINESS-20261012','backend':'real','argv':argv,'cwd':str(snapshot),
          'pins':{'dispatcher':pin(snapshot/'training_workloads/unified_h1/local_dispatch.py'),'executable':pin(runtime),
                  'product':pin(snapshot/'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py'),
                  'workload':pin(snapshot/'training_workloads/unified_h1/formal_unified_h1.py'),'runtime':pin(runtime),'authorization':authorization_pin},
          'method':'unified-h1-v1','evaluation':'metric-h1-v1','runtime':execution['frozen_runtime_root'],'seed':20260909,
          'resources':{'starts':1,'wall_seconds':910,'gpu_seconds':910,'transitions':20160},
          'conditions':[], # Independent PASS evidence predicate is added before Commander proposal.
          'observations':['status','counters','process_identity','process_start','raw_physics','cleanup','partial_failure'],
          'h1':{'workload_root':str(snapshot),'attempt':str(attempt),'mode':'enhanced-b0-diagnostic','config_sha256':hashlib.sha256(config_bytes).hexdigest(),'gpu_devices':['0'],'gpu_accounting':'allocated_device_wall_seconds'}}
    write(output('experiment_spec.draft'),spec)
    for p in sorted(snapshot.rglob('*'),key=lambda q:len(q.parts),reverse=True):
        if p.is_symlink():continue
        p.chmod(p.stat().st_mode & ~0o222)
    snapshot.chmod(snapshot.stat().st_mode & ~0o222)
    write(output('input_manifest'),{'schema_version':'instinctrl-r5-baseline-inputs/1',
          'status':'PREPARED_NOT_RELEASED','snapshot':str(snapshot),'snapshot_commit':commit,'snapshot_tree':tree,
          'input_origin':{'commit':commit,'path':config_origin,'sha256':hashlib.sha256(config_bytes).hexdigest()},
          'registered_config_overlay':'candidate_config.json is exactly the committed snapshot config input; source commit unchanged, overlay diff explicitly hashed in handoff',
          'snapshot_diff_sha256':diff,'handoff_sha256':sha(handoff_path),'expected_authorization_sha256':authorization_pin['sha256'],
          'authorization_issued':False,'approval_issued':False,'read_only':True,
          'role_review':'PENDING_EXACT_INPUT_REVIEW','condition':'ENHANCED_B0_ZERO_TARGET',
          'supplemental_limits':{'initialization_ticks':10,'explicit_resets':3,'automatic_reset_lanes':20160,
              'prelaunch_nvidia_smi_calls':6,'runtime_cuda_memory_observations':634,'postrun_nvidia_smi_audit_calls':2},
          'probe_definitions':'prelaunch: 3 coexistence samples x2 commands; runtime: after init +3 explicit reset gates +630 step gates; postrun: max2 external cleanup audit commands, only after terminal ownership settlement',
          'remaining_goal_live_budget_before_release':remaining,
          'free_disk_bytes_observed':shutil.disk_usage(attempt).free,'generator_sha256':sha(Path(__file__))})
    print(json.dumps({'snapshot':str(snapshot),'commit':commit,'attempt':str(attempt),'status':'PREPARED_NOT_RELEASED'},indent=2))


if __name__=='__main__':main()
