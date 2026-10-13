"""Freeze new exact C0/C1 candidate and reviewable drafts; never release live."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/experiments/r5-zero-correction-20261014'
OLD = ROOT / 'docs/experiments/r5-paired-pilot-20261013'

def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def main():
    from rlflow.policy import stopped
    from rlflow.store import Store
    from rlflow.jobs import owns
    from rlflow.releases import verify
    assert not stopped(ROOT / 'coordination/research')
    state=Store(ROOT/'coordination/research').replay()
    assert not state['paused'] and all(j['status'] in ['FAILED','SUCCEEDED','STOPPED'] and not owns(j.get('identity')) for j in state['jobs'].values())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    relative='docs/experiments/r5-zero-correction-20261014/inputs/snapshot_config.json'
    config_bytes=subprocess.check_output(['git','show',commit+':'+relative],cwd=ROOT)
    config=json.loads(config_bytes); snapshot=Path(config['workload']['path'])
    assert not snapshot.exists()
    subprocess.run(['git','clone','--no-hardlinks','--no-checkout','--local',str(ROOT),str(snapshot)],check=True)
    subprocess.run(['git','-C',str(snapshot),'-c','core.hooksPath=/dev/null','checkout','--detach',commit],check=True)
    (snapshot/'training_workloads/unified_h1/candidate_config.json').write_bytes(config_bytes)
    tree=subprocess.check_output(['git','-C',str(snapshot),'rev-parse','HEAD^{tree}'],text=True).strip()
    from training_workloads.unified_h1.b0_handoff import _candidate_diff
    diff=hashlib.sha256(_candidate_diff(snapshot,commit)).hexdigest()
    old=json.loads((OLD/'launch_inputs.proposed.json').read_text())
    release_id=old['tools_release_content_id'];release=verify(ROOT/'coordination/research/control-releases'/release_id)
    for relative,expected in release['files'].items(): assert sha(ROOT/'tools/rlflow'/relative)==expected
    baseline={'snapshot':str(snapshot),'snapshot_commit':commit,'snapshot_tree':tree,'config_sha256':hashlib.sha256(config_bytes).hexdigest(),'snapshot_diff_sha256':diff,
              'tools_release_path':str(ROOT/'coordination/research/control-releases'/release_id),'physical_baseline':'ef68ebb5682b85aa83044fdea1c13b7d0edbf325','read_only':True}
    from training_workloads.unified_h1.budget import required_mode_budget
    from training_workloads.unified_h1.pilot_contract import pilot_residual_weight,PILOT_OUTPUT_CAP_BYTES
    rows=[]
    for group in ['C0','C1']:
        train_attempt=ROOT/'coordination/research/attempts'/f'UNIFIED-H1-R5-{group}-PILOT-TRAIN-ZERO-{commit[:12]}-20261014'
        for mode in ['pilot-training','pilot-evaluation']:
            training=mode=='pilot-training'; oldrow=next(r for r in old['entries'] if r['group']=='PLR' and r['mode']==mode)
            row=deepcopy(oldrow);spec=row['experiment_proposal'];auth=row['authorization_proposal']
            attempt=train_attempt if training else ROOT/'coordination/research/attempts'/f'UNIFIED-H1-R5-{group}-PILOT-EVAL-ZERO-{commit[:12]}-20261014'
            assert not attempt.exists()
            order=f'UNIFIED-H1-R5-{group}-PILOT-'+('TRAIN' if training else 'EVAL')
            params={'group':group,'residual_neutrality_weight':pilot_residual_weight(group),'output_cap_bytes':PILOT_OUTPUT_CAP_BYTES}
            if not training: params['checkpoints']=[{'path':str(train_attempt/f'results/pilot-training/checkpoints/u{u}.pt'),'sha256':None,'update':u} for u in [16,64]]
            auth.update(order_id=order,method_sha256=config['method']['proposal_sha256'],config_sha256=baseline['config_sha256'],product_commit=commit,product_tree=tree,workload_commit=commit,workload_tree=tree,attempt=str(attempt),parameters=params,budgets=required_mode_budget(mode,params))
            auth_path=attempt/'commands/AUTHORIZATION.json'
            argv=[config['execution']['python_path'],'-B','-m','training_workloads.unified_h1.formal_unified_h1','--authorization',str(auth_path),'--attempt',str(attempt),'--mode',mode]
            pin=lambda p:{'path':str(p),'sha256':sha(p)}
            spec.update(id=attempt.name,stage_id='INSTINCTRL-R5-ZERO-CORRECTION-20261014',argv=argv,cwd=str(snapshot),method=config['method']['version'],conditions=[],resources={'starts':1,'wall_seconds':3610 if training else 1810,'gpu_seconds':3610 if training else 1810,'transitions':32768 if training else 12800})
            spec['pins'].update(dispatcher=pin(snapshot/'training_workloads/unified_h1/local_dispatch.py'),product=pin(snapshot/'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py'),workload=pin(snapshot/'training_workloads/unified_h1/formal_unified_h1.py'),authorization={'path':str(auth_path),'sha256':None})
            spec['h1'].update(workload_root=str(snapshot),attempt=str(attempt),config_sha256=baseline['config_sha256'])
            row.update(group=group,mode=mode,experiment_proposal=spec,authorization_proposal=auth,output=str(attempt/'results'/mode),authorization_issued=False,actor_learning_rate=.0001,critic_learning_rate=.0003,residual_neutrality_weight=pilot_residual_weight(group))
            row.pop('expected_authorization',None)
            if training:
                name=f'inputs/{group}-training-authorization.proposed.json';write(OUT/name,dict(auth,status='AUTHORIZED'))
                spec['pins']['authorization']['sha256']=sha(OUT/name)
                row['expected_authorization']={'path':name,'sha256':sha(OUT/name),'status':'PROPOSED_BYTES_ONLY_NOT_ISSUED'}
            row['missing_bindings']=['new independent prelive and exact Commander approval'] if training else ['accepted actual u16/u64 source/SHA','independent actual training acceptance','exact eval Commander approval']
            rows.append(row)
    for p in sorted(snapshot.rglob('*'),key=lambda q:len(q.parts),reverse=True):
        if not p.is_symlink(): p.chmod(p.stat().st_mode & ~0o222)
    snapshot.chmod(snapshot.stat().st_mode & ~0o222)
    packet={'schema_version':'instinctrl-r5-zero-correction-launch-inputs/1','status':'UNISSUED_PENDING_INDEPENDENT_PRELIVE_AND_COMMANDER','stage_id':'INSTINCTRL-R5-ZERO-CORRECTION-20261014','baseline':baseline,'tools_release_content_id':release_id,'method_sha256':config['method']['proposal_sha256'],'first_rollout_gate':old['first_rollout_gate'],'single_variable':'zero-command residual-neutrality coefficient w0 versus w.05; bothactorLR1e-4','entries':rows,'no_automatic_retries':True,'max_training_update_per_group':64,'user_task_authority_sha256':sha(OUT/'user_task_authority.json'),'goal_resource_caps':json.loads((OUT/'goal_budget.json').read_text())['caps'],'generator_sha256':sha(Path(__file__)),'historical_control_acceptance_reused_only':'ef68 physical source and accepted original commandB0 Z/T/S; newrewardcandidate not inheriting oldreceipt approval','evaluation_release_policy':'Accepted actualtraining/firstgate/checkpoint source/hash binding and newexactCommander; originalmetric-h1-v1 allfailures/32lanes/T0/startup unchanged','stop_paths':old['stop_paths'],'evidence_collection':old['entries'][0]['evidence_collection']}
    write(OUT/'launch_inputs.proposed.json',packet)
    write(OUT/'preparation-verification.json',{'result':'PASS_STATIC_ONLY','snapshot_commit':commit,'snapshot_tree':tree,'config_sha256':baseline['config_sha256'],'overlay_diff_sha256':diff,'source_files_checked':sum(len(config[k]['source_sha256']) for k in ['product','workload']),'tool_release_files_checked':len(release['files']),'disk_free_bytes':shutil.disk_usage(ROOT).free,'old_terminal_jobs':len(state['jobs']),'new_live_starts':0,'GPU_queries':0,'new_optimizer_updates':0})
    print(json.dumps(baseline))

if __name__=='__main__': main()
