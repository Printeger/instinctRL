"""Prepare exact first fixed-policy W control packet; no issuance or live start."""
from copy import deepcopy
from pathlib import Path
import json
from tools.r5_w_commander import ROOT,BASE,MASTER_ID,MASTER_EVIDENCE,read,save,pin,sha

def main():
    from rlflow.releases import verify,files
    from training_workloads.unified_h1.w_closure import validate_parameters,required_budget
    E=BASE/'experiments/R5-W-E1-CONTROL-01';source=read(E/'readonly-snapshot.json');snapshot=Path(source['snapshot'])
    configpath=snapshot/'training_workloads/unified_h1/candidate_config.json';config=read(configpath)
    baseline=read(BASE/'BASELINE_IDENTITY.json');oldconfig=read(Path(baseline['origin_config']['path']))
    delta={k:dict(origin=v,execution=config['product']['source_sha256'][k]) for k,v in oldconfig['product']['source_sha256'].items() if v!=config['product']['source_sha256'][k]}
    compat=dict(schema_version='instinctrl-w-checkpoint-compatibility/1',scope='W_MARGIN_AND_CURRICULUM_INTEGRATION',origin_source_identity=baseline['origin_source_identity'],execution_source_identity=source['source_identity'],origin_config=baseline['origin_config'],execution_config=pin(configpath),changed_product_sources=delta)
    cp=E/'checkpoint-compatibility.json';save(cp,compat,new=True)
    p=dict(experiment_id=E.name,group='R1',training_seed=20260908,evaluation_seed=20260909,start_update=512,target_update=512,checkpoint_updates=[],evaluation_updates=[512],command_profile='mixed-center',residual_neutrality_weight=.05,anchor_weight=4.,actor_lr=1e-4,critic_lr=3e-4,mean_prior_weight=10.,mean_prior_contract=pin(snapshot/'coordination/contracts/R5_ZERO_COMMAND_MEAN_PRIOR_V1.md'),output_cap_bytes=8*1024**3,master_authority=pin(BASE/'MASTER_AUTHORITY.json'),w_closure_contract=pin(snapshot/'coordination/contracts/R5_W_SAFETY_CLOSURE_V1.md'),checkpoint_source_identity=baseline['origin_source_identity'],checkpoint_compatibility=pin(cp),initialization_checkpoint=baseline['checkpoint'],origin_method_hashes=baseline['origin_method_hashes'],evaluation_checkpoint_origin='BASELINE',ics_clearance_margin_m=.15)
    card=E/'EXPERIMENT_CARD.json';save(card,dict(experiment_id=E.name,profiles={'R1':p},source=source,primary_variable='ENGINEERING_ICS_CLEARANCE_MARGIN_0_TO_0P15_FIXED_ORIGINAL_R1_U512',RL_training_transitions=0,scoring='unchanged metric-h1-v1 plus all32 actual approach requirement'),new=True)
    p=deepcopy(p);p['experiment_card']=pin(card);p['checkpoints']=[baseline['checkpoint']]
    mode='research-formal-evaluation';validate_parameters(mode,p);budget=required_budget(mode,p)
    old=ROOT/'docs/experiments/r5-formal-readiness-auto/experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01'
    spec=read(old/'inputs/R1-research-formal-evaluation.authorized.json')
    auth=read(Path(spec['pins']['authorization']['path']));attempt=ROOT/'coordination/research/attempts'/f'{E.name}-R1-{mode.upper()}-{source["snapshot_commit"][:12]}'
    assert not attempt.exists();actual=attempt/'commands/AUTHORIZATION.json'
    auth.update(order_id=E.name+'-R1-'+mode.upper(),attempt=str(attempt),mode=mode,budgets=budget,parameters=p,**source['source_identity'])
    proposed=E/'inputs/R1-research-formal-evaluation-AUTH.proposed.json';save(proposed,auth,new=True)
    spec.update(id=attempt.name,stage_id=MASTER_ID,cwd=str(snapshot),seed=p['evaluation_seed'],conditions=[],resources=dict(starts=1,wall_seconds=1810,gpu_seconds=1810,transitions=32960),argv=[config['execution']['python_path'],'-B','-m','training_workloads.unified_h1.formal_unified_h1','--authorization',str(actual),'--attempt',str(attempt),'--mode',mode])
    for k,rel in [('dispatcher','training_workloads/unified_h1/local_dispatch.py'),('workload','training_workloads/unified_h1/formal_unified_h1.py'),('product','isaaclab-training/instinctlab/envs/unified_h1_direct_env.py')]:spec['pins'][k]=pin(snapshot/rel)
    spec['pins']['authorization']=dict(path=str(actual),sha256=sha(proposed))
    spec['h1'].update(workload_root=str(snapshot),attempt=str(attempt),mode=mode,config_sha256=sha(configpath))
    parent=read(old/'launch_inputs.proposed.json');releasepath=Path(parent['baseline']['tool_release_path']);release=verify(releasepath)
    assert {n:sha(f) for n,f in files(ROOT/'tools/rlflow').items()}==release['files']
    sb={k:source[k] for k in ('snapshot','snapshot_commit','snapshot_tree','config_sha256','snapshot_diff_sha256','read_only')};sb.update(tool_release_id=release['id'],tool_release_path=str(releasepath))
    packet=dict(schema_version='instinctrl-w-exact-inputs/1',status='PREPARED_NOT_APPROVED',experiment_id=E.name,baseline=sb,master_authority=pin(BASE/'MASTER_AUTHORITY.json'),stage_allocation=pin(BASE/'stages/W-E1-001.json'),experiment_card=pin(card),entries=[dict(group='R1',mode=mode,experiment_proposal=spec,authorization_proposal=auth,expected_authorization=pin(proposed))],first_rollout_gate=parent['first_rollout_gate'],stop_paths=parent['stop_paths'],generator=pin(Path(__file__)))
    save(E/'launch_inputs.proposed.json',packet,new=True)
    print(json.dumps(pin(E/'launch_inputs.proposed.json')))
if __name__=='__main__':main()
