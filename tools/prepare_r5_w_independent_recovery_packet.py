"""Prepare the exact independently reviewed recovery after FINAL0; never submit."""
from pathlib import Path
from copy import deepcopy
from tools.r5_w_commander import ROOT, BASE, read, save, pin, sha


def main():
    from training_workloads.unified_h1.w_independent import validate_parameters, required_budget
    from training_workloads.unified_h1.w_closure import validate_compatibility
    old=BASE/'experiments/R5-W-L1-INDEPENDENT-01'
    new=BASE/'experiments/R5-W-L1-INDEPENDENT-RECOVERY1'
    source=read(new/'readonly-snapshot.json');snapshot=Path(source['snapshot'])
    configpath=snapshot/'training_workloads/unified_h1/candidate_config.json';config=read(configpath)
    packet=deepcopy(read(old/'launch_inputs.proposed.json'));row=packet['entries'][0]
    auth=row['authorization_proposal'];spec=row['experiment_proposal'];p=auth['parameters']
    compat=read(old/'checkpoint-compatibility.json')
    compat.update(execution_source_identity=source['source_identity'],execution_config=pin(configpath))
    cp=new/'checkpoint-compatibility.json';save(cp,compat,new=True)
    validate_compatibility(pin(cp),p['checkpoint_source_identity'],dict(config,resolved_identity=source['source_identity']))
    proto=new/'INDEPENDENT_PROTOCOL.json'
    save(proto,read(old/'INDEPENDENT_PROTOCOL.json'),new=True)
    p.update(experiment_id=new.name,evaluation_seed=20261020,evaluation_updates=[640],checkpoints=[p['checkpoints'][0]],
             checkpoint_compatibility=pin(cp),independent_validation_protocol=pin(proto))
    p.pop('experiment_card')
    card=new/'EXPERIMENT_CARD.json'
    save(card,dict(experiment_id=new.name,profiles={'R1':{k:v for k,v in p.items() if k!='checkpoints'}},
        primary_variable='INDEPENDENT_RESET_SEED_20261020_FIXED_FIRST_PASS_U640',RL_training_transitions=0,
        source=source,actual_seed=20261020,canonical_scoring_seed=20260909,
        scoring='Separate protocol validates actual seed and all160 case/lanes, then only seed metadata of memory view is canonicalized for unchanged original evaluate_checkpoint. All physics/timing/commands/initial-state/validity gates retained. Raw unchanged; original per-case primary-seed mismatch diagnostics retained.',
        limitation='Evaluation initial positions fixed by original suite; independent reset seed is not a new initial-state distribution or demonstrated broad generalization. E1 engineering gain remains separate.',
        original_primary_authorization=read(proto)['primary_authorization'],evaluation_status='NOT_EXECUTED'),new=True)
    p['experiment_card']=pin(card);mode='research-formal-evaluation'
    validate_parameters(mode,p);auth['budgets']=required_budget(mode,p)
    attempt=ROOT/'coordination/research/attempts'/f'{new.name}-R1-{mode.upper()}-{source["snapshot_commit"][:12]}'
    assert not attempt.exists();actual=attempt/'commands/AUTHORIZATION.json'
    auth.update(order_id=new.name+'-R1-'+mode.upper(),attempt=str(attempt),**source['source_identity'])
    proposed=new/'inputs/R1-research-formal-evaluation-AUTH.proposed.json';save(proposed,auth,new=True)
    spec.update(id=attempt.name,cwd=str(snapshot),seed=20261020,conditions=[],resources=dict(starts=1,wall_seconds=1810,gpu_seconds=1810,transitions=32960))
    for flag,value in [('--authorization',str(actual)),('--attempt',str(attempt))]:spec['argv'][spec['argv'].index(flag)+1]=value
    for key,rel in [('dispatcher','training_workloads/unified_h1/local_dispatch.py'),('workload','training_workloads/unified_h1/formal_unified_h1.py'),('product','isaaclab-training/instinctlab/envs/unified_h1_direct_env.py')]:spec['pins'][key]=pin(snapshot/rel)
    spec['pins']['authorization']=dict(path=str(actual),sha256=sha(proposed))
    spec['h1'].update(workload_root=str(snapshot),attempt=str(attempt),config_sha256=sha(configpath))
    row['expected_authorization']=pin(proposed)
    packet['baseline'].update({k:source[k] for k in ('snapshot','snapshot_commit','snapshot_tree','config_sha256','snapshot_diff_sha256','read_only')})
    packet.update(experiment_id=new.name,stage_allocation=pin(BASE/'stages/W-V1-002.json'),experiment_card=pin(card),generator=pin(Path(__file__)))
    save(new/'launch_inputs.proposed.json',packet,new=True)


if __name__=='__main__':main()
