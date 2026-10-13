"""CPU verification and science for the preregistered contemporaneous C0/C1 pair."""
from collections import Counter
import argparse
import json
from pathlib import Path
import torch
from tools.analyze_r5_paired_pilot import training,evaluation,stats,vectors,digest,read
from training_workloads.unified_h1.pilot_contract import pilot_residual_weight

BASE=Path(__file__).resolve().parents[1]/'docs/experiments/r5-zero-correction-20261014'

def verify_reward(attempt,group):
    path=attempt/'results/pilot-training/learning_log/compact_rollouts.jsonl'
    values=[];means=[];terms=[];velocities=[];resets=0;max_error=0.;rows=0
    with path.open() as stream:
        for line in stream:
            r=json.loads(line);corr=torch.tensor(r['pre_reset']['control']['v_corr_b'],dtype=torch.float64)
            command=torch.tensor(r['command_b'],dtype=torch.float64)
            actual=torch.tensor(r['reward_components']['reward_zero_extra'],dtype=torch.float64).reshape(32)
            expected=-pilot_residual_weight(group)*(command.norm(dim=-1)<=.001)*(corr/.1).square().sum(-1)
            error=float((actual-expected).abs().max());max_error=max(max_error,error)
            assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-5),'actual coefficient/decoded correction mismatch'
            values.extend(corr.tolist());means.extend((torch.tensor(r['action_mean'],dtype=torch.float64)[:,1:4]-.5).tolist())
            terms.extend(actual.tolist());velocities.extend(r['pre_reset']['task']['actual_velocity_b'])
            resets+=int(torch.tensor(r['terminated']).logical_or(torch.tensor(r['truncated'])).sum());rows+=1
    assert rows==1024
    return {'actual_reward_formula':'PASS_ALL32768_SAMPLES','raw_path':str(path),'raw_sha256':digest(path),
            'maximum_absolute_formula_error':max_error,'sampled_issued_residual':vectors(torch.tensor(values)),
            'recorded_actor_mean_residual':vectors(torch.tensor(means)), 'actual_velocity_b':vectors(torch.tensor(velocities)),
            'new_reward_zero_extra':stats(terms),'reset_lanes':resets}

def compare_inputs(train_rows):
    initial=[torch.load(Path(r['checkpoints'][0]['path']),map_location='cpu',weights_only=True) for r in train_rows.values()]
    for section in ['actor','critic']:
        assert initial[0][section].keys()==initial[1][section].keys()
        assert all(torch.equal(initial[0][section][k],initial[1][section][k]) for k in initial[0][section]),section
    for k in ['source_identity','network_config','ppo_config','training_seed','curriculum','counters']:
        assert initial[0][k]==initial[1][k],k
    for r in initial:
        assert r['actor_optimizer']['state']==r['critic_optimizer']['state']=={}
        assert r['ppo_config']['actor_learning_rate']==1e-4 and r['ppo_config']['critic_learning_rate']==3e-4
    left,right=[dict(r['method_hashes']) for r in initial]
    assert left.pop('pilot_reward_binding')!=right.pop('pilot_reward_binding') and left==right
    del initial
    first=[torch.load(Path(r['checkpoints'][0]['path']).parents[1]/'pilot_diagnostic/update_1_full_input.pt',map_location='cpu',weights_only=True) for r in train_rows.values()]
    result={}
    for field in ['actions','old_log_probabilities','old_alpha','old_beta','privileged']:
        a,b=first[0][field],first[1][field]
        assert a.shape==b.shape and torch.equal(a,b),field
        result[field]={'shape':list(a.shape),'bit_exact_equal':True}
    assert first[0]['policy'].keys()==first[1]['policy'].keys()
    for field in first[0]['policy']:
        a,b=first[0]['policy'][field],first[1]['policy'][field]
        assert a.shape==b.shape and torch.equal(a,b),field
        result['policy.'+field]={'shape':list(a.shape),'bit_exact_equal':True}
    for field in ['advantages','returns']:
        a,b=first[0][field],first[1][field]
        result[field]={'bit_exact_equal':torch.equal(a,b),'difference_C1_minus_C0':stats(b-a),
                       'scope':'Allowed real learning-signal difference due solely preregistered reward; not required equal.'}
    return result

def pair():
    packet=read(BASE/'launch_inputs.fd-fix-proposed.json');source=packet['baseline']['snapshot_commit']
    trains={g:read(BASE/f'{g}-training-data-analysis.json') for g in ['C0','C1']}
    evaluations={g:read(BASE/f'{g}-evaluation-data-analysis.json') for g in ['C0','C1']}
    assert all(r['scientific_training_coverage_complete'] and r['source_commit']==source for r in trains.values())
    assert all(r['coverage_complete'] and r['source_commit']==source for r in evaluations.values())
    comparison=compare_inputs(trains);groups={}
    for g in ['C0','C1']:
        nodes={}
        for u in [16,64]:
            node=evaluations[g]['u16_u64'][str(u)];episodes=node['episodes'];assert len(episodes)==32
            raw=Path(node['raw_path']);rows=[json.loads(line) for line in raw.open()];assert rows[0]['kind']=='T0'
            steps=[r for r in rows if r['kind']=='STEP'];assert len(steps)==200 and [r['step'] for r in steps]==list(range(200))
            actual=torch.tensor([r['actual_velocity_b'] for r in steps]);reset=torch.tensor([r['reset_mask'] for r in steps])
            nodes[str(u)]={'original_gate':node['original_gate'],'episodes':episodes,'raw_path':str(raw),'raw_sha256':digest(raw),
                'speed_rms_m_s':stats([r['metrics']['speed'] for r in episodes]),'horizontal_drift_m':stats([r['metrics']['drift'] for r in episodes]),'height_deviation_m':stats([r['metrics']['height'] for r in episodes]),
                'actual_evaluation_velocity_b':vectors(actual.reshape(-1,3)),'actual_reset_lanes':int(reset.sum()),
                'engineering':{k:sum(bool(r['engineering'][k]) for r in episodes) for k in ['missing','reset_interrupted','collision','height_failure','evidence_valid']},
                'failure_reasons':dict(Counter(r['reason'] for r in episodes)),
                'scope':'All32 lanes, sensorT0/startup/originalthresholds; no reset splice or exclusion. Evalv_corr not logged by originalevaluator; trainingactualv_corr is separatelyidentified.'}
        probes=trains[g]['actual_first_update_probe']
        groups[g]={'weight':pilot_residual_weight(g),'actor_lr':1e-4,'critic_lr':3e-4,'training_updates':64,'training_transitions':32768,'training_checkpoint_writes':7,
            'live_first_rollout_gate':trains[g]['first_real_rollout_gate'],'training_reward_validation':trains[g]['actual_reward_validation'],
            'first_update_analytic_beta_kl_by_minibatch':[r['analytic_beta_kl_post']['mean'] for r in probes],
            'first_update_parameter_delta_l2_by_minibatch':[r['parameter_delta_l2_post'] for r in probes],
            'first_update_actor_gradient_norms_by_minibatch':[{'unclipped':r['unclipped_gradient_norm'],'clipped':r['clipped_gradient_norm']} for r in probes],
            'per_update_actor_gradient_norms':[r['actor_pre_clip_gradient_norm'] for r in trains[g]['per_update']],
            'first_update_mean_residual_xyz_by_minibatch':[[a['mean'] for a in r['mean_v_corr_b_post']['xyz']] for r in probes],
            'actual_training_correction_u16':trains[g]['executed_training_corrections'][15], 'actual_training_correction_u64':trains[g]['executed_training_corrections'][63],
            'Z_u16_u64':nodes,'all_original_Z_nodes_pass':evaluations[g]['all_original_gates_pass']}
    a,b=groups['C0']['Z_u16_u64']['64'],groups['C1']['Z_u16_u64']['64']
    delta={k:{'C0':a[k]['max'],'C1':b[k]['max'],'C1_minus_C0':b[k]['max']-a[k]['max']} for k in ['speed_rms_m_s','horizontal_drift_m','height_deviation_m']}
    successes=[groups[g]['Z_u16_u64']['64']['original_gate']['successes'] for g in ['C0','C1']]
    positive=(successes[1]>successes[0] or (b['horizontal_drift_m']['max']<a['horizontal_drift_m']['max']*.9 and b['speed_rms_m_s']['max']<=max(.08,a['speed_rms_m_s']['max']) and b['height_deviation_m']['max']<=max(.2,a['height_deviation_m']['max'])))
    return {'schema_version':'instinctrl-r5-zero-correction-pair-results/1','status':'COMPLETE_ACTUAL_PAIR_PENDING_FINAL_INDEPENDENT_REVIEW',
        'candidate_scientific_status':'VALID_EXPERIMENT_POSITIVE' if positive else 'VALID_EXPERIMENT_NEGATIVE',
        'shared_source_commit':source,'config_sha256':packet['baseline']['config_sha256'],'method_sha256':packet['method_sha256'],
        'single_effective_variable':'w0 versus w.05','preregistered_primary_comparison':'u64','diagnostic_node':'u16',
        'first_actual_rollout_comparability':comparison,'init_actor_critic_bit_exact_equal':True,'PPO_config_equal':True,
        'groups':groups,'primary_u64_max_lane_deltas':delta,'primary_u64_successes':{'C0':successes[0],'C1':successes[1]},
        'historical_PLR_only':{'speed':.0565321147441864,'drift':1.0888241529464722,'height':.04665946960449219,'not_contemporaneous_control':True},
        'original_Z_both_pilot_nodes_pass_C1':groups['C1']['all_original_Z_nodes_pass'], 'u64_original_Z_gate_pass_C1':b['original_gate']['pass'],'long_training_authorized':False,'automatic_retries':0,
        'limits':['One pairedseed, notstatistical/generalizationproof','Newreward objectiveexperiment, notpaperinnovation','Actualtrainingresidual anddeterministicZ areseparatetrajectories','No missingfailures dropped orresetsplicing','NativeFAILED/INCOMPLETE/fullreservations preservedseparately','OriginalZonly; T/S/X/W andfulltrainingnotvalidated']}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['training','evaluation','pair'],required=True);parser.add_argument('--group',choices=['C0','C1']);args=parser.parse_args();torch.set_num_threads(2)
    packet=read(BASE/'launch_inputs.fd-fix-proposed.json');source=packet['baseline']['snapshot_commit']
    if args.mode=='pair': result=pair();target=BASE/'pair_analysis.json'
    else:
        assert args.group is not None
        r=next(r for r in packet['entries'] if r['group']==args.group and r['mode']=='pilot-'+args.mode);attempt=Path(r['authorization_proposal']['attempt'])
        result=(training if args.mode=='training' else evaluation)(attempt,args.group,source_commit=source)
        if args.mode=='training':result['actual_reward_validation']=verify_reward(attempt,args.group)
        target=BASE/f'{args.group}-{args.mode}-data-analysis.json'
    assert not target.exists();result.update(analysis_tool_sha256=digest(Path(__file__)),CPU_only=True,analysis_optimizer_updates=0,missing_samples_filled=False)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'output':str(target),'coverage':result.get('scientific_training_coverage_complete',result.get('coverage_complete')),'status':result.get('candidate_scientific_status')}))

if __name__=='__main__':main()
