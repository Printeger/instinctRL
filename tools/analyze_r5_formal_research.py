"""CPU-only actual research evidence: exact jobs, all data and original Z gates."""
from pathlib import Path
import argparse,json
from collections import Counter
import torch
from tools.analyze_r5_paired_pilot import training,stats,vectors,digest,read,jsonl
from tools.analyze_r5_zero_correction_pair import compare_inputs
from training_workloads.unified_h1.evaluation import replay_raw_case,evaluate_episode,case_outcome
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'docs/experiments/r5-formal-readiness-auto'
def bound_entry(D,group,mode):
 name=group+'-'+mode;specpath=D/'inputs'/(name+'.authorized.json');spec=read(specpath);job=read(D/(name+'-final-job.json'));assert job['spec']==spec
 jd=ROOT/'coordination/research/jobs'/job['id'];request=read(jd/'request.json');assert request['spec']==spec and request['job_id']==job['id']
 receipt=read(jd/'receipt.json');assert receipt==job['receipt'] and receipt['status'] in ['FAILED','STOPPED','SUCCEEDED']
 ack=read(jd/'lifeguard-ack.json');assert ack['no_survivors'] and ack['descendant_cleanup']['quiescent'] and ack['descendant_cleanup']['stable_scans']>=2
 actual=Path(spec['pins']['authorization']['path']);assert digest(actual)==spec['pins']['authorization']['sha256'];auth=read(actual);assert auth['mode']==mode and auth['parameters']['group']==group
 config=read(Path(spec['cwd'])/'training_workloads/unified_h1/candidate_config.json');assert digest(Path(spec['cwd'])/'training_workloads/unified_h1/candidate_config.json')==auth['config_sha256']
 expectedargv=[config['execution']['python_path'],'-B','-m','training_workloads.unified_h1.formal_unified_h1','--authorization',str(actual),'--attempt',auth['attempt'],'--mode',mode];assert spec['argv']==expectedargv and spec['h1']['attempt']==auth['attempt']
 return Path(auth['attempt']),auth,config,{'spec_path':str(specpath),'spec_sha256':digest(specpath),'job_id':job['id'],'native_status':receipt['status'],'native_exit_code':receipt['exit_code'],'formal_counter_completeness':job['h1_evidence']['counter_completeness'],'formal_transitions':job['h1_evidence']['transitions'],'authorization_sha256':digest(actual),'source_commit':auth['product_commit'],'owned_cleanup':'NORMAL_ACK_2STABLE_NO_SURVIVORS'}

def training_analysis(D,group):
 attempt,auth,config,binding=bound_entry(D,group,'research-training');p=auth['parameters'];source=auth['product_commit'];result=training(attempt,group,source_commit=source,research_profile=p,research_config=config)
 path=attempt/'results/research-training/learning_log/compact_rollouts.jsonl';maxerror=0.;rows=0;resets=0;terms=[];anchor=[];commands=[]
 for r in jsonl(path):
  corr=torch.tensor(r['pre_reset']['control']['v_corr_b'],dtype=torch.float64);cmd=torch.tensor(r['command_b'],dtype=torch.float64)
  actual=torch.tensor(r['reward_components']['reward_zero_extra'],dtype=torch.float64).reshape(32);expected=-p['residual_neutrality_weight']*(cmd.norm(dim=-1)<=.001)*(corr/.1).square().sum(-1)
  assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-5);maxerror=max(maxerror,float((actual-expected).abs().max()));assert cmd.eq(0).all()
  terms.extend(actual.tolist());anchor.extend(torch.tensor(r['reward_components']['reward_anchor']).reshape(32).tolist());rows+=1;resets+=int(torch.tensor(r['terminated']).logical_or(torch.tensor(r['truncated'])).sum())
 assert rows==p['max_updates']*16
 result.update(binding=binding,actual_parameters=p,actual_reward_validation={'all_samples':rows*32,'maximum_absolute_formula_error':maxerror,'zero_command_all':True,'zero_extra':stats(terms),'anchor_reward':stats(anchor),'reset_lanes':resets,'raw_path':str(path),'raw_sha256':digest(path)},CPU_only=True,optimizer_calls=0,missing_samples_filled=False)
 return result

def evaluation_analysis(D,group):
 attempt,auth,config,binding=bound_entry(D,group,'research-evaluation');p=auth['parameters'];output=attempt/'results/research-evaluation';nodes={}
 for ck in p['checkpoints']:
  assert digest(Path(ck['path']))==ck['sha256'];u=ck['update'];raw=output/'evaluation_raw'/f'research_{group}_u{u}'/'Z.jsonl'
  traces=replay_raw_case(str(raw),case='Z');assert len(traces)==32 and [t.env_index for t in traces]==list(range(32))
  episodes=[evaluate_episode(t) for t in traces];producer=read(output/f'research_{group}_u{u}_Z.json');assert json.loads(json.dumps(episodes))==producer['episodes']
  rows=jsonl(raw);assert rows[0]['kind']=='T0' and rows[0]['seed']==p['evaluation_seed'];steps=[r for r in rows if r['kind']=='STEP'];assert len(steps)==200 and [r['step'] for r in steps]==list(range(200))
  gate=case_outcome([bool(r['success']) for r in episodes],['d'+str(i%8) for i in range(32)],expected_subgroups=['d'+str(i) for i in range(8)])
  actual=torch.tensor([r['actual_velocity_b'] for r in steps]);corr=torch.tensor([r['v_corr_b'] for r in steps]);final=torch.tensor([r['v_final_b'] for r in steps]);root=torch.tensor([r['root_state_w'] for r in steps]);action=torch.tensor([r['issued_action'] for r in steps]);assert torch.allclose(action[:,:,1:4]-.5,corr,atol=6e-8,rtol=0)
  for k,v in [('actual',actual),('corr',corr),('final',final),('root',root),('action',action)]:assert torch.isfinite(v).all(),k
  nodes[str(u)]={'checkpoint':ck,'raw_path':str(raw),'raw_sha256':digest(raw),'episodes':episodes,'original_gate':gate,'speed_rms_m_s':stats([r['metrics']['speed'] for r in episodes]),'horizontal_drift_m':stats([r['metrics']['drift'] for r in episodes]),'height_deviation_m':stats([r['metrics']['height'] for r in episodes]),'failure_reasons':dict(Counter(r['reason'] for r in episodes)),'engineering':{k:sum(bool(r['engineering'][k]) for r in episodes) for k in ['missing','reset_interrupted','collision','height_failure','evidence_valid']},'actual_reset_lanes':sum(sum(r['reset_mask']) for r in steps),'actual_evaluation_v_corr_b':vectors(corr.reshape(-1,3)),'actual_evaluation_v_final_b':vectors(final.reshape(-1,3)),'actual_evaluation_velocity_b':vectors(actual.reshape(-1,3)),'actual_evaluation_velocity_world':{**vectors(root[:,:,7:10].reshape(-1,3)),'frame':'world'},'policy':'Deterministic mean action, not sampledtraining. Fields actualsame-clock snapshots, no extra sensor/model sample.'}
 result={'group':group,'mode':'research-evaluation','source_commit':auth['product_commit'],'actual_parameters':p,'binding':binding,'nodes':nodes,'coverage_complete':all(len(n['episodes'])==32 for n in nodes.values()),'all_original_gates_pass':all(n['original_gate']['pass'] for n in nodes.values()),'metric':'Unchanged originalZ sensorT0/startup/200steps32lanes/allsubgroups/allfailures','CPU_only':True,'optimizer_calls':0,'missing_samples_filled':False}
 return result

def paired_results(D, directories, evaluation_directories=None):
 evaluation_directories=directories if evaluation_directories is None else evaluation_directories
 cardpath=D/'EXPERIMENT_CARD.json';card=read(cardpath)
 trains={g:read(directories[g]/(g+'-training-analysis.json')) for g in ['R0','R1']}
 ev={g:read(evaluation_directories[g]/(g+'-evaluation-analysis.json')) for g in trains}
 source=None
 for g in trains:
  t,e=trains[g],ev[g]
  assert t['scientific_training_coverage_complete'] and e['coverage_complete']
  assert t['group']==e['group']==g
  expected={**card['profiles'][g],'experiment_card':{'path':str(cardpath),'sha256':digest(cardpath)}}
  assert t['actual_parameters']==expected,'paired training must match the original frozen scientific card'
  assert {k:v for k,v in e['actual_parameters'].items() if k!='checkpoints'}==expected
  assert t['binding']['source_commit']==e['binding']['source_commit']==t['source_commit']==e['source_commit']
  if source is None:source=t['source_commit']
  assert source==t['source_commit'],'paired source mismatch'
  checkpoints={r['label']:r for r in t['checkpoints']}
  for ck in e['actual_parameters']['checkpoints']:
   actual=checkpoints['u'+str(ck['update'])]
   assert ck['path']==actual['path'] and ck['sha256']==actual['sha256'],'evaluation must use this actual training entry'
  assert list(e['nodes'])==[str(u) for u in card['predefined_late_nodes']]
 assert trains['R0']['binding']['job_id']!=trains['R1']['binding']['job_id']
 compare=compare_inputs(trains,method_binding_key='research_effective_method')
 return {'status':'COMPLETE_ACTUAL_PAIR_PENDING_INDEPENDENT_SCIENCE','experiment_id':card['experiment_id'],'source_commit':source,'first_actual_rollout_comparison':compare,'training_analyses':{g:str(directories[g]/(g+'-training-analysis.json')) for g in trains},'evaluation_analyses':{g:str(evaluation_directories[g]/(g+'-evaluation-analysis.json')) for g in trains},'groups':ev,'candidate_both_predefined_nodes_pass':ev['R1']['all_original_gates_pass'],'primary_node':card['primary_node'],'scientific_scope':'One developmentseed; notREADY beforecandidatefreeze/twonewtrainingseeds/nonzerotransition/formalpackage/finalreview','old_C1_reference':'Historicalonly; same-versionR0 isstrictcontemporaneousreference','optimizer_calls':0}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('experiment_id');ap.add_argument('mode',choices=['training','evaluation','pair']);ap.add_argument('--group',choices=['R0','R1']);ap.add_argument('--r0-entry');ap.add_argument('--r1-entry');ap.add_argument('--r0-evaluation-entry');ap.add_argument('--r1-evaluation-entry');args=ap.parse_args();torch.set_num_threads(2);D=BASE/'experiments'/args.experiment_id
 if args.mode in ['training','evaluation']:
  assert args.group;r=(training_analysis if args.mode=='training' else evaluation_analysis)(D,args.group);target=D/(args.group+'-'+args.mode+'-analysis.json')
 else:
  directories={g:(BASE/'experiments'/entry if entry else D) for g,entry in [('R0',args.r0_entry),('R1',args.r1_entry)]}
  evaluation_directories={g:(BASE/'experiments'/entry if entry else directories[g]) for g,entry in [('R0',args.r0_evaluation_entry),('R1',args.r1_evaluation_entry)]}
  r=paired_results(D,directories,evaluation_directories);target=D/'pair-results.json'
 assert not target.exists(),target;target.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'output':str(target),'coverage':r.get('scientific_training_coverage_complete',r.get('coverage_complete')),'candidate_both_predefined_nodes_pass':r.get('candidate_both_predefined_nodes_pass')}))
if __name__=='__main__':main()
