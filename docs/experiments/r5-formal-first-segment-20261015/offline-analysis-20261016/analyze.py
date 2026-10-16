"""Read existing formal-segment evidence on CPU; never issue acceptance or run a policy.

Run from repository root using the existing navrl Python and
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tools/rlflow:isaaclab-training.
Pass --output a fresh directory to reproduce without overwriting old evidence.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
CASE = 'R5-FORMAL-MEAN10-FIRST-SEGMENT-01'
BASE = ROOT/'docs/experiments/r5-formal-readiness-auto/experiments'/CASE
SNAP = ROOT/'coordination/research/workspaces/R5-formal-mean10-integration-01-20261015'
RESULTS = ROOT/'docs/experiments/r5-formal-first-segment-20261015'


def read(p): return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:
        for line in f: yield json.loads(line)
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def pin(p): return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def stats(x):
    x=np.asarray(x,dtype=float).reshape(-1)
    assert np.isfinite(x).all()
    if not len(x): return {'n':0}
    return {'n':len(x),'mean':float(x.mean()),'rms':float(np.sqrt(np.mean(x*x))),
            'min':float(x.min()),'p50':float(np.quantile(x,.5)),
            'p95':float(np.quantile(x,.95)),'max':float(x.max())}
def finite_tree(x):
    if isinstance(x,torch.Tensor): assert torch.isfinite(x).all()
    elif isinstance(x,dict):
        for v in x.values(): finite_tree(v)
    elif isinstance(x,(list,tuple)):
        for v in x: finite_tree(v)
    elif isinstance(x,float): assert np.isfinite(x)
def compare(a,b,path=''):
    if isinstance(a,torch.Tensor):
        return [] if isinstance(b,torch.Tensor) and torch.equal(a,b) else [path]
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():return [path]
        return sum((compare(a[k],b[k],path+'/'+str(k)) for k in a),[])
    if isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):return [path]
        return sum((compare(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))),[])
    return [] if a==b else [path]


def analyze(group):
    from tools.analyze_r5_formal_segment import verify_inventory, validate_update_logs, verify_probe
    from training_workloads.unified_h1.schedule import command_at
    sp=BASE/'inputs'/f'{group}-research-formal-training.authorized.json';spec=read(sp)
    auth=read(spec['pins']['authorization']['path']);p=auth['parameters']
    out=Path(spec['h1']['attempt'])/'results/research-formal-training'
    inventory=verify_inventory(BASE,group,'research-formal-training')
    updates=list(rows(out/'learning_log/updates.jsonl')); mini=list(rows(out/'learning_log/minibatches.jsonl'))
    rollouts=list(rows(out/'learning_log/rollouts.jsonl'))
    validate_update_logs(updates,mini,p['mean_prior_weight'])
    assert [r['update'] for r in rollouts]==list(range(257,513))
    assert all(r['statistics']==u['statistics'] for r,u in zip(rollouts,updates))
    metrics=[]; arrays={k:[] for k in ['zero','reward','speed','error','mean_correction','sample_correction','height','q','reset','projection_ratio','final_ratio','clearance']}
    components={}; terminations=Counter(); per_update=[]; direction_errors={}
    for index,row in enumerate(rows(out/'learning_log/compact_rollouts.jsonl')):
        assert (row['update'],row['step'])==(257+index//16,index%16)
        cmd=np.asarray(row['command_b']);zero=np.linalg.norm(cmd,axis=1)<=.001
        assert np.array_equal(cmd,command_at((row['update']-1)*16+row['step']).numpy())
        pre=row['pre_reset'];task=pre['task'];ctrl=pre['control']
        vel=np.asarray(task['actual_velocity_b']);mean=np.asarray(row['action_mean']);sample=np.asarray(row['action_sample'])
        corr=np.asarray(ctrl['v_corr_b']);final=np.asarray(ctrl['v_final_b'])
        assert np.allclose(corr,sample[:,1:4]-.5,rtol=1e-5,atol=1e-6)
        rc={k:np.asarray(v,dtype=float).reshape(32) for k,v in row['reward_components'].items()}
        assert np.allclose(rc['reward_zero_extra'],-.05*zero*np.sum((corr/.1)**2,axis=1),rtol=1e-5,atol=1e-6)
        assert np.allclose(sum(v for k,v in rc.items() if k!='reward_total'),rc['reward_total'],rtol=1e-5,atol=1e-6)
        q=np.asarray(pre['reward']['diagnostics']['q']).reshape(32).astype(bool)
        err=np.linalg.norm(vel-cmd,axis=1);speed=np.linalg.norm(vel,axis=1)
        assert np.allclose(rc['reward_velocity'],-np.where(q,speed,err),rtol=1e-5,atol=1e-6)
        reset=np.asarray(row['terminated']).reshape(32)|np.asarray(row['truncated']).reshape(32)
        vals={'zero':zero,'reward':rc['reward_total'],'speed':speed,'error':err,
              'mean_correction':np.linalg.norm(mean[:,1:4]-.5,axis=1),
              'sample_correction':np.linalg.norm(corr,axis=1),
              'height':np.abs(np.asarray(pre['root_state_w'])[:,2]-2),
              'q':q,'reset':reset,'clearance':np.asarray(task['exact_clearance_m']).reshape(32),
              'projection_ratio':np.sum(vel*cmd,axis=1)/np.maximum(np.sum(cmd*cmd,axis=1),1e-12),
              'final_ratio':np.linalg.norm(final,axis=1)/np.maximum(np.linalg.norm(cmd,axis=1),1e-12)}
        for k,v in vals.items():assert np.isfinite(v).all();arrays[k].append(v)
        for k,v in rc.items():components.setdefault(k,[]).append(v)
        for k,v in task['termination']['masks'].items():terminations[k]+=int(np.asarray(v).sum())
        for lane in np.flatnonzero(~zero):
            key=','.join(f'{v:.2f}' for v in cmd[lane]);direction_errors.setdefault(key,[]).append(err[lane])
        if row['step']==15:
            u={k:np.concatenate(arrays[k][-16:]) for k in arrays};z=u['zero']
            summary={'update':row['update'],'reward':float(u['reward'].mean()),'zero_reward':float(u['reward'][z].mean()),
                     'nonzero_reward':float(u['reward'][~z].mean()),'zero_speed_rms':stats(u['speed'][z])['rms'],
                     'nonzero_error_rms':stats(u['error'][~z])['rms'],
                     'zero_mean_correction_rms':stats(u['mean_correction'][z])['rms'],
                     'zero_sample_correction_rms':stats(u['sample_correction'][z])['rms']}
            for role,mask in [('zero',z),('nonzero',~z)]:
                for k in components:
                    actual=np.concatenate(components[k][-16:])[mask]
                    saved=rollouts[index//16]['groups'][role][k]
                    assert saved['count']==len(actual) and np.isclose(saved['signed_mean'],actual.mean(),rtol=1e-5,atol=1e-7)
            summary['gae_target_mean']=updates[index//16]['statistics']['returns']['signed_mean']
            summary['explained_variance']=updates[index//16]['statistics']['explained_variance']['value']
            per_update.append(summary)
    assert len(per_update)==256 and index+1==4096
    arrays={k:np.stack(v) for k,v in arrays.items()};components={k:np.stack(v) for k,v in components.items()}
    periods={}
    for name,start,end in [('all',0,256),('first32',0,32),('middle32',112,144),('last32',224,256)]:
        a={k:v[start*16:end*16].reshape(-1) for k,v in arrays.items()};z=a['zero']
        record={'updates':[257+start,256+end],'transitions':len(z),'zero_samples':int(z.sum()),'nonzero_samples':int((~z).sum()),
                'reward':stats(a['reward']),'q_count':int(a['q'].sum()),'reset_count':int(a['reset'].sum())}
        for role,mask in [('zero',z),('nonzero',~z)]:
            record[role]={k:stats(a[k][mask]) for k in ['reward','speed','error','mean_correction','sample_correction','height','clearance']}
            record[role]['reward_components']={k:float(v[start*16:end*16].reshape(-1)[mask].mean()) for k,v in components.items()}
        record['nonzero']['projection_ratio']=stats(a['projection_ratio'][~z]);record['nonzero']['final_ratio']=stats(a['final_ratio'][~z])
        periods[name]=record
    cp_records=[]
    for u in [272,288,320,384,512]:
        path=out/f'checkpoints/u{u}.pt';c=torch.load(path,map_location='cpu',weights_only=True);finite_tree(c)
        assert read(path.with_suffix('.json'))['sha256']==sha(path)
        assert c['counters']=={'update':u,'transition':u*512,'global_policy_step':u*16,'actor_optimizer':u*8,'critic_optimizer':u*8}
        assert c['source_identity']=={k:auth[k] for k in ['product_commit','product_tree','workload_commit','workload_tree','config_sha256']}
        assert c['training_seed']==p['training_seed']
        assert c['method_hashes']['origin_checkpoint_sha256']==p['initialization_checkpoint']['sha256']
        assert c['method_hashes']['reset_distribution']=='center-only-v1'
        assert c['ppo_config']['actor_learning_rate']==p['actor_lr'] and c['ppo_config']['critic_learning_rate']==p['critic_lr']
        for k in ['actor_optimizer','critic_optimizer']:assert {int(v['step']) for v in c[k]['state'].values()}=={8*u}
        cp_records.append({**pin(path),'counters':c['counters'],'method_hashes':c['method_hashes'],'tensor_finiteness':True})
    origin_path=Path(p['initialization_checkpoint']['path']);assert sha(origin_path)==p['initialization_checkpoint']['sha256']
    origin=torch.load(origin_path,map_location='cpu',weights_only=True)
    data=torch.load(out/'pilot_diagnostic/update_257_full_input.pt',map_location='cpu',weights_only=True)
    probe=torch.load(out/'pilot_diagnostic/first_update_probe.pt',map_location='cpu',weights_only=True)
    verify_probe(data,probe,mini,p,origin)
    first_probe=[{'minibatch':i,'KL_post':stats(r['after']['analytic_beta_kl'].numpy()),
                  'mean_correction_post':stats(torch.linalg.vector_norm(r['after']['mean_v_corr_b'],dim=-1).numpy()),
                  'std_post':stats(r['after']['action_std'][:,1:4].numpy())} for i,r in enumerate(probe['minibatches'])]
    episodes_raw=list(rows(out/'learning_log/episodes.jsonl'));blocks=list(rows(out/'learning_log/command_blocks.jsonl'))
    unique={};duplicate_keys=[]
    for e in episodes_raw:
        key=(e['env'],e['episode'],e['reset'])
        if key in unique:
            assert unique[key]==e,'conflicting duplicate episode'
            duplicate_keys.append(key)
        else:unique[key]=e
    episodes=list(unique.values())
    assert sum(e['length'] for e in episodes)==131072
    episode_groups={}
    for eid in sorted({e['episode'] for e in episodes}):
        es=[e for e in episodes if e['episode']==eid]
        episode_groups[str(eid)]={'records':len(es),'lengths':dict(Counter(e['length'] for e in es)),
          'right_censored':sum(e['right_censored'] for e in es),'reward_per_step':sum(e['return'] for e in es)/sum(e['length'] for e in es),
          'max_horizontal_displacement_from_episode_start':stats([e['behavior']['max_horizontal_drift_m'] for e in es]),
          'max_height_deviation':stats([e['behavior']['max_height_deviation_m'] for e in es])}
    assert sum(b['duration_steps'] for b in blocks)==131072
    block_groups={}
    for role in ['zero','nonzero']:
        bs=[b for b in blocks if (np.linalg.norm(b['command'])<=.001)==(role=='zero')]
        continuous=[b for b in bs if not b['reset_interrupted']]
        confirmed=[b for b in continuous if b['stop_confirmation']['status']=='CONFIRMED']
        eligible=[b for b in continuous if np.linalg.norm(b['previous_command'])>.001] if role=='zero' else []
        block_groups[role]={'records':len(bs),'reset_interrupted':sum(b['reset_interrupted'] for b in bs),
          'right_censored':sum(b['right_censored'] for b in bs),'continuous_records':len(continuous),
          'continuous_block_speed_rms':stats([b['speed_rms_m_s'] for b in continuous]),
          'continuous_block_max_displacement':stats([b['max_horizontal_drift_m'] for b in continuous]),
          'stop_eligible_nonreset':len(eligible),'stop_confirmed':len(confirmed),
          'confirmed_stop_time':stats([b['stop_confirmation']['confirmation_time_s'] for b in confirmed]),
          'confirmed_post_stop_drift':stats([b['post_stop_max_drift_m'] for b in confirmed])}
    logs=['updates','minibatches','rollouts','compact_rollouts','episodes','command_blocks']
    result={'group':group,'parameters':p,'inventory_check':inventory,'raw_status':read(out/'status.json'),
      'execution_summary':read(BASE/f'{group}-research-formal-training-execution-summary.json'),
      'first_update_gate':read(out/'pilot_diagnostic/first_update_gate.json'),
      'periods':periods,'per_update':per_update,'termination_counts':dict(terminations),
      'all_minibatches':{k:stats([r[k] for r in mini]) for k in ['actor_auxiliary_loss','actor_loss','value_loss','entropy','approximate_kl','clip_fraction','actor_pre_clip_gradient_norm','critic_pre_clip_gradient_norm']},
      'first_update_saved_probe':first_probe,'checkpoints':cp_records,'episodes':episode_groups,'command_blocks':block_groups,
      'episode_log_duplicates':{'raw_records':len(episodes_raw),'unique_records':len(episodes),'identical_duplicate_keys':duplicate_keys,
          'raw_length_sum':sum(e['length'] for e in episodes_raw),'unique_length_sum':sum(e['length'] for e in episodes),
          'original_file_preserved':True,'deduplication_scope':'Only this descriptive view, by identical (env, episode, reset) records'},
      'directions_error':{k:stats(v) for k,v in direction_errors.items()},
      'input_pins':[pin(sp),pin(out/'formal_initialization.json'),pin(BASE/'live'/f'{group}-research-formal-training'/'INVENTORY.json')]+[pin(out/'learning_log'/f'{n}.jsonl') for n in logs],
      'missing_binding_file':not (out/'actor_mean_prior_binding.json').exists(),
      'scientific_acceptance_granted':False,'evaluation_authorization_issued':False}
    return result,data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);target=args.output/'observations.json';assert not target.exists()
    sources=['training_workloads/unified_h1/schedule.py','training_workloads/unified_h1/logging.py',
             'training_workloads/unified_h1/formal_segment.py','training_workloads/unified_h1/mean_prior.py',
             'isaaclab-training/instinctlab/control/governor.py','isaaclab-training/instinctlab/training/ppo.py',
             'isaaclab-training/instinctlab/task/rewards_h1.py']
    for p in sources:assert sha(ROOT/p)==sha(SNAP/p),'source drift: '+p
    result={'scope':'USER_REQUESTED_DESCRIPTIVE_OFFLINE_ANALYSIS_NOT_FORMAL_ACCEPTANCE',
            'head_at_analysis':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'script':pin(__file__),'original_independent_review':pin(RESULTS/'reviews/joint-actual-training.json'),
            'sources':[pin(SNAP/p) for p in sources], 'groups':{},'actor_forwards':0,'optimizer_calls':0,'GPU_calls':0}
    inputs={}
    for group in ['R0','R1']:
        result['groups'][group],inputs[group]=analyze(group)
        print(group,'CPU descriptive analysis complete',flush=True)
    result['first_512_input_differing_fields']=compare(inputs['R0'],inputs['R1'])
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,11),sharex=True)
    panels=[('reward','Mean immediate reward'),('nonzero_error_rms','Nonzero command tracking RMS (m/s)'),
            ('zero_speed_rms','Zero-command speed RMS (m/s)'),('zero_mean_correction_rms','Zero-command mean correction RMS (m/s)'),
            ('zero_sample_correction_rms','Zero-command sampled correction RMS (m/s)'),('explained_variance','Critic explained variance')]
    for ax,(key,label) in zip(axes.flat,panels):
        for group,color in [('R0','#2463a5'),('R1','#d45b2a')]:
            data=result['groups'][group]['per_update'];x=np.array([r['update'] for r in data]);y=np.array([r[key] for r in data])
            ax.plot(x,y,color=color,alpha=.2,lw=.6)
            ax.plot(x[7:],np.convolve(y,np.ones(8)/8,mode='valid'),color=color,label=group,lw=1.5)
        ax.set_title(label);ax.grid(alpha=.25);ax.legend()
    for ax in axes[-1]:ax.set_xlabel('Absolute update (257-512)')
    fig.suptitle('R5 observed training only: raw traces + trailing 8-update means\nNo formal checkpoint acceptance or Z/T/S/X/W evaluation',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(args.output/'learning_curves.png',dpi=180);fig.savefig(args.output/'learning_curves.svg');plt.close(fig)
    print(str(target),sha(target))


if __name__=='__main__':main()
