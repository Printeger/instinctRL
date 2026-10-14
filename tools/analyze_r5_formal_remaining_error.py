"""Finite CPU attribution from accepted C0/C1 evidence; never changes scoring."""
from pathlib import Path
import hashlib,json,math,statistics
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'docs/experiments/r5-zero-correction-20261014'
OUT=ROOT/'docs/experiments/r5-formal-readiness-auto'
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def stats(values):
 v=[float(x) for x in values];assert v and all(math.isfinite(x) for x in v)
 return {'n':len(v),'mean':statistics.fmean(v),'min':min(v),'max':max(v),'rms':math.sqrt(statistics.fmean(x*x for x in v))}
def vector(values):return [stats([x[i] for x in values]) for i in range(3)]
def flatten(v):return [x[0] for x in v]
def analyze():
 pair=read(OLD/'pair_analysis.complete.json');result={}
 for g in ['C0','C1']:
  inv=read(OLD/f'live-{g}-training/INVENTORY.json');an=read(OLD/f'{g}-training-data-analysis.json');raw=Path(an['actual_reward_validation']['raw_path']);row=next(r for r in inv['files'] if r['original']==str(raw));assert digest(raw)==row['sha256']
  rows=[json.loads(l) for l in raw.read_text().splitlines()];assert len(rows)==1024
  windows={}
  for u in [1,2,16,64]:
   rr=[r for r in rows if r['update']==u];assert len(rr)==16
   anchor=[r['pre_reset']['control']['anchor_metrics'] for r in rr]
   aa={k:stats([x for a in anchor for x in flatten(a[k])]) for k in ['active','usable_fraction','huber_loss','mean_absolute_error_m']}
   codes=[int(x) for a in anchor for x in flatten(a['reset_code'])]
   aa['reset_codes']={str(c):codes.count(c) for c in sorted(set(codes))}
   aa['reward_components']={k:stats([x for r in rr for x in flatten(r['reward_components'][k])]) for k in ['reward_anchor','reward_velocity','reward_zero_extra','reward_smooth','reward_total']}
   windows[str(u)]=aa
  evals={}
  for u in ['16','64']:
   n=pair['groups'][g]['Z_u16_u64'][u];p=Path(n['raw_path']);assert digest(p)==n['raw_sha256'];er=[json.loads(l) for l in p.read_text().splitlines()];t0=er[0];ss=er[1:];assert len(ss)==200 and [r['step'] for r in ss]==list(range(200))
   initial=t0['initial_sensor_position_w_m'];lanes=[]
   for lane in range(32):
    ids=[r['episode_ids'][lane] for r in ss];assert len(set(ids))==1 and not any(r['reset_mask'][lane] for r in ss)
    pos=[initial[lane]]+[r['sensor_positions_w_m'][lane] for r in ss]
    sections={}
    for name,start,end in [('startup_0_to_2s',0,20),('middle_2_to_10s',20,100),('late_10_to_20s',100,200),('last_4s',160,200)]:
     delta=[pos[end][i]-pos[start][i] for i in range(3)];dt=.1*(end-start)
     sections[name]={'sensor_world_displacement_m':delta,'sensor_world_endpoint_secant_m_s':[x/dt for x in delta],'net_XY_displacement_m':math.hypot(*delta[:2]),'actual_body_velocity_m_s':vector([r['actual_velocity_b'][lane] for r in ss[start:end]]),'fixed_T0_XY_distance_at_start_m':math.dist(pos[start][:2],pos[0][:2]),'fixed_T0_XY_distance_at_end_m':math.dist(pos[end][:2],pos[0][:2])}
    lanes.append({'lane':lane,'sections':sections,'full_original_metrics':n['episodes'][lane]['metrics']})
   evals[u]={'raw_path':str(p),'raw_sha256':digest(p),'evaluation_seed':t0['seed'],'lanes':lanes,'original_gate':n['original_gate'],'summary':{name:{'XY_net_displacement':stats([r['sections'][name]['net_XY_displacement_m'] for r in lanes]),'endpoint_secant_world_xyz':vector([r['sections'][name]['sensor_world_endpoint_secant_m_s'] for r in lanes])} for name in lanes[0]['sections']},'limits':'Sensor world displacement exact from recorded positions. Perstep world velocity/attitude and evaluation v_corr/anchor absent: no exact velocity integral/body-world substitution or fabricated feedback. Section boundaries diagnostic only; original score keeps entire T0+200steps.'}
  cp=next(r for r in an['checkpoints'] if str(r.get('label',r.get('name',''))) in ['u64','64']) if any(str(r.get('label',r.get('name',''))) in ['u64','64'] for r in an['checkpoints']) else an['checkpoints'][-1]
  result[g]={'training_path':str(raw),'training_sha256':digest(raw),'anchor_actual_windows':windows,'evaluation_nodes':evals,'training_seed_source':'Actual checkpoint metadata (separately recorded below); not inferred from evaluation seed.'}
 return {'schema_version':'instinctrl-r5-formal-remaining-error/1','source_main_at_goal_start':'56f996d996b83ced3312d44e204b8df6b4f80a9d','CPU_only':True,'GPU_queries':0,'live_starts':0,'optimizer_updates':0,'groups':result,'decision':'Do not default anchor4to8: actual anchor active/usable and alreadylarge, poseconfounded and not explicitly actorreference. First finite falsifiable question: does longer purezero training solve the remaining persistentbias with existing C1 method? Extend both same-source w0/.05 to128 before deciding an information-boundary change; predefinedlate96/128, no mixedschedule. If plateau/unstable, move to measurement-reference representation, notblind durationincrease.','analysis_tool_sha256':digest(Path(__file__))}
if __name__=='__main__':
 p=OUT/'REMAINING_ERROR_ANALYSIS.json';assert not p.exists();r=analyze();p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'output':str(p),'C1_u64_anchor':r['groups']['C1']['anchor_actual_windows']['64'],'C1_u64_motion':r['groups']['C1']['evaluation_nodes']['64']['summary']},ensure_ascii=False))
