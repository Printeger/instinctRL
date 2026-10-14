"""Bounded formal-readiness profiles; old pilot identities remain immutable."""
from pathlib import Path
import hashlib,json,math,re
MODES=('research-training','research-evaluation')
METHOD_VERSION='unified-h1-formal-readiness-v1'
CURRICULUM_VERSION='curriculum-h1-research-pure-hover-v1'
PROFILE_FIELDS={'experiment_id','group','training_seed','evaluation_seed','max_updates','checkpoint_updates','evaluation_updates','command_profile','residual_neutrality_weight','anchor_weight','actor_lr','critic_lr','output_cap_bytes','master_authority','experiment_card'}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate_parameters(mode,value):
 if mode not in MODES or not isinstance(value,dict):raise PermissionError('unknown research profile')
 expected=PROFILE_FIELDS|({'checkpoints'} if mode=='research-evaluation' else set())
 if set(value)!=expected:raise PermissionError('research profile fields mismatch')
 p=dict(value)
 if not isinstance(p['experiment_id'],str) or not re.fullmatch(r'[A-Z0-9-]{3,96}',p['experiment_id']):raise PermissionError('research identity invalid')
 if p['group'] not in {'R0','R1'}:raise PermissionError('research group unknown')
 for k in ['training_seed','evaluation_seed']:
  if type(p[k]) is not int or not 0<=p[k]<2**31:raise PermissionError('research seed invalid')
 if type(p['max_updates']) is not int or p['max_updates'] not in [64,128,256]:raise PermissionError('research horizon outside bound')
 for k in ['checkpoint_updates','evaluation_updates']:
  a=p[k]
  if not isinstance(a,list) or not a or any(type(u) is not int or not 1<=u<=p['max_updates'] for u in a) or sorted(set(a))!=a:raise PermissionError('research nodes invalid')
 if p['checkpoint_updates'][-1]!=p['max_updates'] or not set(p['evaluation_updates'])<=set(p['checkpoint_updates']):raise PermissionError('research nodes not persisted')
 if p['command_profile']!='pure-hover':raise PermissionError('research command distribution not registered')
 for k in ['residual_neutrality_weight','anchor_weight','actor_lr','critic_lr']:
  if type(p[k]) not in [int,float] or not math.isfinite(p[k]) or p[k]<0:raise PermissionError('research scalar invalid')
 if p['actor_lr']!=1e-4 or p['critic_lr']!=3e-4:raise PermissionError('research LR profile unregistered')
 if p['residual_neutrality_weight']!={'R0':0.,'R1':.05}[p['group']]:raise PermissionError('research group/weight mismatch')
 if type(p['output_cap_bytes']) is not int or not 4*1024**3<=p['output_cap_bytes']<=16*1024**3:raise PermissionError('research output limit invalid')
 for k in ['master_authority','experiment_card']:
  pin=p[k]
  if not isinstance(pin,dict) or set(pin)!={'path','sha256'}:raise PermissionError('research delegated binding invalid')
  path=Path(pin['path'])
  if not path.is_absolute() or path.is_symlink() or not path.is_file() or digest(path)!=pin['sha256']:raise PermissionError('research delegated binding drift')
 master=json.loads(Path(p['master_authority']['path']).read_text());card=json.loads(Path(p['experiment_card']['path']).read_text())
 if master['id']!='R5-FORMAL-READINESS-AUTO-01' or master['status']!='REGISTERED_USER_DELEGATION':raise PermissionError('research master delegation invalid')
 if card['experiment_id']!=p['experiment_id'] or card['profiles'][p['group']]!={k:v for k,v in p.items() if k not in ['checkpoints','experiment_card']}:raise PermissionError('research card/profile mismatch')
 if mode=='research-evaluation':
  rows=p['checkpoints']
  if not isinstance(rows,list) or [r.get('update') for r in rows if isinstance(r,dict)]!=p['evaluation_updates']:raise PermissionError('research checkpoint nodes mismatch')
  for r in rows:
   if set(r)!={'path','sha256','update'}:raise PermissionError('research checkpoint fields mismatch')
   f=Path(r['path'])
   if not f.is_absolute() or f.is_symlink() or not f.is_file() or digest(f)!=r['sha256']:raise PermissionError('research checkpoint identity drift')
 return p

def method_hashes(config,p):
 result=dict(config['method']['hashes'])
 effective={k:p[k] for k in ['residual_neutrality_weight','anchor_weight','actor_lr','critic_lr','command_profile']}
 result['research_effective_method']=hashlib.sha256(json.dumps(effective,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return result

def required_budget(mode,p):
 p=validate_parameters(mode,p)
 common={'live_process_starts':1,'simulator_initializations':1,'initialization_ticks':10}
 if mode=='research-training':
  transitions=p['max_updates']*512
  return {**common,'training_transitions':transitions,'nontraining_transitions':0,'optimizer_updates':p['max_updates'],'checkpoint_writes':len(p['checkpoint_updates'])+1,'explicit_resets':1,'automatic_resets':transitions,'wall_seconds':3600 if p['max_updates']<=128 else 7200}
 n=len(p['evaluation_updates']);return {**common,'training_transitions':0,'nontraining_transitions':n*6400,'optimizer_updates':0,'checkpoint_writes':0,'explicit_resets':n,'automatic_resets':n*6400,'wall_seconds':1800}
