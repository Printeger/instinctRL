"""Bounded formal-readiness profiles; old pilot identities remain immutable."""
from pathlib import Path
import hashlib,json,math,re
MODES=('research-r6-final-evaluation','research-r6-candidate-evaluation','research-r6-training','research-r6-evaluation','research-training','research-evaluation','research-transition','research-formal-training','research-formal-evaluation')
TRANSITION_PROTOCOL='zero-directions-reverse-zero112-v1'
INTEGRATION_CONTRACT_SHA256='720845138986bfb053304db915e38fb6bd9558023d65177be53af79bd189a873'
MEAN_PRIOR_CONTRACT_SHA256='d0aa5731c0712d9a50a88e1ae32f29a1eacc2b1c417b925a3e2ba3a416652da9'
METHOD_VERSION='unified-h1-formal-readiness-v1'
CURRICULUM_VERSION='curriculum-h1-research-pure-hover-v1'
PROFILE_FIELDS={'experiment_id','group','training_seed','evaluation_seed','max_updates','checkpoint_updates','evaluation_updates','command_profile','residual_neutrality_weight','anchor_weight','actor_lr','critic_lr','output_cap_bytes','master_authority','experiment_card'}
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def validate_pin(pin):
 if not isinstance(pin,dict) or set(pin)!={'path','sha256'}:raise PermissionError('research input pin schema mismatch')
 path=Path(pin['path'])
 if not path.is_absolute() or path.resolve()!=path or path.is_symlink() or not path.is_file() or digest(path)!=pin['sha256']:raise PermissionError('research input pin drift')
 return path
def validate_parameters(mode,value):
 if mode=='research-r6-final-evaluation':
  from .robustness_final_contract import validate_parameters as final_parameters
  return final_parameters(mode,value)
 if mode=='research-r6-candidate-evaluation':
  from .robustness_candidate_contract import validate_parameters as candidate_parameters
  return candidate_parameters(mode,value)
 if mode=='research-r6-training':
  from .robustness_training_contract import validate_parameters as training
  return training(mode,value)
 if mode=='research-r6-evaluation':
  from .robustness_contract import validate_parameters as validate_r6
  return validate_r6(mode,value)
 if mode in ['research-formal-training','research-formal-evaluation']:
  from .formal_segment import validate_parameters as validate_formal
  return validate_formal(mode,value)
 if mode not in MODES or not isinstance(value,dict):raise PermissionError('unknown research profile')
 expected=PROFILE_FIELDS|({'checkpoints'} if mode in ['research-evaluation','research-transition'] else set())
 mean_prior = 'mean_prior_contract' in value or 'mean_prior_weight' in value
 if mean_prior:expected|={'mean_prior_contract','mean_prior_weight'}
 if mode=='research-transition':expected|={'transition_protocol','integration_contract','checkpoint_source_identity','checkpoint_compatibility','recalculation'}
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
 if mean_prior:
  validate_pin(p['mean_prior_contract'])
  if (p['mean_prior_contract']['sha256']!=MEAN_PRIOR_CONTRACT_SHA256
      or type(p['mean_prior_weight']) not in (int,float)
      or p['mean_prior_weight']!={'R0':0.,'R1':10.}[p['group']]
      or p['residual_neutrality_weight']!=.05 or p['anchor_weight']!=4.):
   raise PermissionError('mean prior registered profile mismatch')
 elif p['residual_neutrality_weight']!={'R0':0.,'R1':.05}[p['group']]:raise PermissionError('research group/weight mismatch')
 if type(p['output_cap_bytes']) is not int or not 4*1024**3<=p['output_cap_bytes']<=16*1024**3:raise PermissionError('research output limit invalid')
 for k in ['master_authority','experiment_card']:
  pin=p[k]
  if not isinstance(pin,dict) or set(pin)!={'path','sha256'}:raise PermissionError('research delegated binding invalid')
  path=Path(pin['path'])
  if not path.is_absolute() or path.is_symlink() or not path.is_file() or digest(path)!=pin['sha256']:raise PermissionError('research delegated binding drift')
 master=json.loads(Path(p['master_authority']['path']).read_text());card=json.loads(Path(p['experiment_card']['path']).read_text())
 if master['id']!='R5-FORMAL-READINESS-AUTO-01' or master['status']!='REGISTERED_USER_DELEGATION':raise PermissionError('research master delegation invalid')
 if card['experiment_id']!=p['experiment_id'] or card['profiles'][p['group']]!={k:v for k,v in p.items() if k not in ['checkpoints','experiment_card']}:raise PermissionError('research card/profile mismatch')
 if mode in ['research-evaluation','research-transition']:
  rows=p['checkpoints']
  if not isinstance(rows,list) or [r.get('update') for r in rows if isinstance(r,dict)]!=p['evaluation_updates']:raise PermissionError('research checkpoint nodes mismatch')
  for r in rows:
   if set(r)!={'path','sha256','update'}:raise PermissionError('research checkpoint fields mismatch')
   f=Path(r['path'])
   if not f.is_absolute() or f.is_symlink() or not f.is_file() or digest(f)!=r['sha256']:raise PermissionError('research checkpoint identity drift')
 if mode=='research-transition':
  from .checkpoint_compatibility import validate_source_identity,pinned_json
  if p['group']!='R1' or p['transition_protocol']!=TRANSITION_PROTOCOL or p['evaluation_seed']!=20260909 or p['evaluation_updates']!=[p['max_updates']]:raise PermissionError('finite transition profile not registered')
  validate_pin(p['integration_contract'])
  if p['integration_contract']['sha256']!=INTEGRATION_CONTRACT_SHA256:raise PermissionError('integration contract differs from registered bytes')
  validate_source_identity(p['checkpoint_source_identity']);pinned_json(p['checkpoint_compatibility'])
  rec=p['recalculation']
  if not isinstance(rec,dict) or set(rec)!={'checkpoint','input','probe'}:raise PermissionError('recalculation binding incomplete')
  ck=rec['checkpoint']
  if not isinstance(ck,dict) or set(ck)!={'path','sha256','update'} or ck['update']!=1:raise PermissionError('recalculation requires exact actualu1')
  validate_pin({k:ck[k] for k in ['path','sha256']});validate_pin(rec['input']);validate_pin(rec['probe'])
  output=Path(p['checkpoints'][0]['path']).parent.parent
  if (Path(ck['path'])!=output/'checkpoints/u1.pt' or Path(rec['input']['path'])!=output/'pilot_diagnostic/update_1_full_input.pt' or Path(rec['probe']['path'])!=output/'pilot_diagnostic/first_update_probe.pt'):raise PermissionError('recalculation must use same actualtraining entry asfinalcheckpoint')
 return p

def method_hashes(config,p):
 result=dict(config['method']['hashes'])
 effective={k:p[k] for k in ['residual_neutrality_weight','anchor_weight','actor_lr','critic_lr','command_profile']}
 if 'mean_prior_contract' in p:
  effective['mean_prior_weight']=p['mean_prior_weight']
  result['mean_prior_contract']=p['mean_prior_contract']['sha256']
 result['research_effective_method']=hashlib.sha256(json.dumps(effective,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return result

def required_budget(mode,p):
 if mode=='research-r6-final-evaluation':
  from .robustness_final_contract import required_budget as final_budget
  return final_budget(mode,p)
 if mode=='research-r6-candidate-evaluation':
  from .robustness_candidate_contract import required_budget as candidate_budget
  return candidate_budget(mode,p)
 if mode=='research-r6-training':
  from .robustness_training_contract import required_budget as training
  return training(mode,p)
 if mode=='research-r6-evaluation':
  from .robustness_contract import required_budget as r6_budget
  return r6_budget(mode,p)
 if mode in ['research-formal-training','research-formal-evaluation']:
  from .formal_segment import required_budget as formal_budget
  return formal_budget(mode,p)
 p=validate_parameters(mode,p)
 common={'live_process_starts':1,'simulator_initializations':1,'initialization_ticks':10}
 if mode=='research-transition':return {**common,'training_transitions':0,'nontraining_transitions':3584,'optimizer_updates':0,'checkpoint_writes':0,'explicit_resets':1,'automatic_resets':3584,'wall_seconds':900}
 if mode=='research-training':
  transitions=p['max_updates']*512
  return {**common,'training_transitions':transitions,'nontraining_transitions':0,'optimizer_updates':p['max_updates'],'checkpoint_writes':len(p['checkpoint_updates'])+1,'explicit_resets':1,'automatic_resets':transitions,'wall_seconds':3600 if p['max_updates']<=128 else 7200}
 n=len(p['evaluation_updates']);return {**common,'training_transitions':0,'nontraining_transitions':n*6400,'optimizer_updates':0,'checkpoint_writes':0,'explicit_resets':n,'automatic_resets':n*6400,'wall_seconds':1800}
