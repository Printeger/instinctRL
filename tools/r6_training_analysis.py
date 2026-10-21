"""Bounded CPU training evidence audit. No physics replay, dispatch or scientific acceptance."""
import io
import json
import math
import pickle
import zlib
from pathlib import Path
from tools.r6_recovery import checked,file_pin
from training_workloads.unified_h1 import robustness_training_plan as contract
from training_workloads.unified_h1 import lossless_evidence as codec
from training_workloads.unified_h1.robustness_curriculum import slices

MAX_FILES=2048
MAX_RECORD=64*1024**2


def require(value,reason):
    if not value:raise ValueError(reason)


def finite_json(data):
    def invalid(value):raise ValueError('NONFINITE_JSON')
    def number(value):
        parsed=float(value)
        if not math.isfinite(parsed):raise ValueError('NONFINITE_JSON')
        return parsed
    return json.loads(data,parse_constant=invalid,parse_float=number)


def checkpoint_metadata(path,plan,recipe,update):
    import torch
    from instinctlab.training.checkpoint_h1 import CHECKPOINT_H1_KEYS,CHECKPOINT_H1_SCHEMA,CHECKPOINT_H1_VERSION,_finite_tree
    require(path.stat().st_size<=plan['storage']['checkpoint_bytes'],'CHECKPOINT_SIZE')
    value=torch.load(path,map_location='cpu',weights_only=True)
    require(isinstance(value,dict) and set(value)==CHECKPOINT_H1_KEYS
        and value['version']==CHECKPOINT_H1_VERSION and value['schema']==CHECKPOINT_H1_SCHEMA,'CHECKPOINT_SCHEMA')
    expected=dict(update=update,transition=update*512,global_policy_step=update*16,actor_optimizer=update*8,critic_optimizer=update*8)
    require(value['counters']==expected and all(type(v)is int for v in value['counters'].values()),'CHECKPOINT_CLOCK')
    require(type(value['training_seed'])is int and value['training_seed']==plan['training_seed']
        and value['source_identity']==recipe['source_identity'],'CHECKPOINT_IDENTITY')
    hashes=dict(plan['initialization']['method_hashes'],r6_recipe=plan['recipe']['sha256'],
        r6_method=recipe['method']['sha256'],curriculum='curriculum-h1-r6-v1')
    require(value['method_hashes']==hashes,'CHECKPOINT_METHOD')
    method=contract.pinned_json(recipe['method'])
    require(value['ppo_config']==dict(method['ppo'],permutation_seed=plan['training_seed']),'CHECKPOINT_PPO')
    require(value['curriculum']==dict(version='curriculum-h1-r6-v1',completed_policy_steps=update*16,
        recipe_sha256=plan['recipe']['sha256'],origin_checkpoint_sha256=contract.BASELINE_SHA,
        branch_seed=plan['training_seed']),'CHECKPOINT_CURRICULUM')
    _finite_tree(value,'checkpoint audit')
    from tools.r6_training_checkpoint_audit import audit
    state_audit=audit(value)
    return dict(update=update,metadata_verified=True,finite_tensor_tree=True,strict_restore_performed=False,state_audit=state_audit)


def analyze(output,authorization_pin):
    output=Path(output).absolute();require(output.resolve()==output and not output.is_symlink(),'OUTPUT_ALIAS')
    auth=finite_json(checked(authorization_pin).read_bytes())
    require(auth.get('mode')=='research-r6-training','TRAINING_MODE')
    plan=contract.validate(auth['parameters']['r6_training_plan']);recipe=contract.pinned_json(plan['recipe'])
    require(auth['budgets']==contract.required_budget(plan),'EXACT_BUDGET')
    require(output==Path(auth['attempt'])/'results/research-r6-training','EXACT_ATTEMPT')
    issues=[];inputs={};streams={};phases=[];checkpoints=[]
    def issue(name,error):issues.append(dict(input=name,reason=str(error)[:300]))
    def pin(path):
        path=Path(path);require(path.resolve()==path and not path.is_symlink(),'INPUT_ALIAS')
        value=file_pin(path);inputs[value['path']]=dict(value,bytes=path.stat().st_size);return value
    pin(authorization_pin['path'])
    def plain(relative):
        path=output/relative
        try:
            require(path.stat().st_size<=MAX_RECORD,'JSON_SIZE');pin(path)
            value=finite_json(path.read_bytes());require(isinstance(value,dict) and bool(value),'JSON_OBJECT');return value
        except (OSError,ValueError,KeyError,TypeError) as error:issue(relative,error);return None
    # Every present compressed stream is checked, including failure/partial streams.
    compressed=output/'compressed_evidence';files=[]
    for path in compressed.rglob('*.r6z'):
        require(len(files)<MAX_FILES,'STREAM_COUNT_BOUND');files.append(path)
    for path in sorted(files):
        name=path.relative_to(compressed).as_posix()[:-4]
        row=dict(logical_name=name,complete=False,rows=0)
        try:
            row['pin']=pin(path)
            with codec.ChunkReader(path,run_id=plan['experiment_id'],logical_name=name,
                    maximum_raw_bytes=plan['storage']['maximum_raw_file_bytes']) as reader:
                row.update(raw_bytes=reader.size,raw_sha256=reader.raw_sha256)
                if name.endswith('.jsonl'):
                    retained=[];episodes={};updates=[]
                    with io.BufferedReader(reader) as buffer:
                        while True:
                            data=buffer.readline(MAX_RECORD+1)
                            if not data:break
                            require(len(data)<=MAX_RECORD and data.endswith(b'\n'),'JSONL_RECORD_BOUNDARY')
                            item=finite_json(data);require(isinstance(item,dict),'JSONL_OBJECT');row['rows']+=1
                            if name.endswith('/updates.jsonl'):
                                require(len(updates)<6000 and type(item['completed_updates'])is int,'UPDATE_BOUND');updates.append(item['completed_updates'])
                            if name.endswith('/episodes.jsonl'):
                                lane=item['env'];require(type(lane)is int and 0<=lane<32,'EPISODE_LANE')
                                label=str(item['start_type'])+':'+str(item['reason'])
                                require(len(label)<=256 and (label in episodes or len(episodes)<256),'EPISODE_LABELS')
                                group=episodes.setdefault(label,dict(count=0,steps=0,return_sum=0.,right_censored=0))
                                require(type(item['length'])is int and item['length']>=0
                                    and type(item['return'])in (int,float) and math.isfinite(item['return'])
                                    and type(item['right_censored'])is bool,'EPISODE_VALUES')
                                group['count']+=1;group['steps']+=item['length'];group['return_sum']+=item['return']
                                require(math.isfinite(group['return_sum']),'EPISODE_SUM_OVERFLOW')
                                group['right_censored']+=int(item['right_censored'])
                            if name in ('learning_log/plan.jsonl',) or name.endswith('/phase_complete.jsonl'):
                                require(not retained,'SINGLE_IDENTITY_RECORD');retained.append(item)
                    if updates:row['updates']=updates
                    if episodes:row['episodes']=episodes
                    if retained:row['record']=retained[0]
                row['complete']=True
        except (OSError,ValueError,KeyError,TypeError,zlib.error) as error:issue(name,error)
        streams[name]=row
    def stream(name,rows=None):
        value=streams.get(name)
        if value is None or not value['complete'] or (rows is not None and value['rows']!=rows) or (name.endswith('.jsonl') and value['rows']==0):
            issue(name,'MISSING_INCOMPLETE_OR_WRONG_ROW_COUNT');return None
        return value
    identity=stream('learning_log/plan.jsonl',1)
    if identity and (contract.object_sha(identity['record'].get('plan'))!=contract.object_sha(plan) or identity['record'].get('source_identity')!=recipe['source_identity']
            or identity['record'].get('physical_continuity') is not False):issue('learning_log/plan.jsonl','PLAN_IDENTITY')
    stream('rng/initialization.pt')
    stream('pilot_diagnostic/first_update_probe.pt')
    for update in range(plan['start_update']+1,min(plan['target_update'],plan['start_update']+2)+1):
        stream(f'pilot_diagnostic/update_{update}_full_input.pt')
    curriculum=contract.pinned_json(recipe['curriculum'])
    for phase in slices(curriculum,plan['start_update'],plan['target_update']):
        prefix=f"phases/{phase['index']}-{phase['start_update']*16}-{phase['target_update']*16}/learning_log"
        count=phase['target_update']-phase['start_update'];end=stream(prefix+'/phase_complete.jsonl',1)
        if end and end['record'].get('phase')!=phase:issue(prefix,'PHASE_IDENTITY')
        update_rows=stream(prefix+'/updates.jsonl',count)
        if update_rows and update_rows.get('updates')!=list(range(phase['start_update']+1,phase['target_update']+1)):
            issue(prefix,'UPDATE_SEQUENCE')
        for name,rows in [('rollouts.jsonl',count),('compact_rollouts.jsonl',count*16),('minibatches.jsonl',count*8)]:stream(prefix+'/'+name,rows)
        stream(prefix+'/accounting.jsonl');episodes=stream(prefix+'/episodes.jsonl')
        if phase['kind']=='robust':
            for name in ('curriculum.jsonl','reset.jsonl','force.jsonl','measurement.jsonl'):stream(prefix+'/'+name)
        phases.append(dict(phase=phase,completed_record=end is not None,
            episodes={} if episodes is None else episodes.get('episodes',{})))
    status=plain('status.json');expected_updates=plan['target_update']-plan['start_update']
    expected=dict(training_transitions=expected_updates*512,nontraining_transitions=0,
        completed_policy_steps=expected_updates*16,real_environment_transitions=expected_updates*512,
        real_optimizer_updates=expected_updates,completed_updates=expected_updates,
        actor_optimizer_steps=expected_updates*8,critic_optimizer_steps=expected_updates*8,
        real_checkpoint_writes=len(plan['checkpoint_updates']),explicit_resets=len(phases))
    if status:
        if status.get('status')!='PASS':issue('status.json','WORKLOAD_NOT_PASS')
        counters=status.get('counters',{})
        if any(type(counters.get(k))is not int or counters[k]!=v for k,v in expected.items()):issue('status.json','TERMINAL_COUNTERS')
    gate=plain('pilot_diagnostic/first_update_gate.json')
    if gate and any((type(gate.get(k))is not type(v) or gate[k]!=v) for k,v in dict(status='PASS',sample_count=512,minibatches=8,policy_rng_consumed=False,
            starting_update=plan['start_update'],first_absolute_update=plan['start_update']+1).items()):issue('first_update_gate','GATE_BINDING')
    actual={path.name for path in (output/'checkpoints').glob('*.pt')}
    if actual!={f'u{n}.pt' for n in plan['checkpoint_updates']}:issue('checkpoints','CHECKPOINT_SET')
    for update in plan['checkpoint_updates']:
        row=dict(update=update,metadata_verified=False,strict_restore_performed=False);path=output/'checkpoints'/f'u{update}.pt'
        sidecar=plain(f'checkpoints/u{update}.json')
        try:
            cp=pin(path);row['pin']=cp
            require(sidecar==dict(cp,bytes=path.stat().st_size),'CHECKPOINT_SIDECAR')
            row.update(checkpoint_metadata(path,plan,recipe,update))
        except (OSError,ValueError,KeyError,TypeError,RuntimeError,EOFError,pickle.UnpicklingError) as error:issue(str(path.relative_to(output)),error)
        checkpoints.append(row)
    probability=None
    probability_name=f"pilot_diagnostic/update_{plan['start_update']+1}_full_input.pt"
    try:
        import torch
        from tools.r6_training_probability import recompute
        origin_path=checked(plan['initialization']['checkpoint'])
        require(origin_path.stat().st_size<=plan['storage']['checkpoint_bytes'],'ORIGIN_CHECKPOINT_SIZE')
        pin(origin_path)
        require(streams.get(probability_name,{}).get('complete') is True,'PROBABILITY_INPUT_INCOMPLETE')
        with codec.ChunkReader(compressed/(probability_name+'.r6z'),run_id=plan['experiment_id'],
                logical_name=probability_name,maximum_raw_bytes=plan['storage']['maximum_raw_file_bytes']) as reader:
            retained=torch.load(reader,map_location='cpu',weights_only=True)
        origin=torch.load(origin_path,map_location='cpu',weights_only=True)
        probability=recompute(origin,retained,plan)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,ArithmeticError,EOFError,pickle.UnpicklingError) as error:
        issue('first_update_probability_replay',error)
    summary=dict(schema_version='r6-training-summary/1',experiment_id=plan['experiment_id'],phase=plan['phase'],
        training_seed=plan['training_seed'],start_update=plan['start_update'],target_update=plan['target_update'],
        expected_counters=expected,observed_counters=None if status is None else status.get('counters'),
        stream_count=len(streams),complete_stream_count=sum(s['complete'] for s in streams.values()),
        checkpoints=checkpoints,phases=phases,issue_count=len(issues),scientific_acceptance=False,
        learning_gain_established=False,physical_replay_performed=False,strict_checkpoint_restore_performed=False,
        rng_state_semantics_verified=False,probability_gate_recomputed=probability is not None,probability_replay=probability)
    report=dict(schema_version='r6-training-analysis/1',authorization=authorization_pin,plan_sha256=contract.object_sha(plan),
        analyzer=file_pin(__file__),codec=file_pin(codec.__file__),
        checkpoint_auditor=file_pin(Path(__file__).with_name('r6_training_checkpoint_audit.py')),
        probability_replayer=file_pin(Path(__file__).with_name('r6_training_probability.py')),inputs=sorted(inputs.values(),key=lambda p:p['path']),
        streams=list(streams.values()),issues=issues,summary=summary,scientific_acceptance=False)
    return report,summary
