"""New R6 baseline-evaluation registration; no inherited R5 live permission.

This first entry measures the fixed u640. Candidate training/evaluation requires
its own reviewed extension; the overall R6 goal is not reduced to this phase.
"""
from pathlib import Path
import hashlib
import json
import math
import re

MODE='research-r6-evaluation'
BASELINE_SHA='aaef7d40de50032cf2c61e0bc7a6e6f277a2f65e252284f3af58c0f25bae5429'
BINDINGS_SHA='6ebe90717fa2571816ba3392841f3690c2c3fe37225c3728f935bb80d3102623'
USER_SHA='b45cbca50cf944ea2b2abe2b6fd57998224784f5de2e2545cc7ae69fc8a2d706'
REQUIREMENTS_SHA='200fb9daa96606ae3a58516219121d2ae243de51a8b1e02d10077a4b15e2f12b'
FIELDS={'experiment_id','group','training_seed','evaluation_seed','output_cap_bytes','wall_seconds',
        'master_authority','experiment_card','baseline_bindings','checkpoint_compatibility','protocol_review','r6'}
R6_FIELDS={'protocol','block_ids','run_id','seed','split','expected_mass_kg','output_quotas'}


def pinned_json(pin):
    from .research_contract import validate_pin
    return json.loads(validate_pin(pin).read_text())


def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def accepted_review(pin,decision,**bindings):
    review=pinned_json(pin)
    if review.get('decision')!=decision or review.get('must_fix')!=[] or not review.get('reviewer') or review['reviewer']=='root':
        raise PermissionError('R6 independent review missing')
    if any(review.get(k)!=v for k,v in bindings.items()):raise PermissionError('R6 review input mismatch')
    return review


def baseline(p):
    if p['baseline_bindings']['sha256']!=BINDINGS_SHA:raise PermissionError('R6 frozen baseline bindings differ')
    b=pinned_json(p['baseline_bindings'])
    from .research_contract import validate_pin
    validate_pin(b['selected_model'])
    if b['selected_model']['sha256']!=BASELINE_SHA or b['ics_clearance_margin_m']!=.15:
        raise PermissionError('R6 frozen u640/margin differs')
    return b


def validate_parameters(mode,value):
    from .research_contract import validate_pin
    if mode!=MODE or not isinstance(value,dict) or set(value) not in (FIELDS,FIELDS|{'native_volume','extension_inputs','result_volume'}):raise PermissionError('R6 baseline evaluation fields/mode')
    p=dict(value)
    if 'native_volume' in p:
        from .native_volume import validate
        validate(p['native_volume'])
        from .extension_inputs import validate as validate_extensions
        validate_extensions(p['extension_inputs'])
    if not isinstance(p['experiment_id'],str) or not re.fullmatch(r'R6-[A-Z0-9-]{1,90}',p['experiment_id']):raise PermissionError('R6 experiment identity')
    if p['group']!='R1' or type(p['training_seed'])is not int or p['training_seed']!=20260908 or type(p['evaluation_seed'])is not int or p['evaluation_seed']!=20260909:
        raise PermissionError('R6 baseline/original-suite seed identity')
    if type(p['wall_seconds'])is not int or not 1<=p['wall_seconds']<=43200:raise PermissionError('R6 job wall bound')
    if type(p['output_cap_bytes'])is not int or not 3*1024**3<p['output_cap_bytes']<=128*1024**3:raise PermissionError('R6 job output bound')
    master=pinned_json(p['master_authority'])
    if master.get('id')!='R6-ROBUSTNESS-AUTO-20261019' or master.get('status')!='REGISTERED_USER_DELEGATION':raise PermissionError('R6 master identity')
    for name,expected in [('user_authority',USER_SHA),('requirements',REQUIREMENTS_SHA)]:
        validate_pin(master[name])
        if master[name]['sha256']!=expected:raise PermissionError('R6 user authority differs')
    baseline(p)
    r=p['r6']
    if not isinstance(r,dict) or set(r)!=R6_FIELDS:raise PermissionError('R6 block request fields')
    if type(r['seed'])is not int or not 0<=r['seed']<2**31 or r['split']!='development':
        raise PermissionError('R6 actual random stream')
    if not isinstance(r['run_id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',r['run_id']):raise PermissionError('R6 run identity')
    if type(r['expected_mass_kg'])not in (int,float) or not math.isfinite(r['expected_mass_kg']):
        raise PermissionError('R6 measured mass type')
    if abs(r['expected_mass_kg']-1.181517923)>1e-8:raise PermissionError('R6 pinned measured mass')
    protocol=pinned_json(r['protocol'])
    if protocol.get('frozen')is not True:raise PermissionError('R6 protocol not frozen')
    from .robustness_prelaunch import POLICY as resource_policy
    if protocol.get('resource_admission')!=resource_policy:raise PermissionError('R6 prelaunch resource policy differs')
    if protocol['old_suite']['seed']!=20260909 or protocol['old_suite']['transitions']!=32960:raise PermissionError('R6 original suite changed')
    review=accepted_review(p['protocol_review'],'PASS_R6_PROTOCOL',protocol_sha256=r['protocol']['sha256'])
    capacity=pinned_json(review['capacity_report'])
    if capacity.get('capacity_pass') is not True or capacity.get('protocol_sha256')!=r['protocol']['sha256']:
        raise PermissionError('R6 capacity is not accepted for this protocol')
    if capacity['maximum_job_output_bytes']<p['output_cap_bytes'] or capacity['maximum_job_wall_seconds']<p['wall_seconds']:
        raise PermissionError('R6 job exceeds reviewed capacity')
    blocks=protocol['blocks'];indexed={b['id']:b for b in blocks}
    if len(indexed)!=16 or len(indexed)!=len(blocks):raise PermissionError('R6 duplicate protocol blocks')
    ids=r['block_ids']
    if not isinstance(ids,list) or not ids or len(set(ids))!=len(ids) or any(key not in indexed for key in ids):raise PermissionError('R6 selected blocks')
    if set(ids)!=set(indexed):raise PermissionError('R6 baseline requires all frozen blocks')
    if set(r['output_quotas'])!=set(ids):raise PermissionError('R6 per-block output quotas')
    required=64*1024**2 # Original suite and metadata belong to the result payload.
    for key in ids:
        b=indexed[key]
        if b['lanes']!=32 or b['steps']!=(200 if b['task']=='reverse' else 330):raise PermissionError('R6 block denominator')
        quota=r['output_quotas'][key]
        if set(quota)!={'range_bytes','audit_bytes'} or any(type(v)is not int or v<=0 or v>2*1024**3 for v in quota.values()):raise PermissionError('R6 quota schema/per-file ceiling')
        required+=sum(quota.values())+16*1024**2 # Complete motion JSON and block result.
    from .result_volume import workload_write_cap
    if required>workload_write_cap(p,minimum_free_bytes=required):
        raise PermissionError('R6 components exceed job output cap')
    card=pinned_json(p['experiment_card'])
    if card.get('experiment_id')!=p['experiment_id'] or card.get('profiles',{}).get('R1')!={k:v for k,v in p.items() if k!='experiment_card'}:
        raise PermissionError('R6 exact card/profile mismatch')
    pinned_json(p['checkpoint_compatibility'])
    return p


def required_budget(mode,p):
    p=validate_parameters(mode,p);r=p['r6'];protocol=pinned_json(r['protocol'])
    indexed={b['id']:b for b in protocol['blocks']}
    transitions=32960+32*sum(indexed[k]['steps'] for k in r['block_ids'])
    return dict(live_process_starts=1,simulator_initializations=1,initialization_ticks=10,
        training_transitions=0,nontraining_transitions=transitions,optimizer_updates=0,checkpoint_writes=0,
        explicit_resets=5+len(r['block_ids']),automatic_resets=transitions,wall_seconds=p['wall_seconds'])


def validate_compatibility(pin,origin,config):
    from .checkpoint_compatibility import validate_source_identity
    from .research_contract import validate_pin
    validate_source_identity(origin);validate_source_identity(config['resolved_identity'])
    record=pinned_json(pin)
    required={'schema_version','origin_source_identity','execution_source_identity','origin_config','execution_config','changed_product_sources','independent_review'}
    if set(record)!=required or record['schema_version']!='r6-baseline-compatibility/1':raise PermissionError('R6 compatibility schema')
    if record['origin_source_identity']!=origin or record['execution_source_identity']!=config['resolved_identity']:raise PermissionError('R6 compatibility source identity')
    old=pinned_json(record['origin_config']);new=pinned_json(record['execution_config'])
    if record['origin_config']['sha256']!=origin['config_sha256'] or record['execution_config']['sha256']!=config['resolved_identity']['config_sha256'] or new!={k:v for k,v in config.items() if k!='resolved_identity'}:raise PermissionError('R6 compatibility config binding')
    for key in ('versions','actor','training','execution'):
        if old[key]!=new[key]:raise PermissionError('R6 baseline invariant config changed: '+key)
    for key in ('version','proposal_sha256','hashes'):
        if old['method'][key]!=new['method'][key]:raise PermissionError('R6 baseline method changed')
    metric='training_workloads/unified_h1/evaluation.py'
    if old['workload']['source_sha256'].get(metric) is None or old['workload']['source_sha256'][metric]!=new['workload']['source_sha256'].get(metric):
        raise PermissionError('R6 original evaluator source changed')
    a=old['product']['source_sha256'];b=new['product']['source_sha256']
    changed={k:{'origin':a.get(k),'execution':b.get(k)} for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    allowed={'isaaclab-training/instinctlab/envs/taslab_direct_env.py','isaaclab-training/instinctlab/envs/unified_h1_direct_env.py',
        *('isaaclab-training/instinctlab/observations/'+n+'.py' for n in ('robustness','robustness_adapter','robustness_physics'))}
    # Newly manifested pre-existing dependencies must match the original
    # checkpoint checkout byte-for-byte; this does not authorize product edits.
    from .config import R6_PRODUCT_SOURCES, PRODUCT_SOURCES
    for relative in (R6_PRODUCT_SOURCES-PRODUCT_SOURCES)-allowed:
        if relative in changed and relative not in a and relative in b:
            validate_pin({'path':str(Path(old['product']['path'])/relative),'sha256':b[relative]})
            allowed.add(relative)
    if not set(changed)<=allowed or changed!=record['changed_product_sources']:raise PermissionError('R6 baseline product integration differs')
    accepted_review(record['independent_review'],'PASS_R6_EXECUTION_COMPATIBILITY',payload_sha256=object_sha({k:v for k,v in record.items() if k!='independent_review'}))
    # Old absolute evidence stays bound to its bytes; current config validation
    # independently checks execution checkout/manifests before this function.
    for section in ('product','workload'):
        for relative,digest in old[section]['source_sha256'].items():
            if Path(relative).is_absolute() or '..' in Path(relative).parts:raise PermissionError('R6 origin source path')
            validate_pin({'path':str(Path(old[section]['path'])/relative),'sha256':digest})
    return dict(origin)


def evaluate_registered(backend):
    from .evaluation import CASES,evaluate_checkpoint,wall_motion_prerequisite
    from training_workloads.scale_003.persistence import atomic_json_replace
    p=validate_parameters(backend.authorization['mode'],backend.research);b=baseline(p)
    env=backend.environment
    env.set_h1_residual_neutrality_weight(.05)
    env.set_h1_research_profile(pure_hover=False,anchor_weight=4.)
    env.set_h1_ics_clearance_margin(.15)
    metadata=backend._load_evaluation_checkpoint(Path(b['selected_model']['path']),actor_learning_rate=1e-4,
        expected_method_hashes=b['method_hashes'],source_identity=b['source_identity'])
    if backend.trainer.update_counter!=640 or metadata['training_seed']!=20260908 or metadata['curriculum']['version']!='curriculum-h1-v1':
        raise PermissionError('R6 actual loaded u640 metadata differs')
    atomic_json_replace(backend.output,'r6_model_binding.json',dict(checkpoint=b['selected_model'],metadata=metadata,
        checkpoint_source_identity=b['source_identity'],method_hashes=b['method_hashes'],ics_clearance_margin_m=.15,
        physical_history_continuity_claimed=False,actor_input_extension=False))
    backend.trainer.actor.eval();backend.trainer.critic.eval();backend._evaluation_rows.clear()
    label='r6_baseline_u640_original'
    for case in CASES:
        backend.resource_check('before_r6_original_case',640)
        backend._evaluate_case(case,direct_b0=False,evaluation_label=label)
    original=evaluate_checkpoint([t for case in CASES for t in backend._evaluation_rows[case]],update=640)
    atomic_json_replace(backend.output,'r6_original_suite.json',original)
    rows=[json.loads(line) for line in (backend.output/'evaluation_raw'/label/'W.jsonl').read_text().splitlines()][1:]
    motion=[wall_motion_prerequisite([r['root_state_w'][lane][7+(lane>=16)] for r in rows]) for lane in range(32)]
    atomic_json_replace(backend.output,'r6_original_wall_motion.json',dict(per_lane=motion,all32=all(motion)))
    new=backend.evaluate_r6_blocks()
    atomic_json_replace(backend.output,'r6_evaluation_complete.json',dict(original=original,new_blocks=new,
        baseline=b['selected_model'],original_seed=20260909,new_seed=p['r6']['seed'],
        scientific_acceptance=False,evidence_status='REQUIRES_INDEPENDENT_RAW_REPLAY'))
