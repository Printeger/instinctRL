"""Separate one-seed replication of the first accepted W trained checkpoint.

Primary w_closure validation and metric-h1-v1 are deliberately reused unchanged.
This evaluation-only protocol grants no training permission.
"""
from pathlib import Path
import json
from . import w_closure as primary
from .checkpoint_compatibility import pinned_json, validate_source_identity
from .research_contract import validate_pin

EXPERIMENT = 'R5-W-L1-INDEPENDENT-01'
RECOVERY_EXPERIMENT = 'R5-W-L1-INDEPENDENT-RECOVERY1'
SEED = 20261020


def protocol(p):
    r = pinned_json(p['independent_validation_protocol'])
    if set(r) != {'schema_version', 'primary_authorization', 'candidate_config', 'candidate_source_identity', 'checkpoint', 'seed'}:
        raise PermissionError('independent protocol fields differ')
    if r['schema_version'] != 'r5-w-independent-seed/1' or type(r['seed']) is not int or r['seed'] != SEED:
        raise PermissionError('independent protocol identity differs')
    auth = pinned_json(r['primary_authorization'])
    identity_keys = ('product_commit','product_tree','workload_commit','workload_tree','config_sha256')
    if auth['status'] != 'AUTHORIZED' or r['candidate_source_identity'] != {k:auth[k] for k in identity_keys} or r['candidate_config']['sha256'] != auth['config_sha256'] or auth['mode'] != primary.EVAL or auth['order_id'] != 'R5-W-L1-EVAL-01-R1-RESEARCH-FORMAL-EVALUATION':
        raise PermissionError('independent primary authorization/source mismatch')
    original = primary.validate_parameters(primary.EVAL, auth['parameters'])
    if original['experiment_id'] != 'R5-W-L1-EVAL-01' or original['evaluation_checkpoint_origin'] != 'CANDIDATE':
        raise PermissionError('independent primary experiment differs')
    if r['checkpoint'] != original['checkpoints'][0] or r['checkpoint']['update'] != 640 or r['checkpoint']['sha256'] != 'aaef7d40de50032cf2c61e0bc7a6e6f277a2f65e252284f3af58c0f25bae5429':
        raise PermissionError('first passing checkpoint differs')
    validate_source_identity(r['candidate_source_identity'])
    config = pinned_json(r['candidate_config'])
    if r['candidate_config']['sha256'] != r['candidate_source_identity']['config_sha256'] or r['candidate_source_identity']['workload_commit'] != '0c6363bbc0d2947a272f533c02e73d2995ac225a':
        raise PermissionError('trained candidate source differs')
    return r, original, config


def validate_parameters(mode, value):
    if mode != primary.EVAL or not isinstance(value, dict):
        raise PermissionError('independent suite is evaluation only')
    p = dict(value)
    if set(p) != primary.FIELDS | {'checkpoints', 'independent_validation_protocol'}:
        raise PermissionError('independent parameters differ')
    r, original, _ = protocol(p)
    if p.get('experiment_id') not in (EXPERIMENT, RECOVERY_EXPERIMENT):
        raise PermissionError('independent experiment identity differs')
    expected = dict(original, experiment_id=p['experiment_id'], evaluation_seed=SEED,
                    evaluation_updates=[640], checkpoints=[r['checkpoint']],
                    experiment_card=p['experiment_card'], checkpoint_compatibility=p['checkpoint_compatibility'],
                    independent_validation_protocol=p['independent_validation_protocol'])
    if p != expected or type(p['evaluation_seed']) is not int:
        raise PermissionError('independent suite changed more than seed/node/identity')
    card = pinned_json(p['experiment_card'])
    pinned_json(p['checkpoint_compatibility'])
    if card['experiment_id'] != p['experiment_id'] or card['profiles']['R1'] != {k:v for k,v in p.items() if k not in ('checkpoints','experiment_card')}:
        raise PermissionError('independent card differs')
    return p


def required_budget(mode, p):
    validate_parameters(mode, p)
    return dict(live_process_starts=1, simulator_initializations=1, initialization_ticks=10,
                training_transitions=0, nontraining_transitions=32960, optimizer_updates=0,
                checkpoint_writes=0, explicit_resets=5, automatic_resets=32960, wall_seconds=1800)


def method_hashes(config, p):
    validate_parameters(primary.EVAL, p)
    r, original, old_config = protocol(p)
    return primary.method_hashes(old_config, original)


def evaluate_independent(traces, *, update):
    """Validate the new seed, then use the frozen metric on a labelled scoring view.

    Only the seed metadata is canonicalized; raw records and all physical,
    timing, command, validity and initial-state gates remain unchanged.
    """
    from dataclasses import replace
    from .evaluation import CASES, evaluate_checkpoint
    if update != 640 or len(traces) != 160:
        raise ValueError('independent complete checkpoint required')
    if {(t.case, t.env_index) for t in traces} != {(c, i) for c in CASES for i in range(32)}:
        raise ValueError('independent full case/lane coverage required')
    if any(type(t.seed) is not int or t.seed != SEED for t in traces):
        raise ValueError('independent actual seed differs')
    result = evaluate_checkpoint([replace(t, seed=CASES[t.case].seed) for t in traces], update=update)
    return dict(protocol='r5-w-independent-seed/1', actual_seed=SEED,
                canonical_scoring_seed=20260909, seed_metadata_only_scoring_view=True,
                original_metric_result=result)


def evaluate_segment(backend):
    from .config import sha256_file
    from .evaluation import CASES, evaluate_checkpoint, wall_motion_prerequisite
    from training_workloads.scale_003.persistence import atomic_json_replace
    p = validate_parameters(backend.authorization['mode'], backend.research)
    r, original, old_config = protocol(p)
    primary.apply_profile(backend)
    ck = r['checkpoint']
    if sha256_file(Path(ck['path'])) != ck['sha256']:
        raise ValueError('independent checkpoint drift')
    meta = backend._load_evaluation_checkpoint(Path(ck['path']), actor_learning_rate=p['actor_lr'],
        expected_method_hashes=primary.method_hashes(old_config, original), source_identity=r['candidate_source_identity'])
    if backend.trainer.update_counter != 640 or meta['training_seed'] != p['training_seed'] or meta['curriculum']['version'] != 'curriculum-h1-v1':
        raise ValueError('independent checkpoint phase differs')
    backend.trainer.actor.eval(); backend.trainer.critic.eval(); backend._evaluation_rows.clear()
    label = 'w_independent_R1_u640'
    for case in CASES:
        backend.resource_check('before_w_independent_case', 640)
        backend._evaluate_case(case, direct_b0=False, evaluation_label=label)
    result = evaluate_independent([t for c in CASES for t in backend._evaluation_rows[c]], update=640)
    atomic_json_replace(backend.output, label+'.json', result)
    rows = [json.loads(line) for line in (backend.output/'evaluation_raw'/label/'W.jsonl').read_text().splitlines()][1:]
    motion = [wall_motion_prerequisite([row['root_state_w'][lane][7+(lane>=16)] for row in rows]) for lane in range(32)]
    atomic_json_replace(backend.output, label+'_goal_motion.json', dict(world_normal_11_sample_approach=motion, all32=all(motion), original_metric_unchanged=True))
    atomic_json_replace(backend.output, 'independent_protocol_binding.json', dict(protocol=r, parameters=p, checkpoint_metadata_source=r['candidate_source_identity'], original_metric_unchanged=True))
