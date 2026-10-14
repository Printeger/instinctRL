"""Exact prospective warm-start formal segment under the existing H1 entry.

This module supplies validation and backend actions, never grants a release.
"""
from pathlib import Path
import re

MODES = ('research-formal-training', 'research-formal-evaluation')
CONTRACT_SHA256 = '675fe7b24c858d9aa24c963acd591f77c2a99b45dc181073bcc39663314791c7'
MEAN_PRIOR_PHASE_SHA256 = '103664d19c5e037d5903c86e2af11b61181a8722e599bdbdd87ae7c7f9414930'
CHECKPOINT_UPDATES = [272, 288, 320, 384, 512]
FIELDS = {'experiment_id', 'group', 'training_seed', 'evaluation_seed', 'start_update',
          'target_update', 'checkpoint_updates', 'evaluation_updates', 'command_profile',
          'residual_neutrality_weight', 'anchor_weight', 'actor_lr', 'critic_lr',
          'output_cap_bytes', 'master_authority', 'experiment_card', 'phase_contract',
          'checkpoint_source_identity', 'checkpoint_compatibility', 'initialization_checkpoint',
          'origin_profile'}


def validate_parameters(mode, value):
    from .research_contract import validate_pin, validate_parameters as research_parameters
    from .checkpoint_compatibility import pinned_json, validate_source_identity
    import json
    if mode not in MODES or not isinstance(value, dict):
        raise PermissionError('unknown formal segment mode')
    expected = FIELDS | ({'checkpoints'} if mode == MODES[1] else set())
    mean_prior = 'mean_prior_weight' in value or 'mean_prior_contract' in value
    if mean_prior:
        expected |= {'mean_prior_weight', 'mean_prior_contract'}
    if set(value) != expected:
        raise PermissionError('formal segment profile fields mismatch')
    p = dict(value)
    if not isinstance(p['experiment_id'], str) or not re.fullmatch(r'[A-Z0-9-]{3,96}', p['experiment_id']):
        raise PermissionError('formal segment identity invalid')
    if (p['group'] not in {'R0', 'R1'} or type(p['training_seed']) is not int or p['training_seed'] != 20260908
            or type(p['evaluation_seed']) is not int or p['evaluation_seed'] != 20260909):
        raise PermissionError('formal segment group/seed unregistered')
    if (type(p['start_update']) is not int or p['start_update'] != 256
            or type(p['target_update']) is not int or p['target_update'] != 512
            or p['checkpoint_updates'] != CHECKPOINT_UPDATES or p['evaluation_updates'] != CHECKPOINT_UPDATES
            or any(type(u) is not int for u in p['checkpoint_updates'] + p['evaluation_updates'])
            or p['command_profile'] != 'mixed-center'):
        raise PermissionError('formal segment phase/clock/nodes unregistered')
    for key, expected_scalar in {'residual_neutrality_weight': .05 if mean_prior else {'R0': 0., 'R1': .05}[p['group']],
                                 'anchor_weight': 4., 'actor_lr': 1e-4, 'critic_lr': 3e-4}.items():
        if type(p[key]) not in (int, float) or p[key] != expected_scalar:
            raise PermissionError('formal segment method scalar unregistered: ' + key)
    if type(p['output_cap_bytes']) is not int or p['output_cap_bytes'] != 8 * 1024**3:
        raise PermissionError('formal segment output cap unregistered')
    for key in ['master_authority', 'experiment_card', 'phase_contract']:
        validate_pin(p[key])
    if p['phase_contract']['sha256'] != (MEAN_PRIOR_PHASE_SHA256 if mean_prior else CONTRACT_SHA256):
        raise PermissionError('formal segment contract differs')
    master = json.loads(Path(p['master_authority']['path']).read_text())
    card = json.loads(Path(p['experiment_card']['path']).read_text())
    if master['id'] != 'R5-FORMAL-READINESS-AUTO-01' or master['status'] != 'REGISTERED_USER_DELEGATION':
        raise PermissionError('formal segment master delegation invalid')
    if (card['experiment_id'] != p['experiment_id']
            or card['profiles'][p['group']] != {k: v for k, v in p.items() if k not in ['checkpoints', 'experiment_card']}):
        raise PermissionError('formal segment card/profile mismatch')
    origin = research_parameters('research-training', p['origin_profile'])
    if mean_prior:
        from .research_contract import MEAN_PRIOR_CONTRACT_SHA256
        validate_pin(p['mean_prior_contract'])
        if (p['mean_prior_contract']['sha256'] != MEAN_PRIOR_CONTRACT_SHA256
                or type(p['mean_prior_weight']) not in (int, float)
                or p['mean_prior_weight'] != {'R0': 0., 'R1': 10.}[p['group']]
                or origin.get('mean_prior_contract') != p['mean_prior_contract']
                or origin.get('mean_prior_weight') != 10.):
            raise PermissionError('formal mean prior phase/origin mismatch')
    elif 'mean_prior_contract' in origin:
        raise PermissionError('mean prior origin requires explicit current formal mean prior phase')
    if (origin['group'] != 'R1' or origin['training_seed'] != p['training_seed']
            or origin['max_updates'] != 256 or origin['anchor_weight'] != p['anchor_weight']
            or origin['actor_lr'] != p['actor_lr'] or origin['critic_lr'] != p['critic_lr']):
        raise PermissionError('formal initialization must be the frozen qualified R1 warmup')
    validate_source_identity(p['checkpoint_source_identity'])
    pinned_json(p['checkpoint_compatibility'])
    ck = p['initialization_checkpoint']
    if not isinstance(ck, dict) or set(ck) != {'path', 'sha256', 'update'} or type(ck['update']) is not int or ck['update'] != 256:
        raise PermissionError('formal initialization checkpoint identity incomplete')
    validate_pin({k: ck[k] for k in ['path', 'sha256']})
    if mode == MODES[1]:
        rows = p['checkpoints']
        if not isinstance(rows, list) or [r.get('update') for r in rows if isinstance(r, dict)] != CHECKPOINT_UPDATES:
            raise PermissionError('formal evaluation checkpoint nodes mismatch')
        for row in rows:
            if set(row) != {'path', 'sha256', 'update'} or type(row['update']) is not int:
                raise PermissionError('formal evaluation checkpoint identity incomplete')
            validate_pin({k: row[k] for k in ['path', 'sha256']})
    return p


def required_budget(mode, p):
    validate_parameters(mode, p)
    common = {'live_process_starts': 1, 'simulator_initializations': 1, 'initialization_ticks': 10}
    if mode == MODES[0]:
        return {**common, 'training_transitions': 131072, 'nontraining_transitions': 0,
                'optimizer_updates': 256, 'checkpoint_writes': 5, 'explicit_resets': 1,
                'automatic_resets': 131072, 'wall_seconds': 7200}
    return {**common, 'training_transitions': 0, 'nontraining_transitions': 164800,
            'optimizer_updates': 0, 'checkpoint_writes': 0, 'explicit_resets': 25,
            'automatic_resets': 164800, 'wall_seconds': 3600}


def method_hashes(config, p):
    from .research_contract import method_hashes as research_hashes
    result = research_hashes(config, p)
    result.update(curriculum='curriculum-h1-v1', formal_phase=p['phase_contract']['sha256'], reset_distribution='center-only-v1',
                  initialization='qualified-warmup-fresh-reset-v1',
                  origin_checkpoint_sha256=p['initialization_checkpoint']['sha256'])
    return result


def train_segment(backend):
    from .config import load_config, sha256_file
    from .research_contract import method_hashes as research_hashes
    from .schedule import ScheduleClock, command_at
    from .driver import UnifiedH1Driver, UnifiedH1TrainingLoop
    from .logging import LearningLog
    from .pilot_diagnostic import FirstUpdateDiagnostic
    from training_workloads.scale_003.persistence import atomic_json_replace
    if backend.authorization['mode'] != MODES[0]:
        raise PermissionError('formal training requires exact mode')
    p = backend.research
    backend._apply_research_profile()
    ck = p['initialization_checkpoint']
    if sha256_file(Path(ck['path'])) != ck['sha256']:
        raise ValueError('formal initialization checkpoint drift')
    meta = backend._load_evaluation_checkpoint(Path(ck['path']), actor_learning_rate=p['actor_lr'],
        expected_method_hashes=research_hashes(load_config(), p['origin_profile']),
        source_identity=p['checkpoint_source_identity'], restore_curriculum_generator=True)
    if (backend.trainer.update_counter != p['start_update'] or meta['training_seed'] != p['training_seed']
            or meta['curriculum']['version'] != 'curriculum-h1-research-pure-hover-v1'):
        raise ValueError('formal warmup actual checkpoint stage/seed/node mismatch')
    from .mean_prior import configure_trainer
    configure_trainer(backend.trainer, p)
    backend._checkpoint_reset_offset = meta['reset_number']
    backend._formal_segment = True
    start = meta['counters']['global_policy_step']
    backend.environment.set_h1_evaluation_case(None)
    backend.environment.set_h1_initial_command(command_at(start, device=backend.environment.device), start)
    backend.resource_check('before_formal_fresh_reset', backend.counters.explicit_resets + 1)
    backend.observation, _ = backend._preserving_reset(seed=p['training_seed'] + meta['reset_number'] + 1)
    journal = LearningLog(backend.output / 'learning_log')
    backend.journal = journal
    backend.driver = UnifiedH1Driver(backend.environment, backend.observation, clock=ScheduleClock(start),
        counters=backend.counters, journal=journal,
        before_step=lambda ordinal: backend.resource_check('before_formal_step', ordinal),
        transition_limit=backend.authorization['budgets']['training_transitions'],
        automatic_reset_limit=backend.authorization['budgets']['automatic_resets'])
    backend.loop = UnifiedH1TrainingLoop(backend.driver, backend.trainer, journal, backend._write_checkpoint,
        before_rollout=lambda ordinal: backend.resource_check('before_formal_rollout', ordinal),
        before_optimizer=lambda ordinal: backend.resource_check('before_formal_optimizer', ordinal),
        update_limit=backend.authorization['budgets']['optimizer_updates'],
        optimizer_step_origin=(p['start_update'] * 8, p['start_update'] * 8),
        diagnostic=FirstUpdateDiagnostic(backend.output / 'pilot_diagnostic', enabled=True,
                                         starting_update=p['start_update']))
    atomic_json_replace(backend.output, 'formal_initialization.json', {
        'scope': 'FULL_TRAINER_RESTORE_THEN_FRESH_SIMULATION_RESET', 'origin': ck,
        'metadata': meta, 'phase': p, 'initial_global_policy_step': start,
        'restored_adam_steps_each': p['start_update'] * 8,
        'physical_history_anchor_actuator_continuity_claimed': False})
    while backend.trainer.update_counter < p['target_update']:
        ordinal = backend.trainer.update_counter + 1
        backend.resource_check('before_formal_update', ordinal)
        backend.loop.run_update(checkpoint_label=f'u{ordinal}' if ordinal in p['checkpoint_updates'] else None)


def evaluate_segment(backend):
    from .config import load_config, sha256_file
    from .evaluation import CASES, evaluate_checkpoint
    from training_workloads.scale_003.persistence import atomic_json_replace
    if backend.authorization['mode'] != MODES[1]:
        raise PermissionError('formal evaluation requires exact mode')
    p = backend.research
    backend._apply_research_profile()
    for row in p['checkpoints']:
        path = Path(row['path'])
        if sha256_file(path) != row['sha256']:
            raise ValueError('formal evaluation checkpoint drift')
        meta = backend._load_evaluation_checkpoint(path, actor_learning_rate=p['actor_lr'],
            expected_method_hashes=method_hashes(load_config(), p),
            source_identity=load_config()['resolved_identity'])
        if (backend.trainer.update_counter != row['update'] or meta['training_seed'] != p['training_seed']
                or meta['curriculum']['version'] != 'curriculum-h1-v1'):
            raise ValueError('formal evaluation actual phase/seed/node mismatch')
        backend.trainer.actor.eval(); backend.trainer.critic.eval(); backend._evaluation_rows.clear()
        label = f'formal_{p["group"]}_u{row["update"]}'
        for case in CASES:
            backend.resource_check('before_formal_evaluation_case', row['update'])
            backend._evaluate_case(case, direct_b0=False, evaluation_label=label)
        result = evaluate_checkpoint([trace for case in CASES for trace in backend._evaluation_rows[case]], update=row['update'])
        atomic_json_replace(backend.output, label + '.json', result)
