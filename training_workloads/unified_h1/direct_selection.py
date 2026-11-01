"""Local checkpoint provenance; shared scientific checks, no job approval chain."""
from pathlib import Path
from . import robustness_contract as base
from . import robustness_training_plan as training
from .research_contract import validate_pin


def verify_publication(commit, *pins):
    """Verify frozen bytes in a real local commit, independent of Store records."""
    import re
    import subprocess
    from tools.r6_direct.import_legacy import ROOT
    from .robustness_final_selection import require
    require(isinstance(commit, str) and re.fullmatch('[a-f0-9]{40}', commit), 'PUBLICATION_COMMIT')
    for pin in pins:
        path = validate_pin(pin)
        require(path.is_relative_to(ROOT), 'PUBLICATION_ROOT')
        data = subprocess.check_output(['git', '-C', str(ROOT), 'show', commit + ':' + str(path.relative_to(ROOT))])
        import hashlib
        require(hashlib.sha256(data).hexdigest() == pin['sha256'], 'PUBLICATION_BYTES')


def training_model(plan, analysis_pin, checkpoint, method):
    from tools.r6_direct.evidence import validate_binding, verify_inputs
    from .robustness_evaluation_selection import require
    bound, ctx, report = validate_binding(analysis_pin)
    contract = ctx['contract']
    require(plan['schema_version'] == 'r6-local-training-plan/1'
            and contract['operation'] == 'train' and contract['parameters']['r6_training_plan'] == plan,
            'DIRECT_TRAINING_CONTRACT')
    require(report['schema_version'] == 'r6-training-analysis/1'
            and report['plan_sha256'] == base.object_sha(plan) and report['issues'] == [], 'TRAINING_REPORT')
    summary = report['summary']
    require(summary['issue_count'] == 0 and summary['probability_gate_recomputed'] is True
            and summary['rng_initialization_audit']['status'] == 'PASS', 'TRAINING_AUDITS')
    updates = plan['target_update'] - plan['start_update']
    expected = dict(training_transitions=updates*512, nontraining_transitions=0,
                    real_optimizer_updates=updates, completed_updates=updates,
                    real_checkpoint_writes=len(plan['checkpoint_updates']))
    require(all(summary['expected_counters'].get(k) == v for k, v in expected.items()), 'EXPECTED_COUNTERS')
    require(all(type(ctx['state']['counters'].get(k)) is int and ctx['state']['counters'][k] == v
                and summary['observed_counters'].get(k) == v for k, v in expected.items()), 'TERMINAL_COUNTERS')
    rows = [r for r in summary['checkpoints'] if r.get('pin') == checkpoint]
    require(len(rows) == 1, 'EXACT_CHECKPOINT_NODE')
    row = rows[0]; update = row['update']
    require(type(update) is int and update in plan['checkpoint_updates'] and row['metadata_verified'] is True
            and row['state_audit']['actor_critic_state_loaded'] is True
            and row['state_audit']['adam_state_loaded'] is True, 'CHECKPOINT_AUDIT')
    require(validate_pin(checkpoint) == ctx['output'] / 'checkpoints' / f'u{update}.pt', 'CHECKPOINT_ATTEMPT')
    seen = verify_inputs(report)
    require({checkpoint['path'], plan['initialization']['checkpoint']['path']} <= seen, 'REPORT_INPUT_MISSING')
    recipe = base.pinned_json(plan['recipe'])
    return dict(checkpoint=checkpoint, update=update, training_seed=plan['training_seed'],
                source_identity=recipe['source_identity'],
                method_hashes=dict(plan['initialization']['method_hashes'], r6_recipe=plan['recipe']['sha256'],
                                  r6_method=recipe['method']['sha256'], curriculum='curriculum-h1-r6-v1'),
                ppo_config=dict(method['ppo'], permutation_seed=plan['training_seed']))


def evaluation_evidence(pin, protocol_pin):
    from tools.r6_direct.evidence import validate_binding, verify_inputs
    from tools.r6_candidate_analysis import validate_binding as validate_model_binding
    from . import robustness_evaluation_selection as selection
    from .robustness_confirmation import require
    bound, ctx, report = validate_binding(pin)
    p = ctx['contract']['parameters']; request = p['r6']
    require(ctx['contract']['operation'] == 'evaluate' and request['split'] == 'development'
            and request['protocol'] == protocol_pin, 'DIRECT_DEVELOPMENT_ONLY')
    selected = selection.validate(p['evaluation_selection'])
    require(selected['selection']['schema_version'] == 'r6-local-evaluation-selection/1', 'DIRECT_SELECTION')
    require(report['schema_version'] == 'r6-motion-replay/1' and report['candidate_binding_issues'] == []
            and all(report[k] == request[k] for k in ('run_id', 'seed', 'split')), 'REPORT_IDENTITY')
    output = ctx['output']
    import json
    binding_path = output / 'r6_model_binding.json'
    validate_model_binding(json.loads(binding_path.read_text()), selected)
    require(report['candidate']['checkpoint'] == selected['model']['checkpoint']
            and report['candidate']['selection_sha256'] == base.object_sha(selected['selection']), 'MODEL_IDENTITY')
    protocol = base.pinned_json(protocol_pin)
    require([r['block_id'] for r in report['blocks']] == [b['id'] for b in protocol['blocks']], 'FULL_BLOCKS')
    required = {str(binding_path), protocol_pin['path']}
    for block, row in zip(protocol['blocks'], report['blocks']):
        require(row['replay_complete'] is True and row['saved_result_equal'] is True and row['total'] == 32
                and row['domain'] == block['domain'] and row['measurement_replay']['bit_exact_delivery'] is True
                and row['physics_replay']['staging_reconstructed'] is True, 'COMPLETE_REPLAY')
        required.update(str(output / 'r6' / block['id'] / n)
                        for n in ('motion_traces.json', 'result.json', 'measurements.bin', 'audit.bin'))
    original = report['summary']['original']
    require(original['replay_complete'] is True and original['saved_result_equal'] is True
            and original['saved_wall_equal'] is True and original['all_gates_pass'] is True
            and original['wall_motion']['all32'] is True
            and original['wall_motion']['per_lane'] == [True] * 32, 'ORIGINAL_REGRESSION')
    label = f"r6_candidate_u{selected['model']['update']}_original"
    require(report['candidate']['original_label'] == label, 'ORIGINAL_LABEL')
    required.update(str(output / 'evaluation_raw' / label / (n + '.jsonl')) for n in ('Z', 'T', 'S', 'X', 'W'))
    required.update(str(output / n) for n in ('r6_original_suite.json', 'r6_original_wall_motion.json'))
    require(required <= verify_inputs(report), 'COMPLETE_INPUT_SET')
    require(ctx['state']['counters']['nontraining_transitions'] == ctx['contract']['budgets']['nontraining_transitions']
            and ctx['state']['counters']['training_transitions'] == 0
            and ctx['state']['counters']['real_optimizer_updates'] == 0, 'EVALUATION_COUNTERS')
    return selected
