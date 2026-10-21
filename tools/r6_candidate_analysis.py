"""CPU candidate replay with exact selection; no model execution or acceptance."""
from pathlib import Path
from tools.r6_recovery import checked, file_pin, RecoveryError
from training_workloads.unified_h1 import robustness_analysis as replay
from training_workloads.unified_h1 import robustness_evaluation_selection as selection
from training_workloads.unified_h1 import robustness_contract as base

MODE='research-r6-candidate-evaluation'


def context(auth, output):
    p=auth['parameters'];selected=selection.validate(p['evaluation_selection'])
    request=p['r6'];comparison=selected['comparison']
    if (auth['mode']!=MODE or Path(auth['attempt'])/'results'/MODE!=Path(output)
            or auth['budgets']!=base.budget_from_parameters(p)
            or any(request[k]!=comparison[k] for k in ('protocol','seed','split','block_ids'))
            or p['training_seed']!=selected['model']['training_seed'] or p['evaluation_seed']!=20260909
            or p['baseline_bindings']!=selected['selection']['baseline_bindings']):
        raise RecoveryError('CANDIDATE_ANALYSIS_AUTHORIZATION')
    return selected


def identity(selected):
    return dict(selection_sha256=base.object_sha(selected['selection']),
        checkpoint=selected['model']['checkpoint'],update=selected['model']['update'],
        method=selected['selection']['method'],comparison=selected['selection']['comparison'],
        original_label=f"r6_candidate_u{selected['model']['update']}_original")


def validate_binding(binding,selected):
    model=selected['model'];method=selected['method'];ident=identity(selected)
    expected=dict(checkpoint=model['checkpoint'],checkpoint_source_identity=model['source_identity'],
        method_hashes=model['method_hashes'],evaluation_selection_sha256=ident['selection_sha256'],
        comparison=ident['comparison'],method=ident['method'],original_evaluation_label=ident['original_label'],
        ics_clearance_margin_m=method['ics_clearance_margin_m'],physical_history_continuity_claimed=False,
        actor_input_extension=False)
    if any(binding.get(k)!=v or type(binding.get(k))is not type(v) for k,v in expected.items()):
        raise RecoveryError('CANDIDATE_ANALYSIS_MODEL_BINDING')
    selection.verify_loaded(selected,binding['metadata'])
    validate_method_binding(binding,selected)


def validate_method_binding(binding,selected):
    model=selected['model'];method=selected['method']
    actual=binding['actual_method_binding'];env=actual['actual_environment']
    expected_ppo=dict(method['ppo'],permutation_seed=model['training_seed'])
    if (actual['schema_version']!='r6-actual-method-binding/1' or actual['method_sha256']!=base.object_sha(method)
            or actual['actual_ppo']!=expected_ppo or actual['mean_prior_weight']!=method['mean_prior_weight']
            or actual['unchanged_weights_adam_moments_and_counters']is not True
            or actual['strict_resume']is not (selected['selection']['role']=='trained_candidate')
            or env['pure_hover']is not False
            or any(env[k]!=method[k] for k in ('residual_neutrality_weight','anchor_weight','ics_clearance_margin_m'))
            or any(not env[k] or any(x!=v for x in env[k]) for k,v in (
                ('pipeline_margins',method['ics_clearance_margin_m']),
                ('actor_optimizer_lrs',method['ppo']['actor_learning_rate']),
                ('critic_optimizer_lrs',method['ppo']['critic_learning_rate'])))):
        raise RecoveryError('CANDIDATE_ANALYSIS_METHOD_BINDING')


def analyze(output,auth_pin):
    output=Path(output);inputs=replay.Inputs();auth=inputs.read(checked(auth_pin),65536)
    selected=context(auth,output);ident=identity(selected);request=auth['parameters']['r6']
    report,summary=replay.analyze(output,checked(request['protocol']),run_id=request['run_id'],
        seed=request['seed'],split=request['split'],original_update=ident['update'],original_label=ident['original_label'])
    issues=[]
    try:
        binding=inputs.read(output/'r6_model_binding.json')
        validate_binding(binding,selected)
    except (OSError,ValueError,KeyError,TypeError,PermissionError,RecoveryError) as error:
        issues.append(type(error).__name__+':'+str(error)[:240])
    inputs.verify()
    report.update(candidate=ident,candidate_analyzer=file_pin(__file__),authorization=auth_pin,
        candidate_binding_issues=issues)
    report['inputs']+=list(inputs.pins.values())
    summary.update(candidate=ident,candidate_binding_issues=issues,learning_gain_established=False,
        runtime_method_execution_verified=False)
    if len(replay.encoded(summary))>replay.MAX_SUMMARY_BYTES:raise ValueError('SUMMARY_SIZE_BOUND')
    return report,summary
