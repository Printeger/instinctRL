"""CPU final replay with exact selection; no model execution or acceptance."""
from pathlib import Path
from tools.r6_recovery import checked, file_pin, RecoveryError
from training_workloads.unified_h1 import robustness_analysis as replay
from training_workloads.unified_h1 import robustness_final_selection as selection
from training_workloads.unified_h1 import robustness_contract as base

MODE='research-r6-final-evaluation'


def context(auth, output):
    p=auth['parameters'];selected=selection.validate(p['final_selection'])
    request=p['r6'];comparison=selected['comparison']
    if (auth['mode']!=MODE or Path(auth['attempt'])/'results'/MODE!=Path(output)
            or auth['budgets']!=base.budget_from_parameters(p)
            or any(request[k]!=comparison[k] for k in ('protocol','seed','split','block_ids'))
            or p['training_seed']!=selected['model']['training_seed'] or p['evaluation_seed']!=20260909
            or p['baseline_bindings']!=selected['selection']['baseline_bindings']):
        raise RecoveryError('FINAL_ANALYSIS_AUTHORIZATION')
    return selected


def identity(selected):
    return dict(selection_sha256=base.object_sha(selected['final_selection']),
        selection=selected['final_selection'],assignment=selected['assignment'],
        checkpoint=selected['model']['checkpoint'],update=selected['model']['update'],
        method=selected['method_pin'],comparison=selected['comparison'],
        original_label=f"r6_final_u{selected['model']['update']}_original")


def validate_binding(binding,selected):
    model=selected['model'];method=selected['method'];ident=identity(selected)
    expected=dict(checkpoint=model['checkpoint'],checkpoint_source_identity=model['source_identity'],
        method_hashes=model['method_hashes'],final_selection=selected['final_selection'],
        final_selection_sha256=ident['selection_sha256'],assignment=selected['assignment'],
        comparison=ident['comparison'],method=ident['method'],original_evaluation_label=ident['original_label'],
        ics_clearance_margin_m=.15 if method is None else method['ics_clearance_margin_m'],
        physical_history_continuity_claimed=False,actor_input_extension=False)
    if any(binding.get(k)!=v or type(binding.get(k))is not type(v) for k,v in expected.items()):
        raise RecoveryError('FINAL_ANALYSIS_MODEL_BINDING')
    selection.verify_loaded(selected,binding['metadata'])
    if method is not None:
        from tools.r6_candidate_analysis import validate_method_binding
        validate_method_binding(binding,selected)
    else:
        actual=binding['actual_method_binding'];env=actual['actual_environment']
        if (actual['schema_version']!='r6-final-baseline-method-binding/1'
                or actual['unchanged_checkpoint_ppo']is not True
                or actual['baseline_method_hashes']!=model['method_hashes']
                or env['pure_hover']is not False or env['residual_neutrality_weight']!=.05
                or env['anchor_weight']!=4. or env['ics_clearance_margin_m']!=.15
                or len(env['pipeline_margins'])!=32 or any(x!=.15 for x in env['pipeline_margins'])):
            raise RecoveryError('FINAL_ANALYSIS_BASELINE_METHOD_BINDING')


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
    report.update(final=ident,final_analyzer=file_pin(__file__),authorization=auth_pin,
        final_binding_issues=issues)
    report['inputs']+=list(inputs.pins.values())
    summary.update(final=ident,final_binding_issues=issues,learning_gain_established=False,
        runtime_method_execution_verified=False)
    if len(replay.encoded(summary))>replay.MAX_SUMMARY_BYTES:raise ValueError('SUMMARY_SIZE_BOUND')
    return report,summary
