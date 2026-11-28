"""CPU analysis of a closed direct run; writes a new, separate result directory."""
import argparse
import json
from pathlib import Path
from tools.r6_direct.dependencies import install
from tools.r6_direct.evidence import context, add_inputs, bind, validate_binding
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.io import atomic


def analyze(input_pin, *, probability_diagnostic=None):
    install()
    ctx = context(input_pin)
    operation = ctx['contract']['operation']
    if probability_diagnostic is not None and operation!='train':
        raise ValueError('PROBABILITY_DIAGNOSTIC_REQUIRES_TRAIN')
    if operation == 'render':
        from tools.r6_video_presentation import verify
        manifest=pin(ctx['output']/'media/RUN_MANIFEST.json')
        source = json.loads(Path(ctx['plan']['source']['path']).read_text())
        from tools.r6_evidence import checked
        checked(ctx['plan']['source'])
        renderer = dict(path=str(Path(ctx['plan']['snapshot'])/'tools/render_r6_isaac_video.py'),
                        sha256=source['sources']['tools/render_r6_isaac_video.py'])
        summary=verify(manifest,ctx['contract']['parameters']['extraction'], renderer_pin=renderer)
        report=dict(schema_version='r6-direct-video-analysis/1',summary=summary,scientific_acceptance=False,
                    inputs=[dict(manifest,bytes=Path(manifest['path']).stat().st_size)])
        add_inputs(report,input_pin,ctx)
    elif operation == 'baseline':
        from training_workloads.unified_h1.robustness_analysis import analyze as replay
        request = ctx['contract']['parameters']['r6']
        report, summary = replay(ctx['output'], Path(request['protocol']['path']),
                                 run_id=request['run_id'], seed=request['seed'], split=request['split'])
        add_inputs(report, input_pin, ctx)
    elif operation == 'dynamic':
        from training_workloads.unified_h1 import command_replay
        from training_workloads.unified_h1.dynamic_assay import audit
        budgets = ctx['contract']['budgets']
        counters = ctx['state']['counters']
        if (counters['nontraining_transitions'] != budgets['nontraining_transitions']
                or counters['explicit_resets'] != budgets['explicit_resets']
                or counters['real_optimizer_updates'] != 0):
            raise ValueError('DYNAMIC_ASSAY_INCOMPLETE_COUNTERS')
        if command_replay.matches(ctx['plan']['run_id']):
            from training_workloads.unified_h1 import command_replay_audit

            summary, pins = command_replay_audit.audit(ctx['output'], ctx['contract']['parameters'], counters)
            pins.extend(pin(Path(module.__file__)) for module in (command_replay, command_replay_audit))
        else:
            raw, pins = audit(ctx['output'], ctx['contract']['parameters'], counters)
            summary = {'models': raw['models'], 'counters': counters, 'witnesses_replayed': True}
        report = dict(summary=summary, scientific_acceptance=False,
                      analyzer=pin(Path(__file__)),
                      inputs=[dict(v, bytes=Path(v['path']).stat().st_size) for v in pins])
        add_inputs(report, input_pin, ctx)
    else:
        from tools import r6_training_analysis, r6_candidate_analysis, r6_final_analysis
        module = {'train': r6_training_analysis, 'evaluate': r6_candidate_analysis,
                  'final': r6_final_analysis}[operation]
        extra=dict(probability_diagnostic=probability_diagnostic) if operation=='train' else {}
        report, summary = module.analyze(ctx['output'], input_pin, direct=True,**extra)
        if operation in ('train', 'evaluate'):
            from instinctlab.control.escape_recovery import MOUNT_PITCH_DEGREES
            from tools.r6_direct.evidence import read
            from training_workloads.unified_h1.lidar_calibration import verify as verify_mount
            from training_workloads.unified_h1.robustness_training_method import action_option_version

            parameters = ctx['contract']['parameters']
            definition = (read(parameters['r6_training_plan']['recipe']) if operation == 'train'
                          else read(parameters['evaluation_selection']['comparison']))
            override = parameters['evaluation_selection'].get('execution_override') if operation == 'evaluate' else None
            pitch = MOUNT_PITCH_DEGREES.get(override or action_option_version(definition['candidate_id']))
            if pitch is not None:
                calibration = pin(ctx['output']/'lidar-calibration.json')
                observed = read(calibration)
                rays = verify_mount(observed, ctx['output'], 'execution', pitch)
                for item in (calibration, rays):
                    report['inputs'].append(dict(item, bytes=Path(item['path']).stat().st_size))
                summary['lidar_mount'] = observed
            if operation == 'evaluate':
                from training_workloads.unified_h1.command_replay import (
                    BRAKE_EXECUTION, CONFIRMED_STANDOFF_EXECUTION, CONFIRMED_TUBE_EXECUTIONS, EDGE_CONTINUATION_EXECUTION,
                    FINITE_EDGE_EXECUTIONS,
                    NEIGHBOR_HISTORY_EXECUTIONS, REGRESSION_IDS, SETTLED_OFFSET_STOP_EXECUTIONS, WALL_TANGENT_EXECUTIONS,
                    ALONG_INTERCEPT_EXECUTIONS, CROSSING_CONTINUATION_EXECUTIONS, CROSSING_FALLBACK_EXECUTIONS,
                    CROSSING_COMPATIBLE_EXECUTIONS, CROSSING_EARLY_EXECUTIONS, CROSSING_SHALLOW_EXECUTIONS, CROSSING_STOP_EXECUTIONS,
                    CROSSING_EXTENDED_RETREAT_EXECUTIONS, CROSSING_RECEDING_HOVER_EXECUTIONS, CROSSING_VERIFIED_HOVER_EXECUTIONS,
                    CROSSING_REFERENCE_EXECUTIONS, CROSSING_SETTLED_EXECUTIONS, CROSSING_SETTLED_MOUNT,
                    BILATERAL_STOPPING_EXECUTION, SAMPLED_TURN_EXECUTIONS, OBSERVED_HEADING_EXECUTIONS, SHALLOW_STEERING_EXECUTIONS,
                    FAR_CROSSING_REFERENCE_EXECUTIONS, HOVER_OWN_SIDE_EXECUTIONS, VERIFIED_RETREAT_ONSET_EXECUTIONS,
                    CLIPPED_RETREAT_ONSET_EXECUTIONS, TUBE_STOPPING_EXECUTION,
                    CROSSING_RESTART_EXECUTIONS, HOVER_REFERENCE_RESERVE_EXECUTIONS, LOCAL_WALL_STOPPING_EXECUTIONS, FRONT_WALL_BRAKING_EXECUTIONS, FACETED_WALL_STOPPING_EXECUTION, WALL_EDGE_REPROOF_EXECUTION,
                    OVERLAPPING_STOP_EXECUTIONS, BRAKE_RECOVERY_EXECUTIONS, BRAKE_TRANSITION_EXECUTION, RETREAT_CONTINUATION_EXECUTIONS, OLDEST_PARTIAL_TURN_EXECUTION,
                    EARLY_DIAGONAL_EXECUTIONS, FINITE_INTERCEPT_EXECUTIONS,
                    INTERCEPT_CHOICE_EXECUTIONS, OPENING_STEERING_EXECUTIONS, RESERVE_DIRECTION_EXECUTIONS,
                    ROTATION_TUBE_EXECUTIONS, TANGENT_MAGNITUDE_EXECUTION,
                    VERIFIED_RESUME_EXECUTION, VERIFIED_ZERO_EXECUTIONS,
                )

                if ctx['plan']['run_id'] in REGRESSION_IDS:
                    binding = pin(ctx['output']/'command-execution.json')
                    observed = read(binding)
                    tube = override != BRAKE_EXECUTION
                    expected = dict(execution=override, checkpoint=parameters['evaluation_selection']['checkpoint'],
                                    mount_pitch_degrees=25, clearance_margin_m=.15,
                                    motion_tube_radius_m=.60 if tube else 0., confirm_motion=tube)
                    if override in NEIGHBOR_HISTORY_EXECUTIONS:
                        expected['uses_neighbor_history'] = True
                    if override == CONFIRMED_STANDOFF_EXECUTION:
                        expected['confirmed_standoff_m'] = .45
                    if override in SETTLED_OFFSET_STOP_EXECUTIONS:
                        expected['settled_offset_stop'] = True
                    if override in WALL_TANGENT_EXECUTIONS:
                        expected['wall_tangent'] = True
                    if override in CONFIRMED_TUBE_EXECUTIONS:
                        expected['confirmed_tube_radius_m'] = .41
                    if override in ROTATION_TUBE_EXECUTIONS:
                        expected['confirmed_tube_align_rotation'] = True
                    if override in FINITE_EDGE_EXECUTIONS:
                        expected['finite_edge'] = True
                    if override == EDGE_CONTINUATION_EXECUTION:
                        expected['edge_continuation'] = True
                    if override == TANGENT_MAGNITUDE_EXECUTION:
                        expected['tangent_magnitude'] = True
                    if override == VERIFIED_RESUME_EXECUTION:
                        expected['verified_resume'] = True
                    if override in VERIFIED_ZERO_EXECUTIONS:
                        expected['verified_zero'] = True
                    if override in INTERCEPT_CHOICE_EXECUTIONS:
                        expected.update(intercept_choice=True, receding_stop=True)
                    if override in EARLY_DIAGONAL_EXECUTIONS:
                        expected['early_diagonal'] = True
                    if override in RESERVE_DIRECTION_EXECUTIONS:
                        expected['reserve_direction'] = True
                    if override in CROSSING_FALLBACK_EXECUTIONS:
                        expected.update(zero_recheck=True, reverse_diagonal=True)
                    if override in OPENING_STEERING_EXECUTIONS:
                        expected['opening_steering'] = True
                    if override in ALONG_INTERCEPT_EXECUTIONS:
                        expected['along_intercept'] = True
                    if override in FINITE_INTERCEPT_EXECUTIONS:
                        expected['finite_intercept'] = True
                    if override in CROSSING_STOP_EXECUTIONS:
                        expected['crossing_stop'] = True
                    if override in CROSSING_CONTINUATION_EXECUTIONS:
                        expected['crossing_stop_continuation'] = True
                    if override in CROSSING_SHALLOW_EXECUTIONS:
                        expected['crossing_stop_shallow'] = True
                    if override in CROSSING_EARLY_EXECUTIONS:
                        expected['crossing_stop_early'] = True
                    if override in CROSSING_COMPATIBLE_EXECUTIONS:
                        expected['crossing_stop_compatible'] = True
                    if override in CROSSING_VERIFIED_HOVER_EXECUTIONS:
                        expected['crossing_stop_verified_hover'] = True
                    if override in CROSSING_RECEDING_HOVER_EXECUTIONS:
                        expected.update(crossing_stop_receding_hover=True, crossing_retreat_own_side=True)
                    if override in CROSSING_EXTENDED_RETREAT_EXECUTIONS:
                        expected['crossing_retreat_extended'] = True
                    if override in CROSSING_SETTLED_EXECUTIONS:
                        expected['settled_crossing_mount_m'] = list(CROSSING_SETTLED_MOUNT)
                    if override in CROSSING_REFERENCE_EXECUTIONS:
                        expected.update(static_scan_stop=True, crossing_reference=True)
                    if override in SHALLOW_STEERING_EXECUTIONS:
                        expected['shallow_steering'] = True
                    if override in OBSERVED_HEADING_EXECUTIONS:
                        expected.update(observed_heading_steering=True, tracked_resume=True)
                    if override in SAMPLED_TURN_EXECUTIONS:
                        expected['crossing_turn_between_samples'] = True
                    if override == BILATERAL_STOPPING_EXECUTION:
                        expected['bilateral_stopping'] = True
                    if override in HOVER_OWN_SIDE_EXECUTIONS:
                        expected['hover_escape_own_side'] = True
                    if override in FAR_CROSSING_REFERENCE_EXECUTIONS:
                        expected['far_crossing_reference'] = True
                    if override in VERIFIED_RETREAT_ONSET_EXECUTIONS:
                        expected['verified_retreat_onset'] = True
                    if override in CLIPPED_RETREAT_ONSET_EXECUTIONS:
                        expected['retain_only_clipped_retreat'] = True
                    if override == TUBE_STOPPING_EXECUTION:
                        expected['tube_stopping'] = True
                    if override in HOVER_REFERENCE_RESERVE_EXECUTIONS:
                        expected['verify_hover_reference_reserve'] = True
                    if override in CROSSING_RESTART_EXECUTIONS:
                        expected['verify_crossing_restart'] = True
                    if override in LOCAL_WALL_STOPPING_EXECUTIONS:
                        expected['local_wall_stopping'] = True
                    if override in OVERLAPPING_STOP_EXECUTIONS:
                        expected['verify_overlapping_stop'] = True
                    if override in BRAKE_RECOVERY_EXECUTIONS:
                        expected['verify_brake_recovery'] = True
                    if override == BRAKE_TRANSITION_EXECUTION:
                        expected['verify_brake_transition'] = True
                    if override in FRONT_WALL_BRAKING_EXECUTIONS:
                        expected['front_wall_braking'] = True
                    if override == FACETED_WALL_STOPPING_EXECUTION:
                        expected['faceted_wall_stopping'] = True
                    if override == WALL_EDGE_REPROOF_EXECUTION:
                        expected['wall_edge_reproof'] = True
                    if override in RETREAT_CONTINUATION_EXECUTIONS:
                        expected['verify_retreat_continuation'] = True
                    if override == OLDEST_PARTIAL_TURN_EXECUTION:
                        expected['verify_oldest_partial_turn'] = True
                    if observed != expected:
                        raise ValueError('COMMAND_REGRESSION_EXECUTION_READBACK')
                    report['inputs'].append(dict(binding, bytes=Path(binding['path']).stat().st_size))
                    summary['command_execution'] = observed
    return report, summary


def supplement_cpu(input_pin, cpu_binding, diagnostic):
    """Reuse a fully byte-verified CPU audit for its exact CUDA-only exception."""
    from tools.r6_direct.probability_evidence import verify

    bound, ctx, report = validate_binding(cpu_binding)
    issue, = report['issues']
    if (bound['run_inputs'] != input_pin or ctx['contract']['operation'] != 'train'
            or issue['input'] != 'first_update_probability_replay'
            or report['summary']['issue_count'] != 1 or report['summary']['probability_gate_recomputed']):
        raise ValueError('DIRECT_CPU_AUDIT_NOT_SUPPLEMENTABLE')
    plan = ctx['contract']['parameters']['r6_training_plan']
    retained = f'pilot_diagnostic/update_{plan["start_update"]+1}_full_input.pt'
    stream, = [row for row in report['streams'] if row['logical_name'] == retained and row['complete']]
    result, extra = verify(diagnostic, ctx['plan']['contract'], stream['pin'])
    report['issues'] = []
    summary = report['summary']
    summary.update(issue_count=0, probability_gate_recomputed=True, probability_replay=result)
    indexed = {row['path']: row for row in report['inputs']}
    for item in (*extra, cpu_binding, bound['report']):
        indexed[item['path']] = dict(item, bytes=Path(item['path']).stat().st_size)
    report['inputs'] = sorted(indexed.values(), key=lambda row:row['path'])
    return report,summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--probability-diagnostic',type=Path,
                        help='Sealed evidence envelope for the exact closed CUDA diagnostic')
    parser.add_argument('--reuse-cpu-binding',type=Path)
    args = parser.parse_args()
    install()
    if stop_paths():
        raise RuntimeError('STOP')
    input_pin = pin(args.inputs)
    ctx = context(input_pin)
    destination = args.destination.absolute()
    originals = (Path(ctx['plan']['attempt_dir']), Path(ctx['plan']['output_root']))
    system_analysis = (ctx['plan'].get('system_results') is True
                       and destination.parent == originals[0]
                       and destination.name in ('analysis-cpu', 'analysis'))
    if (destination.resolve() != destination or destination.exists()
            or (not system_analysis and any(destination == p or p in destination.parents
                                             or destination in p.parents for p in originals))):
        parser.error('new canonical destination separate from originals required')
    if args.reuse_cpu_binding:
        if args.probability_diagnostic is None:
            parser.error('reusing CPU evidence requires the sealed CUDA diagnostic')
        report,summary=supplement_cpu(input_pin,pin(args.reuse_cpu_binding),pin(args.probability_diagnostic))
    else:
        report, summary = analyze(input_pin,probability_diagnostic=None if args.probability_diagnostic is None else pin(args.probability_diagnostic))
    from training_workloads.unified_h1.dynamic_assay import FRONT_STUDY_PREFIX, NEXT_LONG_PREFIX, U1792_STUDY_PREFIX, physical_layout
    from training_workloads.unified_h1.command_replay import LONG_PREFIX
    run_id = ctx['plan']['run_id']
    summary_limit = (2048 if run_id.startswith(LONG_PREFIX) else
                     768 if run_id.startswith((NEXT_LONG_PREFIX, U1792_STUDY_PREFIX, FRONT_STUDY_PREFIX)) and physical_layout(run_id) else
                     384 if physical_layout(run_id) else
                     96 if ctx['contract']['operation'] == 'dynamic' else 48) * 1024
    if len(json.dumps(report, allow_nan=False).encode()) > 16*1024**2 or len(json.dumps(summary, allow_nan=False).encode()) > summary_limit:
        raise ValueError('DIRECT_ANALYSIS_SIZE_BOUND')
    if stop_paths():
        raise RuntimeError('STOP')
    destination.mkdir(parents=True, exist_ok=False)
    atomic(destination/'report.json', report)
    atomic(destination/'summary.json', summary)
    atomic(destination/'binding.json', bind(input_pin, pin(destination/'report.json')))
    print(destination/'summary.json')


if __name__ == '__main__':
    main()
