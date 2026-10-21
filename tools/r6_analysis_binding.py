"""Bind offline motion replay to closed-job originals without accepting science.

This consumer launches nothing. The report is an implementation-produced replay,
not an independent review or permission to dispatch another experiment.
"""
import json
from pathlib import Path

from rlflow.store import atomic, canonical, sha
from tools.r6_recovery import checked, file_pin, RecoveryError
from tools.r6_job_handoff import exact_job, cleanup_evidence, TERMINAL


def read(pin):
    return json.loads(checked(pin).read_text())


def analysis_context(entry, key, store):
    spec = read(entry['request']['spec'])
    job = exact_job(store.replay(), entry, spec)
    if job['status'] not in TERMINAL:
        raise RecoveryError('ANALYSIS_REQUIRES_TERMINAL_JOB')
    _, native = cleanup_evidence(store.root/'jobs'/job['id'], job)
    closure = read(entry['closure']); inventory = read(entry['inventory']); postrun = read(entry['postrun'])
    if (closure['event_key'] != key or closure['job_id'] != job['id'] or closure['spec_sha256'] != sha(spec)
            or closure['native_status'] != native['status'] or closure['native_exit_code'] != native.get('exit_code')
            or inventory['event_key'] != key or inventory['job_id'] != job['id']
            or inventory['spec'] != entry['request']['spec'] or inventory['closure'] != entry['closure']
            or postrun['job_event_key'] != key or postrun['job_id'] != job['id']
            or postrun['closure'] != entry['closure'] or postrun['inventory'] != entry['inventory']
            or postrun.get('postrun_complete') is not True or postrun.get('postrun_queries_used') != 2
            or len(postrun.get('receipts',[])) != 2
            or len({p['path'] for p in postrun['receipts']}) != 2):
        raise RecoveryError('ANALYSIS_JOB_CHAIN_BINDING')
    for receipt in postrun['receipts']:
        checked(receipt)
    attempt = Path(spec['h1']['attempt'])
    if (attempt.parent != store.root.resolve()/'attempts' or inventory['roots']['attempt'] != str(attempt)
            or spec['h1']['mode'] not in ('research-r6-evaluation','research-r6-training','research-r6-candidate-evaluation')):
        raise RecoveryError('ANALYSIS_EXACT_R6_ATTEMPT')
    mode=spec['h1']['mode']
    output = attempt/'results'/mode
    auth_pin = spec['pins']['authorization']; auth = read(auth_pin)
    argv = spec['argv']
    for flag, value in [('--authorization', auth_pin['path']), ('--attempt',str(attempt)), ('--mode',mode)]:
        if argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv) or argv[argv.index(flag)+1] != value:
            raise RecoveryError('ANALYSIS_AUTHORIZATION_ARGV')
    if auth['mode'] != spec['h1']['mode']:
        raise RecoveryError('ANALYSIS_AUTHORIZATION_MODE')
    parameters = auth['parameters']
    if mode=='research-r6-training':
        from training_workloads.unified_h1 import robustness_training_plan as training
        plan=training.validate(parameters['r6_training_plan'])
        if auth['budgets']!=training.required_budget(plan):raise RecoveryError('ANALYSIS_TRAINING_BUDGET')
        return dict(spec=spec,job=job,closure=closure,inventory=inventory,output=output,
            auth_pin=auth_pin,parameters=parameters,request=plan,protocol=None)
    selected=None
    if mode=='research-r6-candidate-evaluation':
        from tools.r6_candidate_analysis import context as candidate_context
        selected=candidate_context(auth,output)
    request = parameters['r6']
    protocol = read(request['protocol'])
    if (protocol.get('frozen') is not True or len(protocol['blocks']) != 16
            or len(set(request['block_ids'])) != 16 or len(request['block_ids']) != 16
            or set(request['block_ids']) != {b['id'] for b in protocol['blocks']}):
        raise RecoveryError('ANALYSIS_PROTOCOL_NOT_FROZEN_OR_INCOMPLETE')
    return dict(spec=spec,job=job,closure=closure,inventory=inventory,output=output,
                auth_pin=auth_pin,parameters=parameters,request=request,protocol=protocol,selected=selected)


def bind_analysis(entry, key, report_pin, store):
    context=analysis_context(entry,key,store)
    if context['spec']['h1']['mode']=='research-r6-training':
        return bind_training_analysis(entry,key,report_pin,context)
    spec=context['spec'];job=context['job'];closure=context['closure'];inventory=context['inventory']
    output=context['output'];auth_pin=context['auth_pin'];parameters=context['parameters']
    request=context['request'];protocol=context['protocol']
    candidate=context.get('selected')
    label='r6_baseline_u640_original'
    if checked(report_pin).stat().st_size > 16*1024**2:
        raise RecoveryError('ANALYSIS_REPORT_SIZE_BOUND')
    report = read(report_pin)
    if report.get('schema_version') != 'r6-motion-replay/1' or report.get('scientific_acceptance') is not False:
        raise RecoveryError('ANALYSIS_REPORT_SCHEMA')
    if candidate is not None:
        from tools import r6_candidate_analysis as ca
        ident=ca.identity(candidate);label=ident['original_label']
        if (report.get('candidate')!=ident or report.get('candidate_analyzer')!=file_pin(ca.__file__)
                or report.get('authorization')!=auth_pin or report['summary'].get('candidate')!=ident
                or not isinstance(report.get('candidate_binding_issues'),list)
                or report['summary'].get('candidate_binding_issues')!=report['candidate_binding_issues']
                or any(report['summary'].get(k)is not False for k in ('learning_gain_established','runtime_method_execution_verified'))):
            raise RecoveryError('ANALYSIS_CANDIDATE_IDENTITY')
    for field in ('run_id','seed','split'):
        if report[field] != request[field]:
            raise RecoveryError('ANALYSIS_RUN_IDENTITY')
    if [r['block_id'] for r in report['blocks']] != [b['id'] for b in protocol['blocks']]:
        raise RecoveryError('ANALYSIS_BLOCK_COVERAGE')
    from training_workloads.unified_h1 import robustness_analysis, robustness
    for name, path in [('analyzer',robustness_analysis.__file__),('scorer',robustness.__file__)]:
        if report[name] != robustness_analysis.pin(path):
            raise RecoveryError('ANALYSIS_IMPLEMENTATION_DRIFT')
    root=Path(__file__).resolve().parents[1]
    for name,relative in [('measurement_replayer','training_workloads/unified_h1/robustness_measurement_replay.py'),
                          ('audit_reader','training_workloads/unified_h1/robustness_audit.py'),
                          ('force_codec','training_workloads/unified_h1/force_archive.py'),
                          ('physics_replayer','training_workloads/unified_h1/robustness_physics_replay.py'),
                          ('delivery_implementation','isaaclab-training/instinctlab/observations/robustness.py')]:
        if report.get(name)!=robustness_analysis.pin(root/relative):
            raise RecoveryError('ANALYSIS_MEASUREMENT_IMPLEMENTATION_DRIFT')
    indexed = {r['path']:r for r in inventory['files']}
    if len(indexed) != len(inventory['files']):
        raise RecoveryError('ANALYSIS_DUPLICATE_INVENTORY_PATH')
    def inventoried(pin):
        row = indexed.get(pin['path'])
        if row is None or row['sha256'] != pin['sha256'] or Path(pin['path']).stat().st_size != row['bytes']:
            raise RecoveryError('ANALYSIS_INPUT_NOT_IN_EXACT_INVENTORY')
        checked(dict(path=pin['path'],sha256=pin['sha256']))
    inventoried(auth_pin)
    seen = set()
    for pin in report['inputs']:
        if pin['path'] in seen:
            raise RecoveryError('ANALYSIS_DUPLICATE_INPUT')
        seen.add(pin['path'])
        external = [request['protocol']]
        if protocol.get('physics_replay',{}).get('mount_source'):
            external.append(protocol['physics_replay']['mount_source'])
        if {k:pin[k] for k in ('path','sha256')} in external:
            checked({k:pin[k] for k in ('path','sha256')})
        elif candidate is not None and {k:pin[k] for k in ('path','sha256')}==auth_pin:
            inventoried(pin)
        else:
            if output not in Path(pin['path']).parents:
                raise RecoveryError('ANALYSIS_OTHER_ATTEMPT_INPUT')
            inventoried(pin)
    if request['protocol']['path'] not in seen:
        raise RecoveryError('ANALYSIS_PROTOCOL_INPUT_MISSING')
    for row in report['blocks']:
        if row['replay_complete'] and any(str(output/'r6'/row['block_id']/name) not in seen
                for name in ('motion_traces.json','result.json')):
            raise RecoveryError('ANALYSIS_COMPLETE_BLOCK_WITHOUT_INPUTS')
        if row.get('measurement_replay',{}).get('bit_exact_delivery') is True and any(
                str(output/'r6'/row['block_id']/name) not in seen for name in ('measurements.bin','audit.bin')):
            raise RecoveryError('ANALYSIS_MEASUREMENT_WITHOUT_INPUTS')
        if row.get('physics_replay',{}).get('staging_reconstructed') is True and str(output/'r6'/row['block_id']/'audit.bin') not in seen:
            raise RecoveryError('ANALYSIS_PHYSICS_WITHOUT_INPUTS')
    original = report['summary']['original']
    if original['replay_complete']:
        required = [output/'evaluation_raw'/label/(case+'.jsonl') for case in ('Z','T','S','X','W')]
        required += [output/'r6_original_suite.json',output/'r6_original_wall_motion.json']
        if any(str(path) not in seen for path in required):
            raise RecoveryError('ANALYSIS_COMPLETE_ORIGINAL_WITHOUT_INPUTS')
    binding_path = output/'r6_model_binding.json'
    issues = []
    model_binding = None
    if str(binding_path) not in indexed:
        issues.append('MODEL_BINDING_MISSING')
    else:
        model_binding = dict(path=str(binding_path),sha256=indexed[str(binding_path)]['sha256'])
        inventoried(model_binding)
        binding = read(model_binding)
        if candidate is not None:
            ca.validate_binding(binding,candidate)
            if str(binding_path) not in seen:raise RecoveryError('ANALYSIS_CANDIDATE_BINDING_INPUT')
        else:
            from training_workloads.unified_h1.robustness_contract import baseline
            baseline_record = baseline(parameters)
            if (binding['checkpoint'] != baseline_record['selected_model']
                    or binding['checkpoint_source_identity'] != baseline_record['source_identity']
                    or binding['method_hashes'] != baseline_record['method_hashes']
                    or binding['ics_clearance_margin_m'] != .15 or binding['actor_input_extension'] is not False
                    or binding['physical_history_continuity_claimed'] is not False):
                raise RecoveryError('ANALYSIS_MODEL_BINDING')
    if candidate is not None:
        if auth_pin['path'] not in seen:raise RecoveryError('ANALYSIS_CANDIDATE_AUTH_INPUT')
        if report['candidate_binding_issues']:issues.append('CANDIDATE_MODEL_EVIDENCE_INCOMPLETE_OR_MISMATCHED')
    if not inventory['output_inventory_complete']:
        issues.append('INVENTORY_INCOMPLETE')
    if any(row.get('measurement_replay',{}).get('bit_exact_delivery') is not True for row in report['blocks']):
        issues.append('MEASUREMENT_REPLAY_INCOMPLETE')
    if any(row.get('physics_replay',{}).get('staging_reconstructed') is not True for row in report['blocks']):
        issues.append('PHYSICS_STAGING_REPLAY_INCOMPLETE')
    if not original['replay_complete'] or not original.get('saved_result_equal') or not original.get('saved_wall_equal') or any(
            not row['replay_complete'] or not row['saved_result_equal'] for row in report['blocks']):
        issues.append('MOTION_EVIDENCE_INCOMPLETE_OR_MISMATCHED')
    return dict(schema_version='r6-bound-motion-analysis/1',job_id=job['id'],event_key=key,
        spec=entry['request']['spec'],authorization=auth_pin,protocol=request['protocol'],
        closure=entry['closure'],inventory=entry['inventory'],postrun=entry['postrun'],
        report=report_pin,model_binding=model_binding,issues=issues,
        native_status=closure['native_status'],native_exit_code=closure['native_exit_code'],
        scientific_acceptance=False,status='BOUND_REPLAY_REQUIRES_CHAIN_AUDIT_AND_INDEPENDENT_REVIEW')


def bind_training_analysis(entry,key,report_pin,context):
    """Bind structural training evidence only; semantic audits remain explicitly outstanding."""
    if checked(report_pin).stat().st_size>16*1024**2:raise RecoveryError('ANALYSIS_REPORT_SIZE_BOUND')
    report=read(report_pin);plan=context['request'];output=context['output'];inventory=context['inventory']
    from training_workloads.unified_h1 import robustness_training_plan as training
    from tools import r6_training_analysis as analyzer
    if (report.get('schema_version')!='r6-training-analysis/1' or report.get('scientific_acceptance') is not False
            or report.get('authorization')!=context['auth_pin'] or report.get('plan_sha256')!=training.object_sha(plan)):
        raise RecoveryError('ANALYSIS_TRAINING_IDENTITY')
    if report.get('analyzer')!=file_pin(analyzer.__file__) or report.get('codec')!=file_pin(analyzer.codec.__file__):
        raise RecoveryError('ANALYSIS_IMPLEMENTATION_DRIFT')
    summary=report['summary']
    if (summary.get('schema_version')!='r6-training-summary/1'
            or any(summary.get(k)!=plan[k] for k in ('experiment_id','phase','training_seed','start_update','target_update'))
            or any(summary.get(k) is not False for k in ('scientific_acceptance','learning_gain_established',
                'physical_replay_performed','strict_checkpoint_restore_performed','rng_state_semantics_verified'))
            or summary.get('issue_count')!=len(report['issues'])):
        raise RecoveryError('ANALYSIS_TRAINING_SUMMARY')
    if report.get('checkpoint_auditor')!=file_pin(Path(__file__).with_name('r6_training_checkpoint_audit.py')):
        raise RecoveryError('ANALYSIS_IMPLEMENTATION_DRIFT')
    if report.get('rng_auditor')!=file_pin(Path(__file__).with_name('r6_training_rng_audit.py')):
        raise RecoveryError('ANALYSIS_IMPLEMENTATION_DRIFT')
    rng_audit=summary.get('rng_initialization_audit')
    if rng_audit is not None and (rng_audit.get('status')!='PASS' or rng_audit.get('kind')!=plan['initialization']['kind']
            or rng_audit.get('training_seed')!=plan['training_seed'] or rng_audit.get('starting_update')!=plan['start_update']
            or any(rng_audit.get(k) is not False for k in ('cuda_seed_derivation_verified','actual_rollout_rng_use_verified','persistence_order_verified','physical_continuity'))):
        raise RecoveryError('ANALYSIS_RNG_AUDIT_RESULT')
    probability=summary.get('probability_replay')
    if type(summary.get('probability_gate_recomputed'))is not bool or summary['probability_gate_recomputed']!=(probability is not None):
        raise RecoveryError('ANALYSIS_PROBABILITY_REPLAY_FLAG')
    if probability is not None and any(type(probability.get(k))is not type(v) or probability[k]!=v for k,v in dict(
            status='PASS',sample_count=512,starting_update=plan['start_update'],float32_atol=1e-6,float32_rtol=1e-5,
            cpu_actor_forward=True,cpu_rng_preserved=True,gpu_sampling_reproduced=False,optimizer_restored=False).items()):
        raise RecoveryError('ANALYSIS_PROBABILITY_REPLAY_RESULT')
    if report.get('probability_replayer')!=file_pin(Path(__file__).with_name('r6_training_probability.py')):
        raise RecoveryError('ANALYSIS_IMPLEMENTATION_DRIFT')
    indexed={r['path']:r for r in inventory['files']}
    if len(indexed)!=len(inventory['files']):raise RecoveryError('ANALYSIS_DUPLICATE_INVENTORY_PATH')
    seen=set()
    for pin in report['inputs']:
        path=Path(pin['path']);row=indexed.get(str(path))
        if str(path) in seen:raise RecoveryError('ANALYSIS_DUPLICATE_INPUT')
        seen.add(str(path))
        if {k:pin[k] for k in ('path','sha256')}==plan['initialization']['checkpoint']:
            if checked(plan['initialization']['checkpoint']).stat().st_size!=pin['bytes']:raise RecoveryError('ANALYSIS_ORIGIN_SIZE')
            continue
        if path!=Path(context['auth_pin']['path']) and output not in path.parents:
            raise RecoveryError('ANALYSIS_OTHER_ATTEMPT_INPUT')
        if row is None or any(pin.get(k)!=row[k] for k in ('sha256','bytes')) or checked(dict(path=str(path),sha256=pin['sha256'])).stat().st_size!=row['bytes']:
            raise RecoveryError('ANALYSIS_INPUT_NOT_IN_EXACT_INVENTORY')
    required={context['auth_pin']['path']}
    if rng_audit is not None:
        required.add(plan['initialization']['checkpoint']['path'])
        required.add(str(output/'compressed_evidence/rng/initialization.pt.r6z'))
    if probability is not None:
        required.add(plan['initialization']['checkpoint']['path'])
        required.add(str(output/'compressed_evidence'/f"pilot_diagnostic/update_{plan['start_update']+1}_full_input.pt.r6z"))
    for name in indexed:
        path=Path(name)
        if (output/'compressed_evidence' in path.parents and path.suffix=='.r6z') or (output/'checkpoints'==path.parent and path.suffix in ('.pt','.json')) or path in (output/'status.json',output/'pilot_diagnostic/first_update_gate.json'):
            required.add(name)
    if not required<=seen:raise RecoveryError('ANALYSIS_TRAINING_INPUT_MISSING')
    for stream in report['streams']:
        pin=stream.get('pin')
        if pin and (pin['path'] not in seen or pin['sha256']!=indexed[pin['path']]['sha256']):
            raise RecoveryError('ANALYSIS_TRAINING_STREAM_BINDING')
    issues=['TRAINING_SEMANTIC_AUDIT_AND_EVALUATION_REQUIRED']
    if report['issues']:issues.append('TRAINING_EVIDENCE_INCOMPLETE_OR_MISMATCHED')
    if not inventory['output_inventory_complete']:issues.append('INVENTORY_INCOMPLETE')
    closure=context['closure']
    return dict(schema_version='r6-bound-training-analysis/1',job_id=context['job']['id'],event_key=key,
        spec=entry['request']['spec'],authorization=context['auth_pin'],protocol=None,
        training_plan_sha256=training.object_sha(plan),closure=entry['closure'],inventory=entry['inventory'],
        postrun=entry['postrun'],report=report_pin,model_binding=None,issues=issues,
        native_status=closure['native_status'],native_exit_code=closure['native_exit_code'],
        scientific_acceptance=False,status='BOUND_STRUCTURAL_TRAINING_REQUIRES_SEMANTIC_AUDIT_AND_EVALUATION')


def record_analysis(journal, key, report_pin, store):
    journal.guard()
    state = journal.read(); entry = journal.canonical_job(key,state=state)
    if state.get('current_job') is not None or state.get('current_model_event') is not None:
        raise RecoveryError('ANALYSIS_HANDOFF_NOT_IDLE')
    result = bind_analysis(entry,key,report_pin,store)
    path = journal.directory/'jobs'/key/'analysis.json'
    if entry.get('analysis'):
        if read(entry['analysis']) != result:
            raise RecoveryError('ANALYSIS_REPLAY_DRIFT')
    elif path.exists():
        if json.loads(path.read_text()) != result:
            raise RecoveryError('ANALYSIS_REPLAY_DRIFT')
    else:
        data = canonical(result)+b'\n'
        if len(data) > 65536:raise RecoveryError('ANALYSIS_BINDING_SIZE_BOUND')
        atomic(path,data)
    entry['analysis'] = file_pin(path)
    # Preserve STOP, overruns and other control failures. This record alone
    # authorizes neither another model call nor another job.
    journal.save(state)
    return result
