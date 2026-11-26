"""Run the frozen R4 continuation and two independent physical evaluations."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.holdout_summary import compare_holdout
from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import DIAG_TEMPLATE, HISTORICAL, destination, load
from tools.r6_direct.instinct_cycle_finish import analyze, command, freeze, guard, wait_closed
from tools.r6_direct.instinct_cycle_report import displacement
from tools.r6_direct.instinct_followup import PROJECT, R3
from tools.r6_direct.instinct_next import Round
from tools.r6_direct.persistence_summary import summarize as summarize_short
from tools.r6_direct.prepare import build
from tools.r6_direct.resource_policy import GPU_FOLLOWUP_SCHEMA, OUTPUT_CAP, THIRD_CAPS
from tools.r6_direct.source import ROOT as SNAPSHOT, manifest, training_identity
from tools.r6_direct.storage import CONTROL_ALLOWANCE, SYSTEM_FLOOR, SYSTEM_ROOT, SYSTEM_STORAGE_SCHEMA
from tools.r6_direct.study_report import compare, score_model
from training_workloads.unified_h1.dynamic_assay import HOLDOUT_RUN, HOLDOUT_SMOKE
from training_workloads.unified_h1.robustness_training_method import ESCAPE_OPTION_CANDIDATE, bound_method_hashes


PREFIX = 'R6-DIRECT-INSTINCT-GPU-R4-LONG-'
SEGMENTS = ((640, 720, 4500), (720, 768, 5500), (768, 896, 9500), (896, 1024, 9500))
CHECKPOINTS = (642, 672, 704, 720, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024)
TRAINING_SEED = 2026112401
HOLDOUT_SEED = 1096583497
GIB = 2**30


class LongStudy(Round):
    tag_prefix = 'instinct-post-r3'

    def __init__(self):
        self.docs = PROJECT / 'long'
        self.ledger = self.docs / 'RESOURCE_LEDGER.json'
        self.eval = HOLDOUT_RUN
        self.snapshot = SNAPSHOT

    def prepare(self):
        if SNAPSHOT == ROOT:
            raise ValueError('Use the clean detached training snapshot')
        option = pin(PROJECT / 'option-check/RESULTS.json')
        option_result = summarize_short(read(option))['paired']
        approach = option_result['approach']
        controls = [option_result[k] for k in ('static_hover', 'static_travel', 'recede')]
        if (approach['candidate_collisions'] * 2 > approach['reference_collisions']
                or any(r['pass_to_collision'] for r in controls)
                or controls[1]['means']['candidate']['progress_m'] < .95 * controls[1]['means']['reference']['progress_m']
                or controls[0]['means']['candidate']['maximum_lateral_displacement_m']
                > controls[0]['means']['reference']['maximum_lateral_displacement_m'] + .005):
            raise ValueError('Development control gate did not pass')
        smoke = SYSTEM_ROOT / HOLDOUT_SMOKE / 'analysis/binding.json'
        _, smoke_ctx, smoke_report = validate_binding(pin(smoke))
        if (smoke_ctx['plan']['run_id'] != HOLDOUT_SMOKE
                or smoke_report['summary']['counters']['nontraining_transitions'] != 7680
                or not smoke_report['summary']['witnesses_replayed']):
            raise ValueError('Independent physical layout smoke is incomplete')
        if shutil.disk_usage(SYSTEM_ROOT).free < 120*GIB + SYSTEM_FLOOR:
            raise ValueError('Current free capacity does not cover the frozen run and 30 GiB floor')

        self.docs.mkdir(exist_ok=False)
        self.write('source.json', manifest())
        source = self.docs / 'source.json'
        source_identity = training_identity(load(source))
        plan_doc = pin(PROJECT / 'LONG_PLAN.md')
        resource_cap = sum(wall+190 for _, _, wall in SEGMENTS) + 4*390 + 2*(3600+190)
        caps = dict(allocated_gpu_seconds=resource_cap, live_wall_seconds=resource_cap, queue_seconds=1800,
                    live_starts=10, simulator_initializations=6, training_transitions=384*512,
                    nontraining_transitions=193600+69120, training_checkpoint_writes=len(CHECKPOINTS),
                    major_scientific_candidates=1, **{OUTPUT_CAP: 120*GIB})
        policy = self.write('RESOURCE_POLICY.json', dict(schema_version=GPU_FOLLOWUP_SCHEMA,
                            user_authority=plan_doc, caps=caps, elapsed_seconds=50400))
        storage = self.write('STORAGE.json', dict(schema_version=SYSTEM_STORAGE_SCHEMA,
                             root=str(SYSTEM_ROOT), new_volume_allocation_bytes=0))
        zero = dict.fromkeys(THIRD_CAPS, 0)
        self.write(self.ledger.name, dict(resource_policy=policy, caps=caps, historical_charged=zero,
                   new_used=zero, protected_confirmation_delivery_repair=zero, status='FROZEN',
                   new_reservations=[dict(id='ROUND-SYSTEM', status='UNRELEASED',
                                          resources={OUTPUT_CAP: 0}, storage_plan=storage)]))

        train = load(R3 / 'round-3/TRAIN-contract.json')['parameters']
        original = train['r6_training_plan']
        recipe = read(original['recipe'])
        curriculum = read(recipe['curriculum'])
        curriculum['phases'][1]['end_offset'] = 384
        recipe.update(candidate_id=ESCAPE_OPTION_CANDIDATE, source_identity=source_identity,
                      branch_updates=384, checkpoint_offsets=[u-640 for u in CHECKPOINTS],
                      development_seeds=[TRAINING_SEED], confirmation_seeds=[2026112402,2026112403,2026112404],
                      curriculum=self.write('curriculum.json', curriculum),
                      diagnosis=[plan_doc, option, pin(smoke), pin(PROJECT/'development-02/RESULTS.json')],
                      hypothesis='With the same fixed side-step option, longer PPO improves ordinary tracking and post-threat recovery.',
                      comparison='Endpoint versus common u640 with the same option and independent frozen qualified product reference.')
        recipe_pin = self.write('recipe.json', recipe)
        storage_limits = dict(original['storage'], payload_bytes=22*GIB, compressed_bytes=20*GIB,
                              plain_bytes=2*GIB, maximum_raw_file_bytes=64*GIB)
        r3_results = load(R3 / 'round-3/RESULTS.json')
        record = dict(plan=plan_doc, source=pin(source), recipe=recipe_pin, segments=SEGMENTS,
                      storage=storage_limits, checkpoint_updates=CHECKPOINTS, training_seed=TRAINING_SEED,
                      initial=original['initialization'], training_parameters={k:v for k,v in train.items()
                          if k not in ('r6_training_plan', 'source_identity', 'experiment_id', 'training_seed', 'output_cap_bytes')},
                      holdout_seed=HOLDOUT_SEED, qualified=r3_results['references']['previous_gpu_r2'],
                      r3=r3_results['candidate'], first_predecessor=pin(SYSTEM_ROOT/HOLDOUT_SMOKE/'control/receipt.json'))
        record_pin = self.write('DESIGN.json', record)
        self.freeze_segment(record, SEGMENTS[0], record['initial'], record['first_predecessor'], publish=False)
        freeze('instinct-post-r3-long-design', 'Freeze the long PPO stream and independent comparison', *sorted(self.docs.iterdir()))
        return record_pin

    def freeze_segment(self, design, segment, initial, predecessor, *, publish=True):
        start, target, wall = segment
        run_id = PREFIX + f'TRAIN-U{target}'
        params = dict(design['training_parameters'], experiment_id=run_id,
                      training_seed=design['training_seed'], source_identity=training_identity(read(design['source'])),
                      output_cap_bytes=design['storage']['payload_bytes'])
        plan = dict(schema_version='r6-local-training-plan/1', experiment_id=run_id, phase='development',
                    recipe=design['recipe'], selection_freeze=None, training_seed=design['training_seed'],
                    start_update=start, target_update=target, wall_seconds=wall, initialization=initial,
                    checkpoint_updates=[n for n in design['checkpoint_updates'] if start < n <= target],
                    storage=design['storage'])
        params['r6_training_plan'] = plan
        path = self.input_path(run_id)
        if path.exists():
            prior = load(path)
            if read(prior['contract'])['parameters'] != params or prior['predecessor_receipt'] != predecessor:
                raise ValueError('Frozen training inputs drifted')
            return path, plan
        contract = self.write(f'TRAIN-U{target}-contract.json', build('contract', dict(operation='train', parameters=params)))
        self.input(run_id, contract, predecessor, snapshot=self.snapshot)
        if publish:
            freeze(f'instinct-post-r3-train-u{target}-inputs', f'Freeze strict continuation through u{target}',
                   path, Path(contract['path']), self.ledger)
        return path, plan

    def launch(self, path):
        job = load(path)
        if not (Path(job['output_root']) / 'receipt.json').exists():
            if Path(job['attempt_dir']).exists():
                raise RuntimeError(f'Existing job still needs closure: {job["run_id"]}')
            command(self.snapshot, 'run', '--inputs', path, system=True, check=False)
        if not (Path(job['output_root']) / 'receipt.json').exists():
            raise RuntimeError(f'No launch receipt for {job["run_id"]}')
        receipt = wait_closed(job, minutes=3)
        self.settle(job['run_id'])
        return job, receipt

    def replay_probability(self, job, plan, result):
        from tools.r6_direct.probability_evidence import audited_helper

        diagnostic_id = job['run_id'] + '-DIAGNOSTIC'
        path = self.input_path(diagnostic_id)
        spec = load(DIAG_TEMPLATE)
        spec.pop('native_volume', None)
        if helper := audited_helper(job['run_id']):
            spec.update(helper)
        logical_name = f'pilot_diagnostic/update_{plan["start_update"]+1}_full_input.pt'
        retained, = (r['pin'] for r in result['streams'] if r['logical_name'] == logical_name and r['complete'])
        spec.update(destination(diagnostic_id), resource_policy=job['resource_policy'],
                    scope='Exact retained first-rollout probability replay; no new simulator transitions',
                    launcher_script=pin(self.snapshot/'tools/r6_direct/cuda_diagnostic.py'),
                    predecessor_receipt=self.settle(job['run_id']),
                    probability_replay=dict(backend='isaaclab_seeded_reset', contract=job['contract'], retained=retained))
        self.write(path.name, spec)
        if not (Path(spec['output_root'])/'receipt.json').exists():
            freeze(self.tag_prefix + '-probability-u'+str(plan['target_update']),
                   'Freeze exact same-device probability replay', path, self.ledger)
            command(self.snapshot, 'cuda_diagnostic', '--plan', path, system=True, check=False)
        receipt = wait_closed(spec, minutes=3)
        if receipt['exit_code'] != 0 or receipt['error'] is not None:
            raise RuntimeError('Exact CUDA probability replay failed')
        predecessor = self.settle(diagnostic_id)
        attempt, control = Path(spec['attempt_dir']), Path(spec['output_root'])
        result_pin = pin(Path(spec['native_dir'])/'probability_result.json')
        if result_pin['sha256'] != pin(attempt/'probability_result.json')['sha256']:
            raise ValueError('Probability result does not match its mirror')
        from tools.r6_direct.instinct_cycle import save
        evidence = attempt/'sealed-evidence.json'
        save(evidence, dict(schema_version='r6-probability-sealed-evidence/1', plan=pin(path),
             run_inputs=pin(control/'RUN_INPUTS.json'), intent=pin(control/'intent.json'), receipt=predecessor,
             result=result_pin, source_trace=spec['source_trace'], origin=plan['initialization']['checkpoint']))
        return evidence, predecessor

    def execute_segment(self, path, plan):
        job, receipt = self.launch(path)
        attempt = Path(job['attempt_dir'])
        destination = attempt/'analysis-cpu'
        result = analyze(self.snapshot, path, destination)
        predecessor = self.settle(job['run_id'])
        if result['issues']:
            if [row['input'] for row in result['issues']] != ['first_update_probability_replay']:
                raise RuntimeError(f'Non-probability training audit issue: {destination}')
            evidence, predecessor = self.replay_probability(job, plan, result)
            cpu = destination
            destination = attempt/'analysis'
            analyze(self.snapshot, path, destination, '--probability-diagnostic', evidence,
                    '--reuse-cpu-binding', cpu/'binding.json')
        analysis = pin(destination/'binding.json')
        _, ctx, verified = validate_binding(analysis)
        if verified['issues'] or not verified['summary']['probability_gate_recomputed']:
            raise ValueError('Training endpoint is not independently verified')
        checkpoint, = (row['pin'] for row in verified['summary']['checkpoints'] if row['update'] == plan['target_update'])
        row = dict(plan=plan, checkpoint=checkpoint, analysis=analysis, counters=ctx['state']['counters'],
                   receipt=pin(Path(job['output_root'])/'receipt.json'), wall_seconds=receipt['wall_seconds'],
                   next_predecessor=predecessor)
        segment = self.docs/f'U{plan["target_update"]}.json'
        if not segment.exists():
            self.write(segment.name, row)
            freeze(f'{self.tag_prefix}-u{plan["target_update"]}-closed', 'Record audited PPO segment', segment, self.ledger)
        elif load(segment) != row:
            raise ValueError('Previously closed training segment differs')
        return row

    def evaluation_inputs(self, design, final, predecessor, *, physical):
        run_id = HOLDOUT_RUN if physical else PREFIX + 'R6'
        path = self.input_path(run_id)
        if path.exists():
            return path
        if physical:
            params = dict(experiment_id=run_id, training_seed=20260908, seed=design['holdout_seed'], wall_seconds=1800,
                          output_cap_bytes=GIB, models=dict(candidate=final['checkpoint'],
                          control=design['initial']['checkpoint'], reference=design['qualified']['checkpoint']))
        else:
            params = load(R3/'round-3/EVAL-contract.json')['parameters']
            selection = params['evaluation_selection']
            recipe = read(design['recipe'])
            comparison = read(selection['comparison'])
            comparison.update(id='INSTINCT-POST-R3-LONG-R6', candidate_id=ESCAPE_OPTION_CANDIDATE,
                              hypothesis=recipe['hypothesis'], method=recipe['method'])
            comparison_pin = self.write('R6-comparison.json', comparison)
            selection.update(checkpoint=final['checkpoint'], training_plan=final['plan'],
                             training_analysis=final['analysis'], method=recipe['method'], comparison=comparison_pin)
            params.update(experiment_id=run_id, training_seed=design['training_seed'])
            params['r6']['run_id'] = run_id
        operation = 'dynamic' if physical else 'evaluate'
        contract_pin = self.write(f'{run_id}-contract.json', build('contract', dict(operation=operation, parameters=params)))
        self.input(run_id, contract_pin, predecessor, snapshot=self.snapshot)
        extra = [] if physical else [Path(comparison_pin['path'])]
        freeze(f'instinct-post-r3-{operation}-inputs', 'Freeze independent candidate evaluation',
               path, Path(contract_pin['path']), self.ledger, *extra)
        return path

    def evaluate(self, path):
        job, receipt = self.launch(path)
        out = Path(job['attempt_dir'])/'analysis'
        report = analyze(self.snapshot, path, out)
        binding = pin(out/'binding.json')
        validate_binding(binding)
        if report.get('candidate_binding_issues'):
            raise RuntimeError('Independent R6 evaluation failed provenance checks')
        return dict(binding=binding, predecessor=self.settle(job['run_id']), wall_seconds=receipt['wall_seconds'])

    def execute(self):
        design = load(self.docs/'DESIGN.json')
        first_names = self.input_path(PREFIX + f'TRAIN-U{design["segments"][0][1]}')
        first = load(first_names)
        self.snapshot = Path(first['snapshot'])
        source = read(design['source'])
        git = ['git', '-C', str(self.snapshot)]
        if (first['source'] != design['source'] or self.snapshot == ROOT
                or subprocess.check_output(git + ['branch', '--show-current'], text=True).strip()
                or subprocess.check_output(git + ['rev-parse', 'HEAD', 'HEAD^{tree}'], text=True).splitlines()
                != [source['source_commit'], source['source_tree']]):
            raise ValueError('Long execution source differs from the frozen detached source')
        for name, digest in source['sources'].items():
            if hashlib.sha256((self.snapshot/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Frozen execution bytes drifted: {name}')
        initial, predecessor, segments = design['initial'], design['first_predecessor'], []
        recipe = read(design['recipe'])
        for coordinate in design['segments']:
            inputs, plan = self.freeze_segment(design, coordinate, initial, predecessor)
            row = self.execute_segment(inputs, plan)
            segments.append(row)
            predecessor = row['next_predecessor']
            initial = dict(kind='RESUME_R6_STREAM', checkpoint=row['checkpoint'],
                           source_identity=recipe['source_identity'], method_hashes=bound_method_hashes(plan, recipe))
        final = segments[-1]
        if sum(row['counters']['training_transitions'] for row in segments) != 196608:
            raise ValueError('Incomplete selected candidate training volume')
        r6 = self.evaluate(self.evaluation_inputs(design, final, predecessor, physical=False))
        held = self.evaluate(self.evaluation_inputs(design, final, r6['predecessor'], physical=True))
        _, _, comparison = validate_binding(r6['binding'])
        model = score_model(dict(run_id=PREFIX+'R6', arm='candidate', pair=1, report=pin(
            Path(r6['binding']['path']).parent/'report.json')))
        references = {k: design[k] for k in ('qualified', 'r3')}
        for entry in (model, *references.values()):
            if entry['seed'] != comparison['seed']:
                raise ValueError('Archived comparison seed drift')
            entry['stop_displacement'] = {k: displacement(entry['report'], k) for k in entry['endpoints']}
        settings = load(HISTORICAL/'FREEZE.json')
        results = self.write('RESULTS.json', dict(segments=segments, terminal=final['checkpoint'],
                             r6=r6, holdout=held, candidate=model, references=references,
                             holdout_comparison=compare_holdout(held['binding']),
                             r6_comparisons={k:compare(settings, [model], [ref]) for k,ref in references.items()},
                             ledger=pin(self.ledger)))
        freeze('instinct-post-r3-long-results', 'Record full GPU continuation and independent evaluation',
               Path(results['path']), self.ledger)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    args = parser.parse_args()
    guard()
    study = LongStudy()
    print(json.dumps(study.prepare() if args.action == 'prepare' else study.execute()), flush=True)


if __name__ == '__main__':
    main()
