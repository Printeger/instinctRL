"""Run and audit either of the two frozen instinct follow-up rounds."""

import argparse
import io
import json
from pathlib import Path

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import (
    DIAG_TEMPLATE, EVAL_TEMPLATE, HISTORICAL, TRAIN_TEMPLATE, destination, load, save,
)
from tools.r6_direct.instinct_cycle_finish import analyze, command, freeze, guard, wait_closed
from tools.r6_direct.instinct_cycle_report import displacement
from tools.r6_direct.io import atomic
from tools.r6_direct.lifecycle import writer
from tools.r6_direct.prepare import build
from tools.r6_direct.resource_policy import NEXT_PREFIX, NEXT_SCHEMA, OUTPUT_CAP, THIRD_CAPS
from tools.r6_direct.settle import settle_time
from tools.r6_direct.source import ROOT as SNAPSHOT, manifest, training_identity
from tools.r6_direct.storage import SYSTEM_ROOT, SYSTEM_STORAGE_SCHEMA
from tools.r6_direct.study_report import compare, score_model
from training_workloads.unified_h1.lossless_evidence import ChunkReader


PROJECT = ROOT / 'docs/experiments/instinct-next-two-cycles'
FIRST = ROOT / 'docs/experiments/instinct-first-cycle/formal'


def auxiliary_support(training_report, run_id):
    summaries = []
    for stream in training_report['streams']:
        name = stream['logical_name']
        if not name.endswith('/minibatches.jsonl'):
            continue
        path = stream['pin']
        if pin(path['path']) != path:
            raise ValueError(f'Minibatch evidence drift {name}')
        with ChunkReader(path['path'], run_id=run_id, logical_name=name,
                         maximum_raw_bytes=stream['raw_bytes']) as raw:
            with io.BufferedReader(raw) as buffer:
                metrics = [json.loads(row)['actor_auxiliary_metrics'] for row in buffer]
        summaries.append(dict(stream=path, minibatches=len(metrics),
                              active=sum(row['range_closure_active'] for row in metrics),
                              above_ceiling=sum(row['range_closure_above_ceiling'] for row in metrics),
                              maximum_loss=max(row['range_closure_loss'] for row in metrics)))
    if sum(row['minibatches'] for row in summaries) != 1024:
        raise ValueError('Missing range closure training support')
    return summaries


class Round:
    project = PROJECT
    resource_schema = NEXT_SCHEMA

    def __init__(self, number):
        self.number = number
        self.docs = self.project / f'round-{number}'
        self.run_prefix = f'{NEXT_PREFIX}R{number}-'
        self.train = self.run_prefix + 'TRAIN'
        self.eval = self.run_prefix + 'EVAL'
        self.diag = self.run_prefix + 'PROBABILITY-DIAGNOSTIC'
        self.candidate = f'R6-CANDIDATE-INSTINCT-NEXT-R{number}'
        self.ledger = self.docs / 'RESOURCE_LEDGER.json'
        self.tag = f'instinct-continuation-r{number}'

    def write(self, name, value):
        return save(self.docs / name, value)

    def input_path(self, run_id):
        return self.docs / f'{run_id}-inputs.json'

    def input(self, run_id, contract, predecessor):
        template = load(FIRST / 'R6-DIRECT-INSTINCT-CYCLE1-TRAIN-01-R3-inputs.json')
        return self.write(self.input_path(run_id).name, dict(destination(run_id),
            schema_version='r6-direct-run-inputs/1', snapshot=str(SNAPSHOT),
            source=pin(self.docs / 'source.json'), contract=contract,
            extension_inputs=template['extension_inputs'], ipc_wrapper=template['ipc_wrapper'],
            resource_policy=pin(self.docs / 'RESOURCE_POLICY.json'), storage_reservation_id='ROUND-SYSTEM',
            result_payload_cap_bytes=read(contract)['parameters']['output_cap_bytes'],
            predecessor_receipt=predecessor))

    def prepare(self):
        if SNAPSHOT == ROOT:
            raise ValueError('Run from the exact detached execution snapshot')
        plan_doc = pin(self.project / 'PLAN.md')
        curriculum, diagnosis, prior_run, hypothesis = self.design(plan_doc)
        self.docs.mkdir(exist_ok=False)
        zero = dict.fromkeys(THIRD_CAPS, 0)
        caps = dict(allocated_gpu_seconds=10000, live_wall_seconds=10000, queue_seconds=540,
                    live_starts=3, simulator_initializations=2, training_transitions=65536,
                    nontraining_transitions=193600, training_checkpoint_writes=5,
                    major_scientific_candidates=1, **{OUTPUT_CAP: 32 * 1024**3})
        authority = self.write('RESOURCE_POLICY.json', dict(schema_version=self.resource_schema,
                               user_authority=plan_doc, caps=caps, elapsed_seconds=18000))
        storage = self.write('STORAGE.json', dict(schema_version=SYSTEM_STORAGE_SCHEMA,
                             root=str(SYSTEM_ROOT), new_volume_allocation_bytes=0))
        self.write(self.ledger.name, dict(resource_policy=authority, caps=caps, historical_charged=zero,
                   new_used=zero, protected_confirmation_delivery_repair=zero, status='FROZEN',
                   new_reservations=[dict(id='ROUND-SYSTEM', status='UNRELEASED',
                                          resources={OUTPUT_CAP: 0}, storage_plan=storage)]))
        source = manifest()
        self.write('source.json', source)
        identity = training_identity(source)
        params = load(TRAIN_TEMPLATE)['parameters']
        plan = params['r6_training_plan']
        recipe = read(plan['recipe'])
        recipe.update(candidate_id=self.candidate, source_identity=identity, diagnosis=diagnosis,
                      hypothesis=hypothesis, comparison='Full matched training and closed R6 paired regression.',
                      curriculum=self.write('curriculum.json', curriculum))
        plan.update(experiment_id=self.train, recipe=self.write('recipe.json', recipe))
        params.update(experiment_id=self.train, source_identity=identity)
        contract = self.write('TRAIN-contract.json', build('contract', dict(operation='train', parameters=params)))
        prior = pin(SYSTEM_ROOT / prior_run / 'control/receipt.json')
        return self.input(self.train, contract, prior)

    def design(self, plan_doc):
        curriculum = load(FIRST / 'curriculum-r3.json')
        diagnosis = [pin(FIRST / 'REPORT.md'), plan_doc]
        prior_run = 'R6-DIRECT-INSTINCT-CYCLE1-EVAL-01'
        if self.number == 2:
            delay = 1
            hypothesis = 'Causally delivered delay after real first scans adds robustness to isolated drops.'
        else:
            previous = self.project / 'round-2' / 'RESULTS.json'
            prior = load(previous)
            delay = int(prior['comparisons']['dropout_only']['guardrails_pass'])
            diagnosis.extend((pin(previous), pin(self.project / 'R3_DESIGN.md')))
            prior_run = f'{NEXT_PREFIX}R2-EVAL'
            hypothesis = 'Additional actor-only range closure supervision improves temporal selectivity.'
        curriculum['phases'][1]['delay_cycles'] = delay
        return curriculum, diagnosis, prior_run, hypothesis

    def settle(self, run_id):
        with writer():
            ledger = load(self.ledger)
            row, = (r for r in ledger['new_reservations'] if r['id'] == run_id)
            if 'settlement' not in row:
                settle_time(ledger, run_id)
                ledger['status'] = 'COMPLETE' if run_id == self.eval else 'CONTROLLED_HANDOFF'
                atomic(self.ledger, ledger)
            return row['receipt']

    def probability(self, train, report):
        spec = load(DIAG_TEMPLATE)
        del spec['native_volume']
        retained, = (x['pin'] for x in report['streams']
                     if x['logical_name'] == 'pilot_diagnostic/update_641_full_input.pt')
        spec.update(destination(self.diag), resource_policy=train['resource_policy'],
                    scope=f'Round {self.number} exact-input probability replay, zero training',
                    launcher_script=pin(SNAPSHOT / 'tools/r6_direct/cuda_diagnostic.py'),
                    predecessor_receipt=self.settle(self.train), probability_replay=dict(
                        backend='isaaclab_seeded_reset', contract=train['contract'], retained=retained))
        path = self.input_path(self.diag)
        self.write(path.name, spec)
        freeze(self.tag + '-probability-inputs', f'Freeze round {self.number} same-device probability replay', path, self.ledger)
        command(SNAPSHOT, 'cuda_diagnostic', '--plan', path, system=True, check=False)
        receipt = wait_closed(spec, minutes=3)
        if receipt['exit_code'] != 0 or receipt['error'] is not None:
            raise RuntimeError('Probability replay did not pass')
        self.settle(self.diag)
        attempt = Path(spec['attempt_dir'])
        control = Path(spec['output_root'])
        result = pin(Path(spec['native_dir']) / 'probability_result.json')
        if result['sha256'] != pin(attempt / 'probability_result.json')['sha256']:
            raise ValueError('Probability result mirror differs')
        plan = read(train['contract'])['parameters']['r6_training_plan']
        sealed = attempt / 'sealed-evidence.json'
        save(sealed, dict(schema_version='r6-probability-sealed-evidence/1', plan=pin(path),
             run_inputs=pin(control / 'RUN_INPUTS.json'), intent=pin(control / 'intent.json'),
             receipt=pin(control / 'receipt.json'), result=result, source_trace=spec['source_trace'],
             origin=plan['initialization']['checkpoint']))
        return sealed

    def evaluate_input(self, analysis, predecessor):
        _, ctx, report = validate_binding(analysis)
        terminal, = (r['pin'] for r in report['summary']['checkpoints'] if r['update'] == 768)
        plan = ctx['contract']['parameters']['r6_training_plan']
        params = load(EVAL_TEMPLATE)['parameters']
        selection = params['evaluation_selection']
        comparison = read(selection['comparison'])
        comparison.update(id=f'INSTINCT-ROUND-{self.number}-PAIRED', candidate_id=self.candidate,
                          hypothesis=read(plan['recipe'])['hypothesis'])
        selection.update(checkpoint=terminal, training_plan=plan, training_analysis=analysis,
                         comparison=self.write('EVAL-comparison.json', comparison))
        params['experiment_id'] = self.eval
        params['r6']['run_id'] = self.eval
        contract = self.write('EVAL-contract.json', build('contract', dict(operation='evaluate', parameters=params)))
        self.input(self.eval, contract, predecessor)

    def references(self):
        references = {m['arm']: m for m in load(HISTORICAL / 'RESULTS.json')['models']
                      if (m['arm'], m['pair']) in (('baseline', 0), ('control', 1))}
        references['dropout_only'], = (m for m in load(FIRST / 'RESULTS.json')['models'] if m['arm'] == 'candidate')
        if self.number == 3:
            references['round_2'] = load(self.project / 'round-2' / 'RESULTS.json')['candidate']
        return references

    def extra_summary(self, training_report):
        if self.number == 3:
            return {'range_closure_training_support': auxiliary_support(training_report, self.train)}
        return {}

    def summarize(self, analysis, evaluation):
        training_report = read(read(analysis)['report'])
        candidate = score_model(dict(run_id=self.eval, arm='candidate', pair=1,
                                     report=pin(evaluation / 'report.json')))
        references = self.references()
        for model in [candidate, *references.values()]:
            if model['seed'] != candidate['seed']:
                raise ValueError('Mismatched archived comparison seed')
            model['stop_displacement'] = {k: displacement(model['report'], k) for k in model['endpoints']}
        settings = load(HISTORICAL / 'FREEZE.json')
        compare_results = {key: compare(settings, [candidate], [model]) for key, model in references.items()}
        summary = dict(round=self.number, candidate=candidate, references=references,
                       comparisons=compare_results, training_analysis=analysis,
                       training_counters=training_report['summary']['observed_counters'],
                       evaluation_analysis=pin(evaluation / 'binding.json'), ledger=pin(self.ledger))
        summary.update(self.extra_summary(training_report))
        return self.write('RESULTS.json', summary)

    def finish(self):
        train_path = self.input_path(self.train)
        train = load(train_path)
        wait_closed(train, minutes=3)
        attempt = Path(train['attempt_dir'])
        cpu = attempt / 'analysis-cpu'
        result = analyze(SNAPSHOT, train_path, cpu)
        analysis = cpu / 'binding.json'
        predecessor_id = self.train
        if result['issues']:
            if [r['input'] for r in result['issues']] != ['first_update_probability_replay']:
                raise RuntimeError(f'Training evidence issues: {cpu / "report.json"}')
            sealed = self.probability(train, result)
            audited = attempt / 'analysis'
            result = analyze(SNAPSHOT, train_path, audited, '--probability-diagnostic', sealed,
                             '--reuse-cpu-binding', analysis)
            analysis = audited / 'binding.json'
            predecessor_id = self.diag
        if result['issues']:
            raise RuntimeError('Unresolved training evidence issues')
        self.evaluate_input(pin(analysis), self.settle(predecessor_id))
        path = self.input_path(self.eval)
        freeze(self.tag + '-eval-inputs', f'Freeze round {self.number} full evaluation', path,
               self.docs / 'EVAL-contract.json', self.docs / 'EVAL-comparison.json', self.ledger)
        command(SNAPSHOT, 'run', '--inputs', path, system=True, check=False)
        evaluation = load(path)
        wait_closed(evaluation, minutes=3)
        out = Path(evaluation['attempt_dir']) / 'analysis'
        result = analyze(SNAPSHOT, path, out)
        if result['candidate_binding_issues']:
            raise RuntimeError('Incomplete candidate evaluation')
        self.settle(self.eval)
        summary = self.summarize(pin(analysis), out)
        freeze(self.tag + '-results', f'Record audited round {self.number} training and evaluation results',
               Path(summary['path']), self.ledger)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round', type=int, choices=(2, 3))
    parser.add_argument('step', choices=('prepare', 'run', 'finish'))
    args = parser.parse_args()
    guard()
    job = Round(args.round)
    if args.step == 'prepare':
        result = job.prepare()
        freeze(job.tag + '-train-inputs', f'Freeze round {job.number} full training inputs', *sorted(job.docs.iterdir()))
    elif args.step == 'run':
        command(SNAPSHOT, 'run', '--inputs', job.input_path(job.train), system=True, check=False)
        result = job.finish()
    else:
        result = job.finish()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
