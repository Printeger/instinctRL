"""Freeze and audit the R3 same-weight persistence comparison."""

import argparse
import json
from pathlib import Path

from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import load
from tools.r6_direct.instinct_cycle_finish import analyze, command, freeze, guard, wait_closed
from tools.r6_direct.instinct_next import Round
from tools.r6_direct.prepare import build
from tools.r6_direct.resource_policy import GPU_FOLLOWUP_SCHEMA, OUTPUT_CAP, THIRD_CAPS
from tools.r6_direct.source import ROOT as SNAPSHOT, manifest
from tools.r6_direct.storage import CONTROL_ALLOWANCE, SYSTEM_ROOT, SYSTEM_STORAGE_SCHEMA, system_native, system_result
from training_workloads.unified_h1.dynamic_assay import HOLDOUT_SMOKE, OPTION_RUN, PERSISTENCE_RUN


PROJECT = ROOT / 'docs/experiments/instinct-post-r3'
R3 = ROOT / 'docs/experiments/instinct-gpu-three'


class PersistenceStudy(Round):
    def __init__(self, *, option=False, holdout_smoke=False):
        directory, self.eval, authority, self.seed = (
            ('holdout-smoke', HOLDOUT_SMOKE, 'HOLDOUT.md', 77162057) if holdout_smoke else
            ('option-check', OPTION_RUN, 'OPTION.md', 705934187) if option else
            ('development-02', PERSISTENCE_RUN, 'DEVELOPMENT.md', 183421779)
        )
        self.docs = PROJECT / directory
        self.ledger = self.docs / 'RESOURCE_LEDGER.json'
        self.tag = 'instinct-post-r3-' + (directory if option or holdout_smoke else 'persistence-02')
        self.authority_file = PROJECT / authority
        self.option = option
        self.holdout_smoke = holdout_smoke

    def prepare(self):
        if SNAPSHOT == ROOT:
            raise ValueError('Prepare from the detached execution snapshot')
        self.docs.mkdir(exist_ok=False)
        source = self.write('source.json', manifest())
        checkpoint = load(R3 / 'round-3/RESULTS.json')['candidate']['checkpoint']
        models = dict(candidate=checkpoint) if self.holdout_smoke else dict(reference=checkpoint, candidate=checkpoint)
        params = dict(experiment_id=self.eval, training_seed=20260908, seed=self.seed, models=models,
                      wall_seconds=900, output_cap_bytes=2**30)
        contract = build('contract', dict(operation='dynamic', parameters=params))
        contract_pin = self.write('DYNAMIC-contract.json', contract)
        caps = dict(allocated_gpu_seconds=1090, live_wall_seconds=1090, queue_seconds=180,
                    live_starts=1, simulator_initializations=1, training_transitions=1,
                    nontraining_transitions=contract['budgets']['nontraining_transitions'],
                    training_checkpoint_writes=1, major_scientific_candidates=1, **{OUTPUT_CAP: 2**30 + CONTROL_ALLOWANCE})
        authority = self.write('RESOURCE_POLICY.json', dict(schema_version=GPU_FOLLOWUP_SCHEMA,
                               user_authority=pin(self.authority_file), caps=caps, elapsed_seconds=1500))
        storage = self.write('STORAGE.json', dict(schema_version=SYSTEM_STORAGE_SCHEMA,
                             root=str(SYSTEM_ROOT), new_volume_allocation_bytes=0))
        zero = dict.fromkeys(THIRD_CAPS, 0)
        self.write(self.ledger.name, dict(resource_policy=authority, caps=caps, historical_charged=zero,
                   new_used=zero, protected_confirmation_delivery_repair=zero, status='FROZEN',
                   new_reservations=[dict(id='ROUND-SYSTEM', status='UNRELEASED',
                                          resources={OUTPUT_CAP: 0}, storage_plan=storage)]))
        prior_id = (OPTION_RUN if self.holdout_smoke else PERSISTENCE_RUN if self.option
                    else 'R6-DIRECT-INSTINCT-GPU-R3-VISIBILITY')
        prior = pin(SYSTEM_ROOT / prior_id / 'control/receipt.json')
        inputs = self.input(self.eval, contract_pin, prior)
        plan = load(Path(inputs['path']))
        system_native(plan)
        system_result(plan, contract, before_launch=True)
        return inputs

    def finish(self):
        path = self.input_path(self.eval)
        inputs = load(path)
        receipt = wait_closed(inputs, minutes=3)
        dest = Path(inputs['attempt_dir']) / 'analysis'
        self.settle(self.eval)
        report = analyze(SNAPSHOT, path, dest)
        result = self.write('RESULTS.json', dict(analysis=pin(dest / 'binding.json'),
                           receipt=pin(Path(inputs['output_root']) / 'receipt.json'),
                           receipt_wall_seconds=receipt['wall_seconds'], summary=report['summary'],
                           ledger=pin(self.ledger)))
        freeze(self.tag + '-results', 'Record same-weight persistence comparison', Path(result['path']), self.ledger)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('prepare', 'run', 'finish'))
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument('--option', action='store_true')
    variant.add_argument('--holdout-smoke', action='store_true')
    args = parser.parse_args()
    guard()
    study = PersistenceStudy(option=args.option, holdout_smoke=args.holdout_smoke)
    if args.step == 'prepare':
        result = study.prepare()
        freeze(study.tag + '-inputs', 'Freeze same-weight persistence comparison', *sorted(study.docs.iterdir()))
    else:
        if args.step == 'run':
            command(SNAPSHOT, 'run', '--inputs', study.input_path(study.eval), system=True, check=False)
            if not (SYSTEM_ROOT / study.eval).exists():
                raise RuntimeError('Launch stopped before allocating the run directory')
        result = study.finish()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
