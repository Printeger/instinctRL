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
from tools.r6_direct.storage import SYSTEM_ROOT, SYSTEM_STORAGE_SCHEMA
from training_workloads.unified_h1.dynamic_assay import PERSISTENCE_RUN


PROJECT = ROOT / 'docs/experiments/instinct-post-r3'
R3 = ROOT / 'docs/experiments/instinct-gpu-three'


class PersistenceStudy(Round):
    def __init__(self):
        self.docs = PROJECT / 'development'
        self.ledger = self.docs / 'RESOURCE_LEDGER.json'
        self.eval = PERSISTENCE_RUN
        self.tag = 'instinct-post-r3-persistence'

    def prepare(self):
        if SNAPSHOT == ROOT:
            raise ValueError('Prepare from the detached execution snapshot')
        self.docs.mkdir(exist_ok=False)
        source = self.write('source.json', manifest())
        checkpoint = load(R3 / 'round-3/RESULTS.json')['candidate']['checkpoint']
        params = dict(experiment_id=self.eval, training_seed=20260908, seed=183421779,
                      models=dict(reference=checkpoint, candidate=checkpoint),
                      wall_seconds=900, output_cap_bytes=2**30)
        contract = build('contract', dict(operation='dynamic', parameters=params))
        contract_pin = self.write('DYNAMIC-contract.json', contract)
        caps = dict(allocated_gpu_seconds=1090, live_wall_seconds=1090, queue_seconds=180,
                    live_starts=1, simulator_initializations=1, training_transitions=1,
                    nontraining_transitions=contract['budgets']['nontraining_transitions'],
                    training_checkpoint_writes=1, major_scientific_candidates=1, **{OUTPUT_CAP: 2**30 + 64*2**20})
        authority = self.write('RESOURCE_POLICY.json', dict(schema_version=GPU_FOLLOWUP_SCHEMA,
                               user_authority=pin(PROJECT / 'DEVELOPMENT.md'), caps=caps, elapsed_seconds=1500))
        storage = self.write('STORAGE.json', dict(schema_version=SYSTEM_STORAGE_SCHEMA,
                             root=str(SYSTEM_ROOT), new_volume_allocation_bytes=0))
        zero = dict.fromkeys(THIRD_CAPS, 0)
        self.write(self.ledger.name, dict(resource_policy=authority, caps=caps, historical_charged=zero,
                   new_used=zero, protected_confirmation_delivery_repair=zero, status='FROZEN',
                   new_reservations=[dict(id='ROUND-SYSTEM', status='UNRELEASED',
                                          resources={OUTPUT_CAP: 0}, storage_plan=storage)]))
        prior = pin(SYSTEM_ROOT / 'R6-DIRECT-INSTINCT-GPU-R3-VISIBILITY/control/receipt.json')
        return self.input(self.eval, contract_pin, prior)

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
    args = parser.parse_args()
    guard()
    study = PersistenceStudy()
    if args.step == 'prepare':
        result = study.prepare()
        freeze(study.tag + '-inputs', 'Freeze same-weight persistence comparison', *sorted(study.docs.iterdir()))
    else:
        if args.step == 'run':
            command(SNAPSHOT, 'run', '--inputs', study.input_path(study.eval), system=True, check=False)
        result = study.finish()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
