"""Run the R4 recovery comparisons through the pinned R6 direct entry."""

import argparse
import json
from statistics import fmean

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle_finish import freeze, guard, wait_closed
from tools.r6_direct.instinct_long import GIB, LongStudy
from tools.r6_direct.resource_policy import GPU_RECOVERY_SCHEMA, OUTPUT_CAP, THIRD_CAPS
from tools.r6_direct.source import ROOT as SNAPSHOT, manifest
from tools.r6_direct.storage import SYSTEM_ROOT, SYSTEM_STORAGE_SCHEMA
from training_workloads.unified_h1.dynamic_assay import (
    RECOVERY_HOLDOUT,
    RECOVERY_MECHANISM,
    RECOVERY_MECHANISM_RETRY,
    RECOVERY_REAR,
)


PROJECT = ROOT / 'docs/experiments/instinct-r4-recovery'
R4 = ROOT / 'docs/experiments/instinct-post-r3/long'
DEVELOPMENT_SEED = 184976539


class RecoveryStudy(LongStudy):
    def __init__(self):
        self.docs = PROJECT
        self.ledger = PROJECT / 'RESOURCE_LEDGER.json'
        self.eval = RECOVERY_HOLDOUT
        self.snapshot = SNAPSHOT
        self.source = None

    def dynamic_input(self, run_id, seed, models, predecessor, *, wall_seconds):
        from tools.r6_direct.prepare import build

        path = self.input_path(run_id)
        params = dict(experiment_id=run_id, training_seed=20260908, seed=seed,
                      models=models, wall_seconds=wall_seconds, output_cap_bytes=GIB)
        contract = self.write(f'{run_id}-contract.json',
                              build('contract', dict(operation='dynamic', parameters=params)))
        self.input(run_id, contract, predecessor, snapshot=self.snapshot, source=self.source)
        return path

    def prepare_mechanism(self):
        if self.snapshot == ROOT:
            raise ValueError('Use the detached execution snapshot')
        self.write('source.json', manifest())
        caps = dict(allocated_gpu_seconds=102000, live_wall_seconds=102000, queue_seconds=5400,
                    live_starts=30, simulator_initializations=20,
                    training_transitions=(1024+128)*512, nontraining_transitions=850000,
                    training_checkpoint_writes=45, major_scientific_candidates=1,
                    **{OUTPUT_CAP: 114*GIB})
        policy = self.write('RESOURCE_POLICY.json', dict(schema_version=GPU_RECOVERY_SCHEMA,
                            user_authority=pin(PROJECT/'DEVELOPMENT.md'), caps=caps, elapsed_seconds=172800))
        storage = self.write('STORAGE.json', dict(schema_version=SYSTEM_STORAGE_SCHEMA,
                             root=str(SYSTEM_ROOT), new_volume_allocation_bytes=0))
        zero = dict.fromkeys(THIRD_CAPS, 0)
        self.write(self.ledger.name, dict(resource_policy=policy, caps=caps, historical_charged=zero,
                   new_used=zero, protected_confirmation_delivery_repair=zero, status='FROZEN',
                   new_reservations=[dict(id='ROUND-SYSTEM', status='UNRELEASED',
                                          resources={OUTPUT_CAP: 0}, storage_plan=storage)]))
        old = read(pin(R4/'DESIGN.json'))
        start = old['initial']['checkpoint']
        previous = pin(SYSTEM_ROOT/'R6-DIRECT-INSTINCT-GPU-R4-LONG-HOLDOUT/control/receipt.json')
        path = self.dynamic_input(RECOVERY_MECHANISM, DEVELOPMENT_SEED,
                                  dict(reference=start, control=start, candidate=start), previous,
                                  wall_seconds=1500)
        freeze('instinct-r4-recovery-mechanism-inputs', 'Freeze fixed-weight recovery comparison',
               self.ledger, PROJECT/'source.json', PROJECT/'RESOURCE_POLICY.json', PROJECT/'STORAGE.json',
               PROJECT/f'{RECOVERY_MECHANISM}-contract.json', path)
        return pin(path)

    def prepare_retry(self):
        if self.snapshot == ROOT:
            raise ValueError('Use the detached execution snapshot')
        failed = read(pin(self.input_path(RECOVERY_MECHANISM)))
        wait_closed(failed, minutes=3)
        state = read(pin(SYSTEM_ROOT/RECOVERY_MECHANISM/'state.json'))
        if state['counters']['nontraining_transitions'] != 0 or 'free CUDA memory' not in (state['error'] or ''):
            raise ValueError('Retry expects the confirmed first-reset memory failure')
        self.source = self.write('source-retry.json', manifest())
        previous = self.settle(RECOVERY_MECHANISM)
        params = read(failed['contract'])['parameters']
        path = self.dynamic_input(RECOVERY_MECHANISM_RETRY, params['seed'], params['models'],
                                  previous, wall_seconds=params['wall_seconds'])
        freeze('instinct-r4-recovery-mechanism-retry-inputs', 'Freeze first-reset memory recovery assay',
               PROJECT/'source-retry.json', PROJECT/f'{RECOVERY_MECHANISM_RETRY}-contract.json', path, self.ledger)
        return pin(path)

    def execute_mechanism(self):
        path = self.input_path(RECOVERY_MECHANISM_RETRY)
        if not path.exists():
            path = self.input_path(RECOVERY_MECHANISM)
        self.source = read(pin(path))['source']
        first = self.evaluate(path)
        start = read(pin(R4/'DESIGN.json'))['initial']['checkpoint']
        rear = self.input_path(RECOVERY_REAR)
        if not rear.exists():
            self.dynamic_input(RECOVERY_REAR, DEVELOPMENT_SEED+1,
                               dict(reference=start, candidate=start), first['predecessor'], wall_seconds=750)
            freeze('instinct-r4-recovery-rear-inputs', 'Freeze matched sensor aspect comparison',
                   PROJECT/f'{RECOVERY_REAR}-contract.json', rear, self.ledger)
        second = self.evaluate(rear)
        _, _, report = validate_binding(first['binding'])
        summary = {}
        for arm, model in report['summary']['models'].items():
            summary[arm] = {}
            for group in model['groups']:
                rows = [r for r in group['lanes'] if r['layout'] == 'open_tube']
                summary[arm][rows[0]['mode']] = dict(
                    episodes=len(rows), collisions=sum(row['collision'] for row in rows),
                    progress_m=fmean(row['progress_m'] for row in rows),
                    recovery_progress_m=fmean(row['recovery_progress_m'] for row in rows),
                    recovered_without_contact=sum(not row['collision'] and row['recovery_progress_m'] >= .5
                                                  for row in rows),
                    tracking_rmse_mps=fmean(row['tracking_rmse_mps'] for row in rows),
                    maximum_horizontal_excursion_m=fmean(row['maximum_horizontal_excursion_m'] for row in rows))
        result = self.write('MECHANISM_RESULTS.json', dict(assay=first, rear=second, open_tube=summary))
        freeze('instinct-r4-recovery-mechanism-results', 'Record fixed-weight physical comparisons',
               PROJECT/'MECHANISM_RESULTS.json', self.ledger)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare-mechanism', 'prepare-retry', 'run-mechanism'))
    args = parser.parse_args()
    guard()
    study = RecoveryStudy()
    execute = {'prepare-mechanism': study.prepare_mechanism, 'prepare-retry': study.prepare_retry,
               'run-mechanism': study.execute_mechanism}
    result = execute[args.action]()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
