"""Freeze and launch the separate Isaac moving-obstacle assay."""

import argparse
from pathlib import Path

from tools.r6_direct.import_legacy import pin
from tools.r6_direct.instinct_cycle import load
from tools.r6_direct.instinct_cycle_finish import analyze, command, freeze, guard, wait_closed
from tools.r6_direct.instinct_gpu import GpuRound, PREVIOUS, SNAPSHOT
from tools.r6_direct.prepare import build
from tools.r6_direct.resource_policy import OUTPUT_CAP, THIRD_CAPS
from tools.r6_direct.source import manifest
from tools.r6_direct.storage import SYSTEM_ROOT, SYSTEM_STORAGE_SCHEMA


class Assay(GpuRound):
    def __init__(self, number, pilot):
        super().__init__(number)
        self.pilot = pilot
        self.docs = self.project / (f'pilot-{pilot:02}' if pilot else f'round-{number}-dynamic')
        self.eval = self.run_prefix + 'DYNAMIC'
        if pilot:
            self.eval += f'-PILOT-{pilot:02}'
        self.ledger = self.docs / 'RESOURCE_LEDGER.json'
        self.tag = f'instinct-gpu-dynamic-pilot-{pilot:02}' if pilot else f'instinct-gpu-r{number}-dynamic'

    def prepare(self):
        if SNAPSHOT == self.project.parents[2]:
            raise ValueError('Use the detached assay snapshot')
        self.docs.mkdir(exist_ok=False)
        reference = load(PREVIOUS/'round-2'/'RESULTS.json')['candidate']['checkpoint']
        models = {'reference': reference}
        previous = GpuRound(1 if self.pilot else self.number)
        if not self.pilot:
            models['candidate'] = load(previous.docs/'RESULTS.json')['candidate']['checkpoint']
        seed = 942810037 if self.pilot else 705934187
        seconds = 450 if self.pilot else 720
        source = self.write('source.json', manifest())
        lesson = pin(self.project/'DYNAMIC_PHYSX_GPU.md')
        params = dict(experiment_id=self.eval, training_seed=20260908, models=models,
                      seed=seed, wall_seconds=seconds, output_cap_bytes=2**30)
        contract = build('contract', dict(operation='dynamic', parameters=params))
        budget = contract['budgets']
        caps = dict(allocated_gpu_seconds=seconds+190,live_wall_seconds=seconds+190,queue_seconds=180,
                    live_starts=1,simulator_initializations=1,training_transitions=1,
                    nontraining_transitions=budget['nontraining_transitions'],training_checkpoint_writes=1,
                    major_scientific_candidates=1, **{OUTPUT_CAP:3*2**30})
        policy = self.write('RESOURCE_POLICY.json', dict(schema_version=self.resource_schema,
                            user_authority=lesson,caps=caps,elapsed_seconds=seconds+300))
        storage = self.write('STORAGE.json',dict(schema_version=SYSTEM_STORAGE_SCHEMA,
                             root=str(SYSTEM_ROOT),new_volume_allocation_bytes=0))
        empty = dict.fromkeys(THIRD_CAPS,0)
        self.write(self.ledger.name,dict(resource_policy=policy,caps=caps,historical_charged=empty,
                   new_used=empty,protected_confirmation_delivery_repair=empty,status='FROZEN',
                   new_reservations=[dict(id='ROUND-SYSTEM',status='UNRELEASED',
                                     resources={OUTPUT_CAP:0},storage_plan=storage)]))
        contract_pin = self.write('DYNAMIC-contract.json', contract)
        assert source == pin(self.docs/'source.json')
        prior_run = previous.eval
        if self.pilot and self.pilot > 1:
            prior_run = Assay(self.number, self.pilot - 1).eval
        predecessor = pin(SYSTEM_ROOT/prior_run/'control/receipt.json')
        return self.input(self.eval, contract_pin, predecessor)

    def finish_assay(self, *, launch):
        path = self.input_path(self.eval)
        if launch:
            command(SNAPSHOT,'run','--inputs',path,system=True,check=False)
        inputs = load(path)
        wait_closed(inputs,minutes=3)
        dest = Path(inputs['attempt_dir'])/'analysis'
        result = analyze(SNAPSHOT,path,dest)
        self.settle(self.eval)
        summary = result['summary']
        evidence = self.write('RESULTS.json',dict(analysis=pin(dest/'binding.json'),summary=summary,
                                                   ledger=pin(self.ledger)))
        freeze(self.tag+'-results',f'Close dynamic assay {self.eval}',Path(evidence['path']),self.ledger)
        return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round',type=int,choices=(2,3))
    parser.add_argument('step',choices=('prepare','run','finish'))
    parser.add_argument('--pilot',type=int,choices=range(1,5))
    args = parser.parse_args()
    if args.pilot and args.round != 2:
        parser.error('pilot precedes the second full round')
    guard()
    assay = Assay(args.round,args.pilot)
    if args.step == 'prepare':
        result = assay.prepare()
        freeze(assay.tag+'-inputs',f'Freeze dynamic assay {assay.eval}',*sorted(assay.docs.iterdir()))
    else:
        result = assay.finish_assay(launch=args.step=='run')
    print(result,flush=True)


if __name__ == '__main__':
    main()
