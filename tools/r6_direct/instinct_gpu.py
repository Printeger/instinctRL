"""Three exact-source Isaac GPU rounds with full R6 regression."""

import argparse
import io
import json

from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import load
from tools.r6_direct.instinct_cycle_finish import command, freeze, guard
from tools.r6_direct.instinct_next import PROJECT as PREVIOUS, Round
from tools.r6_direct.resource_policy import GPU_STUDY_PREFIX, GPU_STUDY_SCHEMA
from tools.r6_direct.source import ROOT as SNAPSHOT
from training_workloads.unified_h1.lossless_evidence import ChunkReader


class GpuRound(Round):
    project = ROOT / 'docs/experiments/instinct-gpu-three'
    resource_schema = GPU_STUDY_SCHEMA

    def __init__(self, number):
        super().__init__(number)
        self.run_prefix = f'{GPU_STUDY_PREFIX}R{number}-'
        self.train = self.run_prefix + 'TRAIN'
        self.eval = self.run_prefix + 'EVAL'
        self.diag = self.run_prefix + 'PROBABILITY-DIAGNOSTIC'
        self.candidate = f'R6-CANDIDATE-INSTINCT-GPU-R{number}'
        self.tag = f'instinct-gpu-r{number}'

    def design(self, plan_doc):
        curriculum = load(PREVIOUS / 'round-2' / 'curriculum.json')
        earlier = pin(PREVIOUS / 'REPORT.md')
        planar = pin(ROOT / 'docs/experiments/instinct-dynamic-five/REPORT.md')
        if self.number != 1:
            raise ValueError('Freeze the follow-up design after its predecessor is evaluated')
        return curriculum, [plan_doc, earlier, planar], 'R6-DIRECT-INSTINCT-NEXT-R3-EVAL', (
            'Paired decoded-velocity supervision on matched current M4 samples '
            'separates approaching and opening histories while retaining command tracking.')

    def references(self):
        return {key: load(PREVIOUS / f'round-{n}' / 'RESULTS.json')['candidate']
                for key, n in (('previous_gpu_r2', 2), ('previous_gpu_r3', 3))}

    def extra_summary(self, training_report):
        rows = []
        for stream in training_report['streams']:
            name = stream['logical_name']
            if not name.endswith('/minibatches.jsonl'):
                continue
            bind = stream['pin']
            if pin(bind['path']) != bind:
                raise ValueError('Minibatch evidence changed')
            with ChunkReader(bind['path'], run_id=self.train, logical_name=name,
                             maximum_raw_bytes=stream['raw_bytes']) as raw:
                with io.BufferedReader(raw) as buffer:
                    rows.extend(json.loads(row)['actor_auxiliary_metrics'] for row in buffer)
        if len(rows) != 1024:
            raise ValueError('Incomplete matched-pair training counts')
        return {'matched_training': dict(minibatches=len(rows),
                                         valid_pairs=sum(r['paired_examples'] for r in rows),
                                         max_pair_loss=max(r['paired_action_loss'] for r in rows))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round', type=int, choices=(1, 2, 3))
    parser.add_argument('step', choices=('prepare', 'run', 'finish'))
    args = parser.parse_args()
    guard()
    job = GpuRound(args.round)
    if args.step == 'prepare':
        result = job.prepare()
        freeze(job.tag + '-train-inputs', f'Freeze GPU development round {job.number}',
               *sorted(job.docs.iterdir()))
    else:
        if args.step == 'run':
            command(SNAPSHOT, 'run', '--inputs', job.input_path(job.train), system=True, check=False)
        result = job.finish()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
