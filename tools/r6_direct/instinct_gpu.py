"""Three exact-source Isaac GPU rounds with full R6 regression."""

import argparse
import io
import json
import math

from tools.r6_direct.evidence import validate_binding
from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import load
from tools.r6_direct.instinct_cycle_finish import command, freeze, guard
from tools.r6_direct.instinct_next import PROJECT as PREVIOUS, Round
from tools.r6_direct.resource_policy import GPU_STUDY_PREFIX, GPU_STUDY_SCHEMA
from tools.r6_direct.source import ROOT as SNAPSHOT
from tools.r6_direct.storage import SYSTEM_ROOT
from training_workloads.unified_h1.lossless_evidence import ChunkReader
from training_workloads.unified_h1.robustness_training_method import HOVER_REFLEX_CANDIDATE


def complete_window_count(rows):
    pending = None
    completed = 0
    for row in rows:
        index = row.get('window_row_index')
        if index is None:
            if pending is not None:
                raise ValueError('Unclosed measurement window')
            continue
        count = row['window_row_count']
        if pending is None:
            pending = (row['kind'], count, 0)
        if pending != (row['kind'], count, index) or len(row['rows']) != 1 or not 2 <= count <= 16:
            raise ValueError('Invalid measurement window fragment')
        if index == count - 1:
            completed += 1
            pending = None
        else:
            pending = (row['kind'], count, index + 1)
    if pending is not None:
        raise ValueError('Incomplete measurement window fragments')
    return completed


class GpuRound(Round):
    project = ROOT / 'docs/experiments/instinct-gpu-three'
    resource_schema = GPU_STUDY_SCHEMA

    def __init__(self, number, *, attempt=1):
        super().__init__(number)
        if attempt != 1 and (number, attempt) != (2, 2):
            raise ValueError('Only the closed round 2 logging failure has a registered retry')
        self.attempt = attempt
        retry = '' if attempt == 1 else f'-retry-{attempt:02}'
        self.run_prefix = f'{GPU_STUDY_PREFIX}R{number}{retry.upper()}-'
        if retry:
            self.docs = self.project / f'round-{number}{retry}'
            self.ledger = self.docs / 'RESOURCE_LEDGER.json'
        self.train = self.run_prefix + 'TRAIN'
        self.eval = self.run_prefix + 'EVAL'
        self.diag = self.run_prefix + 'PROBABILITY-DIAGNOSTIC'
        self.candidate = f'R6-CANDIDATE-INSTINCT-GPU-R{number}'
        if number == 3:
            self.candidate = HOVER_REFLEX_CANDIDATE
        self.tag = f'instinct-gpu-r{number}{retry}'

    def design(self, plan_doc):
        if self.number == 2:
            earlier = GpuRound(1)
            result_path = earlier.docs / 'RESULTS.json'
            previous_result = load(result_path)
            if previous_result['comparisons']['previous_gpu_r2']['guardrails_pass']:
                raise ValueError('R2 design must match the archived R1 outcome')
            pilot, = sorted(self.project.glob('pilot-*/RESULTS.json'))
            support = load(pilot)['summary']
            if not support['witnesses_replayed'] or support['counters']['nontraining_transitions'] != 7680:
                raise ValueError('Missing full physical moving-sphere preflight')
            history = SYSTEM_ROOT / earlier.train / 'matched-history-probe.json'
            pilot_id = f'{self.run_prefix}DYNAMIC-PILOT-{pilot.parent.name.removeprefix("pilot-")}'
            diagnosis = [plan_doc, pin(result_path), pin(history), pin(pilot),
                         pin(self.project / 'R1_LESSON.md'), pin(self.project / 'R2_DESIGN.md'),
                         pin(self.project / 'DYNAMIC_TRAINING_CHECK.md')]
            if self.attempt != 1:
                failed = GpuRound(2)
                diagnosis.extend([pin(failed.docs / 'FAILURE.json'), pin(self.project / 'R2_LOGGING_REPAIR.md')])
                pilot_id = failed.train
            return load(earlier.docs / 'curriculum.json'), diagnosis, pilot_id, (
                'Repeated physical mover encounters and bounded directional reward improve survival while preserving '
                'static command tracking and zero-command holding away from an approaching object.')
        if self.number == 3:
            earlier = completed_round(2)
            static_path = earlier.docs / 'RESULTS.json'
            static = load(static_path)
            counters = static['training_counters']
            if (counters['real_optimizer_updates'], counters['training_transitions'],
                    counters['real_checkpoint_writes']) != (128, 65536, 5):
                raise ValueError('Round 3 requires a complete physical round 2')
            checkpoint = static['candidate']['checkpoint']
            _, eval_context, eval_report = validate_binding(static['evaluation_analysis'])
            if (eval_context['state']['counters']['nontraining_transitions'] != 193600
                    or eval_report['candidate_binding_issues'] or eval_report['candidate']['checkpoint'] != checkpoint):
                raise ValueError('Round 3 requires complete round 2 R6 regression')
            diagnosis = [plan_doc, pin(static_path)]
            for kind, transitions in (('dynamic', 15360), ('visibility', 7680)):
                path = self.project / f'round-2-{kind}' / 'RESULTS.json'
                summary = load(path)['summary']
                if (not summary['witnesses_replayed'] or summary['counters']['nontraining_transitions'] != transitions
                        or summary['models']['candidate']['checkpoint'] != checkpoint):
                    raise ValueError(f'Round 3 requires the complete paired round 2 {kind} assay')
                diagnosis.append(pin(path))
            diagnosis.extend(pin(self.project / name) for name in (
                'R2_LESSON.md', 'R3_DESIGN.md', 'HOVER_REPRESENTATION_CHECK.md',
                'HOVER_ROTATION_CHECK.md', 'HOVER_REFLEX_COMPATIBILITY.md',
            ))
            diagnosis.extend(pin(SYSTEM_ROOT / name / 'result.json') for name in (
                'INSTINCT-GPU-REFLEX-PHYSICAL-OBSERVATIONS', 'INSTINCT-GPU-CLEAR-TRAVEL-LOSS-CHECK-02',
            ))
            geometry = SYSTEM_ROOT / 'INSTINCT-GPU-ESCAPE-DIRECTION-REPLAY' / 'geometry_022.json'
            diagnosis.extend((pin(geometry), pin(geometry.with_name('result.json'))))
            return load(earlier.docs / 'curriculum.json'), diagnosis, f'{GPU_STUDY_PREFIX}R2-VISIBILITY', (
                'A learned lateral escape from reliable LiDAR closure during hover reduces visible dynamic contacts '
                'while preserving ordinary command tracking and quiet holding.')
        curriculum = load(PREVIOUS / 'round-2' / 'curriculum.json')
        earlier = pin(PREVIOUS / 'REPORT.md')
        planar = pin(ROOT / 'docs/experiments/instinct-dynamic-five/REPORT.md')
        return curriculum, [plan_doc, earlier, planar], 'R6-DIRECT-INSTINCT-NEXT-R3-EVAL', (
            'Paired decoded-velocity supervision on matched current M4 samples '
            'separates approaching and opening histories while retaining command tracking.')

    def references(self):
        archived = {f'previous_gpu_r{n}': load(PREVIOUS / f'round-{n}' / 'RESULTS.json')['candidate'] for n in (2, 3)}
        earlier = {f'this_study_r{n}': load(completed_round(n).docs / 'RESULTS.json')['candidate']
                   for n in range(1, self.number)}
        return archived | earlier

    def training_rows(self, training_report, leaf):
        for stream in training_report['streams']:
            name = stream['logical_name']
            if not name.endswith('/' + leaf):
                continue
            bind = stream['pin']
            if pin(bind['path']) != bind:
                raise ValueError(f'Training evidence changed: {name}')
            with ChunkReader(bind['path'], run_id=self.train, logical_name=name,
                             maximum_raw_bytes=stream['raw_bytes']) as raw:
                with io.BufferedReader(raw) as buffer:
                    yield from map(json.loads, buffer)

    def extra_summary(self, training_report):
        minibatches = list(self.training_rows(training_report, 'minibatches.jsonl'))
        rows = [row['actor_auxiliary_metrics'] for row in minibatches]
        if len(rows) != 1024:
            raise ValueError('Incomplete minibatch counts')
        if self.number == 1:
            return {'matched_training': dict(minibatches=len(rows),
                                             valid_pairs=sum(r['paired_examples'] for r in rows),
                                             max_pair_loss=max(r['paired_action_loss'] for r in rows))}

        curriculum = list(self.training_rows(training_report, 'curriculum.jsonl'))
        layouts = [row['payload'] for row in curriculum if row['event'] == 'MOVING_SPHERE_LAYOUT']
        scans = [row['payload'] for row in curriculum if row['event'] == 'MOVING_SPHERE_ACQUISITION']
        if len(layouts) != 1 or not scans or not any(any(row['winning_beams']) for row in scans):
            raise ValueError('Missing physical mover training exposure')
        summary = {'dynamic_training': dict(
            minibatches=len(rows), hover_prior_removed=sum(r.get('hover_prior_removed', 0) for r in rows),
            hover_prior_retained=sum(r.get('hover_prior_retained', 0) for r in rows),
            cohort=layouts[0], sampled_acquisitions=len(scans),
            mover_visible_scans=sum(sum(x > 0 for x in row['winning_beams']) for row in scans),
            closest_sampled_sensor_gap_m=min(d for row in scans for d, active in
                                            zip(row['sphere_sensor_clearance_m'], row['active']) if active),
            complete_split_measurement_windows=complete_window_count(self.training_rows(training_report, 'measurement_windows.jsonl')),
        )}
        if self.candidate == HOVER_REFLEX_CANDIDATE:
            trace = {(row['update'], row['minibatch_ordinal']): row['hover_reflex_gain_parameter']
                     for row in minibatches if 'hover_reflex_gain_parameter' in row}
            expected = {(update, i) for update in range(641, 769) for i in range(8)}
            if trace.keys() != expected or not all(math.isfinite(x) and 0. <= x <= .0625 for x in trace.values()):
                raise ValueError('Incomplete or out-of-bounds hover gain record')
            summary['hover_reflex_training'] = dict(
                first_minibatch_gain=trace[641, 0], phase_0_final_gain=trace[704, 7],
                final_gain=trace[768, 7], peak_gain=max(trace.values()),
                eligible_minibatch_examples=sum(row.get('hover_reflex_targets', 0) for row in rows),
            )
            summary['clear_travel_training'] = dict(
                eligible_minibatch_examples=sum(row.get('clear_travel_examples', 0) for row in rows),
                peak_loss=max(row.get('clear_travel_loss', 0.) for row in rows),
            )
        return summary


def completed_round(number):
    planned = GpuRound(number)
    if number == 2 and (planned.docs / 'FAILURE.json').exists():
        return GpuRound(number, attempt=2)
    return planned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round', type=int, choices=(1, 2, 3))
    parser.add_argument('step', choices=('prepare', 'run', 'finish'))
    parser.add_argument('--attempt', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    guard()
    job = GpuRound(args.round, attempt=args.attempt)
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
