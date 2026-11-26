"""Freeze and run the next instinctRL study through the local R6 supervisor."""

import argparse
import json
from pathlib import Path
import shutil

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import load
from tools.r6_direct.instinct_cycle_finish import freeze, guard
from tools.r6_direct.instinct_long import GIB, LongStudy
from tools.r6_direct.prepare import build
from tools.r6_direct.resource_policy import NEXT_LONG_SCHEMA, OUTPUT_CAP, THIRD_CAPS
from tools.r6_direct.source import ROOT as SNAPSHOT, manifest
from tools.r6_direct.storage import SYSTEM_FLOOR, SYSTEM_ROOT, SYSTEM_STORAGE_SCHEMA
from tools.r6_evidence import checked
from training_workloads.unified_h1.dynamic_assay import NEXT_LONG_PREFIX as PREFIX


PROJECT = ROOT / 'docs/experiments/instinct-post-r4-next-long'
PRIOR = PROJECT.with_name('instinct-post-r4-three')


def rear_pair(binding):
    _, _, report = validate_binding(binding)
    raw = {}
    for arm in ('reference', 'candidate'):
        group, = report['summary']['models'][arm]['groups']
        with checked(group['trace']).open() as stream:
            raw[arm] = [json.loads(line) for line in stream]
    from tools.r6_direct.post_r4_summary import rear

    trajectories = []
    for lane, axis in enumerate(raw['reference'][0]['axis_w']):
        side = (axis[1], -axis[0], 0.)

        def project(value):
            return sum(x * y for x, y in zip(side, value, strict=True))

        pairs = list(zip(raw['reference'], raw['candidate'], strict=True))
        stop = next((i for i, pair in enumerate(pairs)
                     if any(row['terminal'][lane] or row['ics_emergency'][lane] for row in pair)), len(pairs))
        split = next((i for i, (old, new) in enumerate(pairs[:stop])
                      if new['escape_recovery']['redirected'][lane]
                      and abs(project(new['v_final_w'][lane]) - project(old['v_final_w'][lane])) > .02), None)
        lane_metrics = dict(lane=lane, layout=raw['reference'][0]['layout'][lane],
                            split_step=split, common_end_exclusive=stop)
        if split is not None:
            for arm, rows in raw.items():
                origin = rows[split-1]['vehicle_position_w'][lane] if split else rows[0]['initial_position_w'][lane]
                displacement = project(rows[stop-1]['vehicle_position_w'][lane]) - project(origin)
                lane_metrics[arm] = dict(
                    lateral_issued_integral_m=.1*sum(project(row['v_final_w'][lane]) for row in rows[split:stop]),
                    lateral_motion_m=displacement,
                    minimum_sphere_sensor_gap_m=min(row['post']['sphere_sensor_clearance_m'][lane]
                                                     for row in rows[split:stop]))
        trajectories.append(lane_metrics)
    return dict(by_arm=rear(binding), common_window=trajectories)


class NextLongStudy(LongStudy):
    tag_prefix = 'instinct-post-r4-next-long'

    def __init__(self):
        self.docs = PROJECT
        self.ledger = PROJECT / 'RESOURCE_LEDGER.json'
        self.snapshot = SNAPSHOT
        self.eval = PREFIX + 'R6-END'
        if self.snapshot == ROOT:
            raise ValueError('Use a committed detached execution snapshot')

    def setup(self):
        guard()
        policy_path = PROJECT / 'RESOURCE_POLICY.json'
        if policy_path.exists():
            return pin(policy_path)
        free = shutil.disk_usage(SYSTEM_ROOT).free
        if free < SYSTEM_FLOOR + 2 * GIB:
            raise ValueError('Insufficient space for the development assay')
        caps = dict(allocated_gpu_seconds=144000, live_wall_seconds=144000, queue_seconds=21600,
                    live_starts=120, simulator_initializations=96, training_transitions=3000000,
                    nontraining_transitions=3500000, training_checkpoint_writes=192,
                    major_scientific_candidates=32, **{OUTPUT_CAP: free - SYSTEM_FLOOR})
        policy = self.write(policy_path.name, dict(schema_version=NEXT_LONG_SCHEMA,
                            user_authority=pin(PROJECT/'PLAN.md'), caps=caps, elapsed_seconds=604800))
        storage = self.write('STORAGE.json', dict(schema_version=SYSTEM_STORAGE_SCHEMA,
                             root=str(SYSTEM_ROOT), new_volume_allocation_bytes=0))
        zero = dict.fromkeys(THIRD_CAPS, 0)
        self.write(self.ledger.name, dict(resource_policy=policy, caps=caps, historical_charged=zero,
                   new_used=zero, protected_confirmation_delivery_repair=zero, status='FROZEN',
                   new_reservations=[dict(id='ROUND-SYSTEM', status='UNRELEASED',
                                          resources={OUTPUT_CAP: 0}, storage_plan=storage)]))
        freeze(self.tag_prefix+'-resources', 'Bind system storage and the next study resource ledger',
               self.ledger, policy_path, Path(storage['path']))
        return policy

    def predecessor(self):
        prior = [row['id'] for row in load(self.ledger)['new_reservations'] if row['id'].startswith(PREFIX)]
        if prior:
            return self.settle(prior[-1])
        return load(PRIOR/'r3-end.json')['predecessor']

    def freeze_input(self, run_id, operation, params, *extra):
        guard()
        path = self.input_path(run_id)
        if path.exists():
            job = load(path)
            if job['snapshot'] != str(self.snapshot) or read(job['contract'])['parameters'] != params:
                raise ValueError('Requested input differs from the committed contract')
            return path
        source = manifest()
        source_pin = self.write(f'source-{source["source_commit"][:12]}.json', source)
        contract = self.write(f'{run_id}-contract.json', build('contract', dict(operation=operation, parameters=params)))
        self.input(run_id, contract, self.predecessor(), snapshot=self.snapshot, source=source_pin,
                   resource_policy=pin(PROJECT/'RESOURCE_POLICY.json'))
        label = run_id.removeprefix(PREFIX).lower()
        freeze(f'{self.tag_prefix}-{label}-inputs', f'Freeze {run_id} input',
               path, Path(contract['path']), Path(source_pin['path']), self.ledger, *extra)
        return path

    def physical(self, spec_path):
        from tools.r6_direct.post_r4_summary import physical

        self.setup()
        spec = load(spec_path)
        label = spec['label']
        run_id = PREFIX + label
        params = dict(experiment_id=run_id, training_seed=20260908, seed=spec['seed'],
                      models=spec['models'], arm_options=spec['arm_options'],
                      wall_seconds=spec['wall_seconds'], output_cap_bytes=GIB)
        path = self.freeze_input(run_id, 'dynamic', params, spec_path)
        summary = self.docs / f'{label}.json'
        if summary.exists():
            return pin(summary)
        result = self.evaluate(path)
        metrics = rear_pair(result['binding']) if label.startswith('REARPAIR-') else physical(result['binding'])
        result_pin = self.write(summary.name, dict(**result, spec=pin(spec_path), metrics=metrics))
        freeze(f'{self.tag_prefix}-{label.lower()}-closed', f'Record {label} paired physical evidence',
               summary, self.ledger)
        return result_pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('physical',))
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    study = NextLongStudy()
    print(json.dumps(getattr(study, args.operation)(args.spec.resolve())), flush=True)


if __name__ == '__main__':
    main()
