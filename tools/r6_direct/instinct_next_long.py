"""Freeze and run the next instinctRL study through the local R6 supervisor."""

import argparse
import json
from pathlib import Path
import re
import shutil

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.instinct_cycle_finish import freeze, guard
from tools.r6_direct.instinct_long import GIB, LongStudy
from tools.r6_direct.prepare import build
from tools.r6_direct.resource_policy import NEXT_LONG_SCHEMA, OUTPUT_CAP, THIRD_CAPS
from tools.r6_direct.source import ROOT as SNAPSHOT, manifest, training_identity
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

    def prepare_training(self, spec_path):
        from training_workloads.unified_h1.direct_selection import training_model
        from training_workloads.unified_h1.robustness_training_plan import validate_recipe

        self.setup()
        spec = load(spec_path)
        label = spec['label']
        if not re.fullmatch(r'(SHORT|FINE)-[A-Z0-9-]+|FORMAL', label):
            raise ValueError('Use a distinct short, finite or formal training label')
        added = spec['phases'][-1][1] * 512
        bounds = ((1, 32768) if label.startswith('SHORT-') else
                  (32769, 524287) if label.startswith('FINE-') else (524288, float('inf')))
        if not bounds[0] <= added <= bounds[1]:
            raise ValueError('Training step budget differs from the registered study scale')
        if label == 'FORMAL':
            protocol = read(spec['independent_protocol'])
            if (protocol['added_updates'] != spec['phases'][-1][1]
                    or protocol['fixed_endpoint'] != spec['segments'][-1][1]
                    or not spec['coverage_minimums']):
                raise ValueError('Freeze the formal endpoint and coverage alongside the independent protocol')
        source = manifest()
        prior = load(PRIOR/'FORMAL-R3/RESULTS.json')
        prior_recipe = read(read(prior['design'])['recipe'])
        old = prior['segments'][-1]
        method = read(prior_recipe['method'])
        origin = training_model(old['plan'], old['analysis'], prior['end'], method)
        if origin['update'] != old['plan']['target_update'] or origin['update'] != 768:
            raise ValueError('Branch origin must be the audited R3 terminal checkpoint')
        self.docs = PROJECT / label
        self.docs.mkdir(exist_ok=True)
        design_path = self.docs/'DESIGN.json'
        if design_path.exists():
            design = load(design_path)
            if design['decision'] != pin(spec_path) or read(design['source']) != source:
                raise ValueError('A different training specification or source was already frozen')
            return pin(design_path)
        method.update(spec.get('method_overrides', {}))
        method['ppo'].update(spec.get('ppo_overrides', {}))
        curriculum = read(prior_recipe['curriculum'])
        phases = {p['kind']: p for p in curriculum['phases']}
        curriculum['phases'] = [dict(phases[kind], end_offset=end) for kind, end in spec['phases']]
        if 'tube_drive_steps' in spec:
            for phase in curriculum['phases']:
                if phase['kind'] == 'robust':
                    phase['tube_drive_steps'] = spec['tube_drive_steps']
        method_pin = self.write('method.json', method)
        curriculum_pin = self.write('curriculum.json', curriculum)
        seed = spec['training_seed']
        recipe = dict(prior_recipe, candidate_id=spec['candidate_id'], source_identity=training_identity(source),
                      method=method_pin, curriculum=curriculum_pin, diagnosis=[pin(spec_path), old['analysis'],
                                                                             *spec['development_evidence']],
                      checkpoint_offsets=spec['checkpoint_offsets'], branch_updates=spec['phases'][-1][1],
                      development_seeds=[seed], hypothesis=spec['hypothesis'], comparison=spec['comparison'],
                      branch_origin={k: origin[k] for k in ('checkpoint', 'update', 'training_seed')})
        validate_recipe(recipe)
        recipe_pin = self.write('recipe.json', recipe)
        source_pin = save(PROJECT/f'source-{source["source_commit"][:12]}.json', source)
        initial = dict(kind='LOCAL_U768_NEW_STREAM',
                       **{k: origin[k] for k in ('checkpoint', 'source_identity', 'method_hashes')})
        storage = read(prior['design'])['storage']
        storage.update(payload_bytes=spec['payload_gib']*GIB, compressed_bytes=(spec['payload_gib']-1)*GIB)
        params = read(prior['design'])['training_parameters']
        design = self.write('DESIGN.json', dict(decision=pin(spec_path), source=source_pin, recipe=recipe_pin,
                            initial=initial, training_seed=seed, storage=storage,
                            training_parameters=params, segments=spec['segments']))
        freeze(f'{self.tag_prefix}-{spec["label"].lower()}-design', f'Freeze {spec["label"]} training design',
               spec_path, design_path, *(Path(p['path']) for p in (source_pin, recipe_pin, method_pin, curriculum_pin)))
        return design

    def train(self, design_path):
        from training_workloads.unified_h1.recovery_coverage import audit as audit_coverage
        from training_workloads.unified_h1.robustness_training_method import action_option_version, bound_method_hashes

        self.docs = design_path.parent
        self.tag_prefix += '-' + self.docs.name.lower()
        if self.docs.parent != PROJECT:
            raise ValueError('Use the study training directory')
        design = load(design_path)
        source = read(design['source'])
        if source != manifest():
            raise ValueError('Use the exact snapshot in the frozen design')
        identity = training_identity(source)
        recipe = read(design['recipe'])
        start_update = recipe['branch_origin']['update']
        segments = design['segments']
        if (segments[0][0] != start_update
                or segments[-1][1] != start_update + recipe['branch_updates']
                or any(a[1] != b[0] for a, b in zip(segments, segments[1:]))):
            raise ValueError('Design segments do not span the planned training branch')
        initial, storage = design['initial'], design['storage']
        rows = []
        coverage = []
        for start, target, wall in segments:
            run_id = f'{PREFIX}{self.docs.name}-U{target}'
            plan = dict(schema_version='r6-local-training-plan/1', experiment_id=run_id, phase='development',
                        recipe=design['recipe'], selection_freeze=None, training_seed=design['training_seed'],
                        start_update=start, target_update=target, wall_seconds=wall, initialization=initial,
                        checkpoint_updates=[start_update+u for u in recipe['checkpoint_offsets']
                                            if start < start_update+u <= target], storage=storage)
            params = dict(design['training_parameters'], experiment_id=run_id, source_identity=identity,
                          training_seed=design['training_seed'], output_cap_bytes=storage['payload_bytes'],
                          r6_training_plan=plan)
            path = self.freeze_input(run_id, 'train', params)
            row = self.execute_segment(path, plan)
            rows.append(row)
            coverage_path = Path(row['analysis']['path']).parents[1]/'PIPE_COVERAGE.json'
            if not coverage_path.exists():
                save(coverage_path, audit_coverage(row['analysis']))
            coverage.append(pin(coverage_path))
            initial = dict(kind='RESUME_R6_STREAM', checkpoint=row['checkpoint'], source_identity=identity,
                           method_hashes=bound_method_hashes(plan, recipe))
        updates = sum(r['counters']['completed_updates'] for r in rows)
        transitions = sum(r['counters']['training_transitions'] for r in rows)
        if updates != recipe['branch_updates'] or transitions != updates*512:
            raise ValueError('Training evidence must cover the entire frozen branch')
        result_path = self.docs/'RESULTS.json'
        existed = result_path.exists()
        result = self.write(result_path.name, dict(design=pin(design_path), segments=rows, pipe_coverage=coverage,
            start=design['initial']['checkpoint'], end=rows[-1]['checkpoint'], actual_updates=updates,
            actual_training_transitions=transitions, training_supervised_seconds=sum(r['wall_seconds'] for r in rows),
            next_predecessor=rows[-1]['next_predecessor'], execution=action_option_version(recipe['candidate_id'])))
        if not existed:
            freeze(self.tag_prefix+'-closed', f'Record {self.docs.name} complete training evidence',
                   result_path, self.ledger)
        return result

    def r6(self, assignments, role):
        reference = assignments['roles'][role]
        protocol = read(assignments['protocol'])
        resources = read(assignments['r6_resources'])
        template = PRIOR / f'R6-DIRECT-INSTINCT-GPU-R5-RECOVERY-THREE-R6-{"QUALIFIED" if role == "qualified" else "R3-END"}-contract.json'
        params = load(template)['parameters']
        selection = params['evaluation_selection']
        comparison = read(selection['comparison'])
        run_id = PREFIX + 'R6-' + role.upper()
        if role != 'qualified':
            segment = reference['segment']
            plan = segment['plan']
            recipe = read(plan['recipe'])
            selection.update(checkpoint=reference['checkpoint'], training_plan=plan,
                             training_analysis=segment['analysis'], method=recipe['method'])
            comparison.update(candidate_id=recipe['candidate_id'], method=recipe['method'],
                              hypothesis=recipe['hypothesis'])
            params['training_seed'] = plan['training_seed']
            if reference.get('execution_override'):
                selection['execution_override'] = reference['execution_override']
        elif selection['checkpoint'] != reference['checkpoint']:
            raise ValueError('Qualified R6 model differs from the locked assignment')
        comparison.update(id=run_id, seed=protocol['r6']['perturbation_seed'])
        comparison_pin = self.write(run_id+'-comparison.json', comparison)
        selection['comparison'] = comparison_pin
        params.update(experiment_id=run_id, output_cap_bytes=resources['output_cap_bytes'])
        params['r6'].update(run_id=run_id, seed=comparison['seed'], output_quotas=resources['output_quotas'])
        path = self.freeze_input(run_id, 'evaluate', params, Path(comparison_pin['path']))
        summary_path = PROJECT/f'R6-{role.upper()}.json'
        if summary_path.exists():
            return pin(summary_path)
        result = self.write(summary_path.name, self.evaluate(path))
        freeze(f'{self.tag_prefix}-r6-{role}-closed', f'Record complete R6 replay for {role}',
               summary_path, self.ledger)
        return result

    def final_evaluation(self, protocol_path):
        from tools.r6_direct.next_long_eval_resources import forecast
        from tools.r6_direct.post_r4_audit import verify_training

        path = PROJECT/'FORMAL/RESULTS.json'
        formal = load(path)
        decision = read(read(formal['design'])['decision'])
        protocol_pin = pin(protocol_path)
        protocol = read(protocol_pin)
        if (decision['independent_protocol'] != protocol_pin
                or formal['actual_training_transitions'] < 524288
                or formal['segments'][-1]['plan']['target_update'] != protocol['fixed_endpoint']):
            raise ValueError('The independent stream requires the full locked formal endpoint')
        verify_training(formal)
        prior = load(PRIOR/'FORMAL-R3/RESULTS.json')
        if prior['end'] != formal['start'] or prior['end'] != protocol['r3']['checkpoint']:
            raise ValueError('The three required controls do not bind the formal origin')
        same_execution = formal['execution'] == prior['execution']
        roles = {
            'qualified': dict(checkpoint=protocol['qualified'], execution=None),
            'origin': dict(checkpoint=prior['end'], execution=formal['execution'], segment=prior['segments'][-1]),
            'end': dict(checkpoint=formal['end'], execution=formal['execution'], segment=formal['segments'][-1]),
        }
        if not same_execution:
            roles['origin']['execution_override'] = formal['execution']
            roles['prior'] = dict(checkpoint=prior['end'], execution=prior['execution'], segment=prior['segments'][-1])
        arms = ['qualified', *(['prior'] if not same_execution else []), 'origin', 'end']
        resources = read(protocol['r6_resources'])
        assignments_path = PROJECT/'FINAL_ASSIGNMENTS.json'
        existed = assignments_path.exists()
        assignments_pin = self.write(assignments_path.name, dict(
            protocol=protocol_pin, formal=pin(path), r6_resources=protocol['r6_resources'],
            roles=roles, r6_arm_order=arms, prior_in_origin=same_execution,
        ))
        if not existed:
            guard()
            freeze(f'{self.tag_prefix}-independent-assignments', 'Lock the formal endpoint and independent evaluation arms',
                   assignments_path)
        names = dict(reference='qualified', control='origin', candidate='end')
        if not same_execution:
            names['prior'] = 'prior'
        physical_spec = self.write('HOLDOUT-FINAL-SPEC.json', dict(
            label='HOLDOUT-FINAL', seed=protocol['physical']['seed'], wall_seconds=1800,
            models={a: roles[r]['checkpoint'] for a, r in names.items()},
            arm_options={a: roles[r]['execution'] for a, r in names.items()},
            assignments=assignments_pin,
        ))
        pending_physical = not (PROJECT/'HOLDOUT-FINAL.json').exists()
        pending_r6 = sum(not (PROJECT/f'R6-{role.upper()}.json').exists() for role in arms)
        if pending_physical or pending_r6:
            capacity = forecast(resources, remaining_phases=[
                dict(name='paired physical', launch_cap_bytes=GIB, retained_bytes=GIB//3),
            ] if pending_physical else [], remaining_r6_arms=pending_r6)
            if not capacity['fits']:
                raise ValueError(f'Independent series exceeds forecast system capacity: {capacity}')
            if not (PROJECT/'FINAL_RESOURCE_ADMISSION.json').exists():
                self.write('FINAL_RESOURCE_ADMISSION.json', capacity)
                freeze(f'{self.tag_prefix}-independent-admission', 'Record measured prior-to-evaluation capacity',
                       PROJECT/'FINAL_RESOURCE_ADMISSION.json')
        physical = self.physical(Path(physical_spec['path']))
        assignments = read(assignments_pin)
        r6 = {role: self.r6(assignments, role) for role in arms}
        evaluations_path = PROJECT/'EVALUATIONS.json'
        existed = evaluations_path.exists()
        result = self.write(evaluations_path.name, dict(assignments=assignments_pin, physical=physical,
                                                       r6=r6, ledger=pin(self.ledger)))
        if not existed:
            freeze(f'{self.tag_prefix}-independent-closed', 'Record all independent evaluation originals',
                   evaluations_path, self.ledger)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('physical', 'prepare-training', 'train', 'final-evaluation'))
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    study = NextLongStudy()
    print(json.dumps(getattr(study, args.operation.replace('-', '_'))(args.spec.resolve())), flush=True)


if __name__ == '__main__':
    main()
