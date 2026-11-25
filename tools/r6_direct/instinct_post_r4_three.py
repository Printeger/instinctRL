"""Freeze and execute one stage of the three-round R4 continuation."""

import argparse
import json
from pathlib import Path
import shutil

from tools.r6_direct.evidence import read
from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import load
from tools.r6_direct.instinct_cycle_finish import freeze, guard
from tools.r6_direct.instinct_long import GIB, LongStudy
from tools.r6_direct.prepare import build
from tools.r6_direct.resource_policy import OUTPUT_CAP, POST_R4_THREE_SCHEMA, THIRD_CAPS
from tools.r6_direct.source import ROOT as SNAPSHOT, manifest, training_identity
from tools.r6_direct.storage import SYSTEM_FLOOR, SYSTEM_ROOT, SYSTEM_STORAGE_SCHEMA
from training_workloads.unified_h1.dynamic_assay import RECOVERY_HOLDOUT, THREE_PREFIX
from training_workloads.unified_h1.robustness_training_method import action_option_version, bound_method_hashes


PROJECT = ROOT / 'docs/experiments/instinct-post-r4-three'
PREVIOUS = ROOT / 'docs/experiments/instinct-r4-recovery'
PROTOCOL = PROJECT / 'INDEPENDENT_PROTOCOL.json'


class PostR4Study(LongStudy):
    def __init__(self):
        self.docs = PROJECT
        self.ledger = PROJECT / 'RESOURCE_LEDGER.json'
        self.snapshot = SNAPSHOT
        if self.snapshot == ROOT:
            raise ValueError('Use a detached source snapshot')
        self.tag_prefix = 'instinct-post-r4-three'
        self.eval = THREE_PREFIX + 'HOLDOUT-R3'
        self.protocol = load(PROTOCOL)

    def setup(self):
        free = shutil.disk_usage(SYSTEM_ROOT).free
        caps = dict(allocated_gpu_seconds=144000, live_wall_seconds=144000, queue_seconds=17280,
                    live_starts=96, simulator_initializations=64, training_transitions=2000000,
                    nontraining_transitions=2200000, training_checkpoint_writes=110,
                    major_scientific_candidates=12, **{OUTPUT_CAP: free - SYSTEM_FLOOR})
        if free < 85 * GIB:
            raise ValueError('System space cannot cover the retained final evaluation and reserve')
        policy = self.write('RESOURCE_POLICY.json', dict(schema_version=POST_R4_THREE_SCHEMA,
                            user_authority=pin(PROJECT/'PLAN.md'), caps=caps, elapsed_seconds=259200))
        storage = self.write('STORAGE.json', dict(schema_version=SYSTEM_STORAGE_SCHEMA,
                             root=str(SYSTEM_ROOT), new_volume_allocation_bytes=0))
        zero = dict.fromkeys(THIRD_CAPS, 0)
        self.write(self.ledger.name, dict(resource_policy=policy, caps=caps, historical_charged=zero,
                   new_used=zero, protected_confirmation_delivery_repair=zero, status='FROZEN',
                   new_reservations=[dict(id='ROUND-SYSTEM', status='UNRELEASED',
                                          resources={OUTPUT_CAP: 0}, storage_plan=storage)]))
        freeze(self.tag_prefix+'-resources', 'Freeze the R4 continuation resource envelope',
               self.ledger, PROJECT/'RESOURCE_POLICY.json', PROJECT/'STORAGE.json')
        return policy

    def predecessor(self):
        rows = [row for row in load(self.ledger)['new_reservations'] if row['id'].startswith(THREE_PREFIX)]
        if rows:
            return self.settle(rows[-1]['id'])
        return pin(SYSTEM_ROOT / RECOVERY_HOLDOUT / 'control/receipt.json')

    def source(self):
        value = manifest()
        return self.write(f'source-{value["source_commit"][:12]}.json', value)

    def freeze_input(self, run_id, operation, params, *, extra=(), tag=None):
        path = self.input_path(run_id)
        if path.exists():
            job = load(path)
            if job['snapshot'] != str(self.snapshot) or read(job['contract'])['parameters'] != params:
                raise ValueError('The frozen job differs from the requested input')
            return path
        source = self.source()
        prior = self.predecessor()
        contract = self.write(f'{run_id}-contract.json', build('contract', dict(operation=operation, parameters=params)))
        self.input(run_id, contract, prior, snapshot=self.snapshot, source=source,
                   resource_policy=pin(PROJECT/'RESOURCE_POLICY.json'))
        freeze(tag or f'{self.tag_prefix}-{run_id.removeprefix(THREE_PREFIX).lower()}-inputs',
               f'Freeze {run_id} input', path, Path(contract['path']), Path(source['path']), self.ledger, *extra)
        return path

    def rear(self, suffix):
        from instinctlab.control.escape_recovery import LATERAL_VERSION, RECOVERY_VERSION

        initial = load(PREVIOUS/'FORMAL_DESIGN.json')['initial']['checkpoint']
        options = dict(reference=RECOVERY_VERSION, candidate=LATERAL_VERSION,
                       reference_slow=RECOVERY_VERSION, candidate_slow=LATERAL_VERSION)
        return self.physical('REAR-'+suffix, self.protocol['development']['rear_seed'],
                             dict.fromkeys(options, initial), options, 900)

    def physical(self, label, seed, models, options, wall):
        run_id = THREE_PREFIX + label
        params = dict(experiment_id=run_id, training_seed=20260908, seed=seed,
                      models=models, arm_options=options, wall_seconds=wall, output_cap_bytes=GIB)
        path = self.freeze_input(run_id, 'dynamic', params)
        result = self.evaluate(path)
        summary = self.docs / f'{label}.json'
        existing = summary.exists()
        result_pin = self.write(summary.name, result)
        if not existing:
            freeze(f'{self.tag_prefix}-{label.lower()}-closed', f'Record {label} physical replay', summary, self.ledger)
        return result_pin

    def prepare_training(self, spec_path):
        spec = load(spec_path)
        label = spec['label']
        if label not in ('SHORT-R1', 'SHORT-R2', 'SHORT-R3', 'FORMAL-R1', 'FORMAL-R2', 'FORMAL-R3'):
            raise ValueError('Use a registered short or formal round')
        self.docs = PROJECT/label
        self.docs.mkdir(exist_ok=True)
        old = load(PREVIOUS/'FORMAL_DESIGN.json')
        recipe = read(old['recipe'])
        curriculum = read(recipe['curriculum'])
        robust = dict(curriculum['phases'][0], **spec['robust_overrides'])
        replay = curriculum['phases'][1]
        curriculum['phases'] = [dict(robust if kind == 'robust' else replay, end_offset=offset)
                                for kind,offset in spec['phases']]
        method = read(recipe['method'])
        method['ppo'].update(spec['ppo_overrides'])
        method.update(spec['method_overrides'])
        source = self.source()
        training_seed = spec['training_seed']
        recipe.update(candidate_id=spec['candidate_id'], source_identity=training_identity(read(source)),
                      branch_updates=spec['phases'][-1][1], checkpoint_offsets=spec['checkpoint_offsets'],
                      development_seeds=[training_seed], diagnosis=[pin(spec_path), pin(PROTOCOL), *spec['evidence']],
                      curriculum=self.write('curriculum.json', curriculum), method=self.write('method.json', method),
                      hypothesis=spec['hypothesis'], comparison=spec['comparison'])
        from training_workloads.unified_h1.robustness_training_plan import validate_recipe

        recipe_pin = self.write('recipe.json', recipe)
        validate_recipe(recipe)
        design = self.write('DESIGN.json', dict(label=label, source=source, recipe=recipe_pin,
                            decision=pin(spec_path), initial=old['initial'], training_seed=training_seed,
                            training_parameters=old['training_parameters'], segments=spec['segments'],
                            storage=dict(old['storage'], payload_bytes=spec['payload_gib']*GIB,
                                         compressed_bytes=(spec['payload_gib']-1)*GIB)))
        freeze(f'{self.tag_prefix}-{label.lower()}-design', f'Freeze {label} scientific and training inputs',
               spec_path, Path(source['path']), *(self.docs/name for name in ('DESIGN.json', 'curriculum.json', 'method.json', 'recipe.json')))
        return design

    def independent_inputs(self):
        results = [pin(PROJECT/f'FORMAL-R{i}'/'RESULTS.json') for i in range(1, 4)]
        data = [read(item) for item in results]
        for i, row in enumerate(data, 1):
            design = read(row['design'])
            recipe = read(design['recipe'])
            if (design['label'] != f'FORMAL-R{i}' or row['actual_updates'] != recipe['branch_updates']
                    or row['actual_training_transitions'] != recipe['branch_updates']*512
                    or row['segments'][-1]['plan']['target_update'] != 640+recipe['branch_updates']):
                raise ValueError('Formal endpoint coverage before final evaluation')
        prior = load(PREVIOUS/'FORMAL_DESIGN.json')
        path = PROJECT/'FINAL_ASSIGNMENTS.json'
        exists = path.exists()
        frozen = self.write(path.name, dict(protocol=pin(PROTOCOL), formal=results, qualified=prior['qualified'],
                            arm_order=self.protocol['arms'], implementation=self.source()))
        if not exists:
            freeze(f'{self.tag_prefix}-independent-assignments', 'Freeze the three closed endpoints and all paired evaluation arms',
                   path, Path(read(frozen)['implementation']['path']))
        return frozen, data

    def r6(self, assignments, *, role, summary_name, seed, ordinal=None, trained=None):
        if role == 'qualified':
            template = ROOT/'docs/experiments/instinct-next-two-cycles/round-2/EVAL-contract.json'
        elif role in ('origin', 'end') and ordinal in (1, 2, 3):
            template = PREVIOUS/f'R6-DIRECT-INSTINCT-GPU-R5-RECOVERY-R6-{"CANDIDATE" if role == "end" else "ORIGIN"}-contract.json'
        else:
            raise ValueError('Unknown R6 evaluation role')
        status = self.docs/summary_name
        run_id = THREE_PREFIX + f'R6-{status.stem.upper()}'
        params = load(template)['parameters']
        selection = params['evaluation_selection']
        comparison = read(selection['comparison'])
        comparison.update(id=run_id, seed=seed)
        if role != 'qualified':
            design = read(trained['design'])
            recipe = read(design['recipe'])
            comparison.update(candidate_id=recipe['candidate_id'], method=recipe['method'], hypothesis=recipe['hypothesis'])
            selection['method'] = recipe['method']
            if role == 'end':
                final = trained['segments'][-1]
                selection.update(checkpoint=final['checkpoint'], training_plan=final['plan'], training_analysis=final['analysis'])
                params['training_seed'] = design['training_seed']
            else:
                selection['checkpoint'] = design['initial']['checkpoint']
        elif selection['checkpoint'] != read(assignments)['qualified']:
            raise ValueError('Qualified checkpoint drift')
        comparison_pin = self.write(run_id+'-comparison.json', comparison)
        selection['comparison'] = comparison_pin
        params['experiment_id'] = run_id
        params['r6'].update(run_id=run_id, seed=seed)
        path = self.freeze_input(run_id, 'evaluate', params, extra=(Path(comparison_pin['path']),))
        evaluation = self.evaluate(path)
        exists = status.exists()
        result = self.write(status.name, evaluation)
        if not exists:
            freeze(f'{self.tag_prefix}-r6-{status.stem.lower()}-closed', f'Record {status.stem} complete R6 replay', status, self.ledger)
        return result

    def final_evaluation(self):
        assignments, rounds = self.independent_inputs()
        r6_seed = self.protocol['r6']['perturbation_seed']
        physical_seed = self.protocol['physical']['seed']
        qualified = read(assignments)['qualified']
        phys = []
        for ordinal, trained in enumerate(rounds, 1):
            phys.append(self.physical(f'HOLDOUT-R{ordinal}', physical_seed,
                        dict(reference=qualified, control=trained['start'], candidate=trained['end']),
                        dict(reference=None, control=trained['execution'], candidate=trained['execution']), 1800))
        r6 = {'qualified': self.r6(assignments, role='qualified', summary_name='qualified.json', seed=r6_seed)}
        for ordinal, trained in enumerate(rounds, 1):
            for role in ('origin', 'end'):
                name = f'r{ordinal}-{role}'
                r6[name] = self.r6(assignments, role=role, summary_name=f'{name}.json', seed=r6_seed,
                                   ordinal=ordinal, trained=trained)
        path = PROJECT/'EVALUATIONS.json'
        exists = path.exists()
        result = self.write(path.name, dict(assignments=assignments, physical=phys, r6=r6, ledger=pin(self.ledger)))
        if not exists:
            freeze(f'{self.tag_prefix}-independent-closed', 'Record all closed independent evaluation arms', path, self.ledger)
        return result

    def train(self, design_path):
        design = load(design_path)
        self.docs = design_path.parent
        self.tag_prefix += '-' + design['label'].lower()
        if self.docs != PROJECT/design['label'] or design['label'] not in (
                'SHORT-R1', 'SHORT-R2', 'SHORT-R3', 'FORMAL-R1', 'FORMAL-R2', 'FORMAL-R3'):
            raise ValueError('Use the frozen round directory')
        source = read(design['source'])
        if source != manifest():
            raise ValueError('Use the exact training snapshot')
        if design['label'].startswith('FORMAL-'):
            ordinal = int(design['label'][-1])
            for earlier in range(1, ordinal):
                read(pin(PROJECT/f'FORMAL-R{earlier}'/'RESULTS.json'))
        recipe = read(design['recipe'])
        initial = design['initial']
        training_seed = design['training_seed']
        storage = design['storage']
        identity = training_identity(source)
        if (recipe['source_identity'] != identity or design['segments'][0][0] != 640
                or design['segments'][-1][1] != 640 + recipe['branch_updates']
                or any(a[1] != b[0] for a, b in zip(design['segments'], design['segments'][1:]))):
            raise ValueError('Training design clock does not match the recipe')
        rows = []
        for start, target, wall in design['segments']:
            run_id = f'{THREE_PREFIX}{design["label"]}-U{target}'
            plan = dict(schema_version='r6-local-training-plan/1', experiment_id=run_id, phase='development',
                        recipe=design['recipe'], selection_freeze=None, training_seed=training_seed,
                        start_update=start, target_update=target, wall_seconds=wall, initialization=initial,
                        checkpoint_updates=[640+u for u in recipe['checkpoint_offsets'] if start < 640+u <= target],
                        storage=storage)
            params = dict(design['training_parameters'], experiment_id=run_id, training_seed=training_seed,
                          output_cap_bytes=storage['payload_bytes'], source_identity=identity, r6_training_plan=plan)
            path = self.freeze_input(run_id, 'train', params, tag=f'{self.tag_prefix}-u{target}-inputs')
            row = self.execute_segment(path, plan)
            rows.append(row)
            initial = dict(kind='RESUME_R6_STREAM', checkpoint=row['checkpoint'], source_identity=identity,
                           method_hashes=bound_method_hashes(plan, recipe))
        updates = sum(row['counters']['completed_updates'] for row in rows)
        transitions = sum(row['counters']['training_transitions'] for row in rows)
        if updates != recipe['branch_updates'] or transitions != updates*512:
            raise ValueError('Training evidence must cover the entire frozen endpoint')
        existing = (self.docs/'RESULTS.json').exists()
        result = self.write('RESULTS.json', dict(design=pin(design_path), segments=rows,
            start=design['initial']['checkpoint'], end=rows[-1]['checkpoint'], actual_updates=updates,
            actual_training_transitions=transitions, training_supervised_seconds=sum(row['wall_seconds'] for row in rows),
            next_predecessor=rows[-1]['next_predecessor'], execution=action_option_version(recipe['candidate_id'])))
        if not existing:
            freeze(f'{self.tag_prefix}-closed', f'Record {design["label"]} completed training',
                   Path(result['path']), self.ledger)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('setup', 'rear', 'physical', 'prepare-train', 'train', 'final-evaluation'))
    parser.add_argument('spec', nargs='?')
    args = parser.parse_args()
    guard()
    study = PostR4Study()
    if args.stage == 'setup':
        result = study.setup()
    elif args.stage == 'rear':
        result = study.rear(args.spec or '01')
    elif args.stage == 'physical':
        result = study.physical(**load(Path(args.spec)))
    elif args.stage == 'prepare-train':
        result = study.prepare_training(Path(args.spec).resolve())
    elif args.stage == 'train':
        result = study.train(Path(args.spec).resolve())
    else:
        result = study.final_evaluation()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
