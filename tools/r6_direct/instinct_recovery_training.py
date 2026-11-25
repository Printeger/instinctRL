"""Freeze and run recovery training with paired development checkpoints."""

import argparse
import json
from pathlib import Path
from statistics import fmean

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import load
from tools.r6_direct.instinct_cycle_finish import freeze, guard
from tools.r6_direct.instinct_long import GIB
from tools.r6_direct.instinct_recovery import PROJECT, R4, RecoveryStudy
from tools.r6_direct.io import atomic
from tools.r6_direct.lifecycle import writer
from tools.r6_direct.prepare import build
from tools.r6_direct.source import manifest, training_identity
from training_workloads.unified_h1.dynamic_assay import RECOVERY_PREFIX
from training_workloads.unified_h1.robustness_training_method import (
    ESCAPE_RECOVERY_CANDIDATE, bound_method_hashes,
)


SHORT_SEED = 2026112501
SHORT_CHECK_SEED = 185299993
SHORT_UPDATES = (656, 672, 704)


class RecoveryTraining(RecoveryStudy):
    tag_prefix = 'instinct-r4-recovery'

    def __init__(self):
        super().__init__()
        path = PROJECT/'TRAIN_RESOURCE_POLICY.json'
        if path.exists():
            self.resource_policy = pin(path)

    def new_policy(self):
        with writer():
            ledger = load(self.ledger)
            policy = read(ledger['resource_policy'])
            # The accounting admission charges each initialization from u640.
            # Short development and the single formal stream use one mechanism.
            revised = dict(policy['caps'], major_scientific_candidates=2)
            policy['caps'] = revised
            policy['user_authority'] = pin(PROJECT/'SHORT_TRAIN.md')
            path = PROJECT/'TRAIN_RESOURCE_POLICY.json'
            resource = self.write(path.name, policy)
            atomic(self.ledger, dict(ledger, caps=revised, resource_policy=resource))
        return path

    def training_spec(self, series, *, recipe, initial, source, predecessor, training_seed, segments, storage):
        r4 = load(R4/'DESIGN.json')
        return dict(series=series, recipe=recipe, source=source, initial=initial,
                    predecessor=predecessor, training_seed=training_seed, segments=segments,
                    training_parameters=r4['training_parameters'], storage=storage)

    def prepare_short(self):
        if self.snapshot == ROOT:
            raise ValueError('Use the detached training snapshot')
        comparison = read(pin(PROJECT/'MECHANISM_RESULTS.json'))
        reference = comparison['open_tube']['reference']['approach']
        candidate = comparison['open_tube']['candidate']['approach']
        if (candidate['recovered_without_contact'] <= reference['recovered_without_contact']
                or candidate['collisions'] > reference['collisions']):
            raise ValueError('Physical recovery did not improve under fixed weights')
        self.source = self.write('source-training.json', manifest())
        old = load(R4/'DESIGN.json')
        recipe = read(old['recipe'])
        curriculum = read(recipe['curriculum'])
        replay, robust = curriculum['phases']
        # All short endpoints contain exactly their stated count of target-scene
        # updates. The protocol still binds an ordinary replay phase after them.
        curriculum['phases'] = [dict(robust, end_offset=64), dict(replay, end_offset=65)]
        method = read(recipe['method'])
        method['ppo']['actor_learning_rate'] = 3.e-5
        recipe.update(candidate_id=ESCAPE_RECOVERY_CANDIDATE,
                      source_identity=training_identity(read(self.source)),
                      diagnosis=[pin(PROJECT/'DEVELOPMENT.md'), pin(PROJECT/'SHORT_TRAIN.md'),
                                 pin(PROJECT/'MECHANISM_RESULTS.json')],
                      branch_updates=65, checkpoint_offsets=[16, 32, 64, 65],
                      development_seeds=[SHORT_SEED], confirmation_seeds=[2026112511, 2026112512, 2026112513],
                      curriculum=self.write('short-curriculum.json', curriculum),
                      method=self.write('training-method.json', method),
                      hypothesis='PPO with axial approaches improves recovery after the fixed escape option.',
                      comparison='Compare u640 and the checkpoint with identical execution and development layouts.')
        recipe_pin = self.write('short-recipe.json', recipe)
        storage = dict(old['storage'], payload_bytes=5*GIB, compressed_bytes=4*GIB, plain_bytes=GIB)
        segments = [[640, 656, 1450], [656, 672, 1450], [672, 704, 2500]]
        design = self.training_spec('SHORT', recipe=recipe_pin, initial=old['initial'],
                                    source=self.source, predecessor=comparison['rear']['predecessor'],
                                    training_seed=SHORT_SEED, segments=segments, storage=storage)
        design_path = self.write('SHORT_DESIGN.json', design)
        authority = self.new_policy()
        freeze(self.tag_prefix+'-short-design', 'Freeze target-scene PPO development checks',
               PROJECT/'source-training.json', PROJECT/'SHORT_DESIGN.json', PROJECT/'short-recipe.json',
               PROJECT/'short-curriculum.json', PROJECT/'training-method.json', authority, self.ledger)
        return design_path

    def segment_input(self, design, segment, initial, predecessor):
        self.source = design['source']
        start, target, wall = segment
        run_id = f'{RECOVERY_PREFIX}{design["series"]}-TRAIN-U{target}'
        params = dict(design['training_parameters'], experiment_id=run_id,
                      training_seed=design['training_seed'],
                      source_identity=training_identity(read(design['source'])),
                      output_cap_bytes=design['storage']['payload_bytes'])
        offsets = read(design['recipe'])['checkpoint_offsets']
        plan = dict(schema_version='r6-local-training-plan/1', experiment_id=run_id, phase='development',
                    recipe=design['recipe'], selection_freeze=None, training_seed=design['training_seed'],
                    start_update=start, target_update=target, wall_seconds=wall, initialization=initial,
                    checkpoint_updates=[u+640 for u in offsets if start < u+640 <= target],
                    storage=design['storage'])
        params['r6_training_plan'] = plan
        inputs = self.input_path(run_id)
        if inputs.exists():
            previous = load(inputs)
            if read(previous['contract'])['parameters'] != params or previous['predecessor_receipt'] != predecessor:
                raise ValueError('Frozen training inputs drifted')
        else:
            contract = self.write(f'{run_id}-contract.json', build('contract', dict(operation='train', parameters=params)))
            self.input(run_id, contract, predecessor, snapshot=self.snapshot, source=self.source,
                       resource_policy=self.resource_policy)
            freeze(f'{self.tag_prefix}-train-u{target}-inputs', f'Freeze strict continuation through u{target}',
                   inputs, Path(contract['path']), self.ledger)
        return inputs, plan

    def short_check(self, design, row):
        update = row['plan']['target_update']
        run_id = f'{RECOVERY_PREFIX}SHORT-CHECK-U{update}'
        inputs = self.input_path(run_id)
        if not inputs.exists():
            self.dynamic_input(run_id, SHORT_CHECK_SEED, dict(
                candidate=row['checkpoint'], reference=design['initial']['checkpoint']),
                row['next_predecessor'], wall_seconds=1250)
            freeze(f'{self.tag_prefix}-check-u{update}-inputs', f'Freeze development comparison at u{update}',
                   inputs, PROJECT/f'{run_id}-contract.json', self.ledger)
        result = self.evaluate(inputs)
        _, _, report = validate_binding(result['binding'])
        candidate = report['summary']['models']['candidate']['groups']
        reference = report['summary']['models']['reference']['groups']
        details = {}
        for c_group, r_group in zip(candidate, reference, strict=True):
            for layout in ('room_door', 'trunk_aisle', 'street_gap', 'open_tube'):
                c = [r for r in c_group['lanes'] if r['layout'] == layout]
                r = [r for r in r_group['lanes'] if r['layout'] == layout]
                mode = c[0]['mode']
                metrics = ('progress_m', 'recovery_progress_m', 'tracking_rmse_mps', 'maximum_horizontal_excursion_m')
                means = lambda rows: {key: fmean(lane[key] for lane in rows) for key in metrics}
                details[f'{layout}/{mode}'] = dict(candidate=means(c), origin=means(r),
                    new_contacts=sum(x['collision'] and not y['collision'] for x, y in zip(c, r, strict=True)),
                    collisions=[sum(x['collision'] for x in rows) for rows in (c, r)],
                    recovered_without_contact=[sum(not x['collision'] and x['recovery_progress_m'] >= .5
                                                   for x in rows) for rows in (c, r)])
        summary = self.write(f'SHORT_CHECK_U{update}.json', dict(result=result, comparisons=details))
        freeze(f'{self.tag_prefix}-check-u{update}-results', f'Record recovery development assessment at u{update}',
               Path(summary['path']), self.ledger)
        return result

    def execute_short(self):
        design = load(PROJECT/'SHORT_DESIGN.json')
        recipe = read(design['recipe'])
        predecessor, initial = design['predecessor'], design['initial']
        for section in design['segments']:
            inputs, plan = self.segment_input(design, section, initial, predecessor)
            row = self.execute_segment(inputs, plan)
            check = self.short_check(design, row)
            predecessor = check['predecessor']
            initial = dict(kind='RESUME_R6_STREAM', checkpoint=row['checkpoint'],
                           source_identity=recipe['source_identity'], method_hashes=bound_method_hashes(plan, recipe))
        return pin(PROJECT/f'SHORT_CHECK_U{SHORT_UPDATES[-1]}.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare-short', 'run-short'))
    args = parser.parse_args()
    guard()
    study = RecoveryTraining()
    result = study.prepare_short() if args.action == 'prepare-short' else study.execute_short()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
