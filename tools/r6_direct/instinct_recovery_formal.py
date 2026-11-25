"""Run one frozen 1024-update recovery candidate and independent comparisons."""

import argparse
import json
from pathlib import Path
import shutil

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.holdout_summary import compare_holdout, mechanisms
from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import HISTORICAL, load
from tools.r6_direct.instinct_cycle_finish import freeze, guard
from tools.r6_direct.instinct_cycle_report import displacement
from tools.r6_direct.instinct_long import GIB
from tools.r6_direct.instinct_recovery import PROJECT, R4
from tools.r6_direct.instinct_recovery_training import RecoveryTraining, SHORT_UPDATES
from tools.r6_direct.prepare import build
from tools.r6_direct.source import manifest, training_identity
from tools.r6_direct.storage import SYSTEM_FLOOR, SYSTEM_ROOT
from tools.r6_direct.study_report import compare, score_model
from training_workloads.unified_h1.dynamic_assay import RECOVERY_HOLDOUT, RECOVERY_PREFIX
from training_workloads.unified_h1.robustness_training_method import bound_method_hashes


FORMAL_SEED = 2026112502
R6_SEED = 1763541728
HOLDOUT_SEED = 1657421097
TOTAL_UPDATES = 1024


class FormalRecovery(RecoveryTraining):
    def prepare_formal(self):
        if self.snapshot == ROOT:
            raise ValueError('Use the detached formal source')
        old = load(PROJECT/'SHORT_DESIGN.json')
        checks = [pin(PROJECT/f'SHORT_CHECK_U{u}.json') for u in SHORT_UPDATES]
        development = [read(p) for p in checks]
        endpoints = [read(pin(PROJECT/f'U{u}.json')) for u in SHORT_UPDATES]
        measured = load(self.ledger)
        prior_name = f'{RECOVERY_PREFIX}SHORT-TRAIN-U{SHORT_UPDATES[-1]}'
        prior, = [r for r in measured['new_reservations'] if r['id'] == prior_name]
        per_segment = 2*prior['closed_evidence_bytes'] + GIB
        expected_disk = 8*per_segment + 18*GIB
        if shutil.disk_usage(SYSTEM_ROOT).free < SYSTEM_FLOOR + expected_disk:
            raise ValueError('Measured development volume exceeds available formal reserve')

        self.source = self.write('source-formal.json', manifest())
        development_source = read(old['source'])
        formal_source = read(self.source)
        for relative in ('isaaclab-training/instinctlab/control/escape_recovery.py',
                         'training_workloads/unified_h1/escape_option_env.py',
                         'training_workloads/unified_h1/recovery_training_scene.py',
                         'training_workloads/unified_h1/robustness_training_method.py'):
            if development_source['sources'][relative] != formal_source['sources'][relative]:
                raise ValueError(f'Formal mechanism differs from the tested source: {relative}')
        recipe = read(old['recipe'])
        curriculum = read(recipe['curriculum'])
        robust, ordinary = curriculum['phases']
        curriculum['phases'] = [part for start in range(0, TOTAL_UPDATES, 128) for part in (
            dict(robust, end_offset=start+64), dict(ordinary, end_offset=start+128))]
        recipe.update(source_identity=training_identity(formal_source), branch_updates=TOTAL_UPDATES,
                      development_seeds=[FORMAL_SEED], checkpoint_offsets=list(range(64, 1025, 64)),
                      curriculum=self.write('formal-curriculum.json', curriculum),
                      diagnosis=[*recipe['diagnosis'], pin(PROJECT/'FORMAL.md'), *checks])
        recipe_pin = self.write('formal-recipe.json', recipe)
        limits = dict(old['storage'], payload_bytes=11*GIB, compressed_bytes=10*GIB)
        segments = [[640+offset, 640+offset+128, 9600] for offset in range(0, TOTAL_UPDATES, 128)]
        design = self.training_spec('FORMAL', recipe=recipe_pin, initial=old['initial'], source=self.source,
                                    predecessor=development[-1]['result']['predecessor'], training_seed=FORMAL_SEED,
                                    segments=segments, storage=limits)
        design.update(holdout_seed=HOLDOUT_SEED, r6_seed=R6_SEED, fixed_endpoint=1664,
                      qualified=load(R4/'DESIGN.json')['qualified']['checkpoint'],
                      short_endpoints=[row['checkpoint'] for row in endpoints],
                      qualification=checks, expected_formal_storage_bytes=expected_disk,
                      free_bytes_at_freeze=shutil.disk_usage(SYSTEM_ROOT).free)
        design_pin = self.write('FORMAL_DESIGN.json', design)
        freeze(self.tag_prefix+'-formal-design', 'Freeze one 1024-update candidate and both evaluation streams',
               PROJECT/'FORMAL_DESIGN.json', PROJECT/'source-formal.json', PROJECT/'formal-curriculum.json',
               PROJECT/'formal-recipe.json', self.ledger)
        return design_pin

    def r6_input(self, design, endpoint, predecessor, role):
        run_id = f'{RECOVERY_PREFIX}R6-{role.upper()}'
        path = self.input_path(run_id)
        if path.exists():
            return path
        if role == 'qualified':
            template = ROOT/'docs/experiments/instinct-next-two-cycles/round-2/EVAL-contract.json'
        else:
            template = R4/'R6-DIRECT-INSTINCT-GPU-R4-LONG-R6-contract.json'
        params = load(template)['parameters']
        selection = params['evaluation_selection']
        comparison = read(selection['comparison'])
        comparison.update(id=f'INSTINCT-R4-RECOVERY-R6-{role.upper()}', seed=design['r6_seed'])
        if role != 'qualified':
            recipe = read(design['recipe'])
            comparison.update(candidate_id=recipe['candidate_id'], method=recipe['method'],
                              hypothesis=recipe['hypothesis'])
            selection['method'] = recipe['method']
            if role == 'candidate':
                selection.update(checkpoint=endpoint['checkpoint'], training_plan=endpoint['plan'],
                                 training_analysis=endpoint['analysis'])
            else:
                selection.update(role='baseline_control', checkpoint=design['initial']['checkpoint'],
                                 training_plan=None, training_analysis=None)
                params['training_seed'] = 20260908
        elif selection['checkpoint'] != design['qualified']:
            raise ValueError('Qualified evaluation must use the frozen product checkpoint')
        comparison_path = PROJECT/f'R6-{role}-comparison.json'
        selection['comparison'] = self.write(comparison_path.name, comparison)
        params['experiment_id'] = run_id
        params['r6'].update(run_id=run_id, seed=design['r6_seed'])
        if role == 'candidate':
            params['training_seed'] = design['training_seed']
        contract = self.write(f'{run_id}-contract.json', build('contract', dict(operation='evaluate', parameters=params)))
        self.input(run_id, contract, predecessor, snapshot=self.snapshot, source=self.source,
                   resource_policy=self.resource_policy)
        freeze(f'{self.tag_prefix}-r6-{role}-inputs', f'Freeze independent R6 {role} arm',
               path, Path(contract['path']), comparison_path, self.ledger)
        return path

    def evaluate_formal(self, design, endpoint):
        predecessor = endpoint['next_predecessor']
        r6, models = {}, {}
        for role in ('qualified', 'origin', 'candidate'):
            path = self.r6_input(design, endpoint, predecessor, role)
            evaluation = self.evaluate(path)
            r6[role] = evaluation
            predecessor = evaluation['predecessor']
            run_id = load(path)['run_id']
            report = pin(Path(evaluation['binding']['path']).parent/'report.json')
            model = score_model(dict(run_id=run_id, arm=role, pair=1, report=report))
            model['stop_displacement'] = {k: displacement(report, k) for k in model['endpoints']}
            models[role] = model
        path = self.input_path(RECOVERY_HOLDOUT)
        if not path.exists():
            self.dynamic_input(RECOVERY_HOLDOUT, design['holdout_seed'], dict(
                candidate=endpoint['checkpoint'], control=design['initial']['checkpoint'],
                reference=design['qualified']), predecessor, wall_seconds=1800)
            freeze(self.tag_prefix+'-holdout-inputs', 'Freeze the independent four-layout physical evaluation',
                   path, PROJECT/f'{RECOVERY_HOLDOUT}-contract.json', self.ledger)
        physical = self.evaluate(path)
        settings = load(HISTORICAL/'FREEZE.json')
        result = dict(r6=r6, models=models, holdout=physical, holdout_comparison=compare_holdout(physical['binding']),
                      r6_comparisons={role: compare(settings, [models['candidate']], [models[role]])
                                      for role in ('qualified', 'origin')},
                      physical_mechanisms=mechanisms(physical['binding']))
        return result

    def execute_formal(self):
        design = load(PROJECT/'FORMAL_DESIGN.json')
        self.source = design['source']
        if read(self.source)['source_commit'] != manifest()['source_commit']:
            raise ValueError('Use the exact formal detached source')
        recipe = read(design['recipe'])
        predecessor, initial = design['predecessor'], design['initial']
        segments = []
        for section in design['segments']:
            inputs, plan = self.segment_input(design, section, initial, predecessor)
            row = self.execute_segment(inputs, plan)
            predecessor = row['next_predecessor']
            initial = dict(kind='RESUME_R6_STREAM', checkpoint=row['checkpoint'],
                           source_identity=recipe['source_identity'], method_hashes=bound_method_hashes(plan, recipe))
            segments.append(row)
        endpoint = segments[-1]
        updates = sum(row['counters']['completed_updates'] for row in segments)
        transitions = sum(row['counters']['training_transitions'] for row in segments)
        if (updates != TOTAL_UPDATES or transitions != TOTAL_UPDATES*512
                or endpoint['plan']['target_update'] != design['fixed_endpoint']):
            raise ValueError('Incomplete frozen formal training volume')
        evaluation = self.evaluate_formal(design, endpoint)
        result = self.write('FORMAL_RESULTS.json', dict(
            training=segments, actual_updates=updates, actual_training_transitions=transitions,
            training_supervised_seconds=sum(row['wall_seconds'] for row in segments),
            start=design['initial']['checkpoint'], end=endpoint['checkpoint'],
            design=pin(PROJECT/'FORMAL_DESIGN.json'), ledger=pin(self.ledger), **evaluation))
        freeze(self.tag_prefix+'-formal-results', 'Record 1024-update recovery stream and independent evaluation',
               Path(result['path']), self.ledger)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    args = parser.parse_args()
    guard()
    study = FormalRecovery()
    print(json.dumps(study.prepare_formal() if args.action == 'prepare' else study.execute_formal()), flush=True)


if __name__ == '__main__':
    main()
