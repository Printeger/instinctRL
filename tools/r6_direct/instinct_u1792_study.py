"""Run two independent training branches from the one audited u1792 checkpoint."""

import argparse
import json
from pathlib import Path

from tools.r6_direct.evidence import read
from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import load
from tools.r6_direct.instinct_cycle_finish import freeze, guard
from tools.r6_direct.instinct_long import GIB
from tools.r6_direct.instinct_next_long import NextLongStudy
from tools.r6_direct.next_long_eval_resources import forecast
from tools.r6_direct.post_r4_audit import verify_training
from tools.r6_direct.resource_policy import U1792_STUDY_SCHEMA
from training_workloads.unified_h1.dynamic_assay import U1792_STUDY_PREFIX
from training_workloads.unified_h1.robustness_training_plan import U1792_LOCAL_FORK


ROOT_DOCS = ROOT/'docs/experiments/instinct-post-u1792-two-long'
PRIOR = ROOT_DOCS.with_name('instinct-post-r4-next-long')


class TwoLongStudy(NextLongStudy):
    project = ROOT_DOCS/'study'
    tag_prefix = 'instinct-post-u1792-two-long-study'
    run_prefix = U1792_STUDY_PREFIX
    resource_schema = U1792_STUDY_SCHEMA
    origin_result = PRIOR/'FORMAL-RETRY-01/RESULTS.json'
    origin_update = 1792
    branch_kind = U1792_LOCAL_FORK
    training_labels = r'(SHORT|FINE)-[A-Z0-9-]+|FORMAL-(A|B)'

    def predecessor(self):
        rows = [row for row in load(self.ledger)['new_reservations'] if row['id'].startswith(self.run_prefix)]
        if rows:
            return self.settle(rows[-1]['id'])
        return load(ROOT_DOCS/'rear-clearance/PIPE-CLEARANCE-01.json')['predecessor']

    def formal_evaluation(self, protocol_path):
        from instinctlab.control.escape_recovery import REAR_CLEARANCE_25_VERSION

        guard()
        protocol_pin = pin(protocol_path)
        protocol = read(protocol_pin)
        prior = load(self.origin_result)
        formal = {arm: pin(self.project/f'FORMAL-{arm.upper()}/RESULTS.json') for arm in ('a', 'b')}
        execution = read(formal['a'])['execution']
        if execution not in (prior['execution'], REAR_CLEARANCE_25_VERSION):
            raise ValueError('Unknown shared execution for these branches')
        seed_set = set()
        roles = {
            'qualified': dict(checkpoint=protocol['qualified'], execution=None),
            'origin': dict(checkpoint=prior['end'], execution=execution, segment=prior['segments'][-1]),
        }
        if execution != prior['execution']:
            roles['prior'] = dict(roles['origin'], execution=prior['execution'])
            roles['origin']['execution_override'] = execution
        for arm, result_pin in formal.items():
            result = read(result_pin)
            verification = verify_training(result)
            decision = read(verification['decision'])
            segment = result['segments'][-1]
            first = result['segments'][0]['plan']
            seed = protocol['training_seeds'][arm]
            if (decision['independent_protocol'] != protocol_pin or verification['counts']['transitions'] != 524288
                    or first['training_seed'] != seed or seed in seed_set
                    or first['initialization']['kind'] != self.branch_kind
                    or segment['plan']['target_update'] != protocol['fixed_endpoint']
                    or result['start'] != prior['end'] or result['execution'] != execution):
                raise ValueError('The frozen formal branch or common execution does not match')
            seed_set.add(seed)
            roles[arm] = dict(checkpoint=result['end'], execution=result['execution'], segment=segment)
        if roles['a']['checkpoint']['sha256'] == roles['b']['checkpoint']['sha256']:
            raise ValueError('Matched independent seeds produced identical terminal weights')
        arms = ['qualified', *(['prior'] if 'prior' in roles else []), 'origin', 'a', 'b']
        assignments_path = self.project/'FINAL_ASSIGNMENTS.json'
        existed = assignments_path.exists()
        assignments = dict(protocol=protocol_pin, formal=formal, roles=roles,
                           r6_resources=protocol['r6_resources'], r6_arm_order=arms,
                           r6_result_payload_cap_bytes=protocol['r6_result_payload_cap_bytes'])
        self.write(assignments_path.name, assignments)
        if not existed:
            freeze(self.tag_prefix+'-assignments', 'Lock both terminal weights before the shared holdout', assignments_path)
        pending = []
        for arm in formal:
            if not (self.project/f'HOLDOUT-{arm.upper()}.json').exists():
                pending.append(dict(name='physical-'+arm, launch_cap_bytes=GIB, retained_bytes=GIB//3))
        count = sum(not (self.project/f'R6-{arm.upper()}.json').exists() for arm in arms)
        resources = read(protocol['r6_resources'])
        resources['output_cap_bytes'] = protocol['r6_result_payload_cap_bytes']
        capacity = forecast(resources, remaining_phases=pending, remaining_r6_arms=count)
        if not capacity['fits']:
            raise RuntimeError(f'The shared holdout series exceeds measured system capacity: {capacity}')
        physical = {}
        for arm in formal:
            names = dict(candidate=arm, control='origin', reference='qualified')
            if 'prior' in roles:
                names['prior'] = 'prior'
            spec = self.write(f'HOLDOUT-{arm.upper()}-SPEC.json', dict(label=f'HOLDOUT-{arm.upper()}',
                seed=protocol['physical']['seed'], wall_seconds=1800,
                models={key:roles[value]['checkpoint'] for key,value in names.items()},
                arm_options={key:roles[value]['execution'] for key,value in names.items()},
                assignments=pin(assignments_path)))
            physical[arm] = self.physical(Path(spec['path']))
        r6 = {arm:self.r6(assignments, arm) for arm in arms}
        path = self.project/'EVALUATIONS.json'
        existed = path.exists()
        result = self.write(path.name, dict(assignments=pin(assignments_path), physical=physical,
                                          r6=r6, ledger=pin(self.ledger)))
        if not existed:
            freeze(self.tag_prefix+'-independent-closed', 'Record both endpoints and the shared independent evaluation', path, self.ledger)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('physical', 'prepare-training', 'train', 'formal-evaluation'))
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    study = TwoLongStudy()
    print(json.dumps(getattr(study, args.operation.replace('-', '_'))(args.spec.resolve())), flush=True)


if __name__ == '__main__':
    main()
