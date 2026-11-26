"""Evaluate the next front-clearance branch without modifying the closed study ledger."""

import argparse
import json
from pathlib import Path

from instinctlab.control.escape_recovery import FRONT_CLEARANCE_25_VERSION, REAR_CLEARANCE_25_VERSION
from tools.r6_direct.evidence import read
from tools.r6_direct.import_legacy import pin
from tools.r6_direct.instinct_cycle import load
from tools.r6_direct.instinct_cycle_finish import freeze, guard
from tools.r6_direct.instinct_long import GIB
from tools.r6_direct.instinct_u1792_study import ROOT_DOCS, TwoLongStudy
from tools.r6_direct.next_long_eval_resources import forecast
from tools.r6_direct.post_r4_audit import verify_training
from tools.r6_direct.resource_policy import FRONT_STUDY_SCHEMA, OUTPUT_CAP
from tools.r6_direct.storage import SYSTEM_FLOOR
from training_workloads.unified_h1.dynamic_assay import FRONT_STUDY_PREFIX


class FrontStudy(TwoLongStudy):
    project = ROOT_DOCS/'study-front'
    previous_study = ROOT_DOCS/'study'
    tag_prefix = 'instinct-post-u1792-two-long-front-study'
    run_prefix = FRONT_STUDY_PREFIX
    resource_schema = FRONT_STUDY_SCHEMA
    training_labels = r'(SHORT|FINE)-[A-Z0-9-]+|FORMAL-C'

    def previous_evaluations(self):
        return load(self.previous_study/'EVALUATIONS.json')

    def planned_caps(self, free):
        prior_pin = self.previous_evaluations()['ledger']
        previous = read(prior_pin)
        caps = {}
        for key, total in previous['caps'].items():
            reserved = sum(row['resources'].get(key, 0) for row in previous['new_reservations'])
            caps[key] = total - reserved - sum(previous[k][key] for k in (
                'historical_charged','new_used','protected_confirmation_delivery_repair'))
        caps[OUTPUT_CAP] = min(caps[OUTPUT_CAP], free-SYSTEM_FLOOR)
        if any(value <= 0 for value in caps.values()):
            raise ValueError('The parent task has no remaining allocation for a required resource')
        return caps

    def predecessor(self):
        rows = [row for row in load(self.ledger)['new_reservations'] if row['id'].startswith(self.run_prefix)]
        return self.settle(rows[-1]['id']) if rows else read(self.previous_evaluations()['r6']['b'])['predecessor']

    def formal_evaluation(self, protocol_path):
        guard()
        protocol_pin = pin(protocol_path)
        protocol = read(protocol_pin)
        old_evaluations = self.previous_evaluations()
        old = read(old_evaluations['assignments'])
        original_protocol = read(old['protocol'])
        if (protocol['r6'] != original_protocol['r6'] or protocol['physical']['seed'] == original_protocol['physical']['seed']
                or protocol['qualified'] != old['roles']['qualified']['checkpoint']):
            raise ValueError('The comparison requires fresh physical cases and the preassigned matching R6 control stream')
        pin_c = pin(self.project/'FORMAL-C/RESULTS.json')
        result = read(pin_c)
        verification = verify_training(result)
        decision = read(verification['decision'])
        end = result['segments'][-1]
        prior = old['roles']['origin']
        if (decision['independent_protocol'] != protocol_pin or verification['counts']['transitions'] != 524288
                or result['execution'] != FRONT_CLEARANCE_25_VERSION or prior['execution'] != REAR_CLEARANCE_25_VERSION
                or result['start'] != prior['checkpoint'] or end['plan']['target_update'] != protocol['fixed_endpoint']
                or end['plan']['training_seed'] != protocol['training_seeds']['c']
                or result['segments'][0]['plan']['initialization']['kind'] != self.branch_kind):
            raise ValueError('The closed branch differs from the admitted independent plan')
        roles = dict(qualified=old['roles']['qualified'], prior=prior,
                     origin=dict(prior, execution=FRONT_CLEARANCE_25_VERSION, execution_override=FRONT_CLEARANCE_25_VERSION),
                     c=dict(checkpoint=result['end'], execution=result['execution'], segment=end))
        assignments = dict(protocol=protocol_pin, formal={'c':pin_c}, roles=roles,
                           r6_resources=protocol['r6_resources'], r6_arm_order=['qualified','prior','origin','c'],
                           r6_result_payload_cap_bytes=protocol['r6_result_payload_cap_bytes'])
        path = self.project/'FINAL_ASSIGNMENTS.json'
        existed = path.exists()
        assignment_pin = self.write(path.name, assignments)
        if not existed:
            freeze(self.tag_prefix+'-assignments', 'Lock the front-clearance terminal and matched controls', path)
        resources = read(protocol['r6_resources'])
        resources['output_cap_bytes'] = protocol['r6_result_payload_cap_bytes']
        phases = ([] if (self.project/'HOLDOUT-C.json').exists() else
                  [dict(name='holdout-c',launch_cap_bytes=GIB,retained_bytes=GIB//2)])
        remaining = sum(not (self.project/f'R6-{role}.json').exists() for role in ('ORIGIN', 'C'))
        if not forecast(resources, remaining_phases=phases, remaining_r6_arms=remaining)['fits']:
            raise ValueError('The new physical and R6 arms exceed current capacity')
        names = dict(candidate='c', control='origin', reference='qualified', prior='prior')
        physical = self.write('HOLDOUT-C-SPEC.json', dict(label='HOLDOUT-C', seed=protocol['physical']['seed'], wall_seconds=1800,
            models={key:roles[value]['checkpoint'] for key,value in names.items()},
            arm_options={key:roles[value]['execution'] for key,value in names.items()},assignments=assignment_pin))
        closed = self.physical(Path(physical['path']))
        r6 = dict(qualified=old_evaluations['r6']['qualified'], prior=old_evaluations['r6']['origin'],
                  origin=self.r6(assignments,'origin'), c=self.r6(assignments,'c'))
        destination = self.project/'EVALUATIONS.json'
        existed = destination.exists()
        output = self.write(destination.name, dict(assignments=assignment_pin, physical={'c':closed}, r6=r6, ledger=pin(self.ledger)))
        if not existed:
            freeze(self.tag_prefix+'-independent-closed', 'Record the new physical and changed-execution R6 arms', destination, self.ledger)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('physical','prepare-training','train','formal-evaluation'))
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    study = FrontStudy()
    print(json.dumps(getattr(study,args.operation.replace('-','_'))(args.spec.resolve())), flush=True)


if __name__ == '__main__':
    main()
