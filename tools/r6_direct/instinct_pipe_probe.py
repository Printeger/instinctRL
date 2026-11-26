"""Prepare and run a bounded, fixed-u1792 observation diagnostic."""

import argparse
import json
from pathlib import Path
import secrets
import shutil

from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.instinct_cycle_finish import freeze, guard
from tools.r6_direct.instinct_long import GIB, LongStudy
from tools.r6_direct.instinct_post_r4_three import execution_manifest
from tools.r6_direct.prepare import build
from tools.r6_direct.resource_policy import OUTPUT_CAP, PIPE_PROBE_SCHEMA, THIRD_CAPS
from tools.r6_direct.source import ROOT as SNAPSHOT
from tools.r6_direct.storage import CONTROL_ALLOWANCE, SYSTEM_FLOOR, SYSTEM_ROOT, SYSTEM_STORAGE_SCHEMA
from training_workloads.unified_h1.pipe_authority_probe import PIPE_CLEARANCE_RUN, PIPE_PROBE_RUN, PIPE_SWITCH_RUN


PROJECT = ROOT/'docs/experiments/instinct-post-u1792-two-long'
PREVIOUS = PROJECT.with_name('instinct-post-r4-next-long')
TAG = 'instinct-post-u1792-two-long-pipe-probe'


class PipeProbe(LongStudy):
    def __init__(self, *, switch=False, clearance=False):
        self.case = 'rear-clearance' if clearance else 'rear-switch' if switch else None
        self.docs = PROJECT/self.case if self.case else PROJECT
        self.ledger = self.docs/'RESOURCE_LEDGER.json'
        self.snapshot = SNAPSHOT
        self.eval = PIPE_CLEARANCE_RUN if clearance else PIPE_SWITCH_RUN if switch else PIPE_PROBE_RUN
        self.tag_prefix = 'instinct-post-u1792-two-long-'+self.case if self.case else TAG
        if SNAPSHOT == ROOT:
            raise ValueError('Run the probe from a committed detached snapshot')

    def prepare(self):
        from instinctlab.control.escape_recovery import FALLBACK_25_VERSION, REAR_CLEARANCE_25_VERSION, REAR_SWITCH_25_VERSION

        guard()
        path = self.input_path(self.eval)
        if path.exists():
            if load(path)['snapshot'] != str(SNAPSHOT):
                raise ValueError('Use the already frozen probe snapshot')
            return path
        free = shutil.disk_usage(SYSTEM_ROOT).free
        if free < SYSTEM_FLOOR + GIB + CONTROL_ALLOWANCE:
            raise ValueError('The probe payload does not fit above the system floor')
        source = execution_manifest()
        source_pin = self.write('source-'+source['source_commit'][:12]+'.json',source)
        accepted = load(PREVIOUS/'FORMAL-RETRY-01/RESULTS.json')
        caps = dict(allocated_gpu_seconds=2400, live_wall_seconds=2400, queue_seconds=300,
                    live_starts=1, simulator_initializations=1, training_transitions=1,
                    nontraining_transitions=23040 if self.case else 11520, training_checkpoint_writes=1,
                    major_scientific_candidates=1, **{OUTPUT_CAP:GIB+CONTROL_ALLOWANCE})
        authority_path = self.docs/'RESOURCE_POLICY.json'
        authority = save(authority_path, dict(schema_version=PIPE_PROBE_SCHEMA,
            user_authority=pin(self.docs/'DESIGN.md' if self.case else PROJECT/'PLAN.md'),
            caps=caps, elapsed_seconds=86400))
        storage = self.write('STORAGE.json', dict(schema_version=SYSTEM_STORAGE_SCHEMA, root=str(SYSTEM_ROOT),
                                                new_volume_allocation_bytes=0))
        zero = dict.fromkeys(THIRD_CAPS, 0)
        self.write(self.ledger.name, dict(resource_policy=authority, caps=caps, historical_charged=zero,
            new_used=zero, protected_confirmation_delivery_repair=zero, status='FROZEN',
            new_reservations=[dict(id='ROUND-SYSTEM',status='UNRELEASED',resources={OUTPUT_CAP:0}, storage_plan=storage)]))
        parameters = dict(experiment_id=self.eval, training_seed=20260908,
            models={'candidate':accepted['end']}, arm_options={'candidate':FALLBACK_25_VERSION},
            seed=1+secrets.randbelow(2**31-16), wall_seconds=1500, output_cap_bytes=GIB)
        predecessor = load(PREVIOUS/'R6-END.json')['predecessor']
        if self.case:
            from tools.r6_direct.evidence import read

            probe = (PROJECT/'rear-switch/PIPE-SWITCH-01.json' if self.case == 'rear-clearance'
                     else PROJECT/'PIPE-PROBE-01.json')
            probe_inputs = load(PROJECT/f'{PIPE_PROBE_RUN}-inputs.json')
            predecessor = load(probe)['predecessor']
            version = REAR_CLEARANCE_25_VERSION if self.case == 'rear-clearance' else REAR_SWITCH_25_VERSION
            parameters.update(seed=read(probe_inputs['contract'])['parameters']['seed'],
                              models=dict(candidate=accepted['end'],reference=accepted['end']),
                              arm_options=dict(candidate=version, reference=FALLBACK_25_VERSION))
        contract = self.write(self.eval+'-contract.json',build('contract',dict(operation='dynamic',parameters=parameters)))
        self.input(self.eval,contract,predecessor,snapshot=SNAPSHOT,source=source_pin,resource_policy=authority)
        freeze(self.tag_prefix+'-inputs','Freeze the fixed-u1792 mirror pipe experiment',path,self.ledger,authority_path,
               *(Path(item['path']) for item in (source_pin,storage,contract)))
        return path

    def run(self):
        path = self.prepare()
        result_path = self.docs/(self.eval.split('-TWO-', 1)[1]+'.json')
        if result_path.exists():
            return pin(result_path)
        result = self.evaluate(path)
        output = self.write(result_path.name,result)
        freeze(self.tag_prefix+'-closed','Record the closed u1792 mirror pipe experiment',result_path,self.ledger)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run'))
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument('--switch',action='store_true')
    variants.add_argument('--clearance',action='store_true')
    args = parser.parse_args()
    result = getattr(PipeProbe(switch=args.switch, clearance=args.clearance),args.operation)()
    print(json.dumps(pin(result) if isinstance(result,Path) else result))


if __name__=='__main__':
    main()
