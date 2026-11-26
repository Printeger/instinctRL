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
from training_workloads.unified_h1.pipe_authority_probe import PIPE_PROBE_RUN


PROJECT = ROOT/'docs/experiments/instinct-post-u1792-two-long'
PREVIOUS = PROJECT.with_name('instinct-post-r4-next-long')
TAG = 'instinct-post-u1792-two-long-pipe-probe'


class PipeProbe(LongStudy):
    def __init__(self):
        self.docs = PROJECT
        self.ledger = PROJECT/'RESOURCE_LEDGER.json'
        self.snapshot = SNAPSHOT
        if SNAPSHOT == ROOT:
            raise ValueError('Run the probe from a committed detached snapshot')

    def prepare(self):
        from instinctlab.control.escape_recovery import FALLBACK_25_VERSION

        guard()
        path = self.input_path(PIPE_PROBE_RUN)
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
                    nontraining_transitions=11520, training_checkpoint_writes=1,
                    major_scientific_candidates=1, **{OUTPUT_CAP:GIB+CONTROL_ALLOWANCE})
        authority_path = PROJECT/'RESOURCE_POLICY.json'
        authority = save(authority_path, dict(schema_version=PIPE_PROBE_SCHEMA,
            user_authority=pin(PROJECT/'PLAN.md'), caps=caps, elapsed_seconds=86400))
        storage = self.write('STORAGE.json', dict(schema_version=SYSTEM_STORAGE_SCHEMA, root=str(SYSTEM_ROOT),
                                                new_volume_allocation_bytes=0))
        zero = dict.fromkeys(THIRD_CAPS, 0)
        self.write(self.ledger.name, dict(resource_policy=authority, caps=caps, historical_charged=zero,
            new_used=zero, protected_confirmation_delivery_repair=zero, status='FROZEN',
            new_reservations=[dict(id='ROUND-SYSTEM',status='UNRELEASED',resources={OUTPUT_CAP:0}, storage_plan=storage)]))
        parameters = dict(experiment_id=PIPE_PROBE_RUN, training_seed=20260908,
            models={'candidate':accepted['end']}, arm_options={'candidate':FALLBACK_25_VERSION},
            seed=1+secrets.randbelow(2**31-16), wall_seconds=1500, output_cap_bytes=GIB)
        contract = self.write(PIPE_PROBE_RUN+'-contract.json',build('contract',dict(operation='dynamic',parameters=parameters)))
        predecessor = load(PREVIOUS/'R6-END.json')['predecessor']
        self.input(PIPE_PROBE_RUN,contract,predecessor,snapshot=SNAPSHOT,source=source_pin,resource_policy=authority)
        freeze(TAG+'-inputs','Freeze the fixed-u1792 mirror pipe diagnostic',path,self.ledger,authority_path,
               *(Path(item['path']) for item in (source_pin,storage,contract)))
        return path

    def run(self):
        path = self.prepare()
        result_path = PROJECT/'PIPE-PROBE-01.json'
        if result_path.exists():
            return pin(result_path)
        result = self.evaluate(path)
        output = self.write(result_path.name,result)
        freeze(TAG+'-closed','Record the closed u1792 mirror pipe diagnostic',result_path,self.ledger)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run'))
    args = parser.parse_args()
    result = getattr(PipeProbe(),args.operation)()
    print(json.dumps(pin(result) if isinstance(result,Path) else result))


if __name__=='__main__':
    main()
