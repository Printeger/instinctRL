"""One bounded, read-only resource observation; never launch CUDA or simulation."""
import json
import time
from pathlib import Path

from tools.r5_w_commander import BASE, CONTROL, idle, pin
from rlflow.store import Store
from training_workloads.unified_h1.config import resource_binding_pins
from training_workloads.scale_003.resources import collect_resource_sample


def main():
    idle(Store(CONTROL))
    output = BASE / 'resource-admission-v2' / 'observation.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    pins = resource_binding_pins()
    start = time.monotonic()
    with output.open('x') as stream:
        for ordinal in range(61):
            if ordinal:
                time.sleep(max(0, start + ordinal * 5 - time.monotonic()))
            idle(Store(CONTROL))
            if (BASE / 'batch/STOP').exists():
                raise RuntimeError('STOP')
            sample = collect_resource_sample(resource_pins=pins)
            stream.write(json.dumps({'ordinal': ordinal, 'sample': sample}) + '\n')
            stream.flush()
    print(json.dumps({'observation': pin(output), 'elapsed_seconds': time.monotonic()-start,
                      'samples': 61, 'live_starts': 0, 'simulator_initializations': 0}))


if __name__ == '__main__':
    main()
