"""Exact kernel quota for the otherwise writable R6 result-directory exception."""
import json
from pathlib import Path
from .native_volume import validate_volume,require,MAXIMUM
from .research_contract import validate_pin

BASE=Path('/var/lib/instinctrl-r6-results')
ATTEMPTS=Path('/home/mint/rl_dev/instinctRL-dev/coordination/research/attempts')
CONTROL_ALLOWANCE=64*1024**2


def validate(pin, *, output_cap_bytes, expected_target=None, minimum_free_bytes=256*1024**2,
             allow_cpu_fixture=False):
    require(type(output_cap_bytes)is int and MAXIMUM+CONTROL_ALLOWANCE<output_cap_bytes<=128*1024**3,'RESULT_JOB_CAP')
    record=json.loads(validate_pin(pin).read_text())
    target=Path(record['target']);size=record['maximum_backing_bytes']
    require(target.is_absolute() and target.resolve()==target and target.name=='results'
            and target.parent.parent==ATTEMPTS and target.parent.name.startswith('R6-'),'RESULT_TARGET')
    require(expected_target is None or target==Path(expected_target),'RESULT_ATTEMPT_BINDING')
    require(type(size)is int and 16*1024**2<=size==output_cap_bytes-MAXIMUM-CONTROL_ALLOWANCE
            and size%4096==0,'RESULT_AGGREGATE_CAP')
    return validate_volume(pin,base=BASE,maximum=size,image_name='results.ext4',
        expected_target=target,minimum_free_bytes=minimum_free_bytes,allow_cpu_fixture=allow_cpu_fixture)
