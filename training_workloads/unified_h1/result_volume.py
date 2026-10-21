"""Exact kernel quota for the otherwise writable R6 result-directory exception."""
import json
import os
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
    binding=validate_volume(pin,base=BASE,maximum=size,image_name='results.ext4',
        expected_target=target,minimum_free_bytes=minimum_free_bytes,allow_cpu_fixture=allow_cpu_fixture)
    if 'layout' in record:
        require(record['layout']=='fixed-results-v1' and record['status']=='MOUNT_CREATED_NOT_ACCEPTED',
                'RESULT_LAYOUT_STATUS')
        payload=record['minimum_free_bytes'];inodes=record['minimum_free_inodes']
        require(type(payload)is int and payload>0 and size-payload==96*1024**2,
                'RESULT_LAYOUT_PAYLOAD')
        require(type(inodes)is int and 1<=inodes<=size//262144,'RESULT_LAYOUT_INODES')
        # Zero is the explicit postrun/inventory mode: full failed volumes must
        # remain readable and accountable after consuming their free space.
        if minimum_free_bytes:
            require(binding['free_bytes']>=max(payload,minimum_free_bytes),'RESULT_LAYOUT_FREE_SPACE')
            require(os.statvfs(target).f_favail>=inodes,'RESULT_LAYOUT_FREE_INODES')
        binding=dict(binding,payload_cap_bytes=payload)
    return binding


def workload_write_cap(parameters, *, expected_target=None, minimum_free_bytes=256*1024**2):
    """Use the measured fixed-layout payload only for an exact dual-volume job.

    Legacy/default layouts retain the old deduction. This is not authority or
    capacity approval; callers still require the reviewed exact experiment.
    """
    cap=parameters['output_cap_bytes']
    if 'native_volume' not in parameters:return cap-3*1024**3
    from .native_volume import validate as validate_native
    validate_native(parameters['native_volume'])
    binding=validate(parameters['result_volume'],output_cap_bytes=cap,
        expected_target=expected_target,minimum_free_bytes=minimum_free_bytes)
    return binding.get('payload_cap_bytes',cap-3*1024**3)
