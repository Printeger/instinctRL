"""Fresh R6 pre-AppLauncher observation; queue receipts cannot be reused."""
import os
from pathlib import Path
import time

from .resource_admission import POLICY as WINDOW_POLICY,validate_window
from .config import sha256_file
from training_workloads.scale_003.resources import collect_resource_sample
from training_workloads.scale_003.persistence import atomic_json_replace

POLICY=dict(schema_version='r6-prelaunch-admission/1',window_policy=WINDOW_POLICY,
    fresh_window_required=True,maximum_collection_seconds=180,
    maximum_final_sample_age_seconds=8,queue_observation_is_admission=False)


def run_gate(output_root,*,mode,attempt_id,authorization_sha256,resource_pins,stop_paths=(),
             sample_provider=None,sleeper=time.sleep,clock_ns=time.monotonic_ns):
    """No accepted window input: always collect each sample inside this call.

    Existing fixed resource-query tool has its own15second per-query timeout;
    this gate also passes its remaining deadline and STOP guard into each query.
    The outer live watchdog remains responsible for a frozen/stalled process.
    """
    if mode not in ('research-r6-evaluation','research-r6-training'):raise PermissionError('R6 prelaunch mode')
    executable=Path(resource_pins['nvidia_smi_path'])
    if executable.is_symlink() or sha256_file(executable)!=resource_pins['nvidia_smi_sha256']:
        raise PermissionError('R6 prelaunch executable changed')
    samples=[];started=clock_ns()
    binding=dict(schema_version=POLICY['schema_version'],policy=POLICY,mode=mode,attempt_id=attempt_id,
        authorization_sha256=authorization_sha256,pid=os.getpid(),phase='ACTUAL_PRELAUNCH_FRESH_WINDOW',
        executable=dict(path=str(executable),sha256=resource_pins['nvidia_smi_sha256']),started_monotonic_ns=started,
        queue_receipt_used=False)
    def guard():
        if any(Path(p).exists() for p in stop_paths):raise RuntimeError('STOP_DURING_R6_PRELAUNCH')
        if not 0<=clock_ns()-started<=POLICY['maximum_collection_seconds']*10**9:
            raise TimeoutError('R6_PRELAUNCH_COLLECTION_DEADLINE')
    def window():
        stamps=[s.get('monotonic_ns') if isinstance(s,dict) else None for s in samples]
        span=0 if not stamps else stamps[-1]-stamps[0] if type(stamps[0]) is int and type(stamps[-1]) is int else None
        return dict(sample_count=len(samples),span_ns=span,samples=samples)
    try:
        for ordinal in range(WINDOW_POLICY['sample_count']):
            guard()
            if ordinal:sleeper(5);guard()
            sample=(sample_provider() if sample_provider is not None else collect_resource_sample(resource_pins=resource_pins,
                deadline_monotonic_ns=started+POLICY['maximum_collection_seconds']*10**9,before_query=guard))
            samples.append(sample)
            atomic_json_replace(output_root,'prelaunch_resource_observations.json',dict(binding,status='COLLECTING_NOT_ADMITTED',samples=samples))
            guard()
            stamp=sample.get('monotonic_ns')
            if (type(stamp)is not int or stamp<started or not 0<=clock_ns()-stamp<=POLICY['maximum_final_sample_age_seconds']*10**9):
                raise ValueError('R6_PRELAUNCH_STALE_OR_FUTURE_SAMPLE')
        result=validate_window(window())
        guard()
        if clock_ns()-samples[-1]['monotonic_ns']>POLICY['maximum_final_sample_age_seconds']*10**9:
            raise ValueError('R6_PRELAUNCH_FINAL_SAMPLE_STALE')
        record=dict(binding,status='PASS',window=result['window'],statistics=result['statistics'],finished_monotonic_ns=clock_ns())
        atomic_json_replace(output_root,'prelaunch_resource_gate.json',record)
        return record
    except BaseException as error:
        atomic_json_replace(output_root,'prelaunch_resource_gate_failure.json',dict(binding,status='FAIL',
            observed_window=window(),completed_samples=len(samples),error_type=type(error).__name__,message=str(error)))
        raise


def selected(authorization,resource_pins):
    from functools import partial
    if authorization['mode']=='research-r6-training':
        from .robustness_training_contract import validate_parameters
    else:
        from .robustness_contract import validate_parameters
    p=validate_parameters(authorization['mode'],authorization['parameters'])
    base=Path(p['master_authority']['path']).parent
    root=base.parents[2]
    stops=(base/'STOP',base/'batch/STOP',root/'STOP',root/'coordination/STOP',
        root/'coordination/commander/STOP',root/'coordination/executor/STOP')
    return partial(run_gate,resource_pins=resource_pins,stop_paths=stops)
