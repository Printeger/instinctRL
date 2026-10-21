"""Explicit W confirmation resource protocol; legacy SCALE admission is unchanged."""
from pathlib import Path
import os
import time
from statistics import mean, median

from training_workloads.scale_003.resources import collect_resource_sample, GPU_IDENTITY
from training_workloads.scale_003.persistence import atomic_json_replace

EXPERIMENT = 'R5-W-L1-INDEPENDENT-RESOURCE-V2'
# These are scheduling limits, not flight-safety or scientific scoring limits.
POLICY = dict(schema_version='r5-w-resource-admission/2', sample_count=13,
              minimum_span_seconds=60, minimum_gap_seconds=4, maximum_gap_seconds=8,
              mean_utilization_max=8, median_utilization_max=5, peak_utilization_max=20,
              elevated_utilization_threshold=10, elevated_samples_max=4,
              consecutive_elevated_max=1,
              memory_used_max_mib=1024, memory_free_min_mib=10240,
              allowed_app='gnome-remote-desktop-daemon', allowed_app_max_mib=256)


def validate_window(window):
    if not isinstance(window, dict) or set(window) != {'sample_count','span_ns','samples'}:
        raise ValueError('resource window schema')
    samples = window['samples']
    if type(window['sample_count']) is not int or window['sample_count'] != POLICY['sample_count'] or not isinstance(samples,list) or len(samples) != POLICY['sample_count']:
        raise ValueError('resource window count')
    prior = None
    utilization = []
    for sample in samples:
        if not isinstance(sample,dict) or set(sample) != {'monotonic_ns','gpu','compute_apps'}:
            raise ValueError('resource sample schema')
        stamp = sample['monotonic_ns']
        if type(stamp) is not int or stamp < 0:
            raise ValueError('resource timestamp')
        if prior is not None and not POLICY['minimum_gap_seconds']*10**9 <= stamp-prior <= POLICY['maximum_gap_seconds']*10**9:
            raise ValueError('resource sample gap')
        prior = stamp
        gpu = sample['gpu']
        if not isinstance(gpu,dict) or set(gpu) != set(GPU_IDENTITY)|{'memory_used_mib','free_memory_mib','utilization_percent'} or any(type(gpu[k]) is not type(v) or gpu[k] != v for k,v in GPU_IDENTITY.items()):
            raise ValueError('GPU identity')
        for k in ('memory_used_mib','free_memory_mib','utilization_percent'):
            if type(gpu[k]) is not int or gpu[k] < 0:
                raise ValueError('resource numeric value')
        if gpu['memory_used_mib']+gpu['free_memory_mib'] != gpu['memory_total_mib']:
            raise ValueError('memory accounting')
        if gpu['memory_used_mib'] > POLICY['memory_used_max_mib'] or gpu['free_memory_mib'] < POLICY['memory_free_min_mib']:
            raise MemoryError('resource memory limit')
        u = gpu['utilization_percent']
        if u > POLICY['peak_utilization_max']:
            raise RuntimeError('resource utilization peak')
        utilization.append(u)
        apps = sample['compute_apps']
        if not isinstance(apps,list) or len(apps)>1:
            raise RuntimeError('unexpected compute applications')
        for app in apps:
            if not isinstance(app,dict) or set(app) != {'pid','name','used_memory_mib'} or type(app['pid']) is not int or app['pid']<=0 or type(app['name']) is not str or Path(app['name']).name != POLICY['allowed_app'] or type(app['used_memory_mib']) is not int or not 0<=app['used_memory_mib']<=POLICY['allowed_app_max_mib']:
                raise RuntimeError('unexpected compute application')
    if type(window['span_ns']) is not int or window['span_ns'] != samples[-1]['monotonic_ns']-samples[0]['monotonic_ns'] or window['span_ns'] < POLICY['minimum_span_seconds']*10**9:
        raise ValueError('resource window span')
    elevated = sum(u>POLICY['elevated_utilization_threshold'] for u in utilization)
    adjacent = any(a>POLICY['elevated_utilization_threshold'] and b>POLICY['elevated_utilization_threshold'] for a,b in zip(utilization,utilization[1:]))
    if mean(utilization)>POLICY['mean_utilization_max'] or median(utilization)>POLICY['median_utilization_max'] or elevated>POLICY['elevated_samples_max'] or adjacent:
        raise RuntimeError('resource sustained utilization')
    return dict(status='PASS', policy=POLICY, window=window,
                statistics=dict(mean=mean(utilization),median=median(utilization),peak=max(utilization),elevated_samples=elevated))


def collect_window(resource_pins, sample_provider=None, sleeper=time.sleep, samples=None, on_sample=None):
    sample_provider = sample_provider or (lambda: collect_resource_sample(resource_pins=resource_pins))
    samples = [] if samples is None else samples
    for ordinal in range(POLICY['sample_count']):
        if ordinal:
            sleeper(5)
        samples.append(sample_provider())
        if on_sample:
            on_sample(samples)
    return dict(sample_count=len(samples),span_ns=samples[-1]['monotonic_ns']-samples[0]['monotonic_ns'],samples=samples)


def run_gate(output_root, *, mode, attempt_id, authorization_sha256, resource_pins, collector=None):
    window = None
    partial_samples = []
    identity = dict(mode=mode,attempt_id=attempt_id,authorization_sha256=authorization_sha256,pid=os.getpid())
    try:
        def persist(samples):
            atomic_json_replace(output_root,'prelaunch_resource_observations.json',
                dict(status='COLLECTING_NOT_ADMITTED',policy=POLICY,samples=samples,**identity))
        window = collector() if collector else collect_window(resource_pins,samples=partial_samples,on_sample=persist)
        result = dict(validate_window(window), **identity)
        atomic_json_replace(output_root,'prelaunch_resource_gate.json',result)
        return result
    except BaseException as exc:
        if window is None and partial_samples:
            window = dict(sample_count=len(partial_samples),samples=partial_samples,
                          span_ns=partial_samples[-1]['monotonic_ns']-partial_samples[0]['monotonic_ns'],complete=False)
        atomic_json_replace(output_root,'prelaunch_resource_gate_failure.json',
            dict(schema_version=POLICY['schema_version'],status='FAIL',policy=POLICY,
                 observed_window=window,completed_samples=len(partial_samples),error_type=type(exc).__name__,message=str(exc),**identity))
        raise


def select_gate(authorization, resource_pins, legacy_gate):
    if authorization.get('mode') in ('research-r6-evaluation','research-r6-training'):
        from .robustness_prelaunch import selected
        return selected(authorization,resource_pins)
    p = authorization.get('parameters',{})
    if p.get('experiment_id') != EXPERIMENT:
        return legacy_gate
    from .w_independent import validate_parameters
    from .checkpoint_compatibility import pinned_json
    validate_parameters(authorization['mode'],p)
    card = pinned_json(p['experiment_card'])
    if card.get('resource_admission_policy') != POLICY:
        raise PermissionError('resource policy not bound in exact experiment card')
    from functools import partial
    return partial(run_gate,resource_pins=resource_pins)
