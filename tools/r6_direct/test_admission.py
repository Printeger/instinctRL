import copy
import json
from pathlib import Path
import pytest
from tools.r6_direct.admission import CYCLE_POLICY, POLICY
from training_workloads.unified_h1.resource_admission import validate_window


def observed():
    from training_workloads.scale_003.resources import GPU_IDENTITY
    samples=[dict(monotonic_ns=i*5*10**9,
        gpu=dict(GPU_IDENTITY,memory_used_mib=1037,free_memory_mib=GPU_IDENTITY['memory_total_mib']-1037,utilization_percent=3),
        compute_apps=[dict(pid=1,name='/usr/libexec/gnome-remote-desktop-daemon',used_memory_mib=164),
                      dict(pid=2,name='/usr/share/rustdesk/rustdesk',used_memory_mib=218)]) for i in range(13)]
    return dict(samples=samples,sample_count=13,span_ns=60*10**9)



def test_remote_desktop_can_coexist_without_changing_old_policy():
    w=observed(); original=copy.deepcopy(w)
    with pytest.raises(MemoryError):validate_window(w)
    assert validate_window(w,policy=POLICY)['status']=='PASS'
    assert w==original


def test_cycle_remote_desktop_allows_two_known_rustdesk_instances_only():
    window=observed()
    for sample in window['samples']:
        sample['compute_apps'].append(dict(pid=3,name='/usr/share/rustdesk/rustdesk',used_memory_mib=381))
    with pytest.raises(RuntimeError, match='unexpected compute applications'):
        validate_window(window,policy=POLICY)
    assert validate_window(window,policy=CYCLE_POLICY)['status']=='PASS'
    window['samples'][0]['compute_apps'].append(dict(pid=4,name='rustdesk',used_memory_mib=1))
    with pytest.raises(RuntimeError, match='instances'):
        validate_window(window,policy=CYCLE_POLICY)


@pytest.mark.parametrize('fault',['memory','compute','load'])
def test_real_resource_conflicts_remain_rejected(fault):
    w=observed();s=w['samples'][0]
    if fault=='memory':s['gpu'].update(free_memory_mib=9000,memory_used_mib=s['gpu']['memory_total_mib']-9000)
    if fault=='compute':s['compute_apps'][1]['name']='/bin/unrelated-training'
    if fault=='load':s['gpu']['utilization_percent']=99
    with pytest.raises((MemoryError,RuntimeError)):validate_window(w,policy=POLICY)
