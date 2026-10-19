from copy import deepcopy
from functools import partial
import json
import pytest
from training_workloads.unified_h1 import resource_admission as r
from training_workloads.scale_003 import resources as legacy


def window(values=None):
    values=values or [3]*13
    samples=[dict(monotonic_ns=i*5*10**9,gpu=dict(legacy.GPU_IDENTITY,memory_used_mib=560,free_memory_mib=11667,utilization_percent=u),compute_apps=[dict(pid=3950,name='/usr/libexec/gnome-remote-desktop-daemon',used_memory_mib=164)]) for i,u in enumerate(values)]
    return dict(sample_count=len(samples),span_ns=samples[-1]['monotonic_ns'],samples=samples)


def test_isolated_spike_preserves_raw_and_legacy():
    w=window([3]*6+[12]+[3]*6);original=deepcopy(w)
    assert r.validate_window(w)['status']=='PASS' and w==original
    old=dict(sample_count=3,span_ns=10*10**9,samples=deepcopy(w['samples'][:3]))
    old['samples'][1]['gpu']['utilization_percent']=6
    with pytest.raises(RuntimeError,match='5 percent'):legacy.validate_coexistence_window(old)


@pytest.mark.parametrize('values',[[6]*13,[3]*8+[11]*5,[3]*12+[21],[3]*6+[12,12]+[3]*5])
def test_sustained_or_peak_rejected(values):
    with pytest.raises(RuntimeError):r.validate_window(window(values))


@pytest.mark.parametrize('key,value',[('utilization_percent',float('nan')),('utilization_percent',True),('utilization_percent',-1),('memory_used_mib',1100),('free_memory_mib',9000),('name','other'),('index',False)])
def test_resource_mutations_rejected(key,value):
    w=window();w['samples'][5]['gpu'][key]=value
    with pytest.raises((ValueError,MemoryError,RuntimeError)):r.validate_window(w)


@pytest.mark.parametrize('mutation',['count','short','gap','repeat','span','unknown','negative_app','many_apps'])
def test_window_and_process_mutations(mutation):
    w=window()
    if mutation=='count':w['sample_count']=12
    if mutation=='short':w['samples']=w['samples'][:-1]
    if mutation=='gap':w['samples'][6]['monotonic_ns']+=4*10**9
    if mutation=='repeat':w['samples'][6]['monotonic_ns']=w['samples'][5]['monotonic_ns']
    if mutation=='span':w['span_ns']-=1
    if mutation=='unknown':w['samples'][6]['compute_apps'][0]['name']='python'
    if mutation=='negative_app':w['samples'][6]['compute_apps'][0]['used_memory_mib']=-1
    if mutation=='many_apps':w['samples'][6]['compute_apps']*=2
    with pytest.raises((ValueError,RuntimeError)):r.validate_window(w)


def test_failure_record_and_no_launcher(tmp_path):
    w=window([6]*13);calls=[]
    gate=partial(r.run_gate,resource_pins={},collector=lambda:w)
    with pytest.raises(RuntimeError):
        legacy.start_after_coexistence_gate(tmp_path,mode='research-formal-evaluation',attempt_id='test',authorization_sha256='a'*64,gate_runner=gate,launcher_factory=lambda _:calls.append(True))
    failure=json.loads((tmp_path/'prelaunch_resource_gate_failure.json').read_text())
    assert failure['observed_window']==w and failure['policy']==r.POLICY and not calls


def test_collection_failure_persisted(tmp_path):
    def fail():raise OSError('query failed')
    with pytest.raises(OSError):r.run_gate(tmp_path,mode='x',attempt_id='y',authorization_sha256='a'*64,resource_pins={},collector=fail)
    assert json.loads((tmp_path/'prelaunch_resource_gate_failure.json').read_text())['status']=='FAIL'


def test_selection_does_not_change_old_entries():
    sentinel=object()
    assert r.select_gate({'parameters':{'experiment_id':'R5-W-L1-INDEPENDENT-RECOVERY1'}},{},sentinel) is sentinel
    with pytest.raises((PermissionError,KeyError)):
        r.select_gate({'mode':'research-formal-training','parameters':{'experiment_id':r.EXPERIMENT}},{},sentinel)


def test_partial_query_failure_is_durable(tmp_path,monkeypatch):
    samples=window()['samples'];seen=[]
    def query(**kwargs):
        if len(seen)==2:raise OSError('third query failed')
        value=samples[len(seen)];seen.append(value);return value
    original=r.collect_window
    monkeypatch.setattr(r,'collect_resource_sample',query)
    monkeypatch.setattr(r,'collect_window',lambda pins,**kwargs:original(pins,sleeper=lambda _:None,**kwargs))
    with pytest.raises(OSError):r.run_gate(tmp_path,mode='x',attempt_id='y',authorization_sha256='a'*64,resource_pins={})
    failure=json.loads((tmp_path/'prelaunch_resource_gate_failure.json').read_text())
    journal=json.loads((tmp_path/'prelaunch_resource_observations.json').read_text())
    assert failure['observed_window']['samples']==seen==journal['samples']
    assert failure['observed_window']['complete'] is False
