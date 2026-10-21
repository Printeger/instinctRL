"""Synthetic full-denominator scores; publication and closed-chain binding are explicit fixtures."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools import r6_final_collection as m
from tools.r6_recovery import file_pin,RecoveryError
from tools.test_r6_baseline_validation import report_fixture

@pytest.fixture(params=['baseline','confirmation'])
def batch(tmp_path,monkeypatch,request):
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return file_pin(p)
    protocol=json.loads(Path('docs/experiments/r6-robustness-auto/freeze-candidate-01/protocol.json').read_text())
    pp=write('protocol.json',protocol)
    pairs=[('baseline',17+i) for i in range(3)] if request.param=='baseline' else [(role,17) for role in ('baseline','seed1','seed2','seed3')]
    assignments=[dict(model_role=role,checkpoint=write(role+'.pt',{'fixture':role}),evaluation_seed=seed,split='final') for role,seed in pairs]
    value=dict(protocol=pp,assignments=assignments);bp=write('batch.json',value);entries={};reports=[];auths=[];bounds=[]
    for i,assignment in enumerate(assignments):
        selection=dict(schema_version='r6-final-selection/1',batch=bp,assignment_index=i)
        report=report_fixture(protocol);report.update(split='final',seed=assignment['evaluation_seed'],
            final=dict(selection=selection,assignment=assignment,checkpoint=assignment['checkpoint']))
        auth=dict(mode='research-r6-final-evaluation',parameters={'final_selection':selection})
        bound=dict(issues=[],job_id='J-'+str(i),native_status='FAILED',native_exit_code=-9,protocol=pp)
        entries[str(i)]=dict(purpose='confirmation_delivery');reports.append(report);auths.append(auth);bounds.append(bound)
    def refresh():
        for i,b in enumerate(bounds):
            b.update(report=write(str(i)+'-report.json',reports[i]),authorization=write(str(i)+'-auth.json',auths[i]))
            entries[str(i)]['analysis']=write(str(i)+'-bound.json',b)
    refresh()
    monkeypatch.setattr(m,'validate_batch',lambda *a:value)
    monkeypatch.setattr(m,'bind_analysis',lambda entry,*a:json.loads(m.checked(entry['analysis']).read_text()))
    j=SimpleNamespace(canonical_job=lambda key:entries[key])
    return SimpleNamespace(j=j,pin=bp,keys=list(entries),reports=reports,auths=auths,bounds=bounds,entries=entries,refresh=refresh)


def collect(b):
    b.refresh();return m.collect(b.j,b.pin,b.keys,None)


def test_all_required_assignments_and_stress_failures_preserved(batch):
    result=collect(batch)
    assert len(result['results'])==len(batch.keys) and result['all_models_pass']
    assert not result['scientific_acceptance'] and not result['dispatch_authorized']
    assert all(r['native_status']=='FAILED' and r['native_exit_code']==-9 for r in result['results'])
    assert any(not r['passed'] for r in result['results'][0]['gates']['blocks'] if r['domain']=='stress')


def test_complete_target_negative_is_collected_without_success(batch):
    row=next(r for r in batch.reports[-1]['blocks'] if r['domain']=='target');s=row['scoring'];lane=s['lanes'][0]
    lane['motion_pass']=False;s['successes']-=1;s['motion_pass']=False;s['subgroups'][lane['subgroup']]['successes']-=1
    result=collect(batch)
    assert not result['all_models_pass'] and not result['results'][-1]['gates']['target_pass']
    assert not result['scientific_acceptance']


@pytest.mark.parametrize('fault',['missing','duplicate','order','seed','checkpoint','purpose','issues','lane','subgroup','partial','original','same_job'])
def test_final_collection_cannot_hide_incomplete_or_wrong_evidence(batch,fault):
    b=batch;r=b.reports[-1]
    if fault=='missing':b.keys.pop()
    elif fault=='duplicate':b.keys[-1]=b.keys[0]
    elif fault=='order':b.keys.reverse()
    elif fault=='seed':r['seed']+=1
    elif fault=='checkpoint':r['final']['checkpoint']={}
    elif fault=='purpose':b.entries[b.keys[-1]]['purpose']='development'
    elif fault=='issues':b.bounds[-1]['issues']=['missing']
    elif fault=='lane':r['blocks'][0]['scoring']['lanes'].pop()
    elif fault=='subgroup':r['blocks'][0]['scoring']['subgroups']={}
    elif fault=='partial':r['blocks'][-1]['measurement_replay']['bit_exact_delivery']=False
    elif fault=='original':r['summary']['original']['wall_motion']['per_lane'].pop()
    elif fault=='same_job':b.bounds[-1]['job_id']=b.bounds[0]['job_id']
    with pytest.raises(RecoveryError):collect(b)


def test_original_baseline_control_failure_does_not_replace_confirmation_verdict(batch):
    batch.reports[0]['summary']['original']['all_gates_pass']=False
    result=collect(batch)
    assert not result['all_models_pass']
    if len(batch.keys)==4:
        assert result['required_models_pass'] and result['required_assignments']==[1,2,3]
    else:
        assert not result['required_models_pass'] and result['required_assignments']==[0,1,2]
