"""Synthetic complete score matrices; no renderer or simulation."""
import copy
import pytest
from tools import r6_video_selection as m
from tools.test_r6_final_collection import batch,collect
from tools.r6_recovery import RecoveryError


def run(b):
    final=collect(b);protocol=m.read(m.read(b.pin)['protocol'])
    return m.select(final,b.reports,protocol)


def test_deterministic_complete_clips_preserve_failure_and_pair_baseline(batch):
    result=run(batch);assert run(batch)==result and result['full_trajectories']
    assert any(not c['motion_pass'] for c in result['clips'])
    for c in result['clips']:
        assert c['first_policy_step']==0 and c['last_policy_step']==c['frame_count']-1
    if len(batch.keys)==4:
        for c in result['clips']:
            if c['assignment_index']:
                assert any(p['assignment_index']==0 and p['block_id']==c['block_id'] and p['lane']==c['lane'] for p in result['clips'])
    assert len(result['clips'])<=14


def test_missing_outcome_is_explicit_not_synthetic(batch):
    for report in batch.reports:
        for row in report['blocks']:
            score=row['scoring'];score['successes']=32;score['motion_pass']=True
            for lane in score['lanes']:lane['motion_pass']=True
            for group in score['subgroups'].values():group['successes']=group['total']
    result=run(batch)
    assert len(result['unavailable_categories'])==len(batch.keys)
    assert all(c['motion_pass'] for c in result['clips'])


def test_lane_and_condition_selection_is_stable_not_best_score(batch):
    result=run(batch)
    success=next(c for c in result['clips'] if c['assignment_index']==0 and 'representative_success' in c['selection_reasons'])
    assert success['lane']==0
    assert success['domain']=='target'


def test_changed_assignment_rejected(batch):
    final=collect(batch);protocol=m.read(m.read(batch.pin)['protocol'])
    reports=copy.deepcopy(batch.reports);reports[0]['final']['assignment']['evaluation_seed']+=1
    with pytest.raises(RecoveryError,match='REPORT_BINDING'):m.select(final,reports,protocol)


@pytest.fixture
def sources(batch,tmp_path):
    import json
    from tools.r6_recovery import file_pin
    b=batch
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return file_pin(path)
    for i,report in enumerate(b.reports):
        attempt=tmp_path/('attempt'+str(i));b.auths[i]['attempt']=str(attempt)
        report['run_id']='run-'+str(i);report['inputs']=[];files=[]
        for block in report['blocks']:
            root=attempt/'results/research-r6-final-evaluation/r6'/block['block_id']
            for name in ('audit.bin','motion_traces.json','result.json'):
                value=dict(block_id=block['block_id'],run_id=report['run_id'],seed=report['seed'],split='final',audit_records=12) if name=='result.json' else {'fixture':'not real decoded trajectory'}
                pin=write(root/name,value);record=dict(pin,bytes=(root/name).stat().st_size)
                files.append(record);report['inputs'].append(record)
        b.bounds[i]['spec']=write(tmp_path/f'spec-{i}.json',{'cwd':str(tmp_path/'snapshot')})
        b.bounds[i]['inventory']=write(tmp_path/f'inventory-{i}.json',{'files':files})
    b.refresh();return b


def test_source_packet_binds_original_inventory_and_analyzed_hashes(sources):
    b=sources;value=m.prepare(b.j,b.pin,b.keys,None)
    assert value['status']=='VIDEO_SOURCES_SELECTED_NOT_RENDERED' and not value['dispatch_authorized']
    assert value['total_frames']==sum(c['frame_count'] for c in value['clips'])
    for clip in value['clips']:
        assert set(clip['raw'])=={'audit.bin','motion_traces.json','result.json'}
        assert clip['checkpoint']==clip['assignment']['checkpoint']


@pytest.mark.parametrize('fault',['inventory','analysis','raw','identity'])
def test_wrong_source_evidence_cannot_become_video(sources,fault):
    import json
    b=sources
    if fault=='inventory':
        path=m.checked(b.bounds[0]['inventory']);path.write_text('{}')
    elif fault=='analysis':b.reports[0]['inputs']=[];b.refresh()
    elif fault=='raw':
        p=next(x for x in b.reports[0]['inputs'] if x['path'].endswith('audit.bin'));m.checked({k:p[k] for k in ('path','sha256')}).write_bytes(b'changed')
    else:
        b.reports[0]['run_id']='wrong';b.refresh()
    with pytest.raises(RecoveryError):m.prepare(b.j,b.pin,b.keys,None)
