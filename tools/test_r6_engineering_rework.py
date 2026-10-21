"""Actual temporary Git/fake CLI; native snapshot and dispatch are fixtures."""
import copy,json
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools.test_r6_engineering_review import case,source,run,FREE
from tools.test_r6_source_preparation import prepared
from tools import r6_engineering_rework as rework,r6_controller as controller
from tools import r6_source_repair as repair
from tools.r6_recovery import RecoveryError,file_pin


def rejected(b):
    b.builder.with_suffix('.settings').write_text('{"reject":true}')
    source(b);out=run(b)
    with b.j.writer():
        state=b.j.read();state['events'][b.controller]['cursor'].update(
            phase='ENGINEERING_REPAIR_REJECTED',engineering_review=out);b.j.save(state)
    return out


def prepare(b):
    with b.j.writer():
        return rework.prepare(b.j,b.controller,b.store,executable=file_pin(b.builder),free_disk_bytes=FREE)


def test_rejected_source_is_reworked_then_reviewed_before_compatibility(case,monkeypatch):
    b=case;b.builder.with_suffix('.settings').write_text('{"reject_once":true}')
    compat=Mock(side_effect=RuntimeError('compatibility boundary'))
    monkeypatch.setattr(controller.compatibility,'prepare',compat)
    original=copy.deepcopy(b.j.read()['events'][b.key])
    for _ in range(2):
        with b.j.writer(),pytest.raises(RuntimeError,match='compatibility boundary'):
            controller.run(b.j,b.request,b.store,free_disk_bytes=FREE)
    state=b.j.read();owner=state['events'][b.controller];history=owner['engineering_rework_history']
    assert len(history)==1 and owner['cursor']['phase']=='COMPATIBILITY'
    prior=state['events'][history[0]['predecessor']]
    assert Path(prior['source_result']['path']).is_file()
    assert json.loads(Path(history[0]['rejected_review']['path']).read_text())['status']=='ENGINEERING_REPAIR_REVIEW_REJECTED'
    assert len(owner['engineering_source_results'])==2
    assert state['events'][b.key]==original
    assert b.builder.with_suffix('.count').read_text()=='call\ncall\n'
    assert b.builder.with_suffix('.review-count').read_text()=='review\nreview\n'
    assert repair.active(b.j,b.parent)==history[0]['replacement']


def test_crash_after_registration_reuses_reserved_attempt(case,monkeypatch):
    b=case;rejected(b);register=rework.source.register
    def crash(*args,**kwargs):
        register(*args,**kwargs);raise RuntimeError('after reservation')
    monkeypatch.setattr(rework.source,'register',crash)
    with pytest.raises(RuntimeError,match='after reservation'):prepare(b)
    budget=b.j.read()['budget'];monkeypatch.setattr(rework.source,'register',register)
    transition=prepare(b)
    assert b.j.read()['budget']==budget
    assert repair.active(b.j,b.parent)==transition['replacement']
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    assert b.j.read()['events'][b.controller]['cursor']['phase']=='REPAIR_AND_REVIEW'


@pytest.mark.parametrize('damage',['stop','dirty','frozen','wrong_source','pass'])
def test_invalid_rework_never_reserves_source_worker(case,damage):
    b=case;out=rejected(b)
    if damage=='stop':(b.j.directory/'STOP').touch()
    if damage=='dirty':(b.repo/'training_workloads/untracked.py').write_text('dirty')
    if damage in ('frozen','wrong_source','pass'):
        with b.j.writer():
            state=b.j.read();owner=state['events'][b.controller]
            if damage=='frozen':owner['cursor']['confirmation']={}
            if damage=='wrong_source':owner['engineering_source_result']=b.report_pin
            if damage=='pass':owner['cursor']['engineering_review']['status']='ENGINEERING_REPAIR_REVIEW_ACCEPTED'
            b.j.save(state)
    before=sum(e['kind']=='model' for e in b.j.read()['events'].values())
    with pytest.raises(RecoveryError):prepare(b)
    assert sum(e['kind']=='model' for e in b.j.read()['events'].values())==before
    assert b.builder.with_suffix('.count').read_text()=='call\n'


def test_renamed_artifacts_without_implementation_change_fail_before_review(case):
    b=case;rejected(b);prepare(b)
    b.builder.with_suffix('.settings').write_text('{"unchanged_rework":true}')
    with pytest.raises(RecoveryError,match='REWORK_NO_IMPLEMENTATION_CHANGE'):source(b)
    assert b.builder.with_suffix('.review-count').read_text()=='review\n'
