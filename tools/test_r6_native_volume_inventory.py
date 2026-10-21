"""Closed-job native inventory integration; temporary Store and mount substitutes."""
import json
from pathlib import Path
import pytest
from rlflow.store import sha
from tools import test_r6_job_inventory as fixtures
from tools.r6_job_inventory import DISK
from tools.r6_recovery import file_pin,RecoveryError
from training_workloads.unified_h1 import native_volume


@pytest.fixture
def native_job(monkeypatch):
    b=fixtures.InventoryTests();b.setUp()
    try:
        metadata=b.base/'native';metadata.mkdir()
        target=metadata/'output';target.mkdir();(target/'lost+found').mkdir()
        (target/'native-output.log').write_bytes(b'failed native log')
        (target/'cache').mkdir();(target/'cache'/'artifact').write_bytes(b'cached original')
        (metadata/'volume.json').write_text('{}');(metadata/'request.json').write_text('{}')
        native_pin=file_pin(metadata/'volume.json')
        authority=b.base/'authority.json'
        authority.write_text(json.dumps(dict(parameters=dict(native_volume=native_pin))))
        b.spec['h1']['mode']='research-r6-evaluation'
        b.spec['pins']={'authorization':file_pin(authority)}
        specpath=Path(b.request['spec']['path']);specpath.write_text(json.dumps(b.spec))
        b.request['spec']=file_pin(specpath)
        b.jid='J-'+sha({'id':b.spec['id'],'spec':b.spec})[:16]
        b.job.update(id=b.jid,spec=b.spec);b.store.state['jobs']={b.jid:b.job}
        new=b.store.root/'jobs'/b.jid;b.directory.rename(new);b.directory=new
        with b.j.writer():
            state=b.j.read();state['events']={};state['budget']['reservations']=[];b.j.save(state)
            identity=dict(goal_id=state['goal_id'],event_id='CPU-NATIVE-INVENTORY',job_id=b.jid,
                          input_sha256=sha(b.request),role='job')
            b.key,_=b.j.intent(identity,kind='job',request=b.request,
                resources={'live_starts':1,'live_wall_seconds':190,'allocated_gpu_seconds':190,DISK:2*1024**3},
                purpose='development',free_disk_bytes=500*1024**3)
        b.finish();b.native_target=target
        binding=dict(target=str(target),manifest=native_pin,mount_id='CPU',device='CPU',maximum_bytes=1610612736)
        monkeypatch.setattr(native_volume,'validate',lambda *a,**kw:dict(binding))
        yield b
    finally:b.tearDown()


def test_native_originals_are_hashed_and_image_is_charged_once(native_job):
    b=native_job;r=b.record()
    native=[row for row in r['files'] if row['category']=='native']
    assert {row['relative'] for row in native}=={'native-output.log','cache/artifact'}
    external=sum(row['bytes'] for row in r['files'] if row['category']!='native')
    assert r['total_charged_output_bytes']==external+1610612736+4
    observation=b.j.read()['budget']['reservations'][0]['observations'][-1]
    assert observation['resources'][DISK]==r['total_charged_output_bytes']
    assert r['native_exit_code']==-9 and r['output_inventory_complete']
    assert not r['scientific_acceptance']


def test_replay_retains_later_phase_and_detects_external_changes(native_job):
    b=native_job;first=b.record()
    with b.j.writer():
        state=b.j.read();state['status']='MODEL_DECISION_RECORDED';b.j.save(state)
    before=b.j.read();assert b.record()==first and b.j.read()==before
    (b.native_target/'cache'/'artifact').write_bytes(b'changed')
    with pytest.raises(RecoveryError,match='REPLAY_DRIFT'):b.record()
    assert b.j.read()==before


def test_native_symlink_cannot_escape_inventory(native_job):
    b=native_job;(b.native_target/'alias').symlink_to(b.attempt)
    with pytest.raises(RecoveryError,match='SYMLINK_OR_SPECIAL_FILE'):b.record()
    assert 'inventory' not in b.j.read()['events'][b.key]


def test_missing_native_mount_fails_without_omitting_originals(native_job,monkeypatch):
    b=native_job
    def absent(*a,**kw):raise ValueError('MOUNT_ABSENT')
    monkeypatch.setattr(native_volume,'validate',absent)
    with pytest.raises(ValueError,match='MOUNT_ABSENT'):b.record()
    assert 'inventory' not in b.j.read()['events'][b.key]


def test_nested_native_mount_is_not_charged_as_part_of_backing_image(native_job,monkeypatch):
    from types import SimpleNamespace
    b=native_job;external=b.native_target/'cache'/'artifact';original=Path.lstat
    def different_device(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        if path!=external:return value
        return SimpleNamespace(st_dev=value.st_dev+1,st_ino=value.st_ino,st_mode=value.st_mode,
            st_size=value.st_size,st_mtime_ns=value.st_mtime_ns,st_ctime_ns=value.st_ctime_ns)
    monkeypatch.setattr(Path,'lstat',different_device)
    with pytest.raises(RecoveryError,match='CROSS_DEVICE'):b.record()
