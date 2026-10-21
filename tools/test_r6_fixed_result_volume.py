import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools import provision_r6_fixed_result_volume as provision
from training_workloads.unified_h1 import result_volume as result


def test_plan_has_fixed_geometry_and_requires_payload_and_inode_floor(tmp_path,monkeypatch):
    monkeypatch.setattr(provision,'ATTEMPTS',tmp_path/'attempts')
    target=provision.ATTEMPTS/'R6-CPU/results';payload=1024**3;size=payload+96*1024**2
    spec=provision.plan('CPU-QUOTA-03',target,size,payload,1024)
    assert spec['commands'][1]==['/usr/sbin/mkfs.ext4','-q','-F','-m','0','-b','4096','-I','256','-i','262144','-O','^resize_inode','-J','size=32',spec['image']]
    assert not target.exists()
    for amount,count in [(payload+4096,1024),(payload,0),(payload,size//262144+1)]:
        with pytest.raises(ValueError):provision.plan('CPU-QUOTA-03',target,size,amount,count)
    monkeypatch.setattr(provision.os,'geteuid',lambda:0)
    spec['commands'][1].append('unsafe')
    with pytest.raises(ValueError,match='PLAN_DRIFT'):provision.apply(spec)


@pytest.mark.parametrize('available,inodes,status',[(100,10,'MOUNT_CREATED_NOT_ACCEPTED'),(99,10,'MOUNT_CAPACITY_FAILED_PRESERVED'),(100,9,'MOUNT_CAPACITY_FAILED_PRESERVED')])
def test_actual_capacity_fields_preserve_shortfalls(available,inodes,status):
    result=provision.capacity_fields(SimpleNamespace(f_blocks=200,f_frsize=4096,f_bavail=available,f_files=100,f_favail=inodes),dict(minimum_free_bytes=100*4096,minimum_free_inodes=10))
    assert result['status']==status
    assert result['filesystem_available_bytes']==available*4096 and result['filesystem_available_inodes']==inodes


@pytest.mark.parametrize('case',['existing','nonempty','disk_floor'])
def test_provision_rejects_before_allocating_or_mounting(tmp_path,monkeypatch,case):
    monkeypatch.setattr(provision,'BASE',tmp_path/'volumes');monkeypatch.setattr(provision,'ATTEMPTS',tmp_path/'attempts')
    monkeypatch.setattr(provision.os,'geteuid',lambda:0);monkeypatch.setattr(provision,'secure',lambda p:None)
    monkeypatch.setattr(provision.shutil,'disk_usage',lambda p:SimpleNamespace(free=provision.FLOOR if case=='disk_floor' else 100*1024**3))
    target=provision.ATTEMPTS/'R6-CPU/results';target.mkdir(parents=True)
    spec=provision.plan('CPU-QUOTA-03',target,1120*1024**2,1024**3,1024)
    if case=='existing':(provision.BASE/'CPU-QUOTA-03').mkdir(parents=True)
    if case=='nonempty':(target/'original').write_text('preserve')
    run=Mock();monkeypatch.setattr(provision.subprocess,'run',run)
    with pytest.raises((ValueError,FileExistsError)):provision.apply(spec)
    run.assert_not_called()
    if case=='nonempty':assert (target/'original').read_text()=='preserve'


def test_runtime_rechecks_fixed_layout_bytes_and_inodes(tmp_path,monkeypatch):
    attempts=tmp_path/'attempts';target=attempts/'R6-CPU/results';target.mkdir(parents=True)
    monkeypatch.setattr(result,'ATTEMPTS',attempts)
    payload=1024**3;size=payload+96*1024**2
    record=dict(target=str(target),maximum_backing_bytes=size,layout='fixed-results-v1',status='MOUNT_CREATED_NOT_ACCEPTED',minimum_free_bytes=payload,minimum_free_inodes=1024)
    path=tmp_path/'volume.json'
    def pin():
        path.write_text(json.dumps(record));return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    binding=dict(target=str(target),free_bytes=payload)
    monkeypatch.setattr(result,'validate_volume',lambda *a,**kw:binding)
    usage=SimpleNamespace(f_favail=1024);monkeypatch.setattr(result.os,'statvfs',lambda p:usage)
    cap=result.MAXIMUM+result.CONTROL_ALLOWANCE+size
    assert result.validate(pin(),output_cap_bytes=cap)==binding
    usage.f_favail=1023
    with pytest.raises(ValueError,match='FREE_INODES'):result.validate(pin(),output_cap_bytes=cap)
    usage.f_favail=1024;binding['free_bytes']=payload-1
    with pytest.raises(ValueError,match='FREE_SPACE'):result.validate(pin(),output_cap_bytes=cap)
    usage.f_favail=0;binding['free_bytes']=0
    assert result.validate(pin(),output_cap_bytes=cap,minimum_free_bytes=0)==binding
    binding['free_bytes']=payload;record['status']='MOUNT_CAPACITY_FAILED_PRESERVED'
    with pytest.raises(ValueError,match='LAYOUT_STATUS'):result.validate(pin(),output_cap_bytes=cap)
