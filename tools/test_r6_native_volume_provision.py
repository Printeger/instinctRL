"""Provisioning guard tests only: no root, mount, formatter or large allocation."""
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools import provision_r6_native_volume as volume


def test_plan_keeps_fixed_ceiling_and_rejects_path_injection():
    spec=volume.plan('CPU-QUOTA-01')
    assert spec['maximum_backing_bytes']==1610612736
    assert spec['commands'][-1][4]=='loop,nodev,nosuid,noexec'
    for value in ('../other','J-abc','CPU-QUOTA-01/child','CPU-QUOTA-01;true'):
        with pytest.raises(ValueError):volume.plan(value)


def test_unprivileged_apply_fails_before_side_effects(monkeypatch):
    monkeypatch.setattr(volume.os,'geteuid',lambda:1000)
    command=Mock();monkeypatch.setattr(volume.subprocess,'run',command)
    with pytest.raises(PermissionError):volume.apply(volume.plan('CPU-QUOTA-01'))
    command.assert_not_called()


@pytest.fixture
def fake_admin(tmp_path,monkeypatch):
    monkeypatch.setattr(volume,'BASE',tmp_path/'volumes')
    monkeypatch.setattr(volume,'SIZE',4096)
    monkeypatch.setattr(volume.os,'geteuid',lambda:0)
    monkeypatch.setattr(volume,'secure_directory',lambda p:None)
    monkeypatch.setattr(volume.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    commands=Mock(side_effect=RuntimeError('CPU stops before formatting/mounting'))
    monkeypatch.setattr(volume.subprocess,'run',commands)
    return volume.plan('CPU-QUOTA-01'),commands


def test_existing_identity_is_never_formatted(fake_admin):
    spec,commands=fake_admin
    directory=Path(spec['image']).parent;directory.mkdir(parents=True)
    old=directory/'original';old.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):volume.apply(spec)
    assert old.read_bytes()==b'preserve'
    commands.assert_not_called()


def test_disk_floor_prevents_new_image(fake_admin,monkeypatch):
    spec,commands=fake_admin
    monkeypatch.setattr(volume.shutil,'disk_usage',lambda p:SimpleNamespace(free=volume.FLOOR))
    with pytest.raises(RuntimeError,match='FREE_FLOOR'):volume.apply(spec)
    assert not Path(spec['image']).exists()
    commands.assert_not_called()


def test_failed_creation_preserves_partial_and_cannot_be_retried_as_fresh(fake_admin):
    spec,commands=fake_admin
    with pytest.raises(RuntimeError,match='before formatting'):volume.apply(spec)
    assert Path(spec['image']).stat().st_size==4096
    assert Path(spec['target']).stat().st_mode & 0o777 == 0
    assert (Path(spec['image']).parent/'request.json').exists()
    with pytest.raises(FileExistsError):volume.apply(spec)
    assert commands.call_count==1


def test_secure_parent_rejects_symlink_and_world_writable_path(tmp_path):
    link=tmp_path/'link';link.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(RuntimeError):volume.secure_directory(link)
    # This checks the real lstat path; unprivileged ownership also fails closed.
    os.chmod(tmp_path,0o777)
    with pytest.raises(RuntimeError):volume.secure_directory(tmp_path)
