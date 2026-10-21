"""Content identity for existing extension inputs; never downloads or edits them."""
import hashlib
import json
import os
from pathlib import Path
import stat


def tree_identity(root):
    root=Path(root)
    if not root.is_absolute() or root.resolve()!=root or not root.is_dir() or any(c.isspace() for c in str(root)):
        raise ValueError('R6 extension input root')
    digest=hashlib.sha256();count=total=0
    def walk(directory):
        nonlocal count,total
        for path in sorted(directory.iterdir()):
            before=path.lstat();relative=path.relative_to(root).as_posix()
            if stat.S_ISLNK(before.st_mode):
                if not path.resolve(strict=True).is_relative_to(root):raise ValueError('R6 extension external symlink')
                entry=['link',relative,os.readlink(path)]
            elif stat.S_ISDIR(before.st_mode):
                entry=['directory',relative]
            elif stat.S_ISREG(before.st_mode):
                content=hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):content.update(chunk)
                count+=1;total+=before.st_size
                entry=['file',relative,before.st_size,content.hexdigest()]
            else:raise ValueError('R6 extension special file')
            after=path.lstat()
            if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError('R6 extension input changed during hashing')
            digest.update(json.dumps(entry,separators=(',',':')).encode()+b'\n')
            if stat.S_ISDIR(before.st_mode):
                walk(path)
                after=path.lstat()
                if (before.st_ino,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_mtime_ns,after.st_ctime_ns):
                    raise ValueError('R6 extension directory changed during hashing')
    before=root.stat()
    walk(root)
    after=root.stat()
    if (before.st_ino,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('R6 extension root changed during hashing')
    return dict(root=str(root),sha256=digest.hexdigest(),files=count,bytes=total)


def manifest(pin):
    from .research_contract import validate_pin
    record=json.loads(validate_pin(pin).read_text())
    if set(record)!={'schema_version','tree','user_config'} or record['schema_version']!='r6-extension-inputs/2':
        raise ValueError('R6 extension input manifest')
    return record


def user_config_bytes(pin):
    record=manifest(pin);config=record['user_config']
    from .research_contract import validate_pin
    path=validate_pin(config)
    if path.stat().st_size>1024**2:raise ValueError('R6 persistent configuration size')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=config['sha256']:raise ValueError('R6 persistent configuration drift')
    if not isinstance(json.loads(data),dict):raise ValueError('R6 persistent configuration object')
    return data


def validate(pin):
    record=manifest(pin)
    user_config_bytes(pin)
    if tree_identity(record['tree']['root'])!=record['tree']:
        raise ValueError('R6 extension input content mismatch')
    return record['tree']['root']
