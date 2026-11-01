"""Small CPU evidence helpers, independent of experiment orchestration."""
import hashlib
import json
import os
from pathlib import Path


class RecoveryError(RuntimeError):
    """Invalid or changed evidence (also used by historical readers)."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def persist_exact(path, data):
    path = Path(path)
    if path.is_symlink() or path.resolve() != path.absolute():
        raise RecoveryError('EVIDENCE_PATH_ALIAS')
    if path.exists():
        if path.read_bytes() != data:
            raise RecoveryError('EVIDENCE_ALREADY_EXISTS_WITH_DIFFERENT_BYTES')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def file_pin(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def checked(pin):
    path = Path(pin['path'])
    if path.is_symlink() or file_pin(path) != pin:
        raise RecoveryError('PIN_DRIFT')
    return path


def atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
