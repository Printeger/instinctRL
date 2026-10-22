"""Small durable-state helpers for the direct runner."""
import json
import os
from pathlib import Path


def atomic(path, value):
    path=Path(path); temporary=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
