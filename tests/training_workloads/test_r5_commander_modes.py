"""Real CLI refuses premature formal submission before opening a Store."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('mode',['research-formal-training','research-formal-evaluation'])
def test_real_cli_formal_submit_requires_explicit_later_start(mode,tmp_path):
    script=Path(__file__).resolve().parents[2]/'tools/r5_formal_commander.py'
    # An empty PYTHONPATH makes any premature rlflow import fail. The explicit
    # later-start boundary must reject before Store, filesystem or GPU work.
    env={**os.environ, 'PYTHONPATH':'', 'PYTHONDONTWRITEBYTECODE':'1', 'CUDA_VISIBLE_DEVICES':''}
    result=subprocess.run([sys.executable,'-B',str(script),'submit','CPU-NONEXISTENT','R1',mode],
                          env=env,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2
    assert 'explicit later --start-formal' in result.stderr
    assert 'ModuleNotFoundError' not in result.stderr
    assert list(tmp_path.iterdir())==[]
