"""Real kernel counterexamples in disposable CPU subprocesses, never GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from training_workloads.unified_h1.write_boundary import prepare


def test_actual_kernel_boundary_and_descendant(tmp_path):
    output=tmp_path/'quota';output.mkdir()
    source=tmp_path/'input';source.write_text('immutable')
    escape=tmp_path/'escape';(output/'outside-link').symlink_to(escape)
    fd=os.open(source,os.O_WRONLY);os.set_inheritable(fd,True)
    program=r'''
import os,json,subprocess,sys,ctypes
from pathlib import Path
from training_workloads.unified_h1.write_boundary import prepare,enforce
root,source,escape=map(Path,sys.argv[1:4]);old_fd=int(sys.argv[4])
enforce(prepare([root],['/dev/null']))
(root/'allowed').write_text('ok')
assert source.read_text()=='immutable'
results={}
libc=ctypes.CDLL(None,use_errno=True)
for syscall in (452,463,466,467,469):
 assert libc.syscall(syscall,0,0,0,0,0,0)==-1 and ctypes.get_errno()==1
operations={'absolute':lambda:escape.write_text('bad'),'input':lambda:source.write_text('bad'),
 'truncate':lambda:os.truncate(source,0),'metadata':lambda:os.chmod(source,0o777),
 'xattr':lambda:os.setxattr(source,'user.r6probe',b'bad'),
 'symlink':lambda:(root/'outside-link').write_text('bad'),
 'hardlink':lambda:os.link(source,root/'linked_input'),
 'inherited_fd':lambda:os.write(old_fd,b'bad')}
for name,operation in operations.items():
 try:operation()
 except OSError as error:results[name]=error.errno
 else:raise AssertionError('escaped: '+name)
child=subprocess.run([sys.executable,'-B','-c',"from pathlib import Path;Path("+repr(str(escape))+ ").write_text('bad')"],capture_output=True)
assert child.returncode!=0 and b'PermissionError' in child.stderr
results['descendant_denied']=True
print(json.dumps(results))
'''
    try:
        result=subprocess.run([sys.executable,'-B','-c',program,str(output),str(source),str(escape),str(fd)],
            pass_fds=(fd,),capture_output=True,text=True,timeout=20,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
    finally:os.close(fd)
    assert result.returncode==0,result.stderr
    data=json.loads(result.stdout);assert data['descendant_denied']
    assert all(data[k] in (1,9,13,18) for k in data if k!='descendant_denied')
    assert source.read_text()=='immutable' and source.stat().st_mode & 0o777!=0o777
    assert not escape.exists() and (output/'allowed').read_text()=='ok'


def test_policy_rejects_broad_or_symlink_roots(tmp_path):
    link=tmp_path/'alias';link.symlink_to(tmp_path,target_is_directory=True)
    for roots in (['/'],[link],[tmp_path,tmp_path]):
        with pytest.raises(ValueError):prepare(roots)
