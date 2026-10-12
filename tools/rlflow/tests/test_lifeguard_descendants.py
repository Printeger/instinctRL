import json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
from rlflow import experiments,lifeguard

def read_json_when_parseable(path,timeout=5):
 deadline=time.monotonic()+timeout
 while True:
  try:return json.loads(Path(path).read_text())
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>=deadline:raise
   time.sleep(.01)
class DescendantLedgerTests(unittest.TestCase):
 def row(self,pid,pgid=None,start=None):
  return {'pid':pid,'pgid':pgid if pgid is not None else pid,'starttime':str(start if start is not None else pid*10),'boot_id':'boot'}
 def test_double_fork_setsid_descendant_is_ancestry_bound_and_reclaimed(self):
  owner=self.row(100);intermediate=self.row(101,101);escaped=self.row(102,102)
  identities={100:owner,101:intermediate,102:escaped}
  table={pid:{'ppid':ppid,'identity':identities[pid]} for pid,ppid in {100:1,101:100,102:101}.items()}
  active={101,102}
  with tempfile.TemporaryDirectory() as td, mock.patch.object(lifeguard,'_process_table',return_value=table), mock.patch.object(lifeguard,'identity',side_effect=lambda pid:identities.get(pid)), mock.patch.object(lifeguard,'owns',side_effect=lambda row:row['pid'] in active and identities.get(row['pid'])==row), mock.patch.object(lifeguard.os,'kill',side_effect=lambda pid,sig:active.discard(pid)) as kill:
   root=Path(td);ledger=lifeguard.record_descendants(root,owner)
   self.assertEqual(ledger,[intermediate,escaped]);self.assertNotEqual(owner['pgid'],escaped['pgid'])
   evidence=lifeguard.reclaim_descendants(root,ledger)
   self.assertEqual(evidence['survivors'],[]);self.assertEqual({call.args[0] for call in kill.call_args_list},{101,102})
   self.assertEqual(json.loads((root/'descendant-ledger.json').read_text())['descendants'],ledger)
 def test_discovery_window_pid_reuse_is_rejected_before_ledger_admission(self):
  owner=self.row(100);observed=self.row(222,start=1);replacement=self.row(222,start=2)
  table={100:{'ppid':1,'identity':owner},222:{'ppid':100,'identity':observed}}
  with tempfile.TemporaryDirectory() as td, \
       mock.patch.object(lifeguard,'_process_table',return_value=table), \
       mock.patch.object(lifeguard,'identity',side_effect=lambda pid:replacement if pid==222 else owner):
   ledger=lifeguard.record_descendants(Path(td),owner)
   self.assertEqual(ledger,[])
   self.assertEqual(json.loads((Path(td)/'descendant-ledger.json').read_text())['descendants'],[])

 def test_ancestor_pid_reuse_during_discovery_window_rejects_child(self):
  owner=self.row(100);observed_parent=self.row(150,start=1)
  replacement_parent=self.row(150,start=2);child=self.row(222,start=3)
  observed={100:{'ppid':1,'identity':owner},
            150:{'ppid':100,'identity':observed_parent},
            222:{'ppid':150,'identity':child}}
  admission={100:{'ppid':1,'identity':owner},
             150:{'ppid':1,'identity':replacement_parent},
             222:{'ppid':150,'identity':child}}
  with tempfile.TemporaryDirectory() as td, \
       mock.patch.object(lifeguard,'_process_table',side_effect=[observed,admission,admission]), \
       mock.patch.object(lifeguard,'identity',return_value=child):
   ledger=lifeguard.record_descendants(Path(td),owner)
   self.assertEqual(ledger,[])
   self.assertEqual(json.loads((Path(td)/'descendant-ledger.json').read_text())['descendants'],[])

 def test_ancestor_replaced_after_admission_snapshot_is_rejected_before_signal(self):
  owner=self.row(100);observed_parent=self.row(150,start=1);replacement=self.row(150,start=2)
  child=self.row(222,start=3)
  stable={100:{'ppid':1,'identity':owner},
          150:{'ppid':100,'identity':observed_parent},
          222:{'ppid':150,'identity':child}}
  replaced={100:{'ppid':1,'identity':owner},
            150:{'ppid':1,'identity':replacement},
            222:{'ppid':150,'identity':child}}
  with tempfile.TemporaryDirectory() as td, \
       mock.patch.object(lifeguard,'_process_table',side_effect=[stable,stable,replaced]), \
       mock.patch.object(lifeguard,'identity',side_effect=lambda pid:{100:owner,150:replacement,222:child}[pid]), \
       mock.patch.object(lifeguard.os,'kill') as kill:
   ledger=lifeguard.record_descendants(Path(td),owner)
   self.assertEqual(ledger,[])
   self.assertEqual(json.loads((Path(td)/'descendant-ledger.json').read_text())['descendants'],[])
   lifeguard.reclaim_descendants(Path(td),ledger)
   kill.assert_not_called()

 def test_pid_reuse_mismatch_is_never_signalled(self):
  stale=self.row(222,start=1);replacement=self.row(222,start=2)
  with tempfile.TemporaryDirectory() as td, mock.patch.object(lifeguard,'identity',return_value=replacement), mock.patch.object(lifeguard.os,'kill') as kill:
   evidence=lifeguard.reclaim_descendants(Path(td),[stale])
   self.assertFalse(any(action['identity_matched'] for action in evidence['actions']));kill.assert_not_called()

 def test_run_rediscovers_child_forked_after_first_post_done_scan(self):
  owner=self.row(100);captured=self.row(101);late=self.row(102,102)
  active={captured['pid']};reclaims=[];scans=[]
  def discover(_directory,_owner,_known):
   scans.append(len(scans)+1)
   if len(scans)<=2:return [captured]
   return [late] if late['pid'] in active else []
  def reclaim(_directory,rows):
   reclaims.append(list(rows))
   if rows==[captured]:
    # The captured process forks and the child calls setsid only after run()
    # has completed its first post-done ancestry scan.
    self.assertGreaterEqual(len(scans),2)
    active.discard(captured['pid']);active.add(late['pid'])
   elif rows==[late]:active.discard(late['pid'])
   return {'actions':[{'identity':row,'signal':'SIGKILL','identity_matched':True} for row in rows],
           'survivors':[]}
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'lifeguard-done').write_bytes(b'done')
   request={'directory':td,'owner':owner,'kind':'experiment','root':td,
            'limit':10,'id':'late-fork','spec':{},'started_monotonic':time.monotonic()}
   request_path=root/'request.json';request_path.write_text(json.dumps(request))
   with mock.patch.object(lifeguard,'identity',return_value=self.row(os.getpid())), \
        mock.patch.object(lifeguard,'record_descendants',side_effect=discover), \
        mock.patch.object(lifeguard,'reclaim_descendants',side_effect=reclaim), \
        mock.patch.object(lifeguard,'owns',side_effect=lambda row:row==owner or row['pid'] in active):
    lifeguard.run(request_path)
   ack=read_json_when_parseable(root/'lifeguard-ack.json')
  cleanup=ack['descendant_cleanup']
  self.assertTrue(ack['no_survivors']);self.assertTrue(cleanup['quiescent'])
  self.assertEqual(cleanup['discovered'],[captured,late])
  self.assertEqual(reclaims[:2],[[captured],[late]])
  self.assertGreaterEqual(len(scans),5)
  self.assertEqual(cleanup['stable_scans'],cleanup['required_stable_scans'])

 def test_production_receipt_fails_when_guardian_cleanup_fails(self):
  receipt={'job_id':'J-test','spec_sha256':'a'*64,'status':'SUCCEEDED','exit_code':0,
           'error':None,'elapsed_seconds':1.0,'released':True,
           'allocated_seconds':0.5,'definitely_not_released':False}
  with tempfile.TemporaryDirectory() as td, \
       mock.patch.object(lifeguard,'await_normal_cleanup',side_effect=RuntimeError('forced cleanup failure')):
   durable=experiments._finish_worker(td,mock.Mock(),self.row(100),receipt)
   persisted=read_json_when_parseable(Path(td)/'receipt.json')
   self.assertEqual((Path(td)/'lifeguard-done').read_bytes(),b'done')
  self.assertEqual(durable,persisted)
  self.assertEqual(persisted['status'],'FAILED')
  self.assertEqual(persisted['exit_code'],0)
  self.assertEqual(persisted['error'],'RuntimeError: forced cleanup failure')

 def test_quiescence_timeout_fails_closed(self):
  owner=self.row(100);survivor=self.row(101)
  with tempfile.TemporaryDirectory() as td, \
       mock.patch.object(lifeguard,'reclaim_descendants',return_value={'actions':[],'survivors':[survivor]}), \
       mock.patch.object(lifeguard,'record_descendants',return_value=[survivor]), \
       mock.patch.object(lifeguard,'owns',return_value=True):
   cleanup=lifeguard.reclaim_descendants_until_quiescent(td,owner,[survivor],timeout=0)
  self.assertTrue(cleanup['timed_out']);self.assertFalse(cleanup['quiescent'])
  self.assertEqual(cleanup['survivors'],[survivor])

 def test_cancel_race_completes_cleanup_handshake_before_returning(self):
  owner=self.row(100);descendant=self.row(101)
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);request={'directory':td,'owner':owner,'kind':'experiment',
                          'started_monotonic':time.monotonic(),'id':'cancel-test','spec':{}}
   def signal_owner(_pid,_signal):(root/'lifeguard-done').write_bytes(b'done')
   with mock.patch.object(lifeguard,'owns',return_value=True), \
        mock.patch.object(lifeguard,'record_descendants',return_value=[descendant]), \
        mock.patch.object(lifeguard.os,'kill',side_effect=signal_owner), \
        mock.patch.object(lifeguard,'acknowledge_normal_completion') as acknowledge:
    lifeguard.reclaim(request,'STOP/cancel')
   acknowledge.assert_called_once_with(request,[descendant])

 def test_real_double_fork_setsid_survives_owner_sigkill_then_is_reclaimed(self):
  try:Path('/proc/1/stat').read_text()
  except PermissionError:raise unittest.SkipTest('CPU sandbox does not expose process identity/kill boundary')
  helper=r'''import ctypes,json,os,sys,time
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),"subreaper")
path=sys.argv[1]
child=os.fork()
if child==0:
 grandchild=os.fork()
 if grandchild==0:
  os.setsid()
  with open(path,"w") as stream:
   json.dump({"pid":os.getpid(),"ppid":os.getppid(),"sid":os.getsid(0)},stream);stream.flush();os.fsync(stream.fileno())
  while True:time.sleep(1)
 os._exit(0)
os.waitpid(child,0)
while True:time.sleep(1)
'''
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);info=root/'escaped.json';owner=None;ledger=[]
   try:
    owner=subprocess.Popen([sys.executable,'-c',helper,str(info)],start_new_session=True)
    deadline=time.monotonic()+5
    while not info.exists() and time.monotonic()<deadline:time.sleep(.01)
    self.assertTrue(info.exists(),"double-fork descendant did not report identity")
    escaped_pid=read_json_when_parseable(info)['pid'];owner_id=lifeguard.identity(owner.pid)
    self.assertIsNotNone(owner_id)
    while time.monotonic()<deadline:
     ledger=lifeguard.record_descendants(root,owner_id,ledger)
     if any(row['pid']==escaped_pid for row in ledger):break
     time.sleep(.01)
    escaped=next(row for row in ledger if row['pid']==escaped_pid)
    self.assertNotEqual(owner_id['pgid'],escaped['pgid'])
    os.kill(owner.pid,signal.SIGKILL);owner.wait(timeout=2)
    evidence=lifeguard.reclaim_descendants(root,ledger)
    self.assertEqual(evidence['survivors'],[])
    self.assertTrue(any(a['identity']==escaped and a['identity_matched'] for a in evidence['actions']))
    durable=json.loads((root/'descendant-cleanup.json').read_text())
    self.assertEqual(durable['survivors'],[])
   finally:
    if owner is not None and owner.poll() is None:
     owner.kill();owner.wait(timeout=2)
    lifeguard.reclaim_descendants(root,ledger)

 def test_real_live_pid_with_reused_identity_mismatch_is_not_signalled(self):
  try:Path('/proc/1/stat').read_text()
  except PermissionError:raise unittest.SkipTest('CPU sandbox does not expose process identity/kill boundary')
  with tempfile.TemporaryDirectory() as td:
   child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
   try:
    live=lifeguard.identity(child.pid);self.assertIsNotNone(live)
    stale={**live,'starttime':str(int(live['starttime'])-1)}
    evidence=lifeguard.reclaim_descendants(Path(td),[stale])
    self.assertIsNone(child.poll())
    self.assertFalse(any(action['identity_matched'] for action in evidence['actions']))
   finally:
    child.kill();child.wait(timeout=2)


 def test_real_normal_completion_handshake_reclaims_detached_descendant(self):
  try:Path('/proc/self/stat').read_text()
  except PermissionError:raise unittest.SkipTest('CPU sandbox does not expose process identity/kill boundary')
  helper=r'''import json,os,sys,time
from pathlib import Path
from rlflow import lifeguard

def read_json_when_parseable(path,timeout=5):
 deadline=time.monotonic()+timeout
 while True:
  try:return json.loads(Path(path).read_text())
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>=deadline:raise
   time.sleep(.01)
root=Path(sys.argv[1]);lifeguard.enable_subreaper();owner=lifeguard.identity(os.getpid())
guard=lifeguard.start(root,'experiment',owner,root,10,'normal-test',{})
child=os.fork()
if child==0:
 grandchild=os.fork()
 if grandchild==0:
  os.setsid();(root/'escaped.json').write_text(json.dumps(lifeguard.identity(os.getpid())))
  while True:time.sleep(1)
 os._exit(0)
os.waitpid(child,0)
deadline=time.monotonic()+5
def read_json_when_parseable(path):
 while True:
  try:return json.loads(path.read_text())
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>=deadline:raise
   time.sleep(.01)
escaped=read_json_when_parseable(root/'escaped.json')
while time.monotonic()<deadline:
 ledger=read_json_when_parseable(root/'descendant-ledger.json').get('descendants',[])
 if escaped in ledger:break
 time.sleep(.01)
else:raise RuntimeError('guardian did not bind detached descendant')
(root/'lifeguard-done').write_bytes(b'done')
ack=lifeguard.await_normal_cleanup(root,guard,owner)
if ack['descendant_cleanup']['survivors'] or lifeguard.owns(escaped):raise RuntimeError('detached descendant survived')
'''
  with tempfile.TemporaryDirectory() as td:
   result=subprocess.run([sys.executable,'-c',helper,td],cwd=Path(lifeguard.__file__).resolve().parents[1],capture_output=True,text=True,timeout=15)
   self.assertEqual(result.returncode,0,result.stderr)
   root=Path(td);ack=json.loads((root/'lifeguard-ack.json').read_text())
   self.assertTrue(ack['no_survivors']);self.assertTrue(ack['descendant_cleanup']['quiescent']);self.assertEqual(ack['descendant_cleanup']['survivors'],[])
   self.assertEqual(json.loads((root/'descendant-cleanup.json').read_text()),ack['descendant_cleanup'])

 def test_normal_completion_cleanup_failure_is_not_acknowledged_as_success(self):
  owner=self.row(100);survivor=self.row(101)
  failure={'actions':[],'discovered':[survivor],'rounds':[],
           'survivors':[survivor],'stable_scans':0,'required_stable_scans':2,
           'quiescent':False,'timed_out':True}
  with tempfile.TemporaryDirectory() as td, mock.patch.object(lifeguard,'reclaim_descendants_until_quiescent',return_value=failure):
   ack=lifeguard.acknowledge_normal_completion({'directory':td,'owner':owner},[survivor])
   self.assertFalse(ack['no_survivors'])
   lifeguard.atomic(Path(td)/'descendant-cleanup.json',lifeguard.canonical(ack['descendant_cleanup']))
   guard=mock.Mock();guard.wait.return_value=0
   with self.assertRaisesRegex(RuntimeError,'cleanup failed'):
    lifeguard.await_normal_cleanup(td,guard,owner)

if __name__=='__main__':unittest.main()
