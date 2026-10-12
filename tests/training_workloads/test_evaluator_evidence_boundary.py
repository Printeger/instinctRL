import hashlib,json,tempfile,unittest
from pathlib import Path
from training_workloads.unified_h1.evidence_boundary import verify_delivery_identity,verify_evaluator_boundary,verify_postrun,verify_predecessor
ROOT=Path(__file__).resolve().parents[2]
class EvidenceBoundaryTests(unittest.TestCase):
 def test_active_successor_delivery_identity_is_consistent_and_fails_closed(self):
  current_package='INSTINCTRL-R5-STAGE1-IDENTITY-BOUNDARY-SUCCESSOR-20260929'
  current_task='T-d219cc10961c75be'
  predecessor_diff='e58bfa16aa6ca448fb730e42582369b427a7454daec1fae3e351b0597c185f97'
  source=ROOT/'docs/history/consolidation-20261012/r5/b0_enhanced_delivery'
  # The immutable delivery is predecessor evidence.  Materialize a current
  # identity representation to exercise the successor's fail-closed reader.
  with self.assertRaisesRegex(PermissionError,'identity mismatch'):
   verify_delivery_identity(source)
  with tempfile.TemporaryDirectory() as td:
   copy=Path(td)/'delivery';copy.mkdir()
   replacements={
    'INSTINCTRL-R5-STAGE1-RESET-AVAILABILITY-CLEANUP-SUCCESSOR-20260929':current_package,
    'T-e4cb1d07a9fa12a7':current_task,
    'dd95d3f64d903bae131da7cf7d05be5d3718fef579027eed7d8f6e826185917b':predecessor_diff,
   }
   for name in ('handoff_contract.json','b0_enhanced_live_root_cause.json','B0_ENHANCED_LIVE_ROOT_CAUSE_REPORT.md'):
    text=(source/name).read_text()
    for stale,current in replacements.items():text=text.replace(stale,current)
    (copy/name).write_text(text)
   expected=verify_delivery_identity(copy)
   self.assertEqual(expected,{'package_id':current_package,'task_id':current_task,
      'experiment_id':'UNIFIED-H1-R5-STAGE1-B0-RESET-AVAILABILITY-CLEANUP-R1-20260929',
      'predecessor_diff_sha256':predecessor_diff})
   original=json.loads((copy/'handoff_contract.json').read_text())
   stale_values=(('package_id','stale-package'),('stage_id','stale-stage'),
                 ('task_id','stale-task'))
   for key,value in stale_values:
    with self.subTest(key=key):
     changed={**original,key:value};(copy/'handoff_contract.json').write_text(json.dumps(changed))
     with self.assertRaisesRegex(PermissionError,'identity mismatch'):verify_delivery_identity(copy)
   changed=json.loads(json.dumps(original))
   changed['accepted_predecessor_candidate']['diff_sha256']='0'*64
   (copy/'handoff_contract.json').write_text(json.dumps(changed))
   with self.assertRaisesRegex(PermissionError,'identity mismatch'):verify_delivery_identity(copy)

 def test_protected_predecessor_bytes_are_readable_and_hash_bound(self):
  result=verify_predecessor(ROOT/'docs/history/consolidation-20261012/r5/b0_enhanced_delivery/evaluator_evidence_boundary.json',ROOT/'docs/history/consolidation-20261012/r5/predecessor_evidence')
  self.assertIn('live_failure/job/receipt.json',result['files'])
  self.assertIn('live_failure/report/b0_enhanced_live_root_cause.json',result['files'])
 def test_missing_future_postrun_fails_closed(self):
  with self.assertRaisesRegex(PermissionError,'postrun evidence is absent'):
   verify_evaluator_boundary(ROOT/'docs/history/consolidation-20261012/r5/b0_enhanced_delivery/evaluator_evidence_boundary.json',ROOT/'docs/history/consolidation-20261012/r5/predecessor_evidence')
 def test_postrun_job_and_report_bytes_are_hash_verified(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);job=root/'job';report=root/'report';job.mkdir();report.mkdir()
   (job/'receipt.json').write_bytes(b'{"status":"FAILED"}\n');(report/'result.json').write_bytes(b'{"result":"INCONCLUSIVE"}\n')
   rows=[]
   for kind,base,name in (("job",job,"receipt.json"),("report",report,"result.json")):
    path=base/name;data=path.read_bytes();rows.append({'kind':kind,'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
   manifest=root/'postrun-manifest.json';manifest.write_text(json.dumps({'schema_version':'instinctrl-enhanced-b0-postrun-evidence/2','package_id':'INSTINCTRL-R5-STAGE1-RESET-AVAILABILITY-CLEANUP-SUCCESSOR-20260929','experiment_id':'UNIFIED-H1-R5-STAGE1-B0-RESET-AVAILABILITY-CLEANUP-R1-20260929','roots':{'job':str(job.resolve()),'report':str(report.resolve())},'files':rows}))
   self.assertEqual(verify_postrun(manifest)['status'],'VERIFIED')
   (job/'receipt.json').write_bytes(b'tampered')
   with self.assertRaisesRegex(PermissionError,'bytes/hash mismatch'):verify_postrun(manifest)
 def test_production_sealer_binds_complete_job_and_report_inventory(self):
  from rlflow.h1 import seal_postrun_manifest
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);job=root/'job';report=root/'report';job.mkdir();report.mkdir()
   (job/'receipt.json').write_text('{}');(report/'status.json').write_text('{}')
   spec={'id':'UNIFIED-H1-R5-STAGE1-B0-RESET-AVAILABILITY-CLEANUP-R1-20260929','h1':{'mode':'enhanced-b0-diagnostic'}}
   manifest=seal_postrun_manifest(spec,job,report)
   value=json.loads(manifest.read_text());self.assertEqual({r['kind'] for r in value['files']},{'job','report'})
   self.assertEqual(verify_postrun(manifest)['status'],'VERIFIED')
   (report/'unsealed.json').write_text('{}')
   self.assertNotIn('unsealed.json',{r['path'] for r in value['files']})
   with self.assertRaisesRegex(PermissionError,'inventory mismatch'):
    verify_postrun(manifest)
   (report/'unsealed.json').unlink()
   (job/'receipt-link.json').symlink_to(job/'receipt.json')
   with self.assertRaisesRegex(PermissionError,'symlink'):
    verify_postrun(manifest)
 def test_reserved_basename_is_excluded_only_for_canonical_job_manifest(self):
  from rlflow.h1 import seal_postrun_manifest
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);job=root/'job';report=root/'report';job.mkdir();report.mkdir()
   (job/'receipt.json').write_text('{}');(report/'status.json').write_text('{}')
   (job/'nested').mkdir();(report/'nested').mkdir()
   (job/'nested/postrun-evidence-manifest.json').write_text('job nested')
   (report/'postrun-evidence-manifest.json').write_text('report root')
   (report/'nested/postrun-evidence-manifest.json').write_text('report nested')
   spec={'id':'UNIFIED-H1-R5-STAGE1-B0-RESET-AVAILABILITY-CLEANUP-R1-20260929','h1':{'mode':'enhanced-b0-diagnostic'}}
   manifest=seal_postrun_manifest(spec,job,report)
   rows={(row['kind'],row['path']) for row in json.loads(manifest.read_text())['files']}
   self.assertIn(('job','nested/postrun-evidence-manifest.json'),rows)
   self.assertIn(('report','postrun-evidence-manifest.json'),rows)
   self.assertIn(('report','nested/postrun-evidence-manifest.json'),rows)
   self.assertNotIn(('job','postrun-evidence-manifest.json'),rows)
   self.assertEqual(verify_postrun(manifest)['status'],'VERIFIED')
   value=json.loads(manifest.read_text())
   for omitted in (('job','nested/postrun-evidence-manifest.json'),
                   ('report','postrun-evidence-manifest.json'),
                   ('report','nested/postrun-evidence-manifest.json')):
    with self.subTest(omitted=omitted):
     changed={**value,'files':[row for row in value['files'] if (row['kind'],row['path'])!=omitted]}
     manifest.write_text(json.dumps(changed))
     with self.assertRaisesRegex(PermissionError,'inventory mismatch'):
      verify_postrun(manifest)

if __name__=='__main__':unittest.main()
