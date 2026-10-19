import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import torch
from tools.test_r6_physics_adapter import Env
from training_workloads.unified_h1.force_archive import MAGIC,encode_force,decode_force
from training_workloads.unified_h1.robustness_audit import AuditLog,read_audit
from training_workloads.unified_h1.logging import _jsonable


class ForceArchiveTests(unittest.TestCase):
    def row(self,dtype=torch.float32):
        e=Env();e.reset();e._r6_physics['ticks']=1000;e.tick()
        p=e.force_rows[0]
        for k,v in p.items():
            if isinstance(v,torch.Tensor):p[k]=v.to(dtype)
        p['dtype']=str(dtype);p['rotor_forces_local_n'][0,0,0]=-0.
        return dict(schema_version='r6-audit/1',run_id='CPU',block_id='CPU',sequence=0,kind='force',payload=p)

    def mutate(self,raw,fn):
        offset=len(MAGIC);size=struct.unpack('<I',raw[offset:offset+4])[0]
        header=json.loads(raw[offset+4:offset+4+size]);fn(header)
        new=json.dumps(header,separators=(',',':')).encode()
        return MAGIC+struct.pack('<I',len(new))+new+raw[offset+4+size:]

    def test_exact_float32_and_float64_roundtrip_and_negative_zero(self):
        for dtype in (torch.float32,torch.float64):
            row=self.row(dtype);raw=encode_force(row);decoded=decode_force(raw)
            self.assertEqual(decoded,_jsonable(row))
            self.assertTrue(torch.signbit(torch.tensor(decoded['payload']['rotor_forces_local_n'][0][0][0])))
            self.assertLess(len(raw),len(json.dumps(_jsonable(row)).encode()))

    def test_invalid_arrays_not_coerced(self):
        for field,value in [('mass_kg',torch.ones(32,6)),('mass_kg',torch.full((32,5),float('nan'))),
                            ('mass_kg',torch.ones(32,5,dtype=torch.int64))]:
            row=self.row();row['payload'][field]=value
            with self.assertRaises(ValueError):encode_force(row)

    def test_corrupt_shape_offset_and_dtype_rejected(self):
        raw=encode_force(self.row())
        for key,value in [('shape',[32,6]),('offset',1),('dtype','|O'),('bytes',0)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                decode_force(self.mutate(raw,lambda h:h['arrays'][0].update({key:value})))

    def test_truncation_and_trailing_data_rejected(self):
        raw=encode_force(self.row())
        for value in (raw[:3],raw[:-1],raw+b'X'):
            with self.assertRaises(ValueError):decode_force(value)

    def test_mixed_legacy_and_binary_audit_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'audit.bin';row=self.row()
            with AuditLog(path,run_id='CPU',block_id='CPU',maximum_bytes=1024**2) as log:
                log.append('force',{'block_tick':0,'fixture_only':True})
                log.append('force',row['payload'])
            records=list(read_audit(path,run_id='CPU',block_id='CPU',expected_records=2))
            self.assertEqual(records[0]['payload'],{'block_tick':0,'fixture_only':True})
            self.assertEqual(records[1]['payload'],_jsonable(row['payload']))
            with path.open('ab') as stream:stream.write(b'x')
            with self.assertRaisesRegex(ValueError,'EXTRA_RECORDS'):
                list(read_audit(path,run_id='CPU',block_id='CPU',expected_records=2))

    def test_quota_checked_before_binary_record_write(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'audit.bin'
            with AuditLog(path,run_id='CPU',block_id='CPU',maximum_bytes=1) as log:
                with self.assertRaisesRegex(ValueError,'QUOTA_BEFORE_WRITE'):log.append('force',self.row()['payload'])
            self.assertEqual(path.stat().st_size,0)


if __name__=='__main__':unittest.main()
