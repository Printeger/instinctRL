import tempfile
from pathlib import Path
import unittest
import json
import struct
import numpy as np
from training_workloads.unified_h1.range_archive import RangeArchive,read_archive,encode,decode,ARRAY_KEYS


def frame():
    r=np.array([[.1,2.,40.,-0.]],dtype=np.float32)
    v=np.array([[True,True,False,True]])
    return dict(clean_ranges=r,delivered_ranges=r.copy(),clean_validity=v,delivered_validity=v.copy(),
                acquisition_time=np.array([[0.]],dtype=np.float32),delivery_time=np.array([[.1]],dtype=np.float32),
                frame_age=np.array([[.1]],dtype=np.float32),drop=np.array([[False]]),
                frame_available=np.array([[True]]),noise_rng_identity=['a'*64])


class ArchiveTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'ranges.bin'
    def tearDown(self):self.tmp.cleanup()
    def writer(self,cap=100000):return RangeArchive(self.path,run_id='CPU',block_id='S-noise',maximum_bytes=cap)
    def read(self,count):return list(read_archive(self.path,run_id='CPU',block_id='S-noise',expected_frames=count))

    def delayed_pair(self):
        rng=np.random.default_rng(123);f=frame();g=frame()
        for obj in (f,g):
            obj['clean_ranges']=rng.uniform(.1,40,(1,4096)).astype(np.float32)
            obj['clean_validity']=np.ones((1,4096),dtype=bool);obj['delivered_validity']=obj['clean_validity'].copy()
        f['delivered_ranges']=f['clean_ranges'].copy();g['delivered_ranges']=f['clean_ranges'].copy()
        return f,g

    def test_delayed_frame_reference_saves_bytes_and_requires_exact_previous(self):
        f,g=self.delayed_pair();previous={k:f[k].copy() for k in ARRAY_KEYS}
        old=encode(g,sequence=1,run_id='CPU',block_id='S-noise')
        new=encode(g,sequence=1,run_id='CPU',block_id='S-noise',previous=previous)
        self.assertLess(len(new),len(old)*.8)
        length=struct.unpack('<I',new[:4])[0];header=json.loads(new[4:4+length]);payload=new[4+length:]
        with self.assertRaisesRegex(ValueError,'PREVIOUS_FRAME_REQUIRED'):decode(header,payload)
        restored=decode(header,payload,previous=previous)
        self.assertEqual(restored['delivered_ranges'].tobytes(),g['delivered_ranges'].tobytes())
        previous['clean_ranges'].fill(1.);previous['delivered_ranges'].fill(1.)
        with self.assertRaises(ValueError):decode(header,payload,previous=previous)

    def test_reader_mutation_cannot_corrupt_next_frame_reference(self):
        f,g=self.delayed_pair()
        with self.writer() as out:out.append(f);out.append(g)
        reader=read_archive(self.path,run_id='CPU',block_id='S-noise',expected_frames=2)
        first=next(reader);first['clean_ranges'].fill(1.);first['delivered_ranges'].fill(1.)
        self.assertEqual(next(reader)['delivered_ranges'].tobytes(),g['delivered_ranges'].tobytes())

    def test_writer_copies_previous_and_reads_legacy_independent_frames(self):
        f,g=self.delayed_pair();first=f['clean_ranges'].copy()
        with self.writer() as out:
            out.append(f);f['clean_ranges'].fill(1.);f['delivered_ranges'].fill(1.);out.append(g)
        self.assertEqual(self.read(2)[0]['clean_ranges'].tobytes(),first.tobytes())
        legacy=encode(g,sequence=0,run_id='CPU',block_id='S-noise')
        self.path.write_bytes(legacy)
        self.assertEqual(self.read(1)[0]['delivered_ranges'].tobytes(),g['delivered_ranges'].tobytes())

    def test_delayed_noise_uses_previous_xor_and_preserves_exact_float_bits(self):
        f,g=self.delayed_pair();g['delivered_ranges']+=np.float32(.01)
        previous={k:f[k].copy() for k in ARRAY_KEYS}
        record=encode(g,sequence=1,run_id='CPU',block_id='S-noise',previous=previous)
        size=struct.unpack('<I',record[:4])[0];header=json.loads(record[4:4+size]);payload=record[4+size:]
        self.assertTrue(any('previous_xor_base' in item for item in header['fields']))
        restored=decode(header,payload,previous=previous)
        self.assertEqual(restored['delivered_ranges'].tobytes(),g['delivered_ranges'].tobytes())
        for item in header['fields']:
            if 'previous_sequence' in item:item['previous_sequence']+=1
        with self.assertRaisesRegex(ValueError,'PREVIOUS_FRAME_REQUIRED'):decode(header,payload,previous=previous)

    def test_bit_exact_roundtrip_dedup_signed_zero_and_float64(self):
        f=frame();g=frame();g['delivered_ranges']=g['clean_ranges'].astype(np.float64)+.001
        with self.writer() as out:out.append(f);out.append(g)
        for original,restored in zip((f,g),self.read(2)):
            for key in original:
                if key=='noise_rng_identity':self.assertEqual(original[key],restored[key])
                else:
                    self.assertEqual(original[key].dtype,restored[key].dtype)
                    self.assertEqual(original[key].tobytes(),restored[key].tobytes())
        self.assertTrue(np.signbit(self.read(2)[0]['clean_ranges'][0,3]))

    def test_quota_rejects_before_write_and_keeps_previous_frame(self):
        f=frame();size=len(encode(f,sequence=0,run_id='CPU',block_id='S-noise'))
        with self.writer(cap=size) as out:
            out.append(f)
            with self.assertRaises(ValueError):out.append(f)
        self.assertEqual(self.path.stat().st_size,size)
        self.assertEqual(len(self.read(1)),1)
        with self.assertRaises(ValueError):self.read(2)

    def test_truncation_and_corruption_rejected(self):
        with self.writer() as out:out.append(frame())
        original=self.path.read_bytes()
        for raw in (original[:-1],original[:2],original[:-1]+bytes([original[-1]^1])):
            self.path.write_bytes(raw)
            with self.assertRaises(ValueError):self.read(1)

    def test_duplicate_open_cannot_overwrite_evidence(self):
        with self.writer() as out:out.append(frame())
        before=self.path.read_bytes()
        with self.assertRaises(FileExistsError):self.writer()
        self.assertEqual(self.path.read_bytes(),before)

    def test_extra_frame_and_wrong_identity_rejected(self):
        with self.writer() as out:out.append(frame());out.append(frame())
        with self.assertRaises(ValueError):self.read(1)
        with self.assertRaises(ValueError):
            list(read_archive(self.path,run_id='OTHER',block_id='S-noise',expected_frames=2))

    def test_nonfinite_and_missing_fields_rejected(self):
        f=frame();f['clean_ranges'][0,0]=np.nan
        with self.assertRaises(ValueError):encode(f,sequence=0,run_id='CPU',block_id='S-noise')
        del f['clean_ranges']
        with self.assertRaises(ValueError):encode(f,sequence=0,run_id='CPU',block_id='S-noise')


if __name__=='__main__':unittest.main()
