import tempfile
from pathlib import Path
import unittest
import numpy as np
from training_workloads.unified_h1.range_archive import RangeArchive,read_archive,encode


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
