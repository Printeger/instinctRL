import copy
from pathlib import Path
import tempfile
import unittest
import torch

from instinctlab.observations.robustness import RangeDelivery
from instinctlab.observations.actor import RAY_COUNT
from training_workloads.unified_h1.range_archive import RangeArchive
from training_workloads.unified_h1.robustness_audit import AuditLog
from training_workloads.unified_h1.robustness_measurement_replay import replay_delivery,reset_schedule


class MeasurementReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.block=dict(id='CPU-noise-delay',lanes=32,steps=5,noise_std_m=.01,delay_cycles=1,drop_policy_steps=[3])
        self.resets=[torch.ones(32,dtype=torch.bool)]+[torch.zeros(32,dtype=torch.bool) for _ in range(5)]
        self.resets[4][7]=True

    def frames(self):
        c=RangeDelivery(lanes=32,rays=8,seed=17,stream_id='development/'+self.block['id'],sigma_m=.01,delay_cycles=1)
        counts=torch.zeros(32,1);frames=[]
        for sequence,mask in enumerate(self.resets):
            c.reset(mask.nonzero().flatten());counts[mask]=0
            ranges=torch.full((32,8),2.+sequence*.02);valid=torch.ones(32,8,dtype=torch.bool)
            ranges[:,0]=40.;valid[:,0]=False
            frame=c.deliver(ranges,valid,counts*.1,torch.full((32,1),sequence==3,dtype=torch.bool))
            frames.append(frame);counts+=1
        return frames

    def write(self,frames):
        path=self.root/'ranges.bin'
        with RangeArchive(path,run_id='CPU',block_id=self.block['id'],maximum_bytes=1024**2) as archive:
            for frame in frames:archive.append(frame)
        return path

    def replay(self,path,**kw):
        return replay_delivery(path,self.block,run_id='CPU',seed=kw.get('seed',17),split='development',
            resets=kw.get('resets',self.resets),expected_rays=kw.get('rays',8))

    def test_full_noise_delay_drop_reset_bit_exact(self):
        result=self.replay(self.write(self.frames()))
        self.assertTrue(result['bit_exact_delivery']);self.assertEqual(result['reset_lane_events'],33)
        self.assertFalse(result['scientific_acceptance']);self.assertEqual(result['frames'],6)

    def test_production_beam_count_with_default_replay_contract(self):
        block=dict(self.block,steps=1,drop_policy_steps=[])
        chain=RangeDelivery(lanes=32,rays=RAY_COUNT,seed=17,stream_id='development/'+block['id'],sigma_m=.01,delay_cycles=1)
        path=self.root/'full.bin'
        with RangeArchive(path,run_id='CPU',block_id=block['id'],maximum_bytes=32*1024**2) as archive:
            for sequence in range(2):
                archive.append(chain.deliver(torch.full((32,RAY_COUNT),2.),torch.ones(32,RAY_COUNT,dtype=torch.bool),
                    torch.full((32,1),sequence*.1),torch.zeros(32,1,dtype=torch.bool)))
        result=replay_delivery(path,block,run_id='CPU',seed=17,split='development',resets=self.resets[:2])
        self.assertEqual(result['rays'],21240);self.assertTrue(result['bit_exact_delivery'])

    def test_reencoded_tamper_rejected_not_just_archive_hash(self):
        for field in ('delivered_ranges','acquisition_time','frame_age','delivered_validity','frame_available','drop'):
            with self.subTest(field=field):
                frames=self.frames();v=frames[2][field]
                if v.dtype==torch.bool:v[0,0]=~v[0,0]
                else:v[0,0]+=.01
                path=self.write(frames)
                with self.assertRaisesRegex(ValueError,'DELIVERY_MISMATCH'):self.replay(path)
                path.unlink()

    def test_seed_label_changes_cannot_pass(self):
        with self.assertRaisesRegex(ValueError,'RNG_IDENTITY'):
            self.replay(self.write(self.frames()),seed=18)

    def test_rng_labels_cannot_hide_wrong_noise(self):
        frames=self.frames();other=RangeDelivery(lanes=32,rays=8,seed=18,stream_id='development/'+self.block['id'])
        for f in frames:f['noise_rng_identity']=other.streams
        with self.assertRaisesRegex(ValueError,'DELIVERY_MISMATCH'):
            self.replay(self.write(frames),seed=18)

    def test_unrecorded_reset_rejected(self):
        masks=[x.clone() for x in self.resets];masks[4].zero_()
        with self.assertRaisesRegex(ValueError,'timestamp must advance'):
            self.replay(self.write(self.frames()),resets=masks)

    def test_missing_frame_and_wrong_beam_denominator(self):
        path=self.write(self.frames()[:-1])
        with self.assertRaisesRegex(ValueError,'MISSING_OR_PARTIAL'):self.replay(path)
        with self.assertRaisesRegex(ValueError,'BEAM_DENOMINATOR'):self.replay(path,rays=9)

    def audit(self,bad=False):
        path=self.root/'audit.bin'
        with AuditLog(path,run_id='CPU',block_id=self.block['id'],maximum_bytes=1024**2) as log:
            log.append('identity',dict(block=self.block,seed=17,split='development'))
            log.append('initial',{})
            for step in range(5):
                log.append('step_attempt',dict(policy_step=step,input_measurement_sequence=step))
                for _ in range(20):log.append('force',{})
                log.append('step_complete',dict(policy_step=step,output_measurement_sequence=step+1,
                    reset_mask=[0]*32 if bad else self.resets[step+1].tolist()))
            records=log.records
        return path,records

    def test_audit_schedule_complete_and_exact(self):
        path,count=self.audit()
        result=reset_schedule(path,self.block,run_id='CPU',seed=17,split='development',expected_records=count)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(result,self.resets)))
        with self.assertRaisesRegex(ValueError,'AUDIT_PROTOCOL_IDENTITY'):
            reset_schedule(path,self.block,run_id='CPU',seed=18,split='development',expected_records=count)

    def test_audit_reset_types_not_coerced(self):
        path,count=self.audit(bad=True)
        with self.assertRaisesRegex(ValueError,'AUDIT_RESET_MASK'):
            reset_schedule(path,self.block,run_id='CPU',seed=17,split='development',expected_records=count)


if __name__=='__main__':unittest.main()
