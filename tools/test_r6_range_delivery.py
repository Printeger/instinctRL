import unittest
import torch
from instinctlab.observations.robustness import RangeDelivery


class DeliveryTests(unittest.TestCase):
    def new(self,**kw):return RangeDelivery(lanes=2,rays=64,seed=17,stream_id='development/S-noise',**kw)
    def step(self,chain,t,value=2.,drop=False):
        return chain.deliver(torch.full((2,64),value),torch.ones((2,64),dtype=torch.bool),
                             torch.full((2,1),float(t)),torch.full((2,1),drop,dtype=torch.bool))

    def test_delay_and_drop_retain_acquisition_time(self):
        c=self.new(delay_cycles=1)
        first=self.step(c,0.)
        self.assertFalse(bool(first['frame_available'].any()))
        self.assertFalse(bool(first['delivered_validity'].any()))
        second=self.step(c,.1,value=3.)
        self.assertTrue(torch.equal(second['delivered_ranges'],torch.full((2,64),2.)))
        self.assertTrue(torch.equal(second['acquisition_time'],torch.zeros((2,1))))
        dropped=self.step(c,.2,value=4.,drop=True)
        self.assertTrue(torch.equal(dropped['delivered_ranges'],second['delivered_ranges']))
        self.assertTrue(torch.equal(dropped['acquisition_time'],second['acquisition_time']))
        self.assertTrue(torch.allclose(dropped['frame_age'],torch.full((2,1),.2)))
        fourth=self.step(c,.3,value=5.)
        self.assertTrue(torch.equal(fourth['delivered_ranges'],torch.full((2,64),4.)))

    def test_noise_is_bounded_reproducible_and_changes_by_seed(self):
        c=self.new(sigma_m=.01);d=self.new(sigma_m=.01)
        a=self.step(c,0.);b=self.step(d,0.)
        self.assertTrue(torch.equal(a['delivered_ranges'],b['delivered_ranges']))
        self.assertFalse(torch.equal(a['delivered_ranges'],a['clean_ranges']))
        self.assertLessEqual(float((a['delivered_ranges']-a['clean_ranges']).abs().max()),.030001)
        e=RangeDelivery(lanes=2,rays=64,seed=18,stream_id='development/S-noise',sigma_m=.01)
        self.assertFalse(torch.equal(a['delivered_ranges'],self.step(e,0.)['delivered_ranges']))

    def test_zero_noise_does_not_consume_random_stream(self):
        c=self.new();before=c.rng_states();self.step(c,0.)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,c.rng_states())))

    def test_invalid_beams_not_noised_and_input_unmodified(self):
        c=self.new(sigma_m=.03)
        ranges=torch.full((2,64),40.); valid=torch.zeros((2,64),dtype=torch.bool)
        before=c.rng_states()
        result=c.deliver(ranges,valid,torch.zeros(2,1),torch.zeros(2,1,dtype=torch.bool))
        self.assertTrue(torch.equal(result['delivered_ranges'],ranges))
        self.assertFalse(all(torch.equal(a,b) for a,b in zip(before,c.rng_states())))
        result['delivered_ranges'].zero_()
        self.assertTrue(torch.equal(ranges,torch.full((2,64),40.)))
        self.assertTrue(torch.equal(c.last['ranges'],ranges))

    def test_selected_reset_cannot_leak_previous_episode_frame(self):
        c=self.new(delay_cycles=1);self.step(c,0.);self.step(c,.1)
        c.reset(torch.tensor([0],dtype=torch.long))
        times=torch.tensor([[0.],[.2]])
        r=c.deliver(torch.full((2,64),3.),torch.ones(2,64,dtype=torch.bool),times,torch.zeros(2,1,dtype=torch.bool))
        self.assertFalse(bool(r['frame_available'][0]))
        self.assertTrue(bool(r['frame_available'][1]))

    def test_common_noise_survives_different_validity_masks(self):
        a=self.new(sigma_m=.01);b=self.new(sigma_m=.01)
        full=torch.full((2,64),2.);partial=full.clone();partial[:,:17]=40.
        valid=torch.ones((2,64),dtype=torch.bool);mask=valid.clone();mask[:,:17]=False
        time=torch.zeros((2,1));drop=torch.zeros((2,1),dtype=torch.bool)
        x=a.deliver(full,valid,time,drop);y=b.deliver(partial,mask,time,drop)
        self.assertTrue(torch.equal(x['delivered_ranges'][:,17:],y['delivered_ranges'][:,17:]))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.rng_states(),b.rng_states())))
        self.assertTrue(torch.equal(self.step(a,.1)['delivered_ranges'],self.step(b,.1)['delivered_ranges']))

    def test_timestamp_regression_rejected(self):
        c=self.new();self.step(c,.2)
        with self.assertRaises(ValueError):self.step(c,.1)
        with self.assertRaises(ValueError):self.step(c,.2)

    def test_actor_history_and_ics_extract_same_delivered_ranges(self):
        from instinctlab.observations.actor import ActorObservationHistory,RAY_COUNT
        from instinctlab.control.pipeline import extract_lidar_history
        c=RangeDelivery(lanes=2,rays=RAY_COUNT,seed=17,stream_id='integration',sigma_m=.01)
        r=c.deliver(torch.full((2,RAY_COUNT),2.),torch.ones(2,RAY_COUNT,dtype=torch.bool),
                    torch.zeros(2,1),torch.zeros(2,1,dtype=torch.bool))
        z=torch.zeros(2,3)
        policy=ActorObservationHistory().update(ranges_m=r['delivered_ranges'],validity=r['delivered_validity'],
            sensor_frame_time_s=r['acquisition_time'],policy_time_s=r['delivery_time'],
            body_angular_velocity_rad_s=z,gravity_direction_body=torch.tensor([[0.,0.,-1.]]*2),
            commanded_body_velocity_m_s=z,previous_issued_body_velocity_m_s=z)
        ranges,validity,reliability=extract_lidar_history(policy['lidar_grid'])
        self.assertTrue(torch.equal(ranges[:,-1].reshape(2,-1),r['delivered_ranges']))
        self.assertEqual(set(policy),{'lidar_grid','state_vec'})
        self.assertEqual(tuple(policy['state_vec'].shape),(2,52))


if __name__=='__main__':unittest.main()
