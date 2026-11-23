import torch

from instinctlab.observations.robustness import RangeDelivery


def test_delayed_training_primes_only_reset_lanes_and_retains_causal_age():
    chain=RangeDelivery(lanes=2,rays=2,seed=1,stream_id='test',delay_cycles=1,prime_after_reset=True)
    valid=torch.ones((2,2),dtype=torch.bool)
    drops=torch.tensor([[True],[True]])
    first=chain.deliver(torch.full((2,2),2.),valid,torch.full((2,1),1.),~drops)
    assert first['reset_prime'].all() and first['frame_available'].all()
    second=chain.deliver(torch.full((2,2),3.),valid,torch.full((2,1),1.125),~drops)
    assert torch.equal(second['delivered_ranges'],first['delivered_ranges'])
    assert second['frame_age'].flatten().tolist()==[.125,.125]
    chain.reset(torch.tensor([0]))
    third=chain.deliver(torch.full((2,2),4.),valid,torch.full((2,1),1.25),drops)
    assert third['requested_drop'].all()
    assert third['drop'].flatten().tolist()==[False,True]
    assert third['reset_prime'].flatten().tolist()==[True,False]
    assert third['delivered_ranges'].tolist()==[[4.,4.],[2.,2.]]
    assert third['acquisition_time'].flatten().tolist()==[1.25,1.]


def test_no_synthetic_coverage_and_default_evaluation_has_old_startup():
    valid=torch.zeros((1,1),dtype=torch.bool)
    for prime in (False,True):
        chain=RangeDelivery(lanes=1,rays=1,seed=1,stream_id='test',delay_cycles=1,prime_after_reset=prime)
        row=chain.deliver(torch.tensor([[40.]]),valid,torch.tensor([[1.]]),valid)
        assert row['frame_available'].item() is prime
        assert not row['delivered_validity'].item()
        assert ('reset_prime' in row) is prime
