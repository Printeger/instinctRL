"""CPU evidence that commit/reveal seeds affect actual R6 reset/noise draws."""
import copy
import json
from pathlib import Path

import pytest
import torch

from training_workloads.unified_h1.robustness_holdout import commitment, reveal
from training_workloads.unified_h1.robustness_scenarios import reset_spec
from instinctlab.observations.robustness import RangeDelivery


def fixture():
    nonce = '12' * 32
    return nonce, commitment(nonce, protocol_sha256='34' * 32, freeze_sha256='56' * 32)


def test_commitment_contains_no_nonce_or_seed_and_replay_is_exact():
    nonce, record = fixture()
    assert nonce not in json.dumps(record)
    assert set(record) == {'schema_version', 'protocol_sha256', 'freeze_sha256', 'nonce_commitment'}
    assert reveal(record, nonce) == reveal(copy.deepcopy(record), nonce)
    seeds = reveal(record, nonce)['realization_seeds']
    assert len(set(seeds)) == 3 and all(0 <= seed < 2**31 for seed in seeds)


@pytest.mark.parametrize('field', ['protocol_sha256', 'freeze_sha256', 'nonce_commitment', 'schema_version'])
def test_tampered_binding_cannot_reveal(field):
    nonce, record = fixture()
    record[field] = '78' * 32
    with pytest.raises(ValueError):
        reveal(record, nonce)


def test_wrong_nonce_and_extra_field_rejected():
    nonce, record = fixture()
    with pytest.raises(ValueError):
        reveal(record, 'ab' * 32)
    with pytest.raises(ValueError):
        reveal(dict(record, model='candidate'), nonce)


def test_three_realizations_change_actual_reset_and_noise_but_pair_exactly():
    nonce, record = fixture()
    seeds = reveal(record, nonce)['realization_seeds']
    root = Path(__file__).resolve().parents[1]
    blocks = json.loads((root / 'docs/experiments/r6-robustness-auto/TEST_PROTOCOL.json').read_text())['blocks']
    block = next(b for b in blocks if b['id'] == 'S-noise')
    def sample(seed):
        reset = reset_spec(block, seed=seed, split='final', lane=0, mount_position_m=[0, 0, .05])
        chain = RangeDelivery(lanes=32, rays=2, seed=seed, stream_id='final/S-noise',
                              sigma_m=.01, delay_cycles=0)
        frame = chain.deliver(torch.full((32, 2), 2.), torch.ones(32, 2, dtype=torch.bool),
                              torch.zeros(32, 1), torch.zeros(32, 1, dtype=torch.bool))
        return reset['root_position_local_m'], frame
    samples = [sample(seed) for seed in seeds]
    assert len({tuple(position) for position, _ in samples}) == 3
    for seed, (position, frame) in zip(seeds, samples):
        paired_position, paired_frame = sample(seed)
        assert paired_position == position
        for key in frame:
            if isinstance(frame[key], torch.Tensor):
                assert torch.equal(frame[key], paired_frame[key])
    # Compare delivered values, not seed labels or RNG metadata.
    ranges = [frame['delivered_ranges'] for _, frame in samples]
    assert all(not torch.equal(ranges[i], ranges[j]) for i in range(3) for j in range(i))
