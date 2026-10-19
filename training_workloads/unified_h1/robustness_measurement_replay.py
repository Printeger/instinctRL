"""Streaming CPU verification of archived delivery, not physical sensor truth.

Replays the versioned production impairment algorithm. Independent review must
still validate that algorithm and actor/ICS wiring, reset physics and clocks.
"""
from pathlib import Path
import numpy as np
import torch

from instinctlab.observations.actor import RAY_COUNT
from instinctlab.observations.robustness import RangeDelivery
from .range_archive import read_archive, ARRAY_KEYS
from .robustness_audit import read_audit
from .robustness_analysis import pin


def reset_schedule(path, block, *, run_id, seed, split, expected_records):
    """Extract exact post-step resets; never infer resets from a regressing clock."""
    steps=block['steps']
    if type(expected_records) is not int or not 0<expected_records<=2+32+steps*(20+2+32):
        raise ValueError('AUDIT_RECORD_COUNT_BOUND')
    resets=[torch.ones(32,dtype=torch.bool)]
    attempts=0; completions=0; initial=0; identities=0; forces=0
    for row in read_audit(path,run_id=run_id,block_id=block['id'],expected_records=expected_records):
        payload=row['payload'];kind=row['kind']
        if kind=='identity':
            if row['sequence']!=0 or payload['block']!=block or payload['seed']!=seed or payload['split']!=split:
                raise ValueError('AUDIT_PROTOCOL_IDENTITY')
            identities+=1
        elif kind=='initial':initial+=1
        elif kind=='step_attempt':
            if attempts!=completions or payload['policy_step']!=attempts or payload['input_measurement_sequence']!=attempts:
                raise ValueError('AUDIT_ATTEMPT_ORDER')
            attempts+=1
        elif kind=='step_complete':
            if attempts!=completions+1 or payload['policy_step']!=completions or payload['output_measurement_sequence']!=completions+1:
                raise ValueError('AUDIT_COMPLETION_ORDER')
            mask=payload['reset_mask']
            if not isinstance(mask,list) or len(mask)!=32 or any(type(v) is not bool for v in mask):
                raise ValueError('AUDIT_RESET_MASK')
            resets.append(torch.tensor(mask,dtype=torch.bool));completions+=1
        elif kind=='force':forces+=1
        elif kind!='reset':raise ValueError('AUDIT_UNKNOWN_KIND')
    if identities!=1 or initial!=1 or attempts!=steps or completions!=steps or forces!=steps*20:
        raise ValueError('AUDIT_COVERAGE')
    return resets


def replay_delivery(path, block, *, run_id, seed, split, resets, expected_rays=RAY_COUNT):
    if block['lanes']!=32 or len(resets)!=block['steps']+1 or split not in ('development','final'):
        raise ValueError('MEASUREMENT_DENOMINATOR')
    path=Path(path)
    if path.is_symlink() or path.absolute()!=path.resolve():raise ValueError('MEASUREMENT_PATH_ALIAS')
    before=pin(path);chain=None;count=0;reset_count=0
    for sequence, frame in enumerate(read_archive(path,run_id=run_id,block_id=block['id'],expected_frames=block['steps']+1)):
        clean=frame['clean_ranges']
        if clean.shape!=(32,expected_rays) or clean.dtype not in (np.dtype('float32'),np.dtype('float64')):
            raise ValueError('MEASUREMENT_BEAM_DENOMINATOR')
        if chain is None:
            chain=RangeDelivery(lanes=32,rays=expected_rays,seed=seed,stream_id=split+'/'+block['id'],
                sigma_m=block['noise_std_m'],delay_cycles=block['delay_cycles'],dtype=torch.from_numpy(clean).dtype)
        mask=resets[sequence]
        if not isinstance(mask,torch.Tensor) or mask.shape!=(32,) or mask.dtype!=torch.bool or mask.device.type!='cpu':
            raise ValueError('MEASUREMENT_RESET_MASK')
        if sequence==0 and not bool(mask.all()):raise ValueError('MEASUREMENT_INITIAL_RESET')
        indices=mask.nonzero().flatten();chain.reset(indices);reset_count+=len(indices)
        drop=torch.full((32,1),sequence in block.get('drop_policy_steps',[]),dtype=torch.bool)
        expected=chain.deliver(torch.from_numpy(clean),torch.from_numpy(frame['clean_validity']),
            torch.from_numpy(frame['delivery_time']),drop)
        if frame['noise_rng_identity']!=expected['noise_rng_identity']:
            raise ValueError(f'MEASUREMENT_RNG_IDENTITY:{sequence}')
        for field in ARRAY_KEYS:
            actual=frame[field];wanted=expected[field].numpy()
            if actual.dtype!=wanted.dtype or actual.shape!=wanted.shape or actual.tobytes()!=wanted.tobytes():
                raise ValueError(f'MEASUREMENT_DELIVERY_MISMATCH:{sequence}:{field}')
        count+=1
    if pin(path)!=before:raise ValueError('MEASUREMENT_INPUT_CHANGED')
    return dict(schema_version='r6-measurement-replay/1',archive=before,block_id=block['id'],run_id=run_id,
        seed=seed,split=split,frames=count,lanes=32,rays=expected_rays,reset_lane_events=reset_count,
        bit_exact_delivery=True,noise_algorithm='R6/range-noise/cpu-v1',scientific_acceptance=False,
        limitation='Replays production impairment only; sensor truth, reset/physics clocks, actor/ICS wiring and independent review remain required')


def replay_block_measurements(folder,block,*,run_id,seed,split,result):
    folder=Path(folder);audit=folder/'audit.bin';before=pin(audit)
    resets=reset_schedule(audit,block,run_id=run_id,seed=seed,split=split,expected_records=result['audit_records'])
    replay=replay_delivery(folder/'measurements.bin',block,run_id=run_id,seed=seed,split=split,resets=resets)
    if pin(audit)!=before:raise ValueError('MEASUREMENT_AUDIT_CHANGED')
    if result['measurement_frames']!=replay['frames'] or result['measurement_bytes']!=replay['archive']['bytes']:
        raise ValueError('MEASUREMENT_RESULT_COUNTS')
    replay['audit']=before
    return replay
