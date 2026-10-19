"""Bounded CPU encoding fixture, not a physical experiment or live-size promise."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from training_workloads.unified_h1.range_archive import encode,decode
import struct

ROOT=Path(__file__).resolve().parents[1]


def run():
    rng=np.random.default_rng(2026101901); results=[]
    for label,valid_fraction,sigma in [('uniform-clean',1.,0.),('uniform-noisy',1.,.03),
                                      ('sparse-noisy',.25,.03),('smooth-noisy',1.,.03)]:
        shape=(32,21240)
        if label=='smooth-noisy':
            clean=np.broadcast_to(np.linspace(.8,40,21240,dtype=np.float32),shape).copy()
        else:clean=rng.uniform(.8,40,size=shape).astype(np.float32)
        valid=rng.uniform(size=shape)<valid_fraction;clean[~valid]=40
        delivered=clean.copy()
        if sigma:
            noise=np.clip(rng.normal(0,sigma,size=int(valid.sum())),-3*sigma,3*sigma).astype(np.float32)
            delivered[valid]=np.clip(delivered[valid]+noise,.1,40)
        f=dict(clean_ranges=clean,clean_validity=valid,delivered_ranges=delivered,
               delivered_validity=valid.copy(),acquisition_time=np.zeros((32,1),dtype=np.float32),
               delivery_time=np.zeros((32,1),dtype=np.float32),frame_age=np.zeros((32,1),dtype=np.float32),
               drop=np.zeros((32,1),dtype=bool),frame_available=np.ones((32,1),dtype=bool),
               noise_rng_identity=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(32)])
        start=time.monotonic();record=encode(f,sequence=0,run_id='CPU-SIZE-FIXTURE',block_id=label)
        seconds=time.monotonic()-start
        size=struct.unpack('<I',record[:4])[0];header=json.loads(record[4:4+size]);restored=decode(header,record[4+size:])
        assert all(f[k].dtype==restored[k].dtype and f[k].shape==restored[k].shape and f[k].tobytes()==restored[k].tobytes() for k in f if k!='noise_rng_identity')
        results.append(dict(label=label,valid_fraction=valid_fraction,sigma_m=sigma,
                            input_array_bytes=sum(v.nbytes for v in f.values() if isinstance(v,np.ndarray)),
                            encoded_bytes=len(record),encoding_seconds=seconds,bitwise_roundtrip=True))
    protocol=json.loads((ROOT/'docs/experiments/r6-robustness-auto/TEST_PROTOCOL.json').read_text())
    steps=sum(b['steps'] for b in protocol['blocks']);noisy=sum(b['steps'] for b in protocol['blocks'] if b['noise_std_m'])
    sizes={r['label']:r['encoded_bytes'] for r in results}
    per_model=(steps-noisy)*sizes['uniform-clean']+noisy*sizes['uniform-noisy']
    return dict(schema_version='r6-cpu-archive-size-fixture/1',status='CPU_FIXTURE_COMPLETE_NOT_LIVE_CAPACITY_PASS',
                cases=results,new_evaluation_policy_steps_per_model=steps,noisy_steps_per_model=noisy,
                independent_uniform_fixture_estimate_bytes_per_model=per_model,
                independent_uniform_fixture_estimate_bytes_final_four_models=4*per_model,
                note='Includes one full32-lane frame per new evaluation policy step; reset frames, original suite, training, motion/force logs, snapshots and video still require allocation. Uniform fixtures are not a mathematical compression bound. CPU encoding time is measured, not simulator throughput.',
                gpu_queries=0,simulator_initializations=0,fixture_seed=2026101901)


if __name__=='__main__':print(json.dumps(run(),indent=2))
