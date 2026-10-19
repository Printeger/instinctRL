"""CPU capacity fixture with actual delivery delay/drop; never final split data."""
from pathlib import Path
import hashlib
import argparse
import json
import struct
import time
import numpy as np
import torch
from instinctlab.observations.robustness import RangeDelivery
from training_workloads.unified_h1.range_archive import encode,decode

ROOT=Path(__file__).resolve().parents[1]


def run(*,temporal=False):
    cases=[];rng=np.random.default_rng(2026102001)
    for sigma in (0.,.01,.03):
        for delay in (0,1):
            chain=RangeDelivery(lanes=32,rays=21240,seed=2026102001,stream_id='CPU-CAPACITY',sigma_m=sigma,delay_cycles=delay)
            sizes=[];seconds=[]
            previous=None
            for sequence in range(4):
                clean=torch.from_numpy(rng.uniform(.8,40,(32,21240)).astype(np.float32))
                frame=chain.deliver(clean,torch.ones_like(clean,dtype=torch.bool),torch.full((32,1),sequence*.1),torch.full((32,1),sequence==3,dtype=torch.bool))
                start=time.monotonic();record=encode(frame,sequence=sequence,run_id='CPU-CAPACITY',block_id=f's{sigma}-d{delay}',previous=previous if temporal else None)
                seconds.append(time.monotonic()-start);sizes.append(len(record))
                size=struct.unpack('<I',record[:4])[0];decoded=decode(json.loads(record[4:4+size]),record[4+size:],previous=previous if temporal else None)
                for key,value in frame.items():
                    if isinstance(value,torch.Tensor):assert value.numpy().tobytes()==decoded[key].tobytes(),key
                previous={k:v.copy() for k,v in decoded.items() if isinstance(v,np.ndarray)}
            cases.append(dict(sigma_m=sigma,delay_cycles=delay,frame_bytes=sizes,encoding_seconds=seconds,bitwise_roundtrip=True))
    protocol_path=ROOT/'docs/experiments/r6-robustness-auto/TEST_PROTOCOL.json';protocol=json.loads(protocol_path.read_text())
    blocks=[]
    for block in protocol['blocks']:
        case=next(c for c in cases if c['sigma_m']==block['noise_std_m'] and c['delay_cycles']==block['delay_cycles'])
        drops=len(block.get('drop_policy_steps',[]))
        estimate=case['frame_bytes'][0]+(block['steps']-drops)*max(case['frame_bytes'][1:3])+drops*case['frame_bytes'][3]
        blocks.append(dict(block_id=block['id'],frames=block['steps']+1,estimated_measurement_bytes=estimate))
    total=sum(b['estimated_measurement_bytes'] for b in blocks)
    return dict(status='CPU_FIXTURE_NOT_CAPACITY_ACCEPTANCE',encoding_mode='previous-frame-v2' if temporal else 'independent-v1',seed=2026102001,shape=[32,21240],dtype='float32',
        cases=cases,blocks=blocks,per_model_measurement_bytes=total,final_four_measurement_bytes=4*total,
        aggregation='Weight normal and dropped delivery by registered block counts; first preliminary v2 output used the largest sampled frame for every step, including blocks with no drop.',
        prior_estimate_limitation='Prior same-frame fixture omitted delayed frames whose current clean and delivered ranges differ across acquisition times.',
        estimate_limitations='Independent uniform clean frame each acquisition; includes initial invalid delay and one dropped delivery. This is not physical data or a worst-case bound. Does not include training, native logs, motion/force/reset audit, snapshots or videos.',
        protocol=dict(path=str(protocol_path),sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest()),
        code=[dict(path=str(ROOT/p),sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in ['tools/measure_r6_capacity_v2.py','training_workloads/unified_h1/range_archive.py','isaaclab-training/instinctlab/observations/robustness.py']],gpu_queries=0,simulator_initializations=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--temporal',action='store_true')
    args=parser.parse_args();print(json.dumps(run(temporal=args.temporal),indent=2))
