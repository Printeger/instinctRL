"""CPU probe of reversible integer-bit deltas; does not implement an archive format.

Production records and decoding are unchanged. Estimate replacing an existing
float32 XOR payload with a smaller zigzag modular integer delta, with an extra
64-byte metadata allowance for every replacement. No rounding of floats occurs.
"""
import hashlib
import json
from pathlib import Path
import struct
import time
from unittest.mock import patch
import zlib
import numpy as np

from tools.measure_r6_capacity_v2 import run
from training_workloads.unified_h1.range_archive import encode


def delta_roundtrip(value, base):
    a = np.ascontiguousarray(value).view('<u4')
    b = np.ascontiguousarray(base).view('<u4')
    difference = np.subtract(a, b, dtype=np.uint32)
    zigzag = (difference << np.uint32(1)) ^ np.subtract(
        np.uint32(0), difference >> np.uint32(31), dtype=np.uint32)
    shuffled = zigzag.view(np.uint8).reshape(-1,4).T.copy().tobytes()
    compressed = zlib.compress(shuffled, level=3)
    decoded = np.frombuffer(zlib.decompress(compressed),dtype=np.uint8).reshape(4,-1).T.copy()
    z = decoded.reshape(-1).view('<u4').reshape(a.shape)
    restored_difference = (z >> np.uint32(1)) ^ np.subtract(
        np.uint32(0), z & np.uint32(1), dtype=np.uint32)
    restored = np.add(restored_difference, b, dtype=np.uint32)
    assert restored.tobytes() == a.tobytes()
    return len(compressed)


def measure():
    # All-bit-pattern arithmetic check includes signed zeros, infinities/NaNs,
    # wraparound and largest unsigned values; these are not measurement inputs.
    rng=np.random.default_rng(2026102002)
    edges=np.array([0,1,0x7fffffff,0x80000000,0xffffffff,0x7f800000,0x80000001],dtype='<u4')
    random=rng.integers(0,2**32,size=10000,dtype=np.uint32)
    values=np.concatenate((edges,random))
    delta_roundtrip(values,np.roll(values,1))
    rows=[]

    def instrument(frame, **kwargs):
        record=encode(frame,**kwargs)
        header_size=struct.unpack('<I',record[:4])[0]
        header=json.loads(record[4:4+header_size])
        arrays={k:v.detach().cpu().numpy() for k,v in frame.items() if hasattr(v,'detach')}
        replacements=[];saving=0
        start=time.monotonic()
        for field in header['fields']:
            if field['dtype']!='<f4':continue
            if 'xor_base' in field:
                base=arrays[field['xor_base']]
            elif 'previous_xor_base' in field:
                base=kwargs['previous'][field['previous_xor_base']]
            else:continue
            size=delta_roundtrip(arrays[field['name']],base)
            gain=max(0,field['length']-size-64)
            saving+=gain
            replacements.append(dict(field=field['name'],xor_bytes=field['length'],
                delta_bytes=size,metadata_allowance=64,saved_bytes=gain))
        rows.append(dict(sequence=kwargs['sequence'],block_id=kwargs['block_id'],
            production_record_bytes=len(record),estimated_record_bytes=len(record)-saving,
            probe_encode_and_verify_seconds=time.monotonic()-start,replacements=replacements))
        return record

    with patch('tools.measure_r6_capacity_v2.encode',side_effect=instrument):
        original=run(temporal=True)
    assert len(rows)==24
    by_case={}
    for row in rows:by_case.setdefault(row['block_id'],[]).append(row)
    root=Path(__file__).resolve().parents[1]
    protocol_path=root/'docs/experiments/r6-robustness-auto/TEST_PROTOCOL.json'
    protocol=json.loads(protocol_path.read_text())
    blocks=[]
    for b in protocol['blocks']:
        samples=by_case[f"s{float(b['noise_std_m'])}-d{b['delay_cycles']}"]
        sizes=[v['estimated_record_bytes'] for v in samples]
        drops=len(b.get('drop_policy_steps',[]))
        estimate=sizes[0]+(b['steps']-drops)*max(sizes[1:3])+drops*sizes[3]
        blocks.append(dict(block_id=b['id'],estimated_measurement_bytes=estimate))
    total=sum(b['estimated_measurement_bytes'] for b in blocks)
    return dict(status='CPU_ENCODING_PROBE_NOT_IMPLEMENTED_OR_ACCEPTED',capacity_pass=False,
        original_per_model_bytes=original['per_model_measurement_bytes'],
        estimated_per_model_bytes=total,saved_per_model_bytes=original['per_model_measurement_bytes']-total,
        rows=rows,blocks=blocks,all_bit_pattern_roundtrip_count=len(values),
        production_record_roundtrip_count=24,original_fixture=original,
        source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        limitations=['Synthetic measurements only; not a worst-case storage bound',
            'Replacement header schema/decoder not implemented; 64 bytes per replacement is a planning allowance',
            'Probe timing includes compression plus reconstruction checks and existing encoding; not production throughput',
            'No production source, protocol, quotas or ledger changed'],gpu_queries=0,simulator_initializations=0)


if __name__=='__main__':print(json.dumps(measure(),indent=2))
