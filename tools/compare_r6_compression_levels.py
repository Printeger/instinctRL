"""Compare immutable CPU codec probes; never authorize execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/experiments/r6-robustness-auto'

def compare(left, right):
    paths = [Path(left), Path(right), BASE/'TEST_PROTOCOL.json',
             BASE/'CAPACITY_COMPOSED_DRAFT_V5.json']
    a, b, protocol, design = [json.loads(p.read_text()) for p in paths]
    assert a['compression_level_probe'] == 3 and b['compression_level_probe'] == 9
    assert a['protocol'] == b['protocol'] and a['code'] == b['code']
    for record in (a,b):
        for pin in [record['protocol'],record['probe_tool'],*record['code']]:
            assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest() == pin['sha256']
    assert [(c['sigma_m'], c['delay_cycles']) for c in a['cases']] == [
        (c['sigma_m'], c['delay_cycles']) for c in b['cases']]
    assert all(c['bitwise_roundtrip'] for x in (a,b) for c in x['cases'])
    rounding = 16*1024**2
    rows = []
    for x in (a,b):
        quota = sum(math.ceil(v['estimated_measurement_bytes']*1.15/rounding)*rounding
                    for v in x['blocks'])
        seconds = sum((v['steps']+1)*max(next(c for c in x['cases']
            if c['sigma_m']==v['noise_std_m'] and c['delay_cycles']==v['delay_cycles'])[
                'encoding_seconds']) for v in protocol['blocks'])
        rows.append(dict(level=x['compression_level_probe'],
            measurement_bytes=x['per_model_measurement_bytes'],
            range_quota_bytes=quota, per_evaluation_encode_seconds=seconds))
    saved = rows[0]['range_quota_bytes']-rows[1]['range_quota_bytes']
    assert rows[0]['range_quota_bytes'] == sum(v['range_quota_bytes'] for v in design['blocks'])
    added_seconds = rows[1]['per_evaluation_encode_seconds']-rows[0]['per_evaluation_encode_seconds']
    projected = design['final_time_seconds']['total_sample_extrapolation']+4*added_seconds
    status = ('CPU_PROBE_REJECT_LEVEL9_AS_CAPACITY_SOLUTION' if projected > 43200
              else 'CPU_PROBE_REQUIRES_FURTHER_CAPACITY_ANALYSIS')
    return dict(status=status, capacity_pass=False,
        inputs=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        levels=rows, range_quota_saved_per_evaluation_bytes=saved,
        added_final_four_evaluation_encode_seconds=4*added_seconds,
        original_final_time_estimate_seconds=design['final_time_seconds']['total_sample_extrapolation'],
        final_time_with_measured_encoding_delta_seconds=projected,
        total_live_wall_cap_seconds=43200,
        rationale='Small storage reduction at substantially higher CPU cost; retain production level3. Timing is a CPU sample extrapolation, not measured live wall or a worst-case proof.',
        coverage='Identical 24-frame CPU fixture per level; clean/noisy, immediate/delayed and dropped frames; original decoder bitwise reconstruction.',
        production_changed=False,ledger_changed=False,gpu_queries=0,simulator_initializations=0,
        unresolved='Original capacity shortfall and native aggregate output bound remain unresolved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('probes', nargs=2)
    args = parser.parse_args()
    print(json.dumps(compare(*args.probes), indent=2))
