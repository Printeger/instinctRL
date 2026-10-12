"""Read-only hash/steady-window check of preserved R4 B0; never starts simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--timeseries',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('output already exists; preserve evidence')
    series=json.loads(args.timeseries.read_text());source=series['source'];trace=Path(source['trace'])
    manifest_path=trace/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if sha(manifest_path)!=source['manifest_sha256']:raise ValueError('manifest identity mismatch')
    if manifest['condition']!='B0' or manifest['lanes']!=32 or manifest['transitions']!=20160:raise ValueError('historical scope mismatch')
    windows={'Z':(146,165),'T':(80,99),'S':(145,164)};sampled={c:[] for c in windows}
    seen={c:[] for c in windows}
    for item in manifest['chunks']:
        path=trace/item['path']
        if path.is_symlink() or not path.resolve().is_relative_to(trace.resolve()):raise ValueError('chunk escaped trace')
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('chunk identity mismatch')
        case=item['case'];step=item['step'];seen[case].append(step)
        if windows[case][0]<=step<=windows[case][1]:
            chunk=torch.load(path,map_location='cpu',weights_only=True)
            if chunk['condition']!='B0' or chunk['case']!=case or chunk['step']!=step:raise ValueError('chunk metadata mismatch')
            state=chunk['post_step']['pre_reset']['root_state_w'].double()
            sampled[case].append(state[:,7:10])
    for case,count in source['case_steps'].items():
        if sorted(seen[case])!=list(range(count)):raise ValueError('noncontiguous case')
    report={'schema_version':'instinctrl-r5-historical-b0-verification/1','timeseries':str(args.timeseries.resolve()),
            'timeseries_sha256':sha(args.timeseries),'trace':str(trace),'manifest_sha256':sha(manifest_path),
            'verified_chunks':sum(map(len,seen.values())),'historical_transitions':20160,
            'new_transitions':0,'simulation_initializations':0,'training_updates':0,'windows':{},
            'scope':'Historical aggregate B0 only. Lacks per-body COM/wrenches/reset-instant evidence; not equivalent to enhanced B0 or current repaired plant.'}
    for case,values in sampled.items():
        velocity=torch.stack(values)
        report['windows'][case]={'steps':list(windows[case]),'velocity_mean_world_m_s':velocity.mean((0,1)).tolist(),
            'lane_time_mean_world_m_s':velocity.mean(0).tolist(),
            'velocity_min_world_m_s':velocity.amin((0,1)).tolist(),'velocity_max_world_m_s':velocity.amax((0,1)).tolist()}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='windows'},indent=2))


if __name__=='__main__':main()
