"""Project heading proposals inside the original onset slew bound before proof."""
import argparse
from pathlib import Path
from unittest.mock import patch

from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked

import replay_egress as replay


original_proof = replay.issued_proof


def projected_proof(option,scan,state,config,fit,targets,horizon):
    if horizon==.60 and len(targets)>2:
        previous = state[0,48:51]
        # Keep both reference actions untouched and re-prove the clipped menu.
        limit = (targets[0]-previous).norm()
        changes = targets[2:]-previous
        clipped = (limit/changes.norm(dim=1,keepdim=True).clamp_min(1.e-6)).clamp(max=1.)
        targets = targets.clone()
        targets[2:] = previous+clipped*changes
    return original_proof(option,scan,state,config,fit,targets,horizon)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane',required=True,type=int,choices=(19,22))
    parser.add_argument('--lag',required=True,type=float,choices=(.25,.35,.45))
    parser.add_argument('--mover',default='native',choices=('native','mirrored','earlier'))
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    assert not args.output.exists()
    file=Path(__file__).resolve()
    registration=file.with_name('INPUTS.json')
    release=load(registration)
    for ref in release.values():
        checked(ref)
    inputs=dict(load(checked(release['crawl_inputs'])),replay_harness=release['handoff_harness'])
    with patch.object(replay,'issued_proof',projected_proof):
        result=replay.run(inputs,args.lane,args.lag,'forecast',args.mover)
    assert not stop_paths(),stop_paths()
    print(save(args.output,dict(inputs=release['crawl_inputs'],replay=release['crawl_replay'],
        direction_inputs=pin(registration),direction_replay=pin(file.with_name('replay_egress.py')),
        slew_projection=pin(file),direction_arm='forecast_projected_slew',plane_reproof=True,**result)))


if __name__=='__main__':
    main()
