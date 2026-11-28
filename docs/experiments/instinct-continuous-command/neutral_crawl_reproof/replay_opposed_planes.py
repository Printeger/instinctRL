"""Compare rotation-only corroboration from opposing stationary wall patches."""
import argparse
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
from instinctlab.control.escape_passage import _return_seed
from instinctlab.control.registered_mover_entry import (
    _RANGE_LIMIT, _observed_planes, _rotations, reprove_plane_alignment,
)
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save

import replay


def reprove_rotation(lidar,state,rays):
    regular = reprove_plane_alignment(lidar,state,rays)
    if regular is not None:
        return regular
    history = state.reshape(4,13)
    if float(history[:,:3].norm(dim=1).max()) > .55:
        return None
    _,_,times = _return_seed(lidar,state,maximum_range_m=_RANGE_LIMIT)
    times = times[0]
    gaps = times.diff()
    if bool(((gaps<.065)|(gaps>.135)).any()):
        return None
    up = -history[-1,3:6].cpu().numpy()
    up /= np.linalg.norm(up)
    forward = np.array([1.,0.,0.])-up*up[0]
    forward_norm = np.linalg.norm(forward)
    if forward_norm < .50:
        return None
    forward /= forward_norm
    basis = np.stack((forward,np.cross(up,forward)),axis=1)
    planes = _observed_planes(lidar.cpu().numpy(),rays.cpu().numpy(),_rotations(history,times),basis,up)
    for i,(first,near_a,_,angle_a,count_a) in enumerate(planes):
        for second,near_b,_,angle_b,count_b in planes[i+1:]:
            if min(count_a,count_b)<28 or first@second>-math.cos(.02):
                continue
            # Opposing face spacing is invariant to in-plane translation.
            width = near_a+near_b
            if np.ptp(width)>.004:
                continue
            error = max(float(np.max(np.abs(angles-angles[-1]))) for angles in (angle_a,angle_b))
            if error<=.006:
                return max(.0015,error),2,float(gaps.min())
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane',type=int,required=True,choices=(19,22))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    folder=Path(__file__).resolve().parent
    inputs=folder/'INPUTS.json'
    with patch.object(replay,'reprove_plane_alignment',reprove_rotation):
        result=replay.run(load(inputs),args.lane,.35,'native','recorded',True)
    assert not stop_paths(), stop_paths()
    print(save(args.output,dict(inputs=pin(inputs),replay=pin(folder/'replay.py'),rotation_prototype=pin(__file__),
                                plane_reproof=True,**result)))


if __name__=='__main__':
    main()
