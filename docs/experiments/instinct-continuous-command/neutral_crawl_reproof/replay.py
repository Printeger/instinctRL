"""Compare the same low-speed stop replay with and without observed-plane renewal."""
import argparse
import gzip
import json
from pathlib import Path
import runpy
from unittest.mock import patch

import numpy as np
import torch
from instinctlab.control.escape_recovery import MOVER_PASSAGE_REPROOF_25
from instinctlab.control.measured_curve_deflection import fresh_motion_history
from instinctlab.control.registered_mover_entry import reprove_plane_alignment
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked, git


def run(inputs, lane, lag, variant, attitude, registration):
    assert not stop_paths(), stop_paths()
    for key, value in inputs.items():
        if key != 'v60_commit':
            checked(value)
    archive = Path(inputs['trace']['path']).parent
    with gzip.open(archive/'candidate-layout_routes.jsonl.gz','rt') as stream:
        initial = json.loads(next(stream))
        rows = [json.loads(next(stream)) for _ in range(254)]
    assert pin(archive/'candidate-layout_routes-clearance.f32') == inputs['physical']
    assert pin(archive/'candidate-lidar-rays.f32') == inputs['rays']
    assert git('rev-parse','HEAD') == inputs['v60_commit'] and not git('branch','--show-current')
    harness = runpy.run_path(str(checked(inputs['replay_harness'])))
    execute, legacy = harness['execute'], harness['StopTrial']
    context = execute.__globals__
    assert legacy.apply.__globals__ is context
    assert context['initial'] == initial
    assert np.array_equal(context['rays'],np.fromfile(archive/'candidate-lidar-rays.f32',dtype='<f4').reshape(-1,3))
    context.update(root=archive, rows=rows, GRAVITY_PREVIEW_RENEWAL_25=MOVER_PASSAGE_REPROOF_25)
    original_fit = context['_fit_return']
    rays = context['ray_tensor']

    class Trial(legacy):
        def apply(self, policy, action, config):
            assert not stop_paths(), stop_paths()
            if not registration:
                return super().apply(policy, action, config)
            scan, state = policy['lidar_grid'], policy['state_vec']
            witness = None
            proof_attempted = False
            failure = None

            def fresh(history):
                nonlocal witness, proof_attempted, failure
                if bool(fresh_motion_history(history)[0]):
                    return True
                proof_attempted = True
                no_gyro = history.clone()
                no_gyro[:, :, :3] = 0.
                if not bool(fresh_motion_history(no_gyro)[0]):
                    failure = 'non_angular_history'
                    return False
                witness = reprove_plane_alignment(scan, state, rays)
                if witness is None:
                    failure = 'plane_alignment'
                return witness is not None

            def fit(*args,**kwargs):
                if witness is None:
                    return original_fit(*args,**kwargs)
                aligned = original_fit(*args,**kwargs,align_gravity=True)
                if aligned is not None:
                    # Existing dynamic margin stays in place. Add a bound on
                    # range displacement for up to 2.65 m of fitted returns.
                    aligned.radius += 3.*witness[0]
                return aligned

            with patch.dict(context, fresh_motion_history=fresh, _fit_return=fit):
                selected = super().apply(policy, action, config)
            if proof_attempted:
                self.last_info['gyro_reproof'] = dict(witness=witness, reason=failure,
                    largest_rate_rad_s=float(state.reshape(4,13)[:,:3].norm(dim=1).max()))
            return selected

    with torch.inference_mode(), patch.dict(context, StopTrial=Trial):
        return execute(lane, {19:176,22:168}[lane], lag, 'crawl', .20, variant, attitude, .065)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane',type=int,required=True,choices=(19,22))
    parser.add_argument('--lag',type=float,required=True,choices=(.25,.35,.45))
    parser.add_argument('--attitude',choices=('recorded','fixed'),default='recorded')
    parser.add_argument('--mover',choices=('native','straight','mirrored','earlier'),default='native')
    parser.add_argument('--reproof',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs_path = Path(__file__).resolve().with_name('INPUTS.json')
    inputs = load(inputs_path)
    result = run(inputs,args.lane,args.lag,args.mover,args.attitude,args.reproof)
    assert not stop_paths(), stop_paths()
    print(save(args.output,dict(inputs=pin(inputs_path),replay=pin(__file__),plane_reproof=args.reproof,**result)))


if __name__ == '__main__':
    main()
