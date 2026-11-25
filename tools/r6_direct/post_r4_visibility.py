"""Recompute an unoccluded MID360 visibility bound from an existing rear assay."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from instinctlab.control.ics import mounted_mid360_rays_body
from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.import_legacy import pin
from tools.r6_direct.instinct_cycle import save
from tools.r6_direct.instinct_cycle_finish import guard
from tools.r6_evidence import checked
from training_workloads.unified_h1.dynamic_assay import visibility_reset


def replay(assay_path):
    assay = pin(assay_path)
    binding = read(assay)['binding']
    _, context, report = validate_binding(binding)
    seed = context['contract']['parameters']['seed']
    group, = report['summary']['models']['reference']['groups']
    trace = group['trace']
    raw = [json.loads(line) for line in checked(trace).read_text().splitlines()]
    centers = np.array([step['pre']['sphere_position_w'] for step in raw])
    sensors = np.array([step['pre']['sensor_position_w'] for step in raw])
    relative = centers - sensors
    block = dict(id='dynamic-0', task='stop', speed_m_s=.25, pulse_acceleration_m_s2=0.,
                 sensor_to_wall_distance_m=None, steps=len(raw))
    yaw = np.radians([visibility_reset(block, seed=seed, lane=lane, mode=9,
                                      mount=(0.,0.,0.))['yaw_degrees'] for lane in range(relative.shape[1])])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    body = relative.copy()
    body[:,:,0] = relative[:,:,0]*cos_yaw + relative[:,:,1]*sin_yaw
    body[:,:,1] = -relative[:,:,0]*sin_yaw + relative[:,:,1]*cos_yaw
    ray_directions = mounted_mid360_rays_body(device='cpu').numpy().astype('float64')
    actual_visible = np.array([step['pre']['winning_beams'] for step in raw]) > 0
    if not actual_visible.any(axis=0).all():
        raise ValueError('Rear trace is missing a recorded first visible step')
    actual = actual_visible.argmax(axis=0)
    proposals = []
    threshold = np.sqrt(np.maximum(0., (body*body).sum(axis=2) - .34**2))
    for degrees in (30, 28, 26, 25, 24, 22):
        delta = math.radians(degrees - 30)
        rays = ray_directions.copy()
        rays[:,0] = math.cos(delta)*ray_directions[:,0] + math.sin(delta)*ray_directions[:,2]
        rays[:,2] = -math.sin(delta)*ray_directions[:,0] + math.cos(delta)*ray_directions[:,2]
        flat = body.reshape(-1,3)
        projections = np.concatenate([(chunk @ rays.T).max(axis=1) for chunk in np.array_split(flat, 16)])
        visible = projections.reshape(threshold.shape) >= threshold
        if not visible.any(axis=0).all():
            raise ValueError('Sphere visibility replay has an unobserved lane')
        first = visible.argmax(axis=0)
        deltas = actual-first
        proposals.append(dict(pitch_degrees=degrees, first_visible_steps=first.tolist(),
                              early_steps=deltas.tolist(), median_early_steps=median(deltas.tolist())))
    return dict(assay=assay, binding=binding, trace=trace, script=pin(Path(__file__)),
        assumptions='Recorded sensor and sphere positions, nominal reset yaw, fixed body pitch and roll in the pre-trigger window. Sphere-only ray intersection without static occlusion. No inference of consumed measurements, controller actions or contacts for the alternative mounts.',
        actual_first_visible_steps=actual.tolist(), candidates=proposals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assay', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    guard()
    print(json.dumps(save(args.output, replay(args.assay))))


if __name__ == '__main__':
    main()
