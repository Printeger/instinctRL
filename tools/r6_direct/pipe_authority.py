"""Replay the unchanged governor and ICS on closed pipe-probe observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from instinctlab.control.governor import GovernorConfig, decode_governor
from instinctlab.control.ics import ICSConfig, attenuate_command
from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.instinct_cycle import save
from tools.r6_direct.instinct_cycle_finish import guard
from tools.r6_direct.import_legacy import pin
from tools.r6_evidence import checked
from training_workloads.unified_h1.pipe_authority_probe import PIPE_PROBE_RUN, replay_frames


AXIAL_WEIGHTS = (0., .45, .85, 1.4, 2.2)
LABELS = tuple(f'{sign:+d}/{weight:g}' for sign in (1, -1) for weight in AXIAL_WEIGHTS) + ('axial',)


def limiting_beams(issued):
    cache = issued.cache
    closing = cache['approach_speed_m_s'].clamp_min(torch.finfo(issued.v_final_b.dtype).tiny)
    ratios = torch.where(cache['active_mask'], cache['v_safe_m_s']/closing, float('inf'))
    value, beam = ratios.min(dim=-1)
    return torch.where((value < 1.) & (issued.metrics['emergency'].flatten() == 0.), beam, -1)


def compare_directions(grid, states, selected, retreat, rays, governor, ics):
    side = torch.stack((-retreat[:, 1], retreat[:, 0], torch.zeros_like(retreat[:, 0])), dim=1)
    unnormalized = [sign*side + weight*retreat for sign in (1, -1) for weight in AXIAL_WEIGHTS] + [retreat]
    directions = torch.nn.functional.normalize(torch.stack(unnormalized,dim=1), dim=2)
    speed = (selected[:, 1:]-.5).norm(dim=1)
    actions = torch.cat((selected[:, :1, None].expand(-1,len(LABELS),-1), .5+speed[:,None,None]*directions), dim=2)
    requests = decode_governor(actions.flatten(0,1), states.repeat_interleave(len(LABELS), dim=0), governor).v_gov_b
    result = attenuate_command(grid.repeat_interleave(len(LABELS), dim=0), rays, requests, .1, ics)
    projected = (result.v_final_b.reshape_as(directions)*directions).sum(dim=-1)
    return projected.tolist(), limiting_beams(result).view(-1,len(LABELS)).tolist()


def first(rows, lane, predicate):
    return next((row['step'] for row in rows if predicate(row, lane)), None)


def summarize_lane(rows, lane, records):
    begin = first(rows,lane,lambda r,k:r['escape_recovery']['active'][k])
    clipped = first(rows,lane,lambda r,k:r['escape_recovery']['active'][k] and r['ics_beta'][k]==0.
                   and not r['ics_emergency'][k])
    start = next((i for i, r in enumerate(records) if r['step']==clipped), None)
    window = [] if start is None else records[start:start+3]
    continuous = []
    if len(window)==3 and window[-1]['step']==window[0]['step']+2:
        continuous = [name for j,name in enumerate(LABELS) if all(r['projected_mps'][j] >= .08 for r in window)]
    return dict(lane=lane,first_escape=begin,first_blocked=clipped,
                first_emergency=first(rows,lane,lambda r,k:r['ics_emergency'][k]),
                contact=first(rows,lane,lambda r,k:r['collision'][k]),
                first_clip_three_step_options=continuous, samples=records)


def audit(result_pin):
    _, ctx, report = validate_binding(read(result_pin)['binding'])
    if ctx['plan']['run_id'] != PIPE_PROBE_RUN or ctx['state']['counters']['nontraining_transitions'] != 11520:
        raise ValueError('Use the closed fixed-u1792 diagnostic')
    summary = report['summary']
    model = summary['models']['candidate']
    torch.set_num_threads(2)
    rays = torch.from_numpy(np.fromfile(checked(model['lidar_mount']['body_rays']),dtype='<f4').reshape(-1,3))
    config = summary['pipe_control_config']
    governor, ics = GovernorConfig(**config['governor']), ICSConfig(**config['ics'])
    if ics.range_rate_filter:
        raise ValueError('Latest-only archives cannot reconstruct a rate-filtered guard')
    maximum_error = 0.
    cohorts = []
    for index, group in enumerate(model['groups']):
        rows = [json.loads(line) for line in checked(group['trace']).read_text().splitlines()]
        records = [[] for _ in range(8)]
        active_lanes = torch.ones(8,dtype=torch.bool)
        for row, (latest, states) in zip(rows,replay_frames(group['approved_frames'],index),strict=True):
            # This frozen ICS uses the newest frame for its action decision. Its
            # diagnostic range-rate cache is not an output of this replay.
            grid = torch.tensor(latest).repeat(1,4,1,1)
            requested = torch.tensor(row['v_gov_b'][24:])
            baseline = attenuate_command(grid,rays,requested,.1,ics)
            expected = torch.tensor(row['v_final_b'][24:])
            if active_lanes.any():
                maximum_error = max(maximum_error,float((baseline.v_final_b-expected)[active_lanes].abs().max()))
            retreat = torch.tensor(row['pipe_retreat_basis_b'])
            available = (active_lanes & torch.tensor(row['escape_recovery']['active'][24:])
                         & (baseline.metrics['emergency'].flatten()==0.) & (retreat.norm(dim=1)>.5))
            if available.any():
                selected = torch.tensor(row['escape_recovery']['selected_action'][24:])
                effects, limit = compare_directions(grid[available],torch.tensor(states)[available],selected[available],
                                                    retreat[available],rays,governor,ics)
                current_limits = limiting_beams(baseline).tolist()
                for lane, effect, ray in zip(available.nonzero().flatten().tolist(),effects,limit,strict=True):
                    current = current_limits[lane]
                    records[lane].append(dict(step=row['step'],age_s=float(states[lane,-1]),
                        beta=row['ics_beta'][lane+24], baseline_limit_ray=current,
                        baseline_limit_raw_m=None if current<0 else float(latest[lane,0].flat[current]),
                        mover_witness_ray=row['pre']['witness_beam'][lane+24],
                        projected_mps=effect, limiting_rays=ray))
            active_lanes &= ~torch.tensor(row['terminal'][24:])
        trials = []
        for lane in range(24,32):
            stop = first(rows,lane,lambda r,k:r['terminal'][k])
            trials.append(summarize_lane(rows if stop is None else rows[:stop+1],lane,records[lane-24]))
        cohorts.append(dict(mode=group['lanes'][24]['mode'],trace=group['trace'],archive=group['approved_frames'],lanes=trials))
    if maximum_error>2e-5:
        raise ValueError(f'Raw observations could not reproduce the original issued command: {maximum_error}')
    return dict(probe=result_pin,declared_candidates=LABELS,raw_baseline_replay_max_issued_error_mps=maximum_error,
                comparison='same observed state and policy tick; alternate actions never passed to the vehicle',
                first_clip_window_steps=3,minimum_issued_projection_mps=.08,cohorts=cohorts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    guard()
    result = audit(pin(args.probe))
    guard()
    print(json.dumps(save(args.output,result)))


if __name__=='__main__':
    main()
