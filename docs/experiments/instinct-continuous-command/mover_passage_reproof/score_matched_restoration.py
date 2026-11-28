"""Freeze N113 opportunity IDs, then compare a closed matched N115 first flight."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from tools.r6_direct.replay_clearance import confirmed_starts, projected_clearance
from training_workloads.unified_h1.escape_holdout_scene import scene_geometry

RUNS = Path('/home/mint/instinctrl-runs')
DOCS = Path('/home/mint/rl_dev/instinctRL-dev/docs/experiments/instinct-continuous-command')
BASELINE = 'R6-DIRECT-INSTINCT-GPU-COMMAND-NOVEL-113'
CANDIDATE = 'R6-DIRECT-INSTINCT-GPU-COMMAND-NOVEL-115'
GATES = DOCS/'long/GATES.json'
STEPS, LANES, SCENE_SEED = 1200, 32, 1999375583
RESTORE_TICKS = 15


def pin(path):
    with path.open('rb') as stream:
        return dict(path=str(path), sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):
    return json.loads(path.read_bytes())


def checked(reference):
    path = Path(reference['path'])
    if pin(path) != reference:
        raise ValueError(f'frozen input changed: {path}')
    return path


def report(run):
    root = RUNS/run
    result = root/'analysis/clearance-and-progress.json'
    analysis = root/'analysis/report.json'
    receipt = root/'control/receipt.json'
    data = read(result)
    bound = read(checked(data['binding']))
    if checked(bound['receipt']) != receipt or checked(bound['report']) != analysis:
        raise ValueError(f'analysis does not bind {run}')
    closure = read(receipt)
    if closure['status'] != 'CLOSED' or closure['remaining']:
        raise ValueError(f'{run} is not closed')
    metrics = data['models']['candidate']['layout_routes']
    summary = read(analysis)['summary']
    flights = summary['models']['candidate']['layout_routes']['lanes']
    if len(metrics) != LANES or len(flights) != LANES or [m['lane'] for m in metrics] != list(range(LANES)):
        raise ValueError(f'invalid lane joins for {run}')
    if (summary['counters']['completed_policy_steps'] != STEPS
            or summary['counters']['nontraining_transitions'] != STEPS*LANES
            or summary['counters']['training_transitions'] != 0):
        raise ValueError(f'incomplete or non-evaluation workload {run}')
    trace = checked(metrics[0]['trace'])
    dense = checked(metrics[0]['physics_tick_clearance']['source'])
    if any(m['trace']!=metrics[0]['trace'] or m['physics_tick_clearance']['source']!=metrics[0]['physics_tick_clearance']['source'] for m in metrics):
        raise ValueError(f'arm archive diverges across lanes for {run}')
    geometry = checked(data['geometry'])
    opportunity = checked(data['opportunities'])
    pins = [pin(p) for p in (result,analysis,receipt,trace,dense,geometry,opportunity)]
    return dict(run=run, metrics=metrics, flights=flights, trace=trace, geometry=geometry,
                opportunities=read(opportunity), pins=pins)


def ingest(arm):
    trace = arm['trace']
    with gzip.open(trace,'rt') as stream:
        initial = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    if [r['step'] for r in rows] != list(range(STEPS)):
        raise ValueError(f'bad step sequence: {trace}')
    after = np.asarray([r['root_state_after_w'] for r in rows],dtype=float)
    before = np.concatenate((np.asarray(initial['root_state_w'])[None], after[:-1]))
    request = np.asarray([[e['velocity_w_m_s'] for e in r['raw_event']] for r in rows],dtype=float)
    mover = np.asarray([r['mover_position_w'] for r in rows],dtype=float)
    return dict(initial=initial,rows=rows,after=after,before=before,request=request,mover=mover)


def windows(base):
    out = []
    for m in base['metrics']:
        events = m['hazard_response_preview']['events']
        for i,event in enumerate(events):
            anchor = event['first_feasible_request_step']
            if anchor is None:
                continue
            end = events[i+1]['start_step'] if i+1 < len(events) else STEPS
            confirmation = event['recovery_confirmed_step']
            out.append(dict(lane=m['lane'],layout=m['layout'],hazard_start=event['start_step'],
                anchor=anchor,end=end,expected_baseline_ticks=None if confirmation is None else confirmation+1-anchor))
    if len(out)!=82 or sum(w['expected_baseline_ticks'] is None or w['expected_baseline_ticks']>RESTORE_TICKS for w in out)!=33:
        raise ValueError('N113 reference cohort drift')
    return out


def freeze():
    arm = report(BASELINE)
    return dict(schema='instinctrl-v60-matched-restoration/1',baseline=BASELINE,candidate=CANDIDATE,
                steps=STEPS,lanes=LANES,scene_seed=SCENE_SEED,
                scorer=pin(Path(__file__).resolve()),production_scorer=pin(Path(__import__(
                    'tools.r6_direct.replay_clearance',fromlist=['']).__file__)),
                gates=pin(GATES),baseline_inputs=arm['pins'],restoration_windows=windows(arm),
                rule=dict(anchor='N113 first feasible packet; end at next N113 risk onset or first-flight end',
                    confirmation='five consecutive feasible nonzero latest commands at measured 60 percent; inclusive fifth packet',
                    missing_denominator='all 82',timeout_ticks=RESTORE_TICKS,minimum_percentage_point_gain=10,
                    nearest_rank_percentile=95,minimum_p95_gain_ticks=3,
                    p95_primary_cohort='completed in both arms, guarded by nonincrease of timeout and missing on all 82 windows'))


def triangles(geometry):
    origin = torch.as_tensor(read(geometry)['env_origins'])
    _,_,vertices,faces = scene_geometry(origin,SCENE_SEED,command_routes=True)
    return np.asarray(vertices)[np.asarray(faces).reshape(-1,3)]


def same_inputs(left,right):
    for key in ('root_state_w','layouts','street_sweep'):
        if left['initial'].get(key)!=right['initial'].get(key):
            raise ValueError(f'matched initial {key} differs')
    if any(a['raw_event']!=b['raw_event'] for a,b in zip(left['rows'],right['rows'],strict=True)):
        raise ValueError('exact per-tick raw Crazy Command events differ')


def matched(arm,raw,registered,mesh,gates):
    result = []
    for m in arm['metrics']:
        lane = m['lane']
        end = m['samples']
        before = raw['before'][:end,lane]
        request = raw['request'][:end,lane]
        actual = raw['after'][:end,lane,7:10]
        mover = raw['mover'][:end,lane] if m['layout']=='street_gap' else None
        radius = m['physics_tick_clearance']['maximum_measured_link_envelope_m']
        nominal,_ = projected_clearance(before[:,:3],before[:,7:10],request,mover,mesh,radius)
        speed = np.linalg.norm(request,axis=1)
        feasible = (speed>.03)&(nominal>=gates[m['layout']]['minimum_clearance_m'])
        tracking = feasible&((request*actual).sum(axis=1)>=.60*speed**2)
        for window in registered:
            if window['lane']!=lane:
                continue
            anchor,stop = window['anchor'],min(end,window['end'])
            offers = confirmed_starts(feasible[anchor:stop])
            found = confirmed_starts(tracking[anchor:stop])
            result.append(dict(candidate_first_feasible=None if not len(offers) else anchor+int(offers[0]),
                               ticks=None if not len(found) else int(found[0])+5))
    return result


def nearest_p95(values):
    if not values:
        return None
    rank = (95*len(values)+99)//100
    return sorted(values,key=lambda x: math.inf if x is None else x)[rank-1]


def score_windows(registration,a,b):
    pairs = [dict(window=w, baseline=x, candidate=y) for w,x,y in zip(registration,a,b,strict=True)]
    if any(p['baseline']['ticks']!=p['window']['expected_baseline_ticks'] for p in pairs):
        raise ValueError('recomputed N113 first-flight clocks differ from frozen analyzer')
    first, second = ([x['ticks'] for x in arm] for arm in (a,b))
    counts = [dict(total=len(arm),missing=arm.count(None),timeout=sum(x is not None and x>RESTORE_TICKS for x in arm),
                  bad=sum(x is None or x>RESTORE_TICKS for x in arm)) for arm in (first,second)]
    primary = [i for i,x in enumerate(first) if x is not None]
    shared = [i for i in primary if second[i] is not None]
    pair_quantiles = {name:[nearest_p95([arm[i] for i in indices]) for arm in (first,second)]
                      for name,indices in (('frozen_baseline_observed',primary),('both_observed',shared),
                                           ('all_registered',list(range(len(first)))))}
    improvement = counts[0]['bad']-counts[1]['bad']
    p,q = pair_quantiles['both_observed']
    return dict(arms=counts,matched_rows=pairs,p95_ticks=pair_quantiles,
                p95_cohort_counts=dict(baseline_observed=len(primary),both_observed=len(shared),all_registered=len(first)),
                timeout_missing_improvement_pp=100*improvement/len(first),
                passes_timeout_missing_branch=10*improvement>=len(first),
                passes_p95_branch=improvement>=0 and p is not None and q is not None and p-q>=3)


def gates_by_lane(arm,raw,gates):
    result = []
    for m,f in zip(arm['metrics'],arm['flights'],strict=True):
        lane,layout = m['lane'],m['layout']
        if layout!=f['layout']:
            raise ValueError(f'lane layout mismatch {lane}')
        gate = gates[layout]
        d = m['physics_tick_clearance']
        tr = m['tracking_in_feasible_moving_windows']
        st = list(m['externally_scheduled_stops'].values())
        pr = list(m['externally_scheduled_progress'].values())
        events = m['hazard_response_preview']['events']
        minimum = d['minimum_clearance_lower_bound_m']
        uninterrupted = (m['samples']==STEPS and f['elapsed_s']==120 and d['samples']==STEPS*20+1
            and all(f[k] is None for k in ('initial_termination_reason','first_contact_step','first_reset_step'))
            and not any(r['collision'][lane] or r['reset'][lane] for r in raw['rows']))
        route = f['route_progress']
        tracking = dict(rmse=tr['rmse_m_s'] is not None and tr['rmse_m_s']<=gate['maximum_tracking_rmse_m_s'],
                        fraction=tr['fraction_at_least_60_percent_of_intent'] is not None and
                        tr['fraction_at_least_60_percent_of_intent']>=gate['minimum_progress_fraction'],
                        slow_span=tr['longest_slow_feasible_window_s']<=gate['maximum_slow_feasible_s'])
        passed = dict(uninterrupted=uninterrupted,clearance=minimum>=gate['minimum_clearance_m'],
            indoor_route=layout!='room_corner' or (route['indoor_door_passed'] and route['indoor_corner_passed']),
            stops=bool(st) and all(x['first_flight_completed_window'] and x['stopped_after_s'] is not None
                and x['stopped_after_s']<=gate['maximum_stop_s'] and x['max_drift_m']<=gate['maximum_stop_drift_m'] for x in st),
            progress=bool(pr) and all(x['met_fraction']>=gate['minimum_progress_fraction']
                and x['measured_progress_m']>=gate['minimum_progress_fraction']*x['requested_progress_m']
                and x['observed_only_rmse_m_s'] is not None
                and x['observed_only_rmse_m_s']<=gate['maximum_tracking_rmse_m_s'] for x in pr),
            **tracking)
        result.append(dict(lane=lane,layout=layout,passed=passed,minimum_clearance_m=minimum,
            feasible_tracking=tr,stops=st,progress=pr,
            intervention_bad=sum(x['intervention_after_s'] is None or x['intervention_after_s']>gate['maximum_intervention_s']
                                 for x in events)))
    return result


def extrema(rows,key,largest):
    present = [(r['lane'],r[key]) for r in rows]
    if not present:
        raise ValueError(f'no layout evidence for {key}')
    missing = next((lane for lane,value in present if value is None),None)
    if missing is not None:
        return dict(lane=missing,value=None)
    lane,value = (max if largest else min)(present,key=lambda item:item[1])
    return dict(lane=lane,value=value)


def by_layout(rows,gates):
    out = {}
    for layout in gates:
        lanes = [r for r in rows if r['layout']==layout]
        flattened=[]
        for r in lanes:
            t=r['feasible_tracking']
            flattened.append(dict(lane=r['lane'],minimum=r['minimum_clearance_m'],rmse=t['rmse_m_s'],
                 along=t['fraction_at_least_60_percent_of_intent'],slow=t['longest_slow_feasible_window_s'],
                 stop_time=None if any(x['stopped_after_s'] is None for x in r['stops']) or not r['stops']
                           else max(x['stopped_after_s'] for x in r['stops']),
                 stop_drift=max((x['max_drift_m'] for x in r['stops'] if x['max_drift_m'] is not None),default=None),
                 progress_fraction=min((x['met_fraction'] for x in r['progress']),default=None),
                 progress_distance=min((x['measured_progress_m']/x['requested_progress_m'] for x in r['progress']),default=None),
                 progress_rmse=None if not r['progress'] or any(x['observed_only_rmse_m_s'] is None for x in r['progress'])
                               else max(x['observed_only_rmse_m_s'] for x in r['progress'])))
        out[layout] = dict(lanes=len(lanes),pass_counts={k:sum(r['passed'][k] for r in lanes) for k in lanes[0]['passed']},
            intervention_bad=sum(r['intervention_bad'] for r in lanes),
            worst={k:extrema(flattened,k,largest=k in ('rmse','slow','stop_time','stop_drift','progress_rmse'))
                   for k in flattened[0] if k!='lane'})
    return out


def compare(frozen, *, self_check=False):
    for item in (frozen['scorer'],frozen['production_scorer'],frozen['gates'],*frozen['baseline_inputs']):
        checked(item)
    if frozen!=freeze():
        raise ValueError('frozen registration or script differs')
    base = report(BASELINE)
    candidate = report(BASELINE if self_check else CANDIDATE)
    gates = read(checked(frozen['gates']))['by_layout']
    for key in ('windows','stop_windows'):
        if base['opportunities'][key]!=candidate['opportunities'][key]:
            raise ValueError(f'predeclared {key} differs')
    if pin(base['geometry'])['sha256']!=pin(candidate['geometry'])['sha256']:
        raise ValueError('scene geometry differs')
    first,second = [ingest(arm) for arm in (base,candidate)]
    same_inputs(first,second)
    mesh = triangles(base['geometry'])
    registered = frozen['restoration_windows']
    restoration = score_windows(registered,matched(base,first,registered,mesh,gates),
                                matched(candidate,second,registered,mesh,gates))
    arm_details = [gates_by_lane(arm,r,gates) for arm,r in ((base,first),(candidate,second))]
    baseline_layout,candidate_layout = [by_layout(rows,gates) for rows in arm_details]
    safety = all(r['passed']['uninterrupted'] and r['passed']['clearance'] for r in arm_details[1])
    original_motion_checks = all(all(r['passed'].values()) for r in arm_details[1])
    return dict(schema=frozen['schema'],self_check=self_check,inputs=[base['pins'],candidate['pins']],
        matched_restoration=restoration,safety_pass=safety,all_original_short_motion_checks_pass=original_motion_checks,
        lane_results=arm_details,layout_worst=[baseline_layout,candidate_layout],
        bounded_pass=(not self_check and safety and original_motion_checks and
                      (restoration['passes_timeout_missing_branch'] or restoration['passes_p95_branch'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','compare','self-check'))
    parser.add_argument('registration',type=Path,nargs='?')
    args = parser.parse_args()
    if args.action=='freeze':
        result = freeze()
    else:
        result = compare(read(args.registration),self_check=args.action=='self-check')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
