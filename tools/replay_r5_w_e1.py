"""CPU-only E1 replay and per-lane timing from hash-bound original records."""
from pathlib import Path
import json
import torch
from tools.r5_w_commander import BASE, read, pin, sha, save
from training_workloads.unified_h1.evaluation import (
    CASES, replay_raw_case, evaluate_checkpoint, sensor_origin_surface_distance,
    wall_motion_prerequisite,
)


def main():
    torch.set_num_threads(2)
    experiment = BASE/'experiments/R5-W-E1-CONTROL-01'
    inventory_path = experiment/'live/R1-research-formal-evaluation/INVENTORY.json'
    inventory = read(inventory_path)
    checked = 0
    for row in inventory['files']:
        for key in ('original', 'archived'):
            if row[key] is not None:
                path = Path(row[key])
                assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
                checked += 1
    output = Path(inventory['roots']['attempt'])/'results/research-formal-evaluation'
    raw = output/'evaluation_raw/w_R1_u512'
    traces = [trace for case in CASES for trace in replay_raw_case(str(raw/(case+'.jsonl')), case=case)]
    result = evaluate_checkpoint(traces, update=512)
    assert json.loads(json.dumps(result)) == read(output/'w_R1_u512.json')
    rows = [json.loads(line) for line in (raw/'W.jsonl').read_text().splitlines()][1:]
    timings = []
    for trace in traces:
        if trace.case != 'W':
            continue
        lane = trace.env_index
        axis = 0 if lane < 16 else 1
        distance, _ = sensor_origin_surface_distance(trace.sensor_positions_w_m, trace.env_origin_w_m.expand(len(rows), -1))
        velocity = [row['root_state_w'][lane][7+axis] for row in rows]
        def first(mask):
            indices = torch.nonzero(mask).flatten()
            return int(indices[0]) if indices.numel() else None
        events = {}
        for name, index in {'trigger_1p1': first(distance <= 1.1), 'q_true': first(trace.q > .5),
                            'beta_below_one': first(trace.ics_beta < 1), 'beta_zero': first(trace.ics_beta == 0),
                            'below_0p8': first(distance < .8), 'minimum_geometry': int(distance.argmin())}.items():
            events[name] = None if index is None else dict(row=index, observation_time_s=(index+1)*.1,
                control_issue_time_s=index*.1, geometry_m=float(distance[index]),
                world_normal_velocity_m_s=velocity[index], beta=float(trace.ics_beta[index]), q=float(trace.q[index]))
        timings.append(dict(lane=lane, wall='x' if axis == 0 else 'y', events=events,
                            world_normal_11_sample_approach=wall_motion_prerequisite(velocity)))
    assert all(row['world_normal_11_sample_approach'] for row in timings)
    report = dict(status='PASS_E1_CPU_REPLAY_NOT_GOAL_COMPLETE', inventory=pin(inventory_path),
                  tool=pin(Path(__file__)), checked_original_and_archived_files=checked,
                  original_result=pin(output/'w_R1_u512.json'), cases=result['cases'],
                  all_gates_pass=result['all_gates_pass'], w_timing=timings,
                  timing='Control issued at i*0.1s; geometry/root observed at (i+1)*0.1s; no cross-frame subtraction.',
                  scientific_attribution='Fixed original R1/u512 with .15m ICS margin: control engineering, not RL.')
    save(experiment/'E1_CPU_REPLAY.json', report, new=True)
    print(json.dumps(dict(status=report['status'], files_checked=checked, cases={k:v['successes'] for k,v in result['cases'].items()})))


if __name__ == '__main__':
    main()
