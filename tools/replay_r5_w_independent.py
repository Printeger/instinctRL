"""CPU replay of the frozen independent protocol; never edits runtime evidence."""
from pathlib import Path
import sys, json, hashlib

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'docs/experiments/r5-w-closure-auto'
EXP = BASE/'experiments/R5-W-L1-INDEPENDENT-RESOURCE-V2'
read = lambda p: json.loads(Path(p).read_text())
def pin(p):
    p = Path(p).resolve()
    with p.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(p), sha256=digest)

def main():
    snapshot = Path(read(EXP/'readonly-snapshot.json')['snapshot'])
    sys.path.insert(0, str(snapshot))
    sys.path.insert(1, str(snapshot/'isaaclab-training'))
    import torch
    from training_workloads.unified_h1 import evaluation as ev, w_independent as wi
    torch.set_num_threads(2)
    assert Path(ev.__file__).is_relative_to(snapshot)
    inventory = read(EXP/'live/R1-research-formal-evaluation/INVENTORY.json')
    checked = 0
    for row in inventory['files']:
        for key in ('original','archived'):
            if row[key]:
                p = Path(row[key])
                assert p.stat().st_size == row['bytes'] and pin(p)['sha256'] == row['sha256'], p
                checked += 1
    output = Path(inventory['roots']['attempt'])/'results/research-formal-evaluation'
    raw = output/'evaluation_raw/w_independent_R1_u640'
    traces, coverage = [], {}
    for case, config in ev.CASES.items():
        p = raw/(case+'.jsonl')
        records = [json.loads(line) for line in p.read_text().splitlines()]
        assert records[0]['kind'] == 'T0' and records[0]['seed'] == 20261020
        assert len(records) == config.steps+1
        assert [r['step'] for r in records[1:]] == list(range(config.steps))
        assert all(len(r['root_state_w']) == 32 for r in records[1:])
        ts = ev.replay_raw_case(str(p), case=case)
        assert len(ts) == 32 and {t.env_index for t in ts} == set(range(32))
        traces.extend(ts)
        coverage[case] = dict(raw=pin(p), seed=records[0]['seed'], lanes=32, steps=config.steps, transitions=32*config.steps)
    result = wi.evaluate_independent(traces, update=640)
    actual = output/'w_independent_R1_u640.json'
    assert json.loads(json.dumps(result)) == read(actual)
    primary = ev.evaluate_checkpoint(traces, update=640)
    assert not primary['all_gates_pass']
    assert sum('CASE_SEED_MISMATCH' in json.dumps(e) for e in primary['episodes']) == 160
    wrows = [json.loads(line) for line in (raw/'W.jsonl').read_text().splitlines()][1:]
    timings = []
    for trace in traces:
        if trace.case != 'W': continue
        lane = trace.env_index; axis = int(lane >= 16)
        velocity = [r['root_state_w'][lane][7+axis] for r in wrows]
        distance,_ = ev.sensor_origin_surface_distance(trace.sensor_positions_w_m,trace.env_origin_w_m.expand(len(wrows),-1))
        events = {}
        for name,mask in [('trigger_1p1',distance<=1.1),('q_true',trace.q>.5),('beta_below_one',trace.ics_beta<1),('beta_zero',trace.ics_beta==0),('below_0p8',distance<.8)]:
            ix = torch.nonzero(mask).flatten()
            i = int(ix[0]) if ix.numel() else None
            events[name] = None if i is None else dict(row=i,observation_time_s=(i+1)*.1,control_issue_time_s=i*.1,geometry_m=float(distance[i]),world_normal_velocity_m_s=velocity[i])
        score = next(e for e in result['original_metric_result']['episodes'] if e['case']=='W' and e['env_index']==lane)
        motion = ev.wall_motion_prerequisite(velocity)
        assert motion and score['success'] and score['metrics']['min_distance']>=.8 and score['metrics']['stop_time']<=3
        timings.append(dict(lane=lane,wall='xy'[axis],world_normal_11_sample_approach=motion,events=events,metrics=score['metrics']))
    report = dict(status='PASS_COMPLETE_INDEPENDENT_RAW_REPLAY',tool=pin(__file__),snapshot=pin(EXP/'readonly-snapshot.json'),
        evaluator=pin(ev.__file__),independent_evaluator=pin(wi.__file__),inventory=pin(EXP/'live/R1-research-formal-evaluation/INVENTORY.json'),
        checked_original_and_archived_files=checked,coverage=coverage,actual_result=pin(actual),dictionary_equal=True,
        primary_actual_seed_diagnostic_count=160,primary_all_gates_pass=False,original_metric_result=result['original_metric_result'],w_timing=timings,
        limitation='Actual seed20261020, original fixed suite initial states; only seed metadata canonicalized in an in-memory scoring view. No broader generalization or isolated RL safety gain.')
    dest=BASE/'INDEPENDENT_RESOURCE_V2_CPU_REPLAY.json'
    assert not dest.exists()
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(checked=checked,coverage=coverage,all_gates_pass=result['original_metric_result']['all_gates_pass'])))

if __name__ == '__main__': main()
