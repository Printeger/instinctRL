"""Bounded CPU-only W temporal attribution from immutable formal raw traces."""
from pathlib import Path
import hashlib
import json
import torch
from training_workloads.unified_h1.evaluation import (replay_raw_case, evaluate_episode,
    sensor_origin_surface_distance, stopping_confirmation, wall_motion_prerequisite)

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'docs/experiments/r5-formal-readiness-auto/experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01'
OUT = ROOT/'docs/experiments/r5-w-closure-auto'

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def first(mask):
    ids = torch.nonzero(mask).flatten()
    return int(ids[0]) if ids.numel() else None

def analyze():
    torch.set_num_threads(2)
    results = []; pins = []; checkpoints = []
    for group in ('R0','R1'):
        spec = json.loads((OLD/'inputs'/f'{group}-research-formal-evaluation.authorized.json').read_text())
        ap = spec['pins']['authorization']; assert sha(ap['path']) == ap['sha256']
        auth = json.loads(Path(ap['path']).read_text()); pins.append(ap)
        for ck in auth['parameters']['checkpoints']:
            assert sha(ck['path']) == ck['sha256']
            checkpoints.append(dict(group=group, **ck))
        invpath = OLD/'live'/f'{group}-research-formal-evaluation'/'INVENTORY.json'
        inv = json.loads(invpath.read_text()); pins.append(dict(path=str(invpath),sha256=sha(invpath)))
        for node in (272,288,320,384,512):
            suffix = f'/evaluation_raw/formal_{group}_u{node}/W.jsonl'
            matches = [r for r in inv['files'] if r['original'].endswith(suffix)]
            assert len(matches) == 1
            pin = matches[0]; path = Path(pin['original'])
            assert path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256']
            pins.append(dict(path=str(path),sha256=pin['sha256']))
            raw = [json.loads(line) for line in path.read_text().splitlines()]
            initial, rows = raw[0], raw[1:]
            root = torch.tensor([r['root_state_w'] for r in rows])
            for trace in replay_raw_case(str(path), case='W'):
                lane = trace.env_index; axis = 0 if lane < 16 else 1
                assert not any(r['reset_mask'][lane] for r in rows)
                score = evaluate_episode(trace)
                assert score['engineering']['evidence_valid']
                d,_ = sensor_origin_surface_distance(trace.sensor_positions_w_m, trace.env_origin_w_m.expand(len(rows),-1))
                d0,_ = sensor_origin_surface_distance(trace.initial_sensor_position_w_m[None],trace.env_origin_w_m[None])
                world_v = root[:,lane,7+axis]
                speed = trace.velocity_b.norm(dim=-1)
                trigger = first(d <= 1.1)
                confirm = stopping_confirmation(speed.tolist(),zero_effective_index=trigger or 0,
                    effective_boundary='OBSERVATION' if trigger is not None else 'PRE_STEP')
                ids = {'trigger_1p1':trigger,'q_true':first(trace.q > .5),
                    'beta_below_one':first(trace.ics_beta < 1), 'beta_zero':first(trace.ics_beta == 0),
                    'geometry_below_0p8':first(d < .8), 'minimum_geometry':int(d.argmin()),
                    'stop_confirmed':confirm.get('confirmation_index'),
                    'trigger_plus_3s':None if trigger is None else min(trigger+30,len(rows)-1)}
                events = {}
                for name,index in ids.items():
                    if index is None: events[name] = None; continue
                    r = rows[index]
                    events[name] = dict(row=index,observation_time_s=(index+1)*.1,
                        control_issued_time_s=index*.1, geometry_m=float(d[index]),
                        previous_observation_geometry_m=float(d[index-1] if index else d0[0]),
                        world_normal_velocity_m_s=float(world_v[index]), body_velocity_m_s=r['actual_velocity_b'][lane],
                        ray_clearance_m=r['ray_clearance_m'][lane],q=r['q'][lane],beta=r['ics_beta'][lane],
                        residual_b=r['v_corr_b'][lane],issued_action=r['issued_action'][lane],
                        ics_output_b=r['v_final_b'][lane],ics_output_w=r['v_final_w'][lane])
                bz=ids['beta_zero']; bl=ids['beta_below_one']
                results.append(dict(group=group,node=node,lane=lane,wall='x' if axis==0 else 'y',
                    t0_geometry_m=float(d0[0]), events=events,original_metrics=score['metrics'],
                    original_success=score['success'],confirmation=confirm,
                    world_motion_11_samples=wall_motion_prerequisite(world_v.tolist()),
                    body_motion_11_samples=wall_motion_prerequisite(trace.velocity_b[:,axis].tolist()),
                    min_world_normal_velocity_m_s=float(world_v.min()), max_world_normal_velocity_m_s=float(world_v.max()),
                    max_post_beta_zero_speed_m_s=None if bz is None else float(speed[bz:].max()),
                    post_beta_zero_geometry_loss_m=None if bz is None else float(d[bz]-d[bz:].min()),
                    post_first_attenuation_geometry_loss_m=None if bl is None else float(d[bl]-d[bl:].min())))
    aggregates=[]
    for group in ('R0','R1'):
        for node in (272,288,320,384,512):
            for wall in ('x','y'):
                rr=[r for r in results if (r['group'],r['node'],r['wall'])==(group,node,wall)]
                def extent(values): return dict(min=min(values),mean=sum(values)/len(values),max=max(values))
                aggregates.append(dict(group=group,node=node,wall=wall,lanes=len(rr),
                    minimum_geometry_m=extent([r['original_metrics']['min_distance'] for r in rr]),
                    stop_time_s=extent([r['original_metrics']['stop_time'] for r in rr]),
                    beta_zero_geometry_m=extent([r['events']['beta_zero']['geometry_m'] for r in rr]),
                    beta_first_previous_geometry_m=extent([r['events']['beta_below_one']['previous_observation_geometry_m'] for r in rr]),
                    beta_zero_world_velocity_m_s=extent([r['events']['beta_zero']['world_normal_velocity_m_s'] for r in rr]),
                    post_zero_geometry_loss_m=extent([r['post_beta_zero_geometry_loss_m'] for r in rr]),
                    q_before_attenuation=sum(r['events']['q_true']['row']<r['events']['beta_below_one']['row'] for r in rr)))
    obj=dict(schema_version='r5-w-causal-diagnosis/1',status='OFFLINE_ATTRIBUTION_NOT_LIVE_VALIDATION',
        pins=pins,checkpoints=checkpoints,lanes=results,aggregates=aggregates,
        timing='row i: beta/output staged at i*0.1s; geometry/root/body velocity observed at (i+1)*0.1s. Previous geometry is an observation proxy, not reconstructed exact ICS ray input.',
        unknown=['Exact dense measured reliable ICS beam histories at decision time','Substep motor/braking dynamics'],
        action_counts=dict(simulation=0,training=0,actor_forward=0,GPU_query=0))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'W_CAUSAL_DIAGNOSIS.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    print(json.dumps(aggregates,indent=2))

if __name__=='__main__': analyze()
