"""Frozen training and paired closed-loop assessment on the planar research plant."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import torch
import torch.nn.functional as F

from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.instinct_cycle_finish import guard

from .policy import FrozenR6, Policy, clearance_guard, command_heading, features
from .world import ACTIONS, DT, MOTIONS, SCENES, STEPS, Arena, Observation, action_vectors, rotate


RUNS = Path.home() / 'instinctrl-runs'
PROJECT = ROOT / 'docs/experiments/instinct-dynamic-five'
R2 = ROOT / 'docs/experiments/instinct-next-two-cycles/round-2/RESULTS.json'
SNAPSHOT = Path(__file__).resolve().parents[2]
DEVELOPMENT = dict(seed=12877013, repeat=8, split='development', sensor=1.)
CONFIRMATION = dict(seed=684921370, repeat=12, split='confirmation', sensor=1.4)
BASELINE_FOLDER = RUNS / 'INSTINCT-DYNAMIC-BASELINE'


def assert_capacity():
    guard()
    if shutil.disk_usage(RUNS).free < 30 * 2**30 + 100 * 2**20:
        raise RuntimeError('Insufficient space above the system reserve')


def check_pin(value):
    if value != pin(value['path']):
        raise ValueError(f'Frozen pin changed {value["path"]}')
    return Path(value['path'])


def save_model(path, model):
    with path.open('xb') as out:
        torch.save(dict(mode=model.mode, objective=model.objective, threat_reference=model.threat_reference,
                        state=model.state_dict()), out)
    return pin(path)


def load_model(value):
    payload = torch.load(check_pin(value), map_location='cpu', weights_only=True)
    model = Policy(payload['mode'], payload['objective'], payload.get('threat_reference', False))
    model.load_state_dict(payload['state'], strict=True)
    return model.eval()


@torch.no_grad()
def collect(config, behavior=None, *, accounting=None):
    arena = Arena(**config['train_world'])
    rng = torch.Generator().manual_seed(config['optimizer_seed'] + 1)
    rows = []
    counts = dict(teacher=0, behavior=0, exploration=0)
    for step in range(STEPS):
        if step % 16 == 0:
            guard()
        t = step * DT
        obs = arena.observe(step)
        active = arena.live.clone()
        reference = config.get('threat_reference', False)
        heading = command_heading(obs, reference) if reference else None
        grid, state = features(obs, config['features'], heading)
        demonstration, risk, choice = arena.teacher(obs, t, config['oracle_horizon'], heading=heading)
        prior = obs.command if behavior is None else behavior.act(obs)
        # Mix instructor and current controller to retain states before successful
        # actions and expose recoveries the current controller actually visits.
        draw = torch.rand(arena.n, generator=rng)
        proposals = torch.where((draw < config['oracle_mix'])[:,None], demonstration, prior)
        actions = action_vectors(obs.command, obs.previous, heading=heading)
        random_actions = torch.randint(actions.shape[1], (arena.n,), generator=rng)
        proposals = torch.where((draw > .96)[:,None], actions[torch.arange(arena.n),random_actions], proposals)
        if active.any():
            counts['teacher'] += int(((draw < config['oracle_mix']) & active).sum())
            counts['behavior'] += int(((draw >= config['oracle_mix']) & (draw <= .96) & active).sum())
            counts['exploration'] += int(((draw > .96) & active).sum())
            rows.append((grid[active], state[active], risk[active], choice[active]))
        arena.advance(clearance_guard(obs, proposals), t)
    if accounting is not None:
        accounting.update(counts)
    return tuple(torch.cat(items) for items in zip(*rows))


def train(config, data):
    torch.manual_seed(config['optimizer_seed'])
    rng = torch.Generator().manual_seed(config['optimizer_seed'])
    model = Policy(config['features'], config['objective'], config.get('threat_reference', False))
    optim = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    train_x, train_state, risk, choice = data
    counts = torch.bincount(choice, minlength=len(ACTIONS))
    weights = (counts.float().median().clamp_min(1) / counts.float().clamp_min(1)).sqrt().clamp(.50, 3.)
    stats, updates = [], 0
    for epoch in range(config['epochs']):
        guard()
        cumulative = 0.
        for idx in torch.randperm(len(choice), generator=rng).split(config['batch_size']):
            prediction = model(train_x[idx], train_state[idx])
            if model.objective == 'imitate':
                loss = F.cross_entropy(prediction, choice[idx], weight=weights)
            else:
                importance = 1 + config['hazard_weight'] * risk[idx]
                if model.objective == 'hybrid':
                    logits, prediction = prediction.chunk(2, dim=-1)
                loss = (F.binary_cross_entropy_with_logits(prediction, risk[idx], reduction='none') * importance).mean()
                if model.objective == 'hybrid':
                    loss += F.cross_entropy(logits, choice[idx], weight=weights)
            if not bool(loss.isfinite()):
                raise FloatingPointError('Nonfinite loss')
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3., error_if_nonfinite=True)
            optim.step()
            cumulative += loss.item()*len(idx)
            updates += 1
        stats.append(cumulative / len(choice))
    return model.eval(), dict(samples=len(choice), epochs=stats, updates=updates,
                              training_exposures=len(choice)*config['epochs'],
                              label_counts=counts.tolist(), risk_positive=int((risk>0).sum()))


@torch.inference_mode()
def evaluate(world, model):
    arena = Arena(**world)
    minimum = torch.full((arena.n,), torch.inf)
    tracking = torch.zeros(arena.n)
    release_position = None
    for step in range(STEPS):
        if step % 16 == 0:
            guard()
        t = step * DT
        obs = arena.observe(step)
        if model == 'operator':
            proposal = obs.command
        elif model == 'slow':
            proposal = .8*obs.command
        elif model == 'oracle':
            proposal, _, _ = arena.teacher(obs, t)
        else:
            proposal = model.act(obs)
        if not bool(proposal.isfinite().all()):
            raise FloatingPointError('Nonfinite policy action')
        if step == 55:
            release_position = arena.position.clone()
        clearance, _ = arena.advance(clearance_guard(obs, proposal), t)
        minimum = torch.minimum(minimum, clearance)
        if step < 55:
            tracking += (arena.velocity - arena.command(t)).square().sum(-1)
    dx = (arena.position - arena.start)[:,0]
    tracking = (tracking / 55).sqrt()
    drift = (arena.position - release_position).norm(dim=-1)
    moved = (arena.position - arena.start).norm(dim=-1)
    out = []
    for k,(scene,motion,rep) in enumerate(arena.codes):
        out.append(dict(scene=SCENES[scene], motion=MOTIONS[motion], replicate=rep,
                        collision=not bool(arena.live[k]), clearance_m=float(minimum[k]),
                        progress_m=float(dx[k]), tracking_rmse_mps=float(tracking[k]),
                        stop_drift_m=float(drift[k]), displacement_m=float(moved[k])))
    return out


def summarize(rows):
    groups = {'all': rows, 'transit': [r for r in rows if r['motion'] in MOTIONS[:4]],
              'safe_transit': [r for r in rows if r['motion'] in ('clear','opening')]}
    groups.update({name:[r for r in rows if r['motion']==name] for name in MOTIONS})
    groups.update({name:[r for r in rows if r['scene']==name] for name in SCENES})
    result = {}
    for name, group in groups.items():
        result[name] = dict(n=len(group), collisions=sum(r['collision'] for r in group),
                            minimum_m=min(r['clearance_m'] for r in group))
        for k in ('progress_m','tracking_rmse_mps','stop_drift_m','displacement_m','clearance_m'):
            result[name][k] = sum(r[k] for r in group)/len(group)
    return result


@torch.inference_mode()
def history_probe(model):
    arena = Arena(seed=251682, repeat=2, split='development', sensor=0.)
    arena.centers[:,0] = arena.position + torch.tensor([1.08,0.])
    # All three probes reach the exact same object, plant pose and operator input.
    outcomes = []
    for rate in (-.70, .0, .70):
        arena.motion_velocity[:,0] = torch.tensor([rate,0.])
        times = torch.tensor([-.3,-.2,-.1,0.]).reshape(1,4,1).repeat(arena.n,1,1)
        ranges, valid = (torch.stack(x,1) for x in zip(*(arena.scan(float(t)) for t in times[0,:,0])))
        yaw = arena.yaw(0.)
        cmd = rotate(arena.command(0.), -yaw)
        previous = torch.zeros_like(cmd)
        gyro = arena.yaw_rate[:,None,None].expand(-1,4,1)
        obs = Observation(ranges, valid, times, times, gyro, cmd, previous)
        result = clearance_guard(obs, model.act(obs))
        outcomes.append((ranges[:,-1].clone(), valid[:,-1].clone(), rotate(result,yaw)))
    if not all(torch.equal(r,outcomes[0][0]) and torch.equal(v,outcomes[0][1]) for r,v,_ in outcomes):
        raise RuntimeError('Unmatched latest measurement')
    moving = arena.motion < 4
    closing, still, opening = [v[moving,0] for _,_,v in outcomes]
    return dict(transit_pairs=int(moving.sum()), closing_mps=float(closing.mean()),
                still_mps=float(still.mean()), opening_mps=float(opening.mean()),
                opening_minus_closing_mps=float((opening-closing).mean()),
                differentiating_pairs=int(((opening-closing)>.05).sum()))


def baseline(path):
    record = load(check_pin(path['r2_summary']))
    return FrozenR6(check_pin(record['candidate']['checkpoint']))


def suite(config, path, confirm=False):
    arena = config['confirmation_world'] if confirm else config['development_world']
    if not path.exists():
        path.mkdir()
    result = {}
    for name, model in [('operator','operator'),('slow','slow'),('oracle','oracle'),('r2_projected',baseline(config))]:
        rows = evaluate(arena, model)
        result[name] = dict(summary=summarize(rows), rows=rows)
        print(name, result[name]['summary']['all'], flush=True)
    return save(path / 'RESULTS.json', result)


def run(path):
    assert_capacity()
    config = load(path)
    for item in (*config['source_files'], config['authority']):
        check_pin(item)
    if config['predecessor']:
        check_pin(config['predecessor'])
    head = subprocess.check_output(['git','-C',str(SNAPSHOT),'rev-parse','HEAD'],text=True).strip()
    if head != config['source_commit'] or SNAPSHOT == ROOT:
        raise RuntimeError('Formal run requires the exact detached snapshot')
    out = RUNS / config['run_id']
    out.mkdir()
    save(out/'INPUTS.json', dict(config=pin(path), torch=torch.__version__, device='cpu'))
    prior = load_model(config['behavior']) if config['behavior'] else None
    accounting = {}
    dataset = collect(config, prior, accounting=accounting)
    with (out/'DATA.pt').open('xb') as f:
        torch.save(dataset, f)
    model, training = train(config, dataset)
    if sum(accounting.values()) != training['samples']:
        raise ValueError('Trajectory source accounting disagrees with training samples')
    training['collection_actions'] = accounting
    assert_capacity()
    checkpoint = save_model(out/'POLICY.pt', model)
    cases = evaluate(config['development_world'], model)
    result = dict(status='COMPLETE', config=pin(path), data=pin(out/'DATA.pt'),
                  checkpoint=checkpoint, training=training, development=dict(summary=summarize(cases), rows=cases),
                  matched_history=history_probe(model))
    value = save(out/'RESULTS.json', result)
    print(json.dumps(dict(result=value, training=training, summary=result['development']['summary'],
                          matched_history=result['matched_history']), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('baselines', 'confirm-baselines','run','confirm'))
    parser.add_argument('config', type=Path)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    assert_capacity()
    torch.set_num_threads(4)
    if args.step == 'run':
        run(args.config)
    elif args.step in ('baselines','confirm-baselines'):
        print(suite(load(args.config), BASELINE_FOLDER if args.step=='baselines'
                    else RUNS/'INSTINCT-DYNAMIC-CONFIRM-BASELINE', confirm=args.step!='baselines'))
    else:
        config = load(args.config)
        model = load_model(load(args.model)['checkpoint'])
        rows = evaluate(config['confirmation_world'], model)
        print(save(RUNS / config['run_id'] / 'CONFIRMATION.json',
                   dict(config=pin(args.config), selected=pin(args.model), summary=summarize(rows), rows=rows,
                        matched_history=history_probe(model))))


if __name__ == '__main__':
    main()
