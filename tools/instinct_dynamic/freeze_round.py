"""Freeze one CPU dynamic research round and its causal predecessor."""
import argparse
from pathlib import Path
import subprocess

from tools.r6_direct.import_legacy import ROOT, pin
from tools.r6_direct.instinct_cycle import save, load
from tools.r6_direct.instinct_cycle_finish import freeze
from .study import SNAPSHOT, PROJECT, RUNS, R2, DEVELOPMENT, CONFIRMATION, assert_capacity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('round', type=int, choices=range(4, 9))
    p.add_argument('--objective', choices=['imitate','risk','hybrid'], required=True)
    p.add_argument('--features', choices=['causal','raw'], default='causal')
    p.add_argument('--behavior', type=int)
    p.add_argument('--sensor', type=float, default=1.)
    p.add_argument('--mix', type=float, default=.78)
    p.add_argument('--repeat', type=int, default=12)
    p.add_argument('--epochs', type=int, default=16)
    p.add_argument('--hazard', type=float, default=4.)
    p.add_argument('--horizon', type=float, default=1.5)
    p.add_argument('--hypothesis', required=True)
    args = p.parse_args()
    if not 0. <= args.mix <= .96 or args.repeat <= 0 or args.epochs <= 0:
        p.error('Collection mixture and training counts are out of range')
    assert_capacity()
    if SNAPSHOT == ROOT or subprocess.check_output(['git','branch','--show-current'],text=True).strip():
        raise RuntimeError('Detached execution snapshot required')
    source = sorted((SNAPSHOT/'tools/instinct_dynamic').glob('*.py'))
    previous = None
    if args.round > 4:
        previous = pin(RUNS / f'INSTINCT-DYNAMIC-R{args.round-1}' / 'RESULTS.json')
        if load(Path(previous['path']))['status'] != 'COMPLETE':
            raise ValueError('Preceding round incomplete')
    behavior = load(RUNS/f'INSTINCT-DYNAMIC-R{args.behavior}'/'RESULTS.json')['checkpoint'] if args.behavior else None
    value = dict(schema='instinct-dynamic-study/1', run_id=f'INSTINCT-DYNAMIC-R{args.round}',
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 source_files=[pin(p) for p in source], authority=pin(PROJECT/'PLAN.md'),
                 predecessor=previous, r2_summary=pin(R2), hypothesis=args.hypothesis, features=args.features,
                 objective=args.objective, behavior=behavior, learning_rate=.0012, optimizer_seed=43102,
                 train_world=dict(seed=920171,repeat=args.repeat,split='train',sensor=args.sensor),
                 oracle_horizon=args.horizon, oracle_mix=args.mix, hazard_weight=args.hazard,
                 batch_size=192, epochs=args.epochs, development_world=DEVELOPMENT, confirmation_world=CONFIRMATION)
    dest = PROJECT/f'ROUND-{args.round}.json'
    print(save(dest,value))
    freeze(f'instinct-dynamic-r{args.round}-inputs', f'Freeze dynamic training round {args.round}', dest)


if __name__=='__main__':
    main()
